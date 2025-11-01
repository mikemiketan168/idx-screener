#!/usr/bin/env python3
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="IDX Screener — Fast", page_icon="🚀", layout="wide")

# ---------- UI THEME ----------
st.markdown("""
<style>
.big {font-size:1.8rem;font-weight:700}
.tag {display:inline-block;padding:.2rem .6rem;border-radius:.6rem;background:#eee;margin-right:.4rem}
.buy {background:#d7ffd9}
.sell{background:#ffd7d7}
.warn{background:#fff2cc}
</style>
""", unsafe_allow_html=True)

# ---------- GLOBAL SESSION ----------
# Setup session yang persistent untuk bypass mobile restrictions
@st.cache_resource
def get_session():
    """Create persistent session with proper headers"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    })
    return session

# ---------- HELPERS ----------
def safe_load_tickers(path="idx_stocks.json"):
    """Load tickers dengan fallback ke default"""
    DEFAULT_TICKERS = [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", 
        "ASII.JK", "UNVR.JK", "GOTO.JK", "BREN.JK",
        "AMMN.JK", "BBNI.JK", "ANTM.JK", "INCO.JK",
        "PTBA.JK", "ADRO.JK", "ITMG.JK", "INDF.JK",
        "ICBP.JK", "KLBF.JK", "MAPI.JK", "SMGR.JK"
    ]
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            tickers = data.get("tickers", []) or data.get("stocks", [])
        elif isinstance(data, list):
            tickers = data
        else:
            return DEFAULT_TICKERS
            
        tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
        tickers = [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
        result = sorted(list(dict.fromkeys(tickers))) if tickers else DEFAULT_TICKERS
        return result
        
    except FileNotFoundError:
        st.info("📌 File idx_stocks.json tidak ditemukan. Menggunakan 20 saham blue chip IDX.")
        return DEFAULT_TICKERS
    except Exception as e:
        st.warning(f"⚠️ Error load JSON: {e}. Menggunakan default tickers.")
        return DEFAULT_TICKERS

@st.cache_data(ttl=600, show_spinner=False)
def fetch_one(ticker: str, period="6mo", interval="1d", max_retries=3):
    """Fetch data dengan multiple fallback methods untuk mobile compatibility"""
    session = get_session()
    
    for attempt in range(max_retries):
        try:
            time.sleep(0.5 + (attempt * 0.5))  # Progressive delay
            
            # METHOD 1: Try yf.Ticker with session
            try:
                stock = yf.Ticker(ticker, session=session)
                df = stock.history(period=period, interval=interval, auto_adjust=False, timeout=20)
                if df is not None and not df.empty and len(df) >= 50:
                    # Success with method 1
                    pass
                else:
                    raise ValueError("Empty dataframe from Ticker")
            except Exception as e1:
                # METHOD 2: Fallback to yf.download
                if attempt < max_retries - 1:
                    time.sleep(1)
                    df = yf.download(
                        ticker, 
                        period=period, 
                        interval=interval, 
                        auto_adjust=False, 
                        progress=False, 
                        threads=False,
                        timeout=20
                    )
                else:
                    return None
            
            # Validasi data
            if df is None or df.empty:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            
            # Normalize column names
            df.columns = [col.capitalize() if isinstance(col, str) else col for col in df.columns]
            
            # Ensure columns exist
            need = {"Open","High","Low","Close","Volume"}
            if not need.issubset(set(df.columns)):
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            
            # Drop NaN rows
            df = df.dropna(subset=["Open","High","Low","Close"])
            if len(df) < 50:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            
            # Calculate indicators
            try:
                df["EMA9"]  = df["Close"].ewm(span=9,  adjust=False).mean()
                df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
                df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
                
                delta = df["Close"].diff()
                gain = np.where(delta>0, delta, 0.0)
                loss = np.where(delta<0, -delta, 0.0)
                roll = 14
                avg_gain = pd.Series(gain, index=df.index).rolling(roll, min_periods=roll).mean()
                avg_loss = pd.Series(loss, index=df.index).rolling(roll, min_periods=roll).mean()
                rs = avg_gain / (avg_loss.replace(0, np.nan))
                df["RSI"] = 100 - (100 / (1 + rs))
                
                # MACD
                exp1 = df["Close"].ewm(span=12, adjust=False).mean()
                exp2 = df["Close"].ewm(span=26, adjust=False).mean()
                df["MACD"] = exp1 - exp2
                df["SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                return None
            
            return df
            
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(2 + attempt)
    
    return None

def score_row(df: pd.DataFrame):
    """Return (score:int, details:dict) from latest row"""
    try:
        row = df.iloc[-1]
    except Exception:
        return 0, {}
    
    score, det = 0, {}

    def add(ok, key, pts):
        nonlocal score
        if ok: 
            score += pts
            det[key]=f"✅ +{pts}"
        else: 
            det[key]="❌ +0"

    add(row["Close"] > row["EMA9"],  "Price>EMA9", 10)
    add(row["Close"] > row["EMA21"], "Price>EMA21",10)
    add(row["Close"] > row["EMA50"], "Price>EMA50",10)
    add(row["EMA9"] > row["EMA21"] > row["EMA50"], "EMA Align", 10)

    rsi = float(row.get("RSI", np.nan))
    if 40 <= rsi <= 70:  
        score += 20
        det[f"RSI({rsi:.1f})"]="✅ +20"
    elif 30 < rsi < 40 or 70 < rsi < 80: 
        score += 10
        det[f"RSI({rsi:.1f})"]="⚠️ +10"
    else: 
        det[f"RSI({rsi if not np.isnan(rsi) else 0:.1f})"]="❌ +0"

    add(row["MACD"] > row["SIGNAL"], "MACD>Signal", 20)

    # Volume pop (20d)
    avg_vol = df["Volume"].tail(20).mean()
    add(row["Volume"] > 1.2*avg_vol if pd.notna(avg_vol) else False, "Vol>20d", 10)

    # Weekly change (approx 5 bars)
    try:
        base = df["Close"].iloc[-5]
        wc = (row["Close"]-base)/base*100
        add(2 <= wc <= 20, f"WkChg({wc:.1f}%)", 10)
    except Exception:
        det["WkChg"]="❌ +0"

    return int(score), det

def signal_from_score(score:int):
    """Convert score to signal"""
    if score >= 70:   return "STRONG BUY", "buy"
    if score >= 60:   return "BUY", "buy"
    if score >= 40:   return "NEUTRAL", "warn"
    return "SELL", "sell"

def strategy_levels(price: float, sig: str):
    """Calculate entry, TP, SL levels"""
    if "BUY" not in sig:
        return {"ideal":None, "aggr":None}
    
    ie = round(price*0.97, 2)
    it1= round(ie*1.10, 2)
    it2= round(ie*1.15, 2)
    isl= round(ie*0.93, 2)
    ae = round(price, 2)
    at1= round(ae*1.08, 2)
    at2= round(ae*1.12, 2)
    asl= round(ae*0.93, 2)
    
    return {
        "ideal":{"entry":ie,"tp1":it1,"tp2":it2,"sl":isl},
        "aggr":{"entry":ae,"tp1":at1,"tp2":at2,"sl":asl}
    }

def analyze_one(ticker, period="6mo"):
    """Analyze single ticker"""
    df = fetch_one(ticker, period=period, interval="1d")
    if df is None: 
        return None
    
    score, _ = score_row(df)
    sig, tag = signal_from_score(score)
    price = float(df["Close"].iloc[-1])
    strat = strategy_levels(price, sig)
    
    return {
        "Ticker": ticker,
        "Price": price,
        "Score": score,
        "Signal": sig,
        "Tag": tag,
        "EntryIdeal": strat["ideal"]["entry"] if strat["ideal"] else None,
        "TP1": strat["ideal"]["tp1"] if strat["ideal"] else None,
        "TP2": strat["ideal"]["tp2"] if strat["ideal"] else None,
        "SL":  strat["ideal"]["sl"]  if strat["ideal"] else None
    }

def batch_scan(tickers, max_workers=2, period="6mo", limit=None, sleep_between=0.5):
    """Batch scan with progress bar - reduced workers for mobile"""
    results = []
    if limit: 
        tickers = tickers[:limit]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(tickers)
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(analyze_one, t, period): t for t in tickers}
        for i, f in enumerate(as_completed(fut)):
            progress_bar.progress((i + 1) / total)
            status_text.text(f"📊 Scanning... {i+1}/{total}")
            t = fut[f]
            try:
                r = f.result(timeout=30)
                if r: 
                    results.append(r)
                    status_text.text(f"✅ {t}: Score {r['Score']}")
            except Exception as e:
                status_text.text(f"⚠️ Skip {t}")
            
            if sleep_between: 
                time.sleep(sleep_between)
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results).sort_values(["Score","Price"], ascending=[False,True])
    return df

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.radio("Mode", ["🔍 Screener","📈 Single Stock"], index=0)
    
    # Detect mobile
    is_mobile = st.checkbox("📱 Mobile Mode (slower, more stable)", value=False, 
                            help="Aktifkan jika pakai Streamlit Mobile App")
    
    min_score = st.slider("Min Score", 0, 100, 60, step=5)
    top_n = st.slider("Top N (show)", 5, 100, 30, step=5)
    
    if is_mobile:
        scan_cap = st.slider("Max tickers", 5, 50, 20, step=5,
                            help="Mode mobile: max 50 ticker")
    else:
        scan_cap = st.slider("Max tickers", 10, 200, 50, step=10,
                            help="Mulai dari 50")
    
    period = st.selectbox("Period", ["3mo","6mo","1y"], index=1)
    
    st.markdown("---")
    st.markdown("### 📊 Info")
    st.caption("Built with yfinance")
    st.caption("Data delay ~15 menit")
    if is_mobile:
        st.warning("📱 Mobile mode: proses lebih lama tapi lebih stabil")

st.markdown('<div class="big">🚀 IDX Screener — Fast</div>', unsafe_allow_html=True)

# ---------- MAIN ----------
tickers = safe_load_tickers()
if not tickers:
    st.error("❌ Ticker list kosong")
    st.stop()

st.info(f"📌 Loaded {len(tickers)} tickers")

if mode == "📈 Single Stock":
    sel = st.selectbox("Pilih saham", tickers, 
                      index=tickers.index("TLKM.JK") if "TLKM.JK" in tickers else 0)
    
    if st.button("🔍 Analyze", type="primary"):
        status = st.empty()
        
        # Try dengan multiple retries dan feedback
        df = None
        for retry in range(3):
            status.info(f"🔄 Mengambil data {sel} (attempt {retry+1}/3)...")
            df = fetch_one(sel, period=period, interval="1d")
            if df is not None:
                status.success("✅ Data berhasil diambil!")
                time.sleep(0.5)
                status.empty()
                break
            if retry < 2:
                status.warning(f"⚠️ Gagal, retry {retry+2}/3...")
                time.sleep(2)
        
        if df is None or df.empty:
            st.error("❌ Data tidak tersedia setelah 3x percobaan.")
            st.info("💡 Tips:\n- Coba saham lain (BBCA.JK, BBRI.JK)\n- Aktifkan '📱 Mobile Mode' di sidebar\n- Gunakan period 3mo\n- Coba lagi 1-2 menit")
            st.stop()
        
        score, det = score_row(df)
        sig, tag = signal_from_score(score)
        price = float(df["Close"].iloc[-1])
        strat = strategy_levels(price, sig)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Price", f"Rp {price:,.0f}")
        c2.metric("Score", f"{score}/100")
        c3.markdown(f'<span class="tag {tag}">{sig}</span>', unsafe_allow_html=True)
        c4.metric("Period", period)

        st.line_chart(df[["Close","EMA9","EMA21","EMA50"]])

        if "BUY" in sig:
            st.success(f"🎯 **Ideal Entry:** Rp {strat['ideal']['entry']:,.0f} | **TP1** {strat['ideal']['tp1']:,.0f} | **TP2** {strat['ideal']['tp2']:,.0f} | **SL** {strat['ideal']['sl']:,.0f}")
        else:
            st.info("⚠️ Belum BUY signal")

        with st.expander("📋 Score Breakdown"):
            st.table(pd.DataFrame.from_dict(det, orient="index", columns=["Status"]))

else:
    # SCREENER MODE
    if st.button("🚀 Run Screener", type="primary"):
        workers = 2 if is_mobile else 3
        sleep = 0.7 if is_mobile else 0.5
        
        with st.spinner(f"Scanning {min(len(tickers), scan_cap)} tickers..."):
            df = batch_scan(tickers, max_workers=workers, period=period, 
                          limit=scan_cap, sleep_between=sleep)
        
        if df.empty:
            st.warning("⚠️ Tidak ada data valid\n\n**Solusi:**\n- Aktifkan 📱 Mobile Mode\n- Kurangi Max tickers jadi 10-20\n- Coba lagi dalam 1-2 menit")
            st.stop()
        
        df = df[df["Score"]>=min_score].head(top_n)
        
        if df.empty:
            st.info(f"📊 Tidak ada saham Score >= {min_score}\n\n**Tips:** Turunkan Min Score atau perbanyak ticker")
            st.stop()
        
        st.success(f"✅ Found {len(df)} stocks!")
        
        show = df[["Ticker","Price","Score","Signal","EntryIdeal","TP1","TP2","SL"]]
        st.dataframe(show, use_container_width=True, height=520)
        
        csv = show.to_csv(index=False).encode()
        st.download_button(
            "📥 Download CSV", 
            data=csv, 
            file_name=f"idx_screener_{time.strftime('%Y%m%d')}.csv", 
            mime="text/csv"
        )
        
        st.markdown("### 🏆 Top 3")
        for i, row in show.head(3).iterrows():
            with st.expander(f"{row['Ticker']} - {row['Score']} ({row['Signal']})"):
                st.markdown(f"""
                - **Price:** Rp {row['Price']:,.0f}
                - **Entry:** Rp {row['EntryIdeal']:,.0f}
                - **TP1:** Rp {row['TP1']:,.0f}
                - **TP2:** Rp {row['TP2']:,.0f}
                - **SL:** Rp {row['SL']:,.0f}
                """)