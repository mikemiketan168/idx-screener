#!/usr/bin/env python3
import streamlit as st
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

# ---------- DETECT MOBILE APP ----------
def is_mobile_app():
    """Detect if running in Streamlit Mobile App"""
    try:
        # Mobile app biasanya tidak punya certain features
        import sys
        # Check if streamlit is running in mobile context
        return 'streamlit.runtime.scriptrunner' not in sys.modules
    except:
        return False

# ---------- ALTERNATIVE DATA FETCHER ----------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_yahoo_api(ticker: str, period="6mo"):
    """Fetch data langsung dari Yahoo Finance API tanpa yfinance library"""
    try:
        # Remove .JK for API
        symbol = ticker.replace(".JK", "")
        
        # Calculate timestamps
        import datetime
        end = int(datetime.datetime.now().timestamp())
        if period == "3mo":
            start = end - (90 * 24 * 3600)
        elif period == "1y":
            start = end - (365 * 24 * 3600)
        else:  # 6mo default
            start = end - (180 * 24 * 3600)
        
        # Yahoo Finance API endpoint
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "period1": start,
            "period2": end,
            "interval": "1d",
            "events": "history"
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        # Parse response
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': quotes['open'],
            'High': quotes['high'],
            'Low': quotes['low'],
            'Close': quotes['close'],
            'Volume': quotes['volume']
        }, index=pd.to_datetime(timestamps, unit='s'))
        
        # Clean data
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if len(df) < 50:
            return None
        
        # Calculate indicators
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
        
        return df
        
    except Exception as e:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_with_yfinance(ticker: str, period="6mo"):
    """Fallback menggunakan yfinance - untuk web browser"""
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d", auto_adjust=False)
        
        if df is None or df.empty or len(df) < 50:
            return None
        
        # Normalize columns
        df.columns = [col.capitalize() if isinstance(col, str) else col for col in df.columns]
        
        # Calculate indicators
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
        
        return df
    except:
        return None

def fetch_one(ticker: str, period="6mo", use_api=False):
    """Smart fetcher - pilih metode terbaik"""
    if use_api:
        # Try Yahoo API first (better for mobile)
        df = fetch_yahoo_api(ticker, period)
        if df is not None:
            return df
        time.sleep(1)
    
    # Fallback to yfinance
    return fetch_with_yfinance(ticker, period)

# ---------- HELPERS ----------
def safe_load_tickers(path="idx_stocks.json"):
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
        return sorted(list(dict.fromkeys(tickers))) if tickers else DEFAULT_TICKERS
    except:
        return DEFAULT_TICKERS

def score_row(df: pd.DataFrame):
    try:
        row = df.iloc[-1]
    except:
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

    avg_vol = df["Volume"].tail(20).mean()
    add(row["Volume"] > 1.2*avg_vol if pd.notna(avg_vol) else False, "Vol>20d", 10)

    try:
        base = df["Close"].iloc[-5]
        wc = (row["Close"]-base)/base*100
        add(2 <= wc <= 20, f"WkChg({wc:.1f}%)", 10)
    except:
        det["WkChg"]="❌ +0"

    return int(score), det

def signal_from_score(score:int):
    if score >= 70:   return "STRONG BUY", "buy"
    if score >= 60:   return "BUY", "buy"
    if score >= 40:   return "NEUTRAL", "warn"
    return "SELL", "sell"

def strategy_levels(price: float, sig: str):
    if "BUY" not in sig:
        return {"ideal":None, "aggr":None}
    
    ie = round(price*0.97, 2)
    it1= round(ie*1.10, 2)
    it2= round(ie*1.15, 2)
    isl= round(ie*0.93, 2)
    
    return {
        "ideal":{"entry":ie,"tp1":it1,"tp2":it2,"sl":isl},
        "aggr":{"entry":None,"tp1":None,"tp2":None,"sl":None}
    }

def analyze_one(ticker, period="6mo", use_api=False):
    df = fetch_one(ticker, period=period, use_api=use_api)
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

def batch_scan(tickers, period="6mo", limit=None, use_api=False):
    results = []
    if limit: 
        tickers = tickers[:limit]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / total)
        status_text.text(f"📊 {i+1}/{total}: {ticker}")
        
        try:
            r = analyze_one(ticker, period, use_api)
            if r:
                results.append(r)
                status_text.text(f"✅ {ticker}: Score {r['Score']}")
        except:
            status_text.text(f"⚠️ Skip {ticker}")
        
        time.sleep(0.8)  # Rate limit protection
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results).sort_values(["Score","Price"], ascending=[False,True])

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    use_api = st.checkbox("🔧 Use Direct API (for Mobile)", value=True,
                          help="Aktifkan jika di mobile app gagal terus")
    
    mode = st.radio("Mode", ["🔍 Screener","📈 Single Stock"], index=1)
    min_score = st.slider("Min Score", 0, 100, 60, step=5)
    top_n = st.slider("Top N", 5, 50, 20, step=5)
    scan_cap = st.slider("Max tickers", 5, 100, 30, step=5)
    period = st.selectbox("Period", ["3mo","6mo","1y"], index=1)
    
    st.markdown("---")
    st.caption("💡 Jika error, centang 'Use Direct API'")

st.markdown('<div class="big">🚀 IDX Screener — Fast</div>', unsafe_allow_html=True)

# ---------- MAIN ----------
tickers = safe_load_tickers()
st.info(f"📌 {len(tickers)} tickers loaded")

if mode == "📈 Single Stock":
    sel = st.selectbox("Pilih saham", tickers, 
                      index=tickers.index("TLKM.JK") if "TLKM.JK" in tickers else 0)
    
    if st.button("🔍 Analyze", type="primary"):
        status = st.empty()
        
        df = None
        for retry in range(3):
            method = "API" if use_api else "yfinance"
            status.info(f"🔄 Attempt {retry+1}/3 ({method})...")
            
            df = fetch_one(sel, period=period, use_api=use_api)
            
            if df is not None:
                status.success("✅ Data loaded!")
                time.sleep(0.3)
                status.empty()
                break
            
            if retry < 2:
                status.warning(f"⚠️ Retry...")
                time.sleep(2)
        
        if df is None:
            st.error("❌ Gagal 3x percobaan")
            st.info("💡 **Solusi:**\n1. Centang '🔧 Use Direct API'\n2. Pilih period '3mo'\n3. Coba saham lain (BBCA.JK)\n4. Buka via browser: idx-screener-auto.streamlit.app")
            st.stop()
        
        score, det = score_row(df)
        sig, tag = signal_from_score(score)
        price = float(df["Close"].iloc[-1])
        strat = strategy_levels(price, sig)

        c1,c2,c3 = st.columns(3)
        c1.metric("Price", f"Rp {price:,.0f}")
        c2.metric("Score", f"{score}/100")
        c3.markdown(f'<span class="tag {tag}">{sig}</span>', unsafe_allow_html=True)

        st.line_chart(df[["Close","EMA9","EMA21","EMA50"]])

        if "BUY" in sig:
            st.success(f"🎯 Entry: Rp {strat['ideal']['entry']:,.0f} | TP1: {strat['ideal']['tp1']:,.0f} | TP2: {strat['ideal']['tp2']:,.0f} | SL: {strat['ideal']['sl']:,.0f}")
        else:
            st.info("⚠️ Wait for better setup")

        with st.expander("📋 Details"):
            st.table(pd.DataFrame.from_dict(det, orient="index", columns=["Status"]))

else:
    if st.button("🚀 Run Screener", type="primary"):
        df = batch_scan(tickers, period=period, limit=scan_cap, use_api=use_api)
        
        if df.empty:
            st.warning("⚠️ No data. Try:\n- Enable 'Use Direct API'\n- Reduce Max tickers to 10\n- Use browser instead")
            st.stop()
        
        df = df[df["Score"]>=min_score].head(top_n)
        
        if df.empty:
            st.info(f"No stocks with Score >= {min_score}")
            st.stop()
        
        st.success(f"✅ {len(df)} stocks found!")
        
        show = df[["Ticker","Price","Score","Signal","EntryIdeal","TP1","TP2","SL"]]
        st.dataframe(show, use_container_width=True)
        
        csv = show.to_csv(index=False).encode()
        st.download_button("📥 CSV", csv, f"screener_{time.strftime('%Y%m%d')}.csv")