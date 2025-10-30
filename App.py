#!/usr/bin/env python3
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
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

# ---------- HELPERS ----------
def safe_load_tickers(path="idx_stocks.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            tickers = data.get("tickers", []) or data.get("stocks", [])
        elif isinstance(data, list):
            tickers = data
        else:
            tickers = []
        tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
        tickers = [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
        return sorted(list(dict.fromkeys(tickers)))
    except Exception as e:
        st.error(f"JSON load error: {e}")
        return []

@st.cache_data(ttl=900)
def fetch_one(ticker: str, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None
        # ensure columns exist
        need = {"Open","High","Low","Close","Volume"}
        if not need.issubset(set(df.columns)):
            return None
        # drop NaN rows
        df = df.dropna(subset=["Open","High","Low","Close"])
        if len(df) < 50:
            return None
        # indicators (fast)
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
    except Exception:
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
        if ok: score += pts; det[key]=f"✅ +{pts}"
        else: det[key]="❌ +0"

    add(row["Close"] > row["EMA9"],  "Price>EMA9", 10)
    add(row["Close"] > row["EMA21"], "Price>EMA21",10)
    add(row["Close"] > row["EMA50"], "Price>EMA50",10)
    add(row["EMA9"] > row["EMA21"] > row["EMA50"], "EMA Align", 10)

    rsi = float(row.get("RSI", np.nan))
    if 40 <= rsi <= 70:  score += 20; det[f"RSI({rsi:.1f})"]="✅ +20"
    elif 30 < rsi < 40 or 70 < rsi < 80: score += 10; det[f"RSI({rsi:.1f})"]="⚠️ +10"
    else: det[f"RSI({rsi if not np.isnan(rsi) else 0:.1f})"]="❌ +0"

    add(row["MACD"] > row["SIGNAL"], "MACD>Signal", 20)

    # volume pop (20d)
    avg_vol = df["Volume"].tail(20).mean()
    add(row["Volume"] > 1.2*avg_vol if pd.notna(avg_vol) else False, "Vol>20d", 10)

    # weekly change (approx 5 bars)
    try:
        base = df["Close"].iloc[-5]
        wc = (row["Close"]-base)/base*100
        add(2 <= wc <= 20, f"WkChg({wc:.1f}%)", 10)
    except Exception:
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
    ie = round(price*0.97, 2)     # wait -3%
    it1= round(ie*1.10, 2)        # +10%
    it2= round(ie*1.15, 2)        # +15%
    isl= round(ie*0.93, 2)        # -7%
    ae = round(price, 2)          # now
    at1= round(ae*1.08, 2)        # +8%
    at2= round(ae*1.12, 2)        # +12%
    asl= round(ae*0.93, 2)        # -7%
    return {
        "ideal":{"entry":ie,"tp1":it1,"tp2":it2,"sl":isl},
        "aggr":{"entry":ae,"tp1":at1,"tp2":at2,"sl":asl}
    }

def analyze_one(ticker, period="6mo"):
    df = fetch_one(ticker, period=period, interval="1d")
    if df is None: return None
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

def batch_scan(tickers, max_workers=5, period="6mo", limit=None, sleep_between=0.15):
    results = []
    if limit: tickers = tickers[:limit]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(analyze_one, t, period): t for t in tickers}
        for i, f in enumerate(as_completed(fut)):
            t = fut[f]
            try:
                r = f.result()
                if r: results.append(r)
            except Exception:
                pass
            if sleep_between: time.sleep(sleep_between)  # rate-limit friendly
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values(["Score","Price"], ascending=[False,True])
    return df

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.radio("Mode", ["🔍 Screener","📈 Single Stock"], index=0)
    min_score = st.slider("Min Score", 0, 100, 60, step=5)
    top_n = st.slider("Top N (show)", 5, 100, 30, step=5)
    scan_cap = st.slider("Max tickers to scan (for speed)", 50, 900, 200, step=50,
                         help="Naikkan pelan2 kalau server kuat. 200 aman di Streamlit free.")
    period = st.selectbox("Period", ["3mo","6mo","1y"], index=1)

st.markdown('<div class="big">🚀 IDX Screener — Fast</div>', unsafe_allow_html=True)

# ---------- MAIN ----------
tickers = safe_load_tickers()
if not tickers:
    st.error("Ticker list kosong. Edit 'idx_stocks.json' dulu.")
    st.stop()

if mode == "📈 Single Stock":
    sel = st.selectbox("Pilih saham", tickers, index=tickers.index("TLKM.JK") if "TLKM.JK" in tickers else 0)
    if st.button("Analyze"):
        with st.spinner(f"Ambil data {sel} ..."):
            df = fetch_one(sel, period=period, interval="1d")
        if df is None or df.empty:
            st.warning("Data tidak tersedia / gagal fetch.")
            st.stop()
        score, det = score_row(df)
        sig, tag = signal_from_score(score)
        price = float(df["Close"].iloc[-1])
        strat = strategy_levels(price, sig)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Price", f"{price:,.0f}")
        c2.metric("Score", f"{score}/100")
        c3.markdown(f'<span class="tag {tag}">{sig}</span>', unsafe_allow_html=True)
        c4.metric("Period", period)

        st.line_chart(df[["Close","EMA9","EMA21","EMA50"]])

        if "BUY" in sig:
            st.success(f"🎯 Ideal Entry: {strat['ideal']['entry']:.0f} | TP1 {strat['ideal']['tp1']:.0f} | TP2 {strat['ideal']['tp2']:.0f} | SL {strat['ideal']['sl']:.0f}")
        else:
            st.info("⚠️ Belum BUY — tunggu setup lebih bagus")

        with st.expander("Score breakdown"):
            st.table(pd.DataFrame.from_dict(det, orient="index", columns=["Status"]))

else:
    if st.button("🚀 Screener Now!"):
        with st.spinner(f"Scanning {min(len(tickers), scan_cap)} tickers..."):
            df = batch_scan(tickers, max_workers=5, period=period, limit=scan_cap)
        if df.empty:
            st.warning("Tidak ada data valid (API limit / koneksi). Kecilkan 'Max tickers' lalu klik lagi.")
            st.stop()
        df = df[df["Score"]>=min_score].head(top_n)
        if df.empty:
            st.info("Tidak ada yang melewati ambang Score. Turunkan 'Min Score' atau perpanjang period.")
            st.stop()
        show = df[["Ticker","Price","Score","Signal","EntryIdeal","TP1","TP2","SL"]]
        st.dataframe(show, use_container_width=True, height=520)
        csv = show.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", data=csv, file_name="idx_screener_fast.csv", mime="text/csv")
