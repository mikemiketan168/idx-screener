#!/usr/bin/env python3
import math, json, re, time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

import streamlit as st
import plotly.graph_objects as go

# =========== UI / THEME ===========
st.set_page_config(page_title="IDX Screener Ultimate", layout="wide", page_icon="🚀")
st.markdown("""
<style>
.main-header {font-size: 3rem; font-weight: 800; letter-spacing: .02em;}
.good {background: linear-gradient(90deg,#22c55e33,#22c55e11); padding:.2rem .5rem; border-radius:.5rem}
.warn {background: linear-gradient(90deg,#eab30833,#eab30811); padding:.2rem .5rem; border-radius:.5rem}
.bad  {background: linear-gradient(90deg,#ef444433,#ef444411); padding:.2rem .5rem; border-radius:.5rem}
.dataframe {font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚀 IDX SCREENER ULTIMATE</div>', unsafe_allow_html=True)

# =========== SETTINGS ===========
DEFAULT_PERIOD   = "6mo"
DEFAULT_INTERVAL = "1d"
TOP_N            = 50           # banyaknya saham ditampilkan di screener
CACHE_TTL_SEC    = 300          # 5 menit

# =========== LOAD TICKERS ===========
@st.cache_data(ttl=CACHE_TTL_SEC)
def load_tickers() -> list[str]:
    """
    Gabungkan dari beberapa file, perbaiki suffix .JK, uppercase, unik & sort.
    File opsional: idx_stocks.json, idx_stocks_extra.json, idx_stocks.txt
    """
    tickers: list[str] = []

    def pull_from_json(name):
        try:
            with open(name, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [str(x) for x in data]
        except Exception:
            pass
        return []

    def pull_from_txt(name):
        try:
            with open(name, "r") as f:
                return [ln.strip() for ln in f if ln.strip()]
        except Exception:
            return []

    tickers += pull_from_json("idx_stocks.json")
    tickers += pull_from_json("idx_stocks_extra.json")
    tickers += pull_from_txt("idx_stocks.txt")

    # normalisasi
    fixed = []
    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue
        if not t.endswith(".JK"):
            # tambahkan .JK jika tidak ada, tapi hanya jika alfanumerik
            base = re.sub(r"[^A-Z0-9]", "", t)
            t = (base + ".JK") if not t.endswith(".JK") else t
            if not t.endswith(".JK"):
                t = base + ".JK"
        fixed.append(t)

    # unik + sort
    uniq = sorted(list(dict.fromkeys(fixed)))
    return uniq

# =========== DATA FETCH ===========
@st.cache_data(ttl=CACHE_TTL_SEC)
def fetch_history(ticker: str, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.dropna()
        return df
    except Exception:
        return pd.DataFrame()

# =========== INDICATORS ===========
def ema(series: pd.Series, length: int):
    if series is None or series.empty:
        return pd.Series(dtype=float)
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def atr(df: pd.DataFrame, length: int = 14):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

def compute_indicators(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    df = df.copy()
    df["EMA9"]  = ema(df["Close"], 9)
    df["EMA21"] = ema(df["Close"], 21)
    df["EMA50"] = ema(df["Close"], 50)
    df["ATR14"] = atr(df, 14)
    return df

# =========== SIGNAL ENGINE ===========
def score_signal_row(row) -> tuple[int, str, float, float, float]:
    """
    Return: (score(0-100), signal, prob, entry_ideal, entry_aggr)
    """
    c, e9, e21, e50, atr = row["Close"], row["EMA9"], row["EMA21"], row["EMA50"], row["ATR14"]

    # guard NaN
    if any(map(lambda x: (x is None) or (isinstance(x, float) and math.isnan(x)), [c, e9, e21, e50])):
        return (0, "WAIT", 0.5, float("nan"), float("nan"))

    # basic structure
    up   = (c > e9 > e21 > e50)
    bull = (c > e21 > e50)
    down = (c < e9 < e21 < e50)

    score = 50
    if up:   score += 40
    elif bull: score += 20
    if c > e9:   score += 5
    if c > e21:  score += 3
    if c > e50:  score += 2
    if down: score = 10

    score = max(0, min(100, score))

    if score >= 90:
        signal, prob = "STRONG BUY", 0.80
    elif score >= 75:
        signal, prob = "BUY", 0.70
    elif score <= 25:
        signal, prob = "SELL", 0.20
    else:
        signal, prob = "HOLD", 0.55

    # entries: ideal = pullback ke EMA21, agresif = breakout high 3 bar
    try:
        entry_ideal = float(e21) if not math.isnan(e21) else float("nan")
    except Exception:
        entry_ideal = float("nan")

    try:
        # breakout over last 3 highs
        entry_aggr = float(row["High_window"])
    except Exception:
        entry_aggr = float("nan")

    return (int(score), signal, float(prob), entry_ideal, entry_aggr)

def targets_closes(row, rr_tp=1.5):
    """
    TP/CL otonom: TP = close + rr * ATR, CL = close - 1 * ATR (atau di bawah EMA21)
    """
    c, atr, e21 = row["Close"], row["ATR14"], row["EMA21"]
    if any(map(lambda x: x is None or (isinstance(x, float) and math.isnan(x)), [c, atr])):
        return (float("nan"), float("nan"))
    tp = c + rr_tp * atr if not math.isnan(atr) else c * 1.02
    cl = max(c - 1.0 * atr, e21 * 0.985 if pd.notna(e21) else c * 0.98) if not math.isnan(atr) else c * 0.98
    return (float(tp), float(cl))

# =========== ONE TICKER ANALYZER ===========
def analyze_single(ticker: str):
    df = fetch_history(ticker)
    if df.empty:
        st.warning("No data. Cek ticker / koneksi.")
        return

    # extra columns
    df["High_window"] = df["High"].rolling(3).max()
    df = compute_indicators(df).dropna()

    last = df.iloc[-1].copy()
    score, signal, prob, entry_ideal, entry_aggr = score_signal_row(last)
    tp, cl = targets_closes(last)

    # Header metrics
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Price", f"Rp {int(round(last['Close'])):,}".replace(",", "."))
    colB.metric("Score", f"{score}/100")
    colC.metric("Signal", signal)
    colD.metric("Probability", f"{prob*100:.1f}%")

    # Entries/Targets
    st.write("### Levels")
    lev1, lev2, lev3 = st.columns(3)
    lev1.write(f"**Entry (Ideal/EMA21):** Rp {0 if math.isnan(entry_ideal) else int(round(entry_ideal)):,}".replace(",", "."))
    lev2.write(f"**Entry (Breakout):** Rp {0 if math.isnan(entry_aggr) else int(round(entry_aggr)):,}".replace(",", "."))
    lev3.write(f"**TP / CL:** Rp {int(round(tp)):,}  /  Rp {int(round(cl)):,}".replace(",", "."))

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"],  name="EMA9"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], name="EMA21"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50"))

    fig.update_layout(height=520, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# =========== MULTI SCREENER ===========
def score_latest_for(ticker: str):
    df = fetch_history(ticker)
    if df.empty or len(df) < 60:
        return None
    df["High_window"] = df["High"].rolling(3).max()
    df = compute_indicators(df).dropna()
    last = df.iloc[-1]
    score, signal, prob, entry_ideal, entry_aggr = score_signal_row(last)
    tp, cl = targets_closes(last)
    price = float(last["Close"])
    return {
        "Ticker": ticker,
        "Price": price,
        "Score": score,
        "Signal": signal,
        "Prob": prob,
        "Entry_Ideal": entry_ideal,
        "Entry_Break": entry_aggr,
        "TP": tp,
        "CL": cl,
    }

def run_screener(tickers: list[str], top_n=TOP_N):
    records = []
    for t in tickers:
        info = score_latest_for(t)
        if info:
            records.append(info)

    if not records:
        st.warning("Tidak ada data yang valid.")
        return

    df = pd.DataFrame(records)
    # sort: score desc, prob desc
    df = df.sort_values(["Score", "Prob"], ascending=[False, False]).head(top_n)

    # formatting
    df["Price"] = df["Price"].round(2)
    for col in ["Entry_Ideal", "Entry_Break", "TP", "CL"]:
        df[col] = df[col].round(2)

    st.dataframe(df.reset_index(drop=True), use_container_width=True)

    st.download_button(
        "⬇️ Download (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"idx_screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

# =========== UI SWITCH ===========
mode = st.sidebar.radio("Mode:", ["📈 Single Stock", "🔎 Multi-Stock Screener"], index=0)
tickers = load_tickers()

if mode == "📈 Single Stock":
    sel = st.selectbox("Stock:", options=tickers, index=0)
    if st.button("📊 Analyze"):
        analyze_single(sel)

else:
    st.write("Tampilkan **Top N** saham terbaik (Score & Prob tertinggi).")
    topn = st.slider("Jumlah hasil (Top N)", 10, 200, TOP_N, step=10)
    if st.button("🚀 Screener Now!"):
        run_screener(tickers, top_n=topn)
