# ==============================
# 🚀 IDX Screener Ultimate — HYBRID AUTO (Safe + Aggressive)
# Author: Mike x Mentor
# ==============================
import json, math, time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="IDX Screener Ultimate", layout="wide")

# ---------- UI STYLES ----------
st.markdown("""
<style>
.main-header{font-size:3rem;font-weight:800;letter-spacing:.02em}
.ok{color:#87f3} .bad{color:#f88}
.buy-signal{background:linear-gradient(90deg,#1f8,#16c);padding:.2rem .5rem;border-radius:.4rem;color:#fff}
.sell-signal{background:linear-gradient(90deg,#f55,#f90);padding:.2rem .5rem;border-radius:.4rem;color:#fff}
.small{opacity:.7;font-size:.85rem}
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
@st.cache_data(ttl=3600, show_spinner=False)
def load_tickers() -> list:
    """Load tickers from local json; sanitize."""
    try:
        with open('idx_stocks.json', 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and "tickers" in data:
            tickers = data["tickers"]
        else:
            tickers = data
        # sanitize
        tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
        tickers = [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
        tickers = sorted(list(dict.fromkeys(tickers)))
        return tickers
    except Exception as e:
        st.warning(f"⚠️ Tidak bisa membaca idx_stocks.json: {e}")
        return []

def _ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def _atr(df, length=14):
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_history(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    """Download OHLCV; return empty df on failure."""
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna().copy()
        df["EMA9"]  = _ema(df["Close"], 9)
        df["EMA21"] = _ema(df["Close"], 21)
        df["EMA50"] = _ema(df["Close"], 50)
        df["ATR14"] = _atr(df, 14)
        df["RSI14"] = _rsi(df["Close"], 14)
        df["VOL_MA20"] = df["Volume"].rolling(20).mean()
        return df
    except Exception:
        return pd.DataFrame()

def _rsi(close, length=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(length).mean()
    ma_down = down.rolling(length).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def hybrid_signal(df: pd.DataFrame):
    """Return dict: price, score, signal, prob, entry, tp1, tp2, cl, mode."""
    last = df.iloc[-1]
    price = float(last["Close"])
    ema9, ema21, ema50 = float(last["EMA9"]), float(last["EMA21"]), float(last["EMA50"])
    atr = max(0.01, float(last["ATR14"]))
    rsi = float(last["RSI14"])
    vol_boost = 1.0 if last["Volume"] <= 0 else float(last["Volume"] / max(1, last["VOL_MA20"]))

    # Base trend score
    trend_up = 0
    if price > ema9 > ema21 > ema50: trend_up = 1.0
    elif price > ema21 > ema50: trend_up = 0.7
    elif ema9 > ema21 > ema50: trend_up = 0.6
    else: trend_up = 0.0

    # Breakout check (3-day high with volume)
    recent_high = df["High"].rolling(3).max().iloc[-2]  # exclude today
    breakout = price > recent_high and vol_boost > 1.1

    # Mode selector (Hybrid)
    # If clear uptrend -> Safe mode; else if breakout with vol -> Aggressive; else Neutral/Hold
    if trend_up >= 0.7:
        mode = "SAFE"
    elif breakout:
        mode = "AGGR"
    else:
        mode = "NEUTRAL"

    # Entry logic
    if mode == "SAFE":
        # pullback near EMA21 with price > EMA50
        entry_ideal = round((ema21 * 1.00), 2)
        entry_aggr  = round((ema9  * 1.00), 2)
        stop = round(min(ema50, price - 1.2*atr), 2)
        rr_unit = max(0.01, (entry_aggr - stop))
        tp1 = round(entry_aggr + 1.0*rr_unit, 2)
        tp2 = round(entry_aggr + 2.0*rr_unit, 2)
        base_score = 70 + int(15*trend_up) + int(min(15, (vol_boost-1)*10))
        label = "STRONG BUY" if base_score >= 90 else "BUY"
    elif mode == "AGGR":
        entry_aggr = round(max(price, recent_high), 2)
        entry_ideal = round((recent_high*0.99), 2)
        stop = round(entry_aggr - 1.5*atr, 2)
        rr_unit = max(0.01, (entry_aggr - stop))
        tp1 = round(entry_aggr + 1.5*rr_unit, 2)
        tp2 = round(entry_aggr + 3.0*rr_unit, 2)
        base_score = 60 + int(15*vol_boost) + int(10*(rsi>50)) + int(5*(trend_up>0))
        label = "BUY" if base_score >= 70 else "SPEC BUY"
    else:
        # Neutral / downtrend
        entry_ideal = None
        entry_aggr = None
        stop = None
        tp1 = None
        tp2 = None
        base_score = 30 if rsi>45 else 20
        label = "HOLD" if rsi>45 else "SELL"

    # Probability heuristic (bounded)
    prob = np.clip((base_score/100.0) * (0.6 if label in ["SELL"] else 0.8) + (0.05 if vol_boost>1.2 else 0), 0.2, 0.9)

    # Clean rounding
    def r(x): return None if x is None else (round(float(x), 2))
    return dict(
        price=r(price), score=int(np.clip(base_score, 0, 100)),
        signal=label, prob=float(round(prob,2)),
        entry_ideal=r(entry_ideal), entry_aggr=r(entry_aggr),
        tp1=r(tp1), tp2=r(tp2), cl=r(stop),
        mode=mode, ema9=r(ema9), ema21=r(ema21), ema50=r(ema50), rsi=round(float(rsi),1)
    )

def analyze_one(ticker: str):
    df = fetch_history(ticker)
    if df.empty:
        return None
    d = hybrid_signal(df)
    d.update({"ticker": ticker})
    return d, df

def table_format(rows):
    df = pd.DataFrame(rows)
    if df.empty: return df
    order = ["ticker","price","score","signal","prob","mode","entry_ideal","entry_aggr","tp1","tp2","cl","rsi"]
    df = df[order]
    df["prob"] = (df["prob"]*100).round(0).astype(int).astype(str)+"%"
    return df.sort_values(["score","prob"], ascending=False)

# ---------- UI ----------
st.markdown('<div class="main-header">🚀 IDX SCREENER <span class="ok">ULTIMATE</span></div>', unsafe_allow_html=True)
mode = st.sidebar.radio("Mode:", ["📈 Single Stock","🔍 Multi-Stock Screener"], index=1)
tickers = load_tickers()

if mode == "📈 Single Stock":
    if not tickers:
        st.warning("Tambah/benarkan `idx_stocks.json` dulu, ya.")
        st.stop()
    sym = st.selectbox("Stock:", options=tickers, index=min(0, len(tickers)-1))
    btn = st.button("📊 Analyze")
    if btn:
        out = analyze_one(sym)
        if not out:
            st.error("Data tidak tersedia untuk saham itu.")
        else:
            data, df = out
            col1,col2 = st.columns([1,1])
            with col1:
                st.metric("Price", f"Rp {int(round(data['price'])):,}")
                st.metric("Score", f"{data['score']}/100")
                st.markdown(f"**Signal**: <span class='buy-signal'>{data['signal']}</span>", unsafe_allow_html=True)
                st.metric("Probability", f"{int(data['prob']*100)}%")
                st.markdown(f"**Mode**: `{data['mode']}`  |  RSI: **{data['rsi']}**")
                st.write("—")
                st.write(f"**Entry Ideal**: {data['entry_ideal']}")
                st.write(f"**Entry Aggressive**: {data['entry_aggr']}")
                st.write(f"**TP1**: {data['tp1']} | **TP2**: {data['tp2']}")
                st.write(f"**Cut Loss**: {data['cl']}")
                st.caption("Trailing stop: max(EMA21, price - 1.5 * ATR14)")
            with col2:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price")
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'],  name="EMA9"))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name="EMA21"))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name="EMA50"))
                fig.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Tampilkan **Top N** saham terbaik (Score & Prob tertinggi).")
    topn = st.slider("Jumlah hasil (Top N)", 10, 200, 50, step=10)
    if st.button("🚀 Screener Now!"):
        if not tickers:
            st.warning("Tambahkan ticker di `idx_stocks.json` terlebih dahulu.")
            st.stop()
        progress = st.progress(0)
        out_rows = []
        total = len(tickers)
        for i, sym in enumerate(tickers, start=1):
            res = analyze_one(sym)
            if res:
                data, _ = res
                # Skip jika signal SELL & score rendah agar tabel tetap relevan
                if not (data["signal"]=="SELL" and data["score"]<40):
                    out_rows.append(data)
            progress.progress(i/total)
        if not out_rows:
            st.warning("Tidak ada data yang valid.")
        else:
            df = table_format(out_rows).head(topn)
            st.dataframe(df, use_container_width=True)
            st.caption("Catatan: Saham tanpa data valid otomatis di-skip. Top N diurutkan berdasarkan Score & Prob.")

# Footer
st.caption("© IDX Screener Ultimate — Hybrid Auto | EMA9/21/50, ATR14, RSI14, Volume breakout")
