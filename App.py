#!/usr/bin/env python3
# app.py – IDX Power Screener (Speed / Swing / BPJS / BSJP / Value)

import os
import math
import time
import json
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= BASIC CONFIG =================
st.set_page_config(
    page_title="IDX Power Screener – Main",
    page_icon="🚀",
    layout="wide",
)

# ================ CONSTANTS & UTILS ==============
IDX_TICKS = [
    (0, 200, 1),
    (200, 500, 2),
    (500, 2000, 5),
    (2000, 5000, 10),
    (5000, float("inf"), 25),
]


def round_to_tick(price: float, mode: str = "nearest") -> int:
    if price is None or not (price == price) or price <= 0:
        return 0
    tick = 1
    for low, high, t in IDX_TICKS:
        if low <= price < high:
            tick = t
            break
    if mode == "floor":
        return int(math.floor(price / tick) * tick)
    if mode == "ceil":
        return int(math.ceil(price / tick) * tick)
    return int(round(price / tick) * tick)


def normalize_ticker(t: str) -> str:
    t = t.strip().upper()
    return t if t.endswith(".JK") else f"{t}.JK"


def format_idr(x: float) -> str:
    try:
        return f"Rp {x:,.0f}".replace(",", ".")
    except Exception:
        return "-"


def get_jakarta_time() -> datetime:
    return datetime.now(timezone(timedelta(hours=7)))


# ================ SESSION STATE ==================
defaults = {
    "last_scan_results": None,
    "last_scan_time": None,
    "last_scan_strategy": None,
    "scan_count": 0,
    "min_price_filter": 50,
    "min_vol_filter": 1_000_000,
    "min_turnover_filter": 150_000_000,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================ IHSG WIDGET =====================


@st.cache_data(ttl=60)
def fetch_ihsg_data():
    try:
        end = int(datetime.now().timestamp())
        start = end - 3 * 86400
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EJKSE"
        params = {"period1": start, "period2": end, "interval": "1d"}
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [None])[0]
        if not result:
            return None
        q = result["indicators"]["quote"][0]
        close = float(q["close"][-1])
        open_price = float(q["open"][-1])
        high = float(q["high"][-1])
        low = float(q["low"][-1])
        change = close - open_price
        change_pct = (change / open_price) * 100 if open_price else 0.0
        return {
            "price": close,
            "change": change,
            "change_pct": change_pct,
            "high": high,
            "low": low,
            "status": "up" if change >= 0 else "down",
        }
    except Exception:
        return None


def display_ihsg_widget():
    ihsg = fetch_ihsg_data()
    if not ihsg:
        st.info("📊 IHSG data temporarily unavailable")
        return

    status_emoji = "🟢" if ihsg["status"] == "up" else "🔴"
    status_text = "BULLISH" if ihsg["status"] == "up" else "BEARISH"
    now_jkt = get_jakarta_time().strftime("%H:%M:%S")

    st.markdown(
        f"""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                padding: 14px; border-radius: 10px; margin-bottom: 18px;
                border-left: 5px solid {"#22c55e" if ihsg['status']=="up" else "#ef4444"}'>
      <div style='display:flex;justify-content:space-between;align-items:center;'>
        <div>
          <h3 style='margin:0;color:white;'>📊 IHSG Overview</h3>
          <p style='margin:5px 0;color:#e0e7ff;font-size:0.9em;'>Jakarta Composite Index</p>
        </div>
        <div style='text-align:right;'>
          <h2 style='margin:0;color:white;'>{status_emoji} {ihsg['price']:,.2f}</h2>
          <p style='margin:5px 0;color:{"#22c55e" if ihsg['status']=="up" else "#ef4444"};
                    font-size:1.1em;font-weight:bold;'>
            {ihsg['change']:+,.2f} ({ihsg['change_pct']:+.2f}%)
          </p>
        </div>
      </div>
      <div style='margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.2);'>
        <p style='margin:3px 0;color:#e0e7ff;font-size:0.85em;'>
          High: {ihsg['high']:,.2f} | Low: {ihsg['low']:,.2f} | Status: <strong>{status_text}</strong>
        </p>
        <p style='margin:4px 0 0 0;color:#94a3b8;font-size:0.75em;'>
          ⏰ Last update: {now_jkt} WIB (Yahoo, delayed)
        </p>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ================ TICKERS =========================


@st.cache_data(ttl=3600)
def load_tickers():
    try:
        if os.path.exists("idx_stocks.json"):
            with open("idx_stocks.json", "r") as f:
                data = json.load(f)
            return [normalize_ticker(t) for t in data.get("tickers", []) if t]
    except Exception:
        pass
    return ["BBRI.JK", "BBCA.JK", "TLKM.JK", "ASII.JK", "ICBP.JK", "INDF.JK"]


# ================ DATA FETCH ======================

def _yahoo_chart_json(ticker: str, period: str):
    end = int(datetime.now().timestamp())
    days = {"3mo": 90, "6mo": 180, "1y": 365}.get(period, 180)
    start = end - days * 86400
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period1": start, "period2": end, "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}
    data = None
    for i in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                break
        except Exception:
            pass
        time.sleep(0.5 * (i + 1))
    return data


@st.cache_data(ttl=90, show_spinner=False)
def fetch_data(ticker: str, period: str = "6mo"):
    try:
        data = _yahoo_chart_json(ticker, period)
        if not data:
            return None
        result = data.get("chart", {}).get("result", [None])[0]
        if not result:
            return None

        q = result["indicators"]["quote"][0]
        ts = result.get("timestamp", [])
        if not ts or len(ts) != len(q["close"]):
            return None

        df = pd.DataFrame(
            {
                "Open": q["open"],
                "High": q["high"],
                "Low": q["low"],
                "Close": q["close"],
                "Volume": q["volume"],
            },
            index=pd.to_datetime(ts, unit="s"),
        ).dropna()

        if len(df) < 60:
            return None

        # EMAs
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA200"] = df["Close"].ewm(span=min(len(df), 200), adjust=False).mean()

        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Volume & Momentum
        df["VOL_SMA20"] = df["Volume"].rolling(20).mean()
        df["VOL_RATIO"] = df["Volume"] / df["VOL_SMA20"].replace(0, np.nan)
        df["MOM_5D"] = df["Close"].pct_change(5) * 100
        df["MOM_20D"] = df["Close"].pct_change(20) * 100

        # ATR%
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - df["Close"].shift()).abs()
        tr3 = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()
        df["ATR_PCT"] = (df["ATR"] / df["Close"]) * 100

        return df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    except Exception:
        return None


# ================ FILTER & TREND ==================

def apply_liquidity_filter(df: pd.DataFrame):
    try:
        r = df.iloc[-1]
        price = float(r["Close"])
        vol_avg = float(df["Volume"].tail(20).mean())
        min_price = st.session_state["min_price_filter"]
        min_vol = st.session_state["min_vol_filter"]
        min_turnover = st.session_state["min_turnover_filter"]

        if price < min_price:
            return False, f"Price < {min_price}"
        if vol_avg < min_vol:
            return False, f"Avg20Vol < {min_vol}"
        if price * vol_avg < min_turnover:
            return False, "Turnover too low"
        return True, "OK"
    except Exception as e:
        return False, f"Error {e}"


def detect_trend(r: pd.Series) -> str:
    price = float(r["Close"])
    ema9 = float(r["EMA9"])
    ema21 = float(r["EMA21"])
    ema50 = float(r["EMA50"])
    ema200 = float(r["EMA200"])

    if price > ema9 > ema21 > ema50 > ema200:
        return "Strong Uptrend"
    if price > ema50 and ema9 > ema21 > ema50:
        return "Uptrend"
    if abs(price - ema50) / price < 0.03:
        return "Sideways"
    return "Downtrend"


def grade_from_score(score: int):
    if score >= 85:
        return "A+", 90
    if score >= 75:
        return "A", 80
    if score >= 65:
        return "B+", 70
    if score >= 55:
        return "B", 60
    if score >= 45:
        return "C", 50
    return "D", 30


def classify_signal(r: pd.Series, score: int, grade: str, trend: str) -> str:
    rsi = float(r["RSI"]) if r.get("RSI") == r.get("RSI") else 50.0
    volr = float(r["VOL_RATIO"]) if r.get("VOL_RATIO") == r.get("VOL_RATIO") else 1.0
    m5 = float(r["MOM_5D"]) if r.get("MOM_5D") == r.get("MOM_5D") else 0.0
    m20 = float(r["MOM_20D"]) if r.get("MOM_20D") == r.get("MOM_20D") else 0.0

    if (
        trend in ["Strong Uptrend", "Uptrend"]
        and grade in ["A+", "A"]
        and volr > 1.5
        and 45 <= rsi <= 70
        and m5 > 0
        and m20 > 0
    ):
        return "Strong Buy"

    if (
        trend in ["Strong Uptrend", "Uptrend"]
        and grade in ["A+", "A", "B+"]
        and volr > 1.0
        and 40 <= rsi <= 75
    ):
        return "Buy"

    if trend in ["Strong Uptrend", "Uptrend"] and grade in ["A+", "A", "B+", "B"]:
        return "Hold"

    if trend == "Sideways" and grade in ["B+", "B", "C"]:
        return "Hold"

    return "Sell"


# ================ STRATEGY SCORING ===============

def score_speed(df: pd.DataFrame):
    ok, reason = apply_liquidity_filter(df)
    if not ok:
        return 0, {"Rejected": reason}, 0, "F"

    r = df.iloc[-1]
    if r["Close"] < r["EMA21"]:
        return 0, {"Rejected": "Below EMA21"}, 0, "F"

    score, details = 0, {}

    # Trend
    price = float(r["Close"])
    ema9 = float(r["EMA9"])
    ema21 = float(r["EMA21"])
    ema50 = float(r["EMA50"])
    ema200 = float(r["EMA200"])
    if price > ema9 > ema21 > ema50 > ema200:
        score += 40
        details["Trend"] = "🟢 Strong Uptrend"
    elif price > ema50 and ema9 > ema21 > ema50:
        score += 25
        details["Trend"] = "🟡 Uptrend"
    else:
        details["Trend"] = "🟠 Weak"

    # RSI
    rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50.0
    if 50 <= rsi <= 65:
        score += 25
        details["RSI"] = f"🟢 {rsi:.0f}"
    elif 45 <= rsi < 50:
        score += 15
        details["RSI"] = f"🟡 {rsi:.0f}"
    else:
        details["RSI"] = f"⚪ {rsi:.0f}"

    # Volume
    volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
    if volr > 2.0:
        score += 20
        details["Volume"] = f"🟢 {volr:.1f}x"
    elif volr > 1.4:
        score += 15
        details["Volume"] = f"🟡 {volr:.1f}x"
    else:
        score += 5
        details["Volume"] = f"⚪ {volr:.1f}x"

    # Momentum
    m5 = float(r["MOM_5D"])
    m20 = float(r["MOM_20D"])
    if m5 > 3 and m20 > 8:
        score += 15
        details["Momentum"] = f"🟢 +{m5:.1f}% (5D)"
    elif m5 > 1 and m20 > 3:
        score += 10
        details["Momentum"] = f"🟡 +{m5:.1f}%"
    else:
        details["Momentum"] = f"⚪ {m5:.1f}%"

    score = int(score)
    grade, conf = grade_from_score(score)
    return score, details, conf, grade


def score_swing(df: pd.DataFrame):
    ok, reason = apply_liquidity_filter(df)
    if not ok:
        return 0, {"Rejected": reason}, 0, "F"

    r = df.iloc[-1]
    if r["Close"] < r["EMA50"]:
        return 0, {"Rejected": "Below EMA50"}, 0, "F"

    score, details = 0, {}

    # Trend
    price = float(r["Close"])
    ema9 = float(r["EMA9"])
    ema21 = float(r["EMA21"])
    ema50 = float(r["EMA50"])
    ema200 = float(r["EMA200"])
    if price > ema9 > ema21 > ema50 > ema200:
        score += 45
        details["Trend"] = "🟢 Strong Uptrend"
    elif price > ema50 and ema9 > ema21 > ema50:
        score += 30
        details["Trend"] = "🟡 Uptrend"
    else:
        details["Trend"] = "🟠 Weak"

    # Pullback ke EMA21
    dist = abs((price - ema21) / price) * 100
    if dist <= 3:
        score += 15
        details["Pullback"] = f"🟢 {dist:.1f}% vs EMA21"
    elif dist <= 5:
        score += 8
        details["Pullback"] = f"🟡 {dist:.1f}%"

    # RSI
    rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50.0
    if 50 <= rsi <= 65:
        score += 20
        details["RSI"] = f"🟢 {rsi:.0f}"
    elif 45 <= rsi < 50:
        score += 10
        details["RSI"] = f"🟡 {rsi:.0f}"
    else:
        details["RSI"] = f"⚪ {rsi:.0f}"

    # Volume
    volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
    if volr > 1.5:
        score += 15
        details["Volume"] = f"🟢 {volr:.1f}x"
    elif volr > 1.1:
        score += 8
        details["Volume"] = f"🟡 {volr:.1f}x"

    score = int(score)
    grade, conf = grade_from_score(score)
    return score, details, conf, grade


def score_value(df: pd.DataFrame):
    ok, reason = apply_liquidity_filter(df)
    if not ok:
        return 0, {"Rejected": reason}, 0, "F"

    r = df.iloc[-1]
    score, details = 0, {}

    if r["Close"] >= r["EMA200"]:
        score += 25
        details["Trend"] = "🟢 >= EMA200"
    else:
        details["Trend"] = "🟠 < EMA200 (mean reversion)"

    rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50
    if 35 <= rsi <= 50:
        score += 25
        details["RSI"] = f"🟢 {rsi:.0f}"
    elif rsi < 30:
        details["RSI"] = f"🔴 Too weak {rsi:.0f}"
    else:
        score += 10
        details["RSI"] = f"🟡 {rsi:.0f}"

    if r["Close"] <= r["EMA50"]:
        score += 15
        details["Discount"] = "🟢 Price <= EMA50"

    volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
    if 0.9 <= volr <= 1.6:
        score += 15
        details["Volume"] = f"🟢 {volr:.1f}x"
    else:
        details["Volume"] = f"⚪ {volr:.1f}x"

    score = int(score)
    grade, conf = grade_from_score(score)
    return score, details, conf, grade


def score_bpjs(df: pd.DataFrame):
    ok, reason = apply_liquidity_filter(df)
    if not ok:
        return 0, {"Rejected": reason}, 0, "F"

    r = df.iloc[-1]
    if r["Close"] < r["EMA21"]:
        return 0, {"Rejected": "Below EMA21"}, 0, "F"

    score, details = 0, {}

    volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
    if volr < 1.3:
        return 0, {"Rejected": f"Low intraday volume ({volr:.1f}x)"}, 0, "F"

    if volr > 3.0:
        score += 40
        details["Volume"] = f"🟢 {volr:.1f}x"
    elif volr > 2.0:
        score += 30
        details["Volume"] = f"🟡 {volr:.1f}x"
    else:
        score += 15
        details["Volume"] = f"🟠 {volr:.1f}x"

    atr = float(r["ATR_PCT"]) if not np.isnan(r["ATR_PCT"]) else 0.0
    if 2 <= atr <= 6:
        score += 25
        details["ATR"] = f"🟢 {atr:.1f}%"
    elif atr < 1.5:
        details["ATR"] = f"🔴 Too tight {atr:.1f}%"
    else:
        score += 10
        details["ATR"] = f"🟡 {atr:.1f}%"

    rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50
    if 48 <= rsi <= 70:
        score += 20
        details["RSI"] = f"🟢 {rsi:.0f}"
    else:
        details["RSI"] = f"⚪ {rsi:.0f}"

    score = int(score)
    grade, conf = grade_from_score(score)
    return score, details, conf, grade


def score_bsjp(df: pd.DataFrame):
    ok, reason = apply_liquidity_filter(df)
    if not ok:
        return 0, {"Rejected": reason}, 0, "F"

    r = df.iloc[-1]
    if r["Close"] < r["EMA50"]:
        return 0, {"Rejected": "Below EMA50"}, 0, "F"

    score, details = 0, {}

    price = float(r["Close"])
    ema9 = float(r["EMA9"])
    ema21 = float(r["EMA21"])
    ema50 = float(r["EMA50"])
    ema200 = float(r["EMA200"])
    if price > ema9 > ema21 > ema50 > ema200:
        score += 40
        details["Trend"] = "🟢 Strong Uptrend"
    elif price > ema50 and ema9 > ema21 > ema50:
        score += 30
        details["Trend"] = "🟡 Uptrend"
    else:
        details["Trend"] = "🟠 Weak"

    rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50
    if 50 <= rsi <= 68:
        score += 20
        details["RSI"] = f"🟢 {rsi:.0f}"
    else:
        details["RSI"] = f"⚪ {rsi:.0f}"

    volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
    if volr > 1.5:
        score += 15
        details["Volume"] = f"🟢 {volr:.1f}x"
    else:
        details["Volume"] = f"⚪ {volr:.1f}x"

    score = int(score)
    grade, conf = grade_from_score(score)
    return score, details, conf, grade


# ================ TRADE PLAN ======================

def compute_trade_plan(df: pd.DataFrame, strategy: str, trend: str):
    r = df.iloc[-1]
    price = float(r["Close"])

    if strategy == "Swing":
        entry = round_to_tick(price * 0.99)
        tp1 = round_to_tick(entry * 1.06)
        tp2 = round_to_tick(entry * 1.10)
        tp3 = round_to_tick(entry * 1.15)
        sl = round_to_tick(entry * 0.95)
    elif strategy == "Value":
        entry = round_to_tick(price * 0.98)
        tp1 = round_to_tick(entry * 1.15)
        tp2 = round_to_tick(entry * 1.25)
        tp3 = round_to_tick(entry * 1.35)
        sl = round_to_tick(entry * 0.93)
    else:  # Speed, BPJS, BSJP
        entry = round_to_tick(price * 0.995)
        tp1 = round_to_tick(entry * 1.04)
        tp2 = round_to_tick(entry * 1.07)
        tp3 = None
        sl = round_to_tick(entry * 0.97)

    if trend == "Downtrend":
        sl = round_to_tick(entry * 0.96)

    return {
        "entry": entry,
        "entry_aggr": round_to_tick(price),
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
    }


# ================ ANALYZE TICKER ==================

def analyze_ticker(ticker: str, strategy: str, period: str):
    df = fetch_data(ticker, period)
    if df is None or df.empty:
        return None

    if strategy == "Speed":
        score, details, conf, grade = score_speed(df)
    elif strategy == "Swing":
        score, details, conf, grade = score_swing(df)
    elif strategy == "Value":
        score, details, conf, grade = score_value(df)
    elif strategy == "BPJS":
        score, details, conf, grade = score_bpjs(df)
    elif strategy == "BSJP":
        score, details, conf, grade = score_bsjp(df)
    else:
        score, details, conf, grade = score_speed(df)

    if grade in ["D", "F"] or score <= 0:
        return None

    r = df.iloc[-1]
    trend = detect_trend(r)
    signal = classify_signal(r, score, grade, trend)
    plan = compute_trade_plan(df, strategy, trend)

    # Stage 2 nanti akan filter lagi ke Uptrend + Buy/Strong Buy
    return {
        "Ticker": ticker.replace(".JK", ""),
        "Price": float(r["Close"]),
        "Score": score,
        "Confidence": conf,
        "Grade": grade,
        "Trend": trend,
        "Signal": signal,
        "Entry": plan["entry"],
        "Entry_Aggressive": plan["entry_aggr"],
        "TP1": plan["tp1"],
        "TP2": plan["tp2"],
        "TP3": plan["tp3"],
        "SL": plan["sl"],
        "Details": details,
    }


# ================ SESSION HELPERS =================

def save_scan_to_session(df_elite: pd.DataFrame, strategy: str):
    st.session_state["last_scan_results"] = df_elite
    st.session_state["last_scan_time"] = datetime.now()
    st.session_state["last_scan_strategy"] = strategy
    st.session_state["scan_count"] = (st.session_state["scan_count"] or 0) + 1


def display_last_scan_info():
    if st.session_state["last_scan_results"] is None:
        return
    df = st.session_state["last_scan_results"]
    mins = int(
        (datetime.now() - st.session_state["last_scan_time"]).total_seconds() / 60
    )
    st.markdown(
        f"""
    <div style='background:linear-gradient(135deg,#064e3b 0%,#065f46 100%);
               padding:12px;border-radius:8px;margin-bottom:15px;
               border-left:4px solid #10b981;'>
      <p style='margin:0;color:white;font-weight:bold;'>📂 LAST SCAN RESULTS</p>
      <p style='margin:5px 0 0 0;color:#d1fae5;font-size:0.9em;'>
         Strategy: {st.session_state['last_scan_strategy']} |
         Time: {st.session_state['last_scan_time'].strftime('%H:%M:%S')} ({mins} min ago) |
         Picks: {len(df)}
      </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def create_csv_download(df: pd.DataFrame, strategy: str):
    if df is None or df.empty:
        return
    export = df.drop(columns=["Details"], errors="ignore").copy()
    csv = export.to_csv(index=False).encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "💾 Download Results (CSV)",
        data=csv,
        file_name=f"IDX_{strategy}_scan_{ts}.csv",
        mime="text/csv",
    )


def process_ticker(ticker: str, strategy: str, period: str):
    try:
        return analyze_ticker(ticker, strategy, period)
    except Exception:
        return None


def scan_stocks(tickers, strategy, period, limit1, limit2):
    st.info(f"🔍 Stage 1: scanning {len(tickers)} stocks for {strategy}...")
    results = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(process_ticker, t, strategy, period): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            done += 1
            progress.progress(done / len(tickers))
            status.text(f"📊 {done}/{len(tickers)} processed | Found: {len(results)}")
            r = fut.result()
            if r:
                results.append(r)
            time.sleep(0.01)

    progress.empty()
    status.empty()

    if not results:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(results).sort_values(
        ["Score", "Confidence"], ascending=False
    )

    # Stage 1: top N by score
    df1 = df.head(limit1)

    # Stage 2: hanya Uptrend/Strong Uptrend + Signal Buy/Strong Buy
    mask_elite = df1["Trend"].isin(["Uptrend", "Strong Uptrend"]) & df1[
        "Signal"
    ].isin(["Buy", "Strong Buy"])
    df2 = df1[mask_elite].head(limit2)

    save_scan_to_session(df2, strategy)
    return df1, df2


# ================ UI HELPERS ======================

def show_table(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        st.warning("Tidak ada hasil.")
        return

    st.subheader(title)
    show = df.copy()
    if "Price" in show:
        show["Price"] = show["Price"].apply(round_to_tick)

    cols = [
        c
        for c in [
            "Ticker",
            "Price",
            "Score",
            "Confidence",
            "Grade",
            "Trend",
            "Signal",
            "Entry",
            "Entry_Aggressive",
            "TP1",
            "TP2",
            "TP3",
            "SL",
        ]
        if c in show.columns
    ]
    st.dataframe(show[cols], use_container_width=True, hide_index=True)
    create_csv_download(show[cols], title.replace(" ", "_"))

    with st.expander("🔎 Detail per saham"):
        for _, row in show.iterrows():
            st.markdown(
                f"**{row['Ticker']}** | Grade **{row['Grade']}** | Trend **{row['Trend']}** | Signal **{row['Signal']}**"
            )
            tp3 = (
                f" | TP3 {format_idr(row['TP3'])}"
                if "TP3" in row and pd.notna(row["TP3"])
                else ""
            )
            st.caption(
                f"Price {format_idr(row['Price'])} | Entry {format_idr(row['Entry'])}"
                f" (Agg {format_idr(row['Entry_Aggressive'])}) | TP1 {format_idr(row['TP1'])}"
                f" | TP2 {format_idr(row['TP2'])}{tp3} | SL {format_idr(row['SL'])}"
            )
            details = row.get("Details", {})
            if isinstance(details, dict) and details:
                st.write({k: str(v) for k, v in details.items()})
            st.markdown("---")


# =================== MAIN UI ======================

st.title("🚀 IDX Power Screener – Main")
st.caption("Speed / Swing / BPJS / BSJP / Value • Anti-saham tidur • Uptrend only")
display_ihsg_widget()

tickers = load_tickers()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.info(f"📊 Total stocks in universe: **{len(tickers)}**")
    st.caption(f"🕐 Jakarta: {get_jakarta_time().strftime('%H:%M WIB')}")

    st.markdown("---")
    menu = st.radio(
        "🎯 Strategy",
        [
            "⚡ SPEED Trader (1–2d)",
            "🎯 SWING Trader (3–5d)",
            "⚡ BPJS (Intraday)",
            "🌙 BSJP (Overnight)",
            "💎 VALUE Plays (Rebound)",
        ],
    )

    if "SPEED" in menu:
        strategy = "Speed"
    elif "SWING" in menu:
        strategy = "Swing"
    elif "BPJS" in menu:
        strategy = "BPJS"
    elif "BSJP" in menu:
        strategy = "BSJP"
    else:
        strategy = "Value"

    period = st.selectbox("History period", ["3mo", "6mo", "1y"], index=1)

    st.markdown("### 🔧 Liquidity filter")
    st.session_state["min_price_filter"] = st.number_input(
        "Min price (Rp)", 50, 500, st.session_state["min_price_filter"], 10
    )
    st.session_state["min_vol_filter"] = st.number_input(
        "Min avg 20D volume", 100_000, 5_000_000, st.session_state["min_vol_filter"], 100_000
    )
    st.session_state["min_turnover_filter"] = st.number_input(
        "Min turnover", 50_000_000, 500_000_000, st.session_state["min_turnover_filter"], 50_000_000
    )

    st.markdown("### 🎯 Result limits")
    c1, c2 = st.columns(2)
    with c1:
        limit1 = st.slider("Stage 1 Top N", 20, 150, 60, 10)
    with c2:
        limit2 = st.slider("Stage 2 Elite N", 5, 40, 15, 5)

    sublist_input = st.text_input(
        "Custom tickers (optional)", placeholder="BBRI, BBCA, TLKM, RAJA, CUAN"
    )
    run_scan = st.button("🚀 RUN SCAN", type="primary", use_container_width=True)

display_last_scan_info()

# Single scan mode only (no Single Stock di app ini untuk jaga file tetap ringan)
if run_scan:
    run_list = tickers
    if sublist_input:
        parts = [normalize_ticker(x) for x in sublist_input.split(",") if x.strip()]
        if parts:
            run_list = parts
            st.info(
                "Using custom tickers: "
                + ", ".join([p.replace(".JK", "") for p in parts])
            )

    if not run_list:
        st.error("Ticker list kosong.")
    else:
        df1, df2 = scan_stocks(run_list, strategy, period, limit1, limit2)
        if df1.empty and df2.empty:
            st.error("❌ Tidak ada saham yang lolos filter.")
        else:
            show_table(df1, "Stage 1 – Candidates")
            show_table(df2, "Stage 2 – Elite Picks (Uptrend + Buy/Strong Buy)")
else:
    st.info("Tekan **RUN SCAN** untuk mulai pemindaian saham.")
