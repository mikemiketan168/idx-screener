#!/usr/bin/env python3
# app.py – IDX Power Screener v5.2 (Cloud-friendly)

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

# =============== BASIC CONFIG ===============
st.set_page_config(
    page_title="IDX Power Screener v5.2 – Chef Edition",
    page_icon="🚀",
    layout="wide",
)

# =============== CONSTANTS & UTILS ===============
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


# =============== SESSION STATE DEFAULTS ===============
defaults = {
    "last_scan_results": None,
    "last_scan_time": None,
    "last_scan_strategy": None,
    "scan_count": 0,
    "min_price_filter": 50,
    "min_vol_filter": 500_000,
    "min_turnover_filter": 100_000_000,
    "account_size": 10_000_000,
    "risk_pct": 1.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============== IHSG WIDGET (Yahoo API langsung) ===============


@st.cache_data(ttl=180)
def fetch_ihsg_data():
    """Ambil data IHSG (^JKSE) via Yahoo chart API dengan timeout pendek."""
    try:
        end = int(datetime.now().timestamp())
        start = end - 3 * 86400  # 3 hari cukup

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
        ts = result.get("timestamp", [])
        if not ts:
            return None

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

    if ihsg["change_pct"] > 1.5:
        condition = "🔥 Strong uptrend - Good for momentum!"
        guidance = "✅ Excellent for SPEED/SWING trades"
    elif ihsg["change_pct"] > 0.5:
        condition = "📈 Moderate uptrend - Good conditions"
        guidance = "✅ Good for all strategies"
    elif ihsg["change_pct"] > -0.5:
        condition = "➡️ Sideways - Mixed conditions"
        guidance = "⚠️ Be selective, use tight stops"
    elif ihsg["change_pct"] > -1.5:
        condition = "📉 Moderate downtrend - Caution"
        guidance = "⚠️ Focus on VALUE plays, avoid SPEED"
    else:
        condition = "🔻 Strong downtrend - High risk"
        guidance = "❌ Consider sitting out or very selective"

    now_jkt = get_jakarta_time().strftime("%H:%M:%S")

    st.markdown(
        f"""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                padding: 15px; border-radius: 10px; margin-bottom: 20px;
                border-left: 5px solid {"#22c55e" if ihsg['status']=="up" else "#ef4444"}'>
      <div style='display:flex;justify-content:space-between;align-items:center;'>
        <div>
          <h3 style='margin:0;color:white;'>📊 MARKET OVERVIEW</h3>
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
      <div style='margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.2);'>
        <p style='margin:3px 0;color:#e0e7ff;font-size:0.85em;'>
          📊 High: {ihsg['high']:,.2f} | Low: {ihsg['low']:,.2f} | Status: <strong>{status_text}</strong>
        </p>
        <p style='margin:3px 0;color:#fbbf24;font-size:0.9em;'> {condition} </p>
        <p style='margin:3px 0;color:#a5b4fc;font-size:0.85em;'> {guidance} </p>
        <p style='margin:5px 0 0 0;color:#94a3b8;font-size:0.75em;'>
          ⏰ Last update: {now_jkt} WIB | 🔄 Yahoo data may be delayed
        </p>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# =============== TICKERS ===============


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


# =============== FETCH DATA (Yahoo API langsung) ===============

def _yahoo_chart_json(ticker: str, period: str):
    end = int(datetime.now().timestamp())
    days = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}.get(period, 180)
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


@st.cache_data(ttl=60, show_spinner=False)
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

        if len(df) < 50:
            return None

        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA200"] = df["Close"].ewm(span=min(len(df), 200), adjust=False).mean()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df["VOL_SMA20"] = df["Volume"].rolling(20).mean()
        df["VOL_SMA50"] = df["Volume"].rolling(50).mean()
        df["VOL_RATIO"] = df["Volume"] / df["VOL_SMA20"].replace(0, np.nan)

        df["MOM_5D"] = df["Close"].pct_change(5) * 100
        df["MOM_10D"] = df["Close"].pct_change(10) * 100
        df["MOM_20D"] = df["Close"].pct_change(20) * 100

        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                obv.append(obv[-1] + (df["Volume"].iloc[i] or 0))
            elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
                obv.append(obv[-1] - (df["Volume"].iloc[i] or 0))
            else:
                obv.append(obv[-1])
        df["OBV"] = obv
        df["OBV_EMA"] = pd.Series(df["OBV"]).ewm(span=10, adjust=False).mean()

        df["BB_MID"] = df["Close"].rolling(20).mean()
        df["BB_STD"] = df["Close"].rolling(20).std()
        df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
        df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]
        df["BB_WIDTH"] = ((df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"]) * 100

        low14 = df["Low"].rolling(14).min()
        high14 = df["High"].rolling(14).max()
        df["STOCH_K"] = 100 * (df["Close"] - low14) / (high14 - low14)
        df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - df["Close"].shift()).abs()
        tr3 = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()
        df["ATR_PCT"] = (df["ATR"] / df["Close"]) * 100

        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

        df["BODY"] = (df["Close"] - df["Open"]).abs()
        df["RANGE"] = (df["High"] - df["Low"]).replace(0, np.nan)
        df["BODY_RATIO"] = df["BODY"] / df["RANGE"]
        df["POS_IN_RANGE"] = (df["Close"] - df["Low"]) / df["RANGE"]

        return df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    except Exception:
        return None


# =============== CHARTS ===============

def create_chart(df: pd.DataFrame, ticker: str, period_days: int = 60):
    try:
        d = df.tail(period_days).copy()
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{ticker} - Price & EMAs", "Volume", "RSI"),
        )

        fig.add_trace(
            go.Candlestick(
                x=d.index,
                open=d["Open"],
                high=d["High"],
                low=d["Low"],
                close=d["Close"],
                name="Price",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=1,
            col=1,
        )

        colors = {
            "EMA9": "#2196F3",
            "EMA21": "#FF9800",
            "EMA50": "#F44336",
            "EMA200": "#9E9E9E",
        }
        for ema in ["EMA9", "EMA21", "EMA50", "EMA200"]:
            if ema in d.columns:
                fig.add_trace(
                    go.Scatter(
                        x=d.index,
                        y=d[ema],
                        name=ema,
                        line=dict(color=colors[ema], width=1.5),
                    ),
                    row=1,
                    col=1,
                )

        colors_vol = [
            "#ef5350" if c < o else "#26a69a"
            for c, o in zip(d["Close"], d["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=d.index,
                y=d["Volume"],
                name="Volume",
                marker_color=colors_vol,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=d.index,
                y=d["RSI"],
                name="RSI",
                line=dict(color="#9C27B0", width=2),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)

        fig.update_layout(
            height=700,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            template="plotly_dark",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#333")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#333")
        return fig
    except Exception:
        return None


# =============== FILTER & GRADING HELPERS ===============

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
        return True, "Passed"
    except Exception as e:
        return False, f"Error {e}"


def ema_alignment_score(r: pd.Series):
    pts = (
        int(r["EMA9"] > r["EMA21"])
        + int(r["EMA21"] > r["EMA50"])
        + int(r["EMA50"] > r["EMA200"])
        + int(r["Close"] > r["EMA9"])
    )
    label = {
        4: "🟢 Perfect",
        3: "🟡 Strong",
        2: "🟠 Moderate",
        1: "🔴 Weak",
        0: "🔴 Weak",
    }[pts]
    return pts, label


def grade_from_score(score: int):
    if score >= 85:
        return "A+", 85
    if score >= 75:
        return "A", 75
    if score >= 65:
        return "B+", 65
    if score >= 55:
        return "B", 55
    if score >= 45:
        return "C", 45
    return "D", max(score, 0)


# =============== STRATEGY SCORING ===============

def score_general(df: pd.DataFrame):
    try:
        ok, reason = apply_liquidity_filter(df)
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"

        r = df.iloc[-1]
        if r["Close"] < r["EMA50"]:
            return 0, {"Rejected": "Below EMA50"}, 0, "F"
        if r["Close"] < r["EMA21"] < r["EMA50"] < r["EMA200"]:
            return 0, {"Rejected": "Strong downtrend"}, 0, "F"

        score, details = 0, {}
        mom20 = float(r["MOM_20D"])
        volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 0.0

        if mom20 < -8:
            return 0, {"Rejected": f"Strong negative momentum ({mom20:.1f}%)"}, 0, "F"

        penalty = 1.0
        if -8 <= mom20 < -5:
            penalty = 0.6
            details["⚠️ Warning"] = f"Weak momentum {mom20:.1f}%"
        elif -5 <= mom20 < 0:
            penalty = 0.8
            details["⚠️ Warning"] = f"Slight negative momentum {mom20:.1f}%"

        pts, label = ema_alignment_score(r)
        score += {4: 40, 3: 25, 2: 10}.get(pts, 0)
        details["Trend"] = label

        rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50.0
        if 50 <= rsi <= 65:
            score += 25
            details["RSI"] = f"🟢 {rsi:.0f}"
        elif 45 <= rsi < 50:
            score += 20
            details["RSI"] = f"🟡 {rsi:.0f}"
        elif 40 <= rsi < 45:
            score += 10
            details["RSI"] = f"🟠 {rsi:.0f}"
        elif rsi > 70:
            details["RSI"] = f"🔴 Overbought {rsi:.0f}"
        elif rsi < 35:
            details["RSI"] = f"🔴 Oversold {rsi:.0f}"
        else:
            score += 5
            details["RSI"] = f"⚪ {rsi:.0f}"

        if volr > 2.0:
            score += 20
            details["Volume"] = f"🟢 {volr:.1f}x"
        elif volr > 1.5:
            score += 15
            details["Volume"] = f"🟡 {volr:.1f}x"
        elif volr > 1.0:
            score += 5
            details["Volume"] = f"🟠 {volr:.1f}x"
        else:
            details["Volume"] = f"🔴 {volr:.1f}x"

        m5 = float(r["MOM_5D"])
        m10 = float(r["MOM_10D"])
        if m5 > 3 and m10 > 5:
            score += 15
            details["Momentum"] = f"🟢 ST +{m5:.1f}%"
        elif m5 > 1 and m10 > 2:
            score += 10
            details["Momentum"] = f"🟡 ST +{m5:.1f}%"
        elif m5 > 0:
            score += 5
            details["Momentum"] = f"🟠 ST +{m5:.1f}%"
        elif mom20 > 5:
            score += 8
            details["Momentum"] = f"🟡 20D +{mom20:.1f}%"

        score = int(score * penalty)
        grade, conf = grade_from_score(score)
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"


def score_bpjs(df: pd.DataFrame):
    try:
        ok, reason = apply_liquidity_filter(df)
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"

        r = df.iloc[-1]
        if r["Close"] < r["EMA21"]:
            return 0, {"Rejected": "Below EMA21"}, 0, "F"

        score, details = 0, {}
        pts, label = ema_alignment_score(r)
        score += {4: 35, 3: 25, 2: 10}.get(pts, 0)
        details["Trend"] = label

        volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 0.0
        if volr < 1.2:
            return 0, {"Rejected": f"Low intraday liquidity ({volr:.1f}x)"}, 0, "F"

        if volr > 3.0:
            score += 35
            details["Volume"] = f"🟢 {volr:.1f}x"
        elif volr > 2.0:
            score += 25
            details["Volume"] = f"🟡 {volr:.1f}x"
        else:
            score += 10
            details["Volume"] = f"🟠 {volr:.1f}x"

        atr = float(r["ATR_PCT"]) if not np.isnan(r["ATR_PCT"]) else 0.0
        if 2 <= atr <= 6:
            score += 20
            details["ATR"] = f"🟢 {atr:.1f}%"
        elif atr < 1.5:
            details["ATR"] = f"🔴 Too tight {atr:.1f}%"
        else:
            score += 10
            details["ATR"] = f"🟡 {atr:.1f}%"

        rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50
        if 48 <= rsi <= 70:
            score += 15
            details["RSI"] = f"🟢 {rsi:.0f}"
        elif rsi > 75:
            details["RSI"] = f"🔴 Overbought {rsi:.0f}"
        else:
            score += 5
            details["RSI"] = f"⚪ {rsi:.0f}"

        grade, conf = grade_from_score(score)
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"


def score_bsjp(df: pd.DataFrame):
    try:
        ok, reason = apply_liquidity_filter(df)
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"

        r = df.iloc[-1]
        if r["Close"] < r["EMA50"]:
            return 0, {"Rejected": "Below EMA50"}, 0, "F"

        score, details = 0, {}
        pts, label = ema_alignment_score(r)
        score += {4: 40, 3: 30, 2: 15}.get(pts, 0)
        details["Trend"] = label

        if float(r["MACD_HIST"]) > 0:
            score += 15
            details["MACD"] = "🟢 Histogram > 0"
        else:
            details["MACD"] = "🟠 Weak"

        volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
        if volr > 1.5:
            score += 15
            details["Volume"] = f"🟡 {volr:.1f}x"
        elif volr > 1.0:
            score += 8
            details["Volume"] = f"⚪ {volr:.1f}x"

        rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50
        if 50 <= rsi <= 68:
            score += 15
            details["RSI"] = f"🟢 {rsi:.0f}"
        elif rsi > 72:
            details["RSI"] = f"🔴 Overbought {rsi:.0f}"

        grade, conf = grade_from_score(score)
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"


def score_bandar(df: pd.DataFrame):
    try:
        ok, reason = apply_liquidity_filter(df)
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"

        r = df.iloc[-1]
        score, details = 0, {}

        relvol = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
        body_ratio = float(r.get("BODY_RATIO", np.nan))
        pos = float(r.get("POS_IN_RANGE", np.nan))

        if relvol < 1.3:
            return 0, {"Rejected": f"Volume not active ({relvol:.1f}x avg)"}, 0, "F"

        pattern = "-"
        if not np.isnan(body_ratio) and not np.isnan(pos):
            if (
                relvol >= 2.0
                and body_ratio <= 0.6
                and pos >= 0.7
                and r["Close"] >= r["Open"]
            ):
                pattern = "Bandar Masuk (Akumulasi)"
            elif (
                relvol >= 2.0
                and body_ratio <= 0.6
                and pos <= 0.3
                and r["Close"] <= r["Open"]
            ):
                pattern = "Bandar Keluar (Distribusi)"

        trend_local = detect_trend(r)
        if "Masuk" in pattern:
            if trend_local == "Downtrend" or r["Close"] < r["EMA50"]:
                pattern = "Akumulasi Risky (Downtrend)"

        details["RelVol"] = f"{relvol:.1f}x"
        if not np.isnan(body_ratio):
            details["BodyRatio"] = f"{body_ratio:.2f}"
        if not np.isnan(pos):
            details["PosInRange"] = f"{pos:.2f}"
        details["BandarPattern"] = pattern

        score += min(relvol, 5.0) * 12  # max 60

        pts, label = ema_alignment_score(r)
        score += {4: 20, 3: 12, 2: 5}.get(pts, 0)
        details["Trend"] = label

        if r["OBV"] > r["OBV_EMA"]:
            score += 10
            details["OBV"] = "🟢 OBV > OBV-EMA"
        else:
            details["OBV"] = "🟠 OBV <= OBV-EMA"

        if "Masuk" in pattern:
            score += 10
        elif "Keluar" in pattern:
            score += 5

        score = int(min(score, 100))
        grade, conf = grade_from_score(score)
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"


def score_swing(df: pd.DataFrame):
    try:
        ok, reason = apply_liquidity_filter(df)
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"

        r = df.iloc[-1]
        if r["Close"] < r["EMA50"]:
            return 0, {"Rejected": "Below EMA50"}, 0, "F"

        score, details = 0, {}
        pts, label = ema_alignment_score(r)
        score += {4: 45, 3: 30, 2: 15}.get(pts, 0)
        details["Trend"] = label

        dist = abs((r["Close"] - r["EMA21"]) / r["Close"]) * 100
        if dist <= 3:
            score += 15
            details["Pullback"] = f"🟢 EMA21 {dist:.1f}%"
        elif dist <= 5:
            score += 8
            details["Pullback"] = f"🟡 {dist:.1f}%"

        rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50
        if 50 <= rsi <= 65:
            score += 20
            details["RSI"] = f"🟢 {rsi:.0f}"
        elif 45 <= rsi < 50:
            score += 10
            details["RSI"] = f"🟡 {rsi:.0f}"
        elif rsi > 70:
            details["RSI"] = f"🔴 Overbought {rsi:.0f}"

        volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
        if volr > 1.5:
            score += 15
            details["Volume"] = f"🟡 {volr:.1f}x"

        grade, conf = grade_from_score(score)
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"


def score_value(df: pd.DataFrame):
    try:
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

        if r["Close"] <= r["BB_LOWER"]:
            score += 20
            details["BB"] = "🟢 Lower band touch"
        elif r["Close"] <= r["BB_MID"]:
            score += 10
            details["BB"] = "🟡 Below mid"

        volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
        if 0.9 <= volr <= 1.6:
            score += 15
            details["Volume"] = f"🟢 {volr:.1f}x"
        elif volr > 2.5:
            details["Volume"] = f"🔴 Blow-off? {volr:.1f}x"
        else:
            score += 5
            details["Volume"] = f"⚪ {volr:.1f}x"

        grade, conf = grade_from_score(score)
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"


# =============== TREND, SIGNAL, TRADE PLAN ===============

def detect_trend(r: pd.Series) -> str:
    price, ema9, ema21, ema50, ema200 = map(
        float, [r["Close"], r["EMA9"], r["EMA21"], r["EMA50"], r["EMA200"]]
    )
    if price > ema9 > ema21 > ema50 > ema200:
        return "Strong Uptrend"
    if price > ema50 and ema9 > ema21 > ema50:
        return "Uptrend"
    if abs(price - ema50) / price < 0.03:
        return "Sideways"
    return "Downtrend"


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
        and grade in ["A+", "A", "B+", "B"]
        and volr > 1.0
        and 40 <= rsi <= 75
    ):
        return "Buy"
    if trend in ["Strong Uptrend", "Uptrend"] and grade in ["A+", "A", "B+", "B"]:
        return "Hold"
    if trend == "Sideways" and grade in ["B+", "B", "C"]:
        return "Hold"
    return "Sell"


def compute_trade_plan(df: pd.DataFrame, strategy: str, trend: str) -> dict:
    r = df.iloc[-1]
    price = float(r["Close"])
    ema21 = float(r["EMA21"])

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
    else:
        entry = round_to_tick(price * 0.995)
        tp1 = round_to_tick(entry * 1.04)
        tp2 = round_to_tick(entry * 1.07)
        tp3 = None
        sl = round_to_tick(entry * 0.97)

    if trend in ["Strong Uptrend", "Uptrend"] and ema21 < price:
        ema_entry = round_to_tick(ema21 * 1.01)
        if price * 0.9 < ema_entry < price:
            entry = ema_entry

    if trend == "Downtrend":
        sl = round_to_tick(entry * 0.96)

    return {
        "entry_ideal": entry,
        "entry_aggressive": round_to_tick(price),
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
    }


# =============== ORCHESTRATION ===============

def analyze_ticker(ticker: str, strategy: str, period: str):
    df = fetch_data(ticker, period)
    if df is None or df.empty:
        return None

    r = df.iloc[-1]
    if strategy == "BPJS":
        score, details, conf, grade = score_bpjs(df)
    elif strategy == "BSJP":
        score, details, conf, grade = score_bsjp(df)
    elif strategy == "Bandar":
        score, details, conf, grade = score_bandar(df)
    elif strategy == "Swing":
        score, details, conf, grade = score_swing(df)
    elif strategy == "Value":
        score, details, conf, grade = score_value(df)
    else:
        score, details, conf, grade = score_general(df)

    if grade not in ["A+", "A", "B+", "B", "C"]:
        return None

    trend = detect_trend(r)
    signal = classify_signal(r, score, grade, trend)

    if strategy == "Bandar" and "BandarPattern" in details:
        signal = details["BandarPattern"]

    plan = compute_trade_plan(df, strategy, trend)

    res = {
        "Ticker": ticker.replace(".JK", ""),
        "Price": float(r["Close"]),
        "Score": score,
        "Confidence": conf,
        "Grade": grade,
        "Trend": trend,
        "Signal": signal,
        "Entry": plan["entry_ideal"],
        "Entry_Aggressive": plan["entry_aggressive"],
        "TP1": plan["tp1"],
        "TP2": plan["tp2"],
        "SL": plan["sl"],
        "Details": details,
    }
    if plan["tp3"]:
        res["TP3"] = plan["tp3"]
    if strategy == "Bandar":
        res["BandarMode"] = details.get("BandarPattern", "-")
    return res


def save_scan_to_session(df2: pd.DataFrame, df1: pd.DataFrame, strategy: str):
    st.session_state["last_scan_results"] = (df2, df1)
    st.session_state["last_scan_time"] = datetime.now()
    st.session_state["last_scan_strategy"] = strategy
    st.session_state["scan_count"] = (st.session_state["scan_count"] or 0) + 1


def display_last_scan_info():
    if st.session_state["last_scan_results"]:
        df2, _ = st.session_state["last_scan_results"]
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
            Time: {st.session_state['last_scan_time'].strftime('%H:%M:%S')}
            ({mins} min ago) | Found: {len(df2)} elite stocks
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


def process_ticker(t: str, strategy: str, period: str):
    try:
        return analyze_ticker(t, strategy, period)
    except Exception:
        return None


def scan_stocks(
    tickers,
    strategy: str,
    period: str,
    limit1: int,
    limit2: int,
    min_score_stage1: int = 55,
    min_score_stage2: int = 70,
):
    st.info(f"🔍 **STAGE 1**: Scanning {len(tickers)} stocks for {strategy}...")
    results = []
    progress = st.progress(0)
    status = st.empty()

    # Cloud-friendly: max_workers kecil
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(process_ticker, t, strategy, period): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            done += 1
            progress.progress(done / len(tickers))
            status.text(f"📊 {done}/{len(tickers)} | Found: {len(results)}")
            r = fut.result()
            if r:
                results.append(r)
            time.sleep(0.005)

    progress.empty()
    status.empty()

    if not results:
        return pd.DataFrame(), pd.DataFrame()

    df1 = (
        pd.DataFrame(results)
        .sort_values(["Score", "Confidence"], ascending=False)
    )
    df1 = df1[df1["Score"] >= min_score_stage1].head(limit1)

    if df1.empty:
        st.error("❌ Stage 1: Tidak ada yang lolos min score.")
        return pd.DataFrame(), pd.DataFrame()

    st.success(
        f"✅ Stage 1: Found {len(df1)} candidates "
        f"(Avg score: {df1['Score'].mean():.0f})"
    )

    if strategy == "Bandar":
        df2 = df1.copy()
        mask_masuk = df2["Signal"].astype(str).str.contains("Masuk", case=False, na=False)
        if mask_masuk.any():
            df2 = df2[mask_masuk]
        df2 = df2[df2["Score"] >= min_score_stage2].head(limit2)
    else:
        df2 = df1[
            (df1["Grade"].isin(["A+", "A", "B+", "B"]))
            & (df1["Signal"].isin(["Strong Buy", "Buy", "Hold"]))
            & (df1["Score"] >= min_score_stage2)
        ].head(limit2)

    if df2.empty:
        st.error("❌ Stage 2: Tidak ada yang lolos min score / filter.")
        return df1, pd.DataFrame()

    st.success(
        f"🏆 Stage 2: {len(df2)} elite picks "
        f"(Avg conf: {df2['Confidence'].mean():.0f}%)"
    )

    if strategy == "Bandar":
        masuk = df2["Signal"].astype(str).str.contains("Masuk", case=False, na=False).sum()
        keluar = df2["Signal"].astype(str).str.contains("Keluar", case=False, na=False).sum()
        st.info(f"🕵️‍♂️ Bandar Summary – Masuk: {masuk} | Keluar: {keluar}")
    else:
        strong = (df2["Signal"] == "Strong Buy").sum()
        buy = (df2["Signal"] == "Buy").sum()
        hold = (df2["Signal"] == "Hold").sum()
        st.info(f"🎯 Signal Summary – Strong Buy: {strong} | Buy: {buy} | Hold: {hold}")

    save_scan_to_session(df2, df1, strategy)
    return df1, df2


# =============== UI HELPERS ===============

def style_table(df: pd.DataFrame):
    def row_style(row):
        color = ""
        signal = str(row.get("Signal", ""))
        bandar = str(row.get("BandarMode", ""))

        if "Masuk" in signal or "Masuk" in bandar:
            color = "background-color: rgba(37, 99, 235, 0.18);"  # biru
        elif "Keluar" in signal or "Keluar" in bandar:
            color = "background-color: rgba(220, 38, 38, 0.18);"  # merah
        elif signal == "Strong Buy":
            color = "background-color: rgba(22, 163, 74, 0.18);"  # hijau
        elif signal == "Buy":
            color = "background-color: rgba(132, 204, 22, 0.15);"
        return [color] * len(row)

    return df.style.apply(row_style, axis=1)


def show_table(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        st.warning("Tidak ada hasil.")
        return

    st.subheader(title)
    show = df.copy()
    if "Price" in show:
        show["Price"] = show["Price"].apply(lambda x: round_to_tick(x))

    order_cols = [
        "Ticker",
        "Price",
        "Score",
        "Confidence",
        "Grade",
        "Trend",
        "Signal",
        "BandarMode",
        "Entry",
        "Entry_Aggressive",
        "TP1",
        "TP2",
        "TP3",
        "SL",
    ]
    order = [c for c in order_cols if c in show.columns]

    st.dataframe(
        style_table(show[order]),
        use_container_width=True,
        hide_index=True,
    )
    create_csv_download(show[order], title.replace(" ", "_"))

    account_size = st.session_state["account_size"]
    risk_pct = st.session_state["risk_pct"]

    with st.expander("🔎 Detail per saham + Position Size"):
        for _, row in show.iterrows():
            st.markdown(
                f"**{row['Ticker']}** | Grade **{row['Grade']}** | "
                f"Trend **{row['Trend']}** | Signal **{row['Signal']}**"
            )
            tp3 = (
                f" | TP3 {format_idr(row['TP3'])}"
                if "TP3" in row and pd.notna(row["TP3"])
                else ""
            )
            st.caption(
                f"Price {format_idr(row['Price'])} | "
                f"Entry {format_idr(row['Entry'])} "
                f"(Agg {format_idr(row['Entry_Aggressive'])}) | "
                f"TP1 {format_idr(row['TP1'])} | TP2 {format_idr(row['TP2'])}"
                f"{tp3} | SL {format_idr(row['SL'])}"
            )

            if (
                account_size > 0
                and row["Entry"] > 0
                and row["SL"] > 0
                and row["Entry"] > row["SL"]
            ):
                risk_per_share = row["Entry"] - row["SL"]
                max_loss = account_size * risk_pct / 100.0
                if risk_per_share > 0:
                    max_lots = math.floor(max_loss / (risk_per_share * 100))
                    if max_lots > 0:
                        est_value = max_lots * row["Entry"] * 100
                        st.caption(
                            f"📌 Position size (risk {risk_pct:.1f}% dari modal {format_idr(account_size)}): "
                            f"max **{max_lots} lot** (~{format_idr(est_value)})"
                        )

            details = row.get("Details", {})
            if isinstance(details, dict) and details:
                st.write({k: str(v) for k, v in details.items()})
            st.markdown("---")


# =============== MAIN UI ===============

st.title("🚀 IDX Power Screener v5.2 – Chef Edition")
st.caption(
    "BPJS / BSJP / Bandar / Swing / Value + IHSG + Tick-size + Bandarology + Risk Mgmt"
)

display_ihsg_widget()
tickers = load_tickers()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.info(f"📊 Total stocks: {len(tickers)}")
    st.caption(f"🕐 Jakarta: {get_jakarta_time().strftime('%H:%M WIB')}")
    st.markdown("---")

    menu = st.radio(
        "📋 Strategy",
        [
            "⚡ SPEED Trader (1-2d)",
            "🎯 SWING Trader (3-5d)",
            "💎 VALUE Plays (Undervalued)",
            "⚡ BPJS (Day Trading)",
            "🌙 BSJP (Overnight)",
            "🔮 Bandar Tracking",
            "🔍 Single Stock",
        ],
    )
    st.markdown("---")

    if "Single" not in menu:
        period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)

        st.markdown("### 🧪 Liquidity Filters")
        min_price = st.number_input(
            "Min Price (IDR)", min_value=50, max_value=5000, value=50, step=25
        )
        min_vol = st.number_input(
            "Min Avg Volume 20D",
            min_value=0,
            max_value=20_000_000,
            value=500_000,
            step=100_000,
        )
        min_turnover = st.number_input(
            "Min Turnover (Price x Vol)",
            min_value=0,
            max_value=5_000_000_000,
            value=100_000_000,
            step=50_000_000,
        )
        st.session_state["min_price_filter"] = min_price
        st.session_state["min_vol_filter"] = min_vol
        st.session_state["min_turnover_filter"] = min_turnover

        st.markdown("### 🎯 Stage Filters")
        c1, c2 = st.columns(2)
        with c1:
            limit1 = st.slider("Stage 1: Top N", 20, 100, 50, 10)
            min_score_stage1 = st.slider("Min Score Stage 1", 40, 90, 55, 5)
        with c2:
            limit2 = st.slider("Stage 2: Elite", 5, 30, 10, 5)
            min_score_stage2 = st.slider("Min Score Stage 2", 50, 95, 70, 5)

        st.markdown("### 🎯 Custom Tickers (opsional)")
        sublist_input = st.text_input(
            "Filter tickers (comma separated, optional)",
            placeholder="e.g. BBRI, BBCA, TLKM",
        )
        run_scan = st.button("🚀 RUN SCAN", type="primary", use_container_width=True)

    st.markdown("### 💰 Risk Management")
    account_size = st.number_input(
        "Account size (Rp)",
        min_value=0,
        max_value=5_000_000_000,
        value=int(st.session_state["account_size"]),
        step=1_000_000,
    )
    risk_pct = st.slider(
        "Risk per trade (%)", 0.1, 5.0, float(st.session_state["risk_pct"]), 0.1
    )
    st.session_state["account_size"] = float(account_size)
    st.session_state["risk_pct"] = float(risk_pct)

    st.caption(
        "v5.2 – Strong filters, Bandar pattern, risk mgmt – educational only, not advice."
    )

display_last_scan_info()

# =============== ROUTING ===============

if "Single Stock" in menu:
    st.markdown("### 🔍 Single Stock Analysis")
    default_symbol = tickers[0].replace(".JK", "") if tickers else "BBRI"
    selected = st.text_input("Symbol (tanpa .JK)", value=default_symbol).strip().upper()
    strategy_single = st.selectbox(
        "Strategy", ["General", "BPJS", "BSJP", "Bandar", "Swing", "Value"]
    )
    period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)

    if st.button("🔍 ANALYZE", type="primary"):
        full = normalize_ticker(selected)
        with st.spinner(f"Analyzing {selected}..."):
            df = fetch_data(full, period)
            if df is None:
                st.error("❌ Failed to fetch data")
            else:
                res = analyze_ticker(full, strategy_single, period)
                st.markdown("### 📊 Interactive Chart")
                chart = create_chart(df, selected)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                if res is None:
                    st.error("❌ Analysis failed or rejected by filters")
                else:
                    st.markdown(f"## 💎 {res['Ticker']}")
                    st.markdown(
                        f"### Grade: **{res['Grade']}** | "
                        f"Trend: **{res['Trend']}** | Signal: **{res['Signal']}**"
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Price", format_idr(res["Price"]))
                    c2.metric("Score", f"{res['Score']}/100")
                    c3.metric("Confidence", f"{res['Confidence']}%")

                    label_map = {
                        "General": "SPEED PLAN (1–2 Hari)",
                        "BPJS": "BPJS PLAN (Intraday)",
                        "BSJP": "BSJP PLAN (Overnight)",
                        "Bandar": "Bandar PLAN (Momentum/Flow)",
                        "Swing": "SWING PLAN (3–5 Hari)",
                        "Value": "VALUE PLAN (Rebound)",
                    }
                    label = label_map.get(strategy_single, "PLAN")

                    tp3_line = (
                        f"• **TP3:** {format_idr(res['TP3'])}  \n"
                        if "TP3" in res
                        else ""
                    )
                    st.success(
                        f"⚡ **{label}**  \n"
                        f"• **Entry Ideal:** {format_idr(res['Entry'])}  \n"
                        f"• **Entry Agresif:** {format_idr(res['Entry_Aggressive'])}  \n"
                        f"• **TP1:** {format_idr(res['TP1'])}  \n"
                        f"• **TP2:** {format_idr(res['TP2'])}  \n"
                        f"{tp3_line}"
                        f"• **Stop Loss:** {format_idr(res['SL'])}"
                    )

                    account_size = st.session_state["account_size"]
                    risk_pct = st.session_state["risk_pct"]
                    if (
                        account_size > 0
                        and res["Entry"] > 0
                        and res["SL"] > 0
                        and res["Entry"] > res["SL"]
                    ):
                        risk_per_share = res["Entry"] - res["SL"]
                        max_loss = account_size * risk_pct / 100.0
                        if risk_per_share > 0:
                            max_lots = math.floor(max_loss / (risk_per_share * 100))
                            if max_lots > 0:
                                est_value = max_lots * res["Entry"] * 100
                                st.info(
                                    f"📌 Position size (risk {risk_pct:.1f}% dari modal "
                                    f"{format_idr(account_size)}): "
                                    f"max **{max_lots} lot** "
                                    f"(~{format_idr(est_value)})"
                                )

                    st.markdown("**Technical Notes:**")
                    for k, v in res["Details"].items():
                        st.caption(f"• **{k}**: {v}")
else:
    strategy_map = {
        "⚡ SPEED Trader (1-2d)": "General",
        "🎯 SWING Trader (3-5d)": "Swing",
        "💎 VALUE Plays (Undervalued)": "Value",
        "⚡ BPJS (Day Trading)": "BPJS",
        "🌙 BSJP (Overnight)": "BSJP",
        "🔮 Bandar Tracking": "Bandar",
    }
    strategy = strategy_map[menu]
    run_list = tickers

    if "sublist_input" in locals() and sublist_input:
        parts = [normalize_ticker(x) for x in sublist_input.split(",") if x.strip()]
        if parts:
            run_list = parts
            st.info(
                "Using custom tickers: "
                + ", ".join([p.replace(".JK", "") for p in parts])
            )

    if "run_scan" in locals() and run_scan:
        if not run_list:
            st.error(
                "Daftar tickers kosong. Tambahkan idx_stocks.json atau input custom."
            )
        else:
            df1, df2 = scan_stocks(
                run_list,
                strategy,
                period,
                limit1,
                limit2,
                min_score_stage1,
                min_score_stage2,
            )
            if df1.empty and df2.empty:
                st.error("❌ No stocks matched filters.")
            else:
                show_table(df1, "Stage 1 – Candidates")
                if not df2.empty:
                    show_table(df2, "Stage 2 – Elite Picks")
    else:
        st.info("Klik **RUN SCAN** untuk memulai pemindaian saham.")

st.markdown("---")
st.caption(
    "🚀 IDX Power Screener v5.2 | Educational only | Tick-size aware | Bandar pattern | "
    "Risk-based position sizing | Data from Yahoo (may be delayed)."
)