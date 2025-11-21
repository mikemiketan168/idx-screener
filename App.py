#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================== BASIC CONFIG ==================
st.set_page_config(
    page_title="IDX Power Screener v6 – Educational Only",
    page_icon="🚀",
    layout="wide"
)

DISCLAIMER = (
    "⚠️ Educational only – bukan rekomendasi beli/jual. "
    "Gunakan sebagai bahan belajar & riset pribadi."
)

# -------------------- CONSTANTS & UTILS --------------------
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

# -------------------- SESSION STATE --------------------
for k, v in [
    ("last_scan_results", None),
    ("last_scan_time", None),
    ("last_scan_strategy", None),
    ("scan_count", 0),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- IHSG WIDGET --------------------
@st.cache_data(ttl=180)
def fetch_ihsg_data():
    try:
        import yfinance as yf
        ihsg = yf.Ticker("^JKSE")
        hist = ihsg.history(period="1d", interval="1d", auto_adjust=False)
        if hist.empty:
            return None
        current = float(hist["Close"].iloc[-1])
        open_price = float(hist["Open"].iloc[-1])
        high = float(hist["High"].iloc[-1])
        low = float(hist["Low"].iloc[-1])
        change = current - open_price
        change_pct = (change / open_price) * 100 if open_price else 0.0
        return {
            "price": current,
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
        condition, guidance = (
            "🔥 Strong uptrend - Good for momentum!",
            "✅ Cocok untuk SPEED / SWING / GORILLA",
        )
    elif ihsg["change_pct"] > 0.5:
        condition, guidance = (
            "📈 Moderate uptrend - Good conditions",
            "✅ Cukup OK untuk semua strategi",
        )
    elif ihsg["change_pct"] > -0.5:
        condition, guidance = (
            "➡️ Sideways - Mixed conditions",
            "⚠️ Lebih selektif, perhatikan risk",
        )
    elif ihsg["change_pct"] > -1.5:
        condition, guidance = (
            "📉 Moderate downtrend - Caution",
            "⚠️ Fokus defensif / kecilkan posisi",
        )
    else:
        condition, guidance = (
            "🔻 Strong downtrend - High risk",
            "❌ Lebih baik wait & see / kecil sekali",
        )

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
          ⏰ Last update: {datetime.now().strftime('%H:%M:%S')} WIB | 🔄 Refresh: tekan RUN SCAN / ANALYZE
        </p>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -------------------- TICKERS LIST --------------------
@st.cache_data(ttl=3600)
def load_tickers() -> list[str]:
    try:
        if os.path.exists("idx_stocks.json"):
            with open("idx_stocks.json", "r") as f:
                data = json.load(f)
            return [normalize_ticker(t) for t in data.get("tickers", []) if t]
    except Exception:
        pass
    # fallback minimal
    return ["BBRI.JK", "BBCA.JK", "TLKM.JK", "ASII.JK", "ICBP.JK", "INDF.JK"]

# -------------------- FETCH & INDICATORS --------------------
def _yahoo_chart_json(ticker: str, period: str) -> dict | None:
    end = int(datetime.now().timestamp())
    days = {
        "5d": 5,
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
    }.get(period, 180)
    start = end - (days * 86400)
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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
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

        # EMA
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

        # Volume
        df["VOL_SMA20"] = df["Volume"].rolling(20).mean()
        df["VOL_SMA50"] = df["Volume"].rolling(50).mean()
        df["VOL_RATIO"] = df["Volume"] / df["VOL_SMA20"].replace(0, np.nan)

        # Momentum
        df["MOM_5D"] = df["Close"].pct_change(5) * 100
        df["MOM_10D"] = df["Close"].pct_change(10) * 100
        df["MOM_20D"] = df["Close"].pct_change(20) * 100

        # OBV
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

        # Bollinger
        df["BB_MID"] = df["Close"].rolling(20).mean()
        df["BB_STD"] = df["Close"].rolling(20).std()
        df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
        df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]
        df["BB_WIDTH"] = ((df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"]) * 100

        # Stochastic
        low14 = df["Low"].rolling(14).min()
        high14 = df["High"].rolling(14).max()
        rng = (high14 - low14).replace(0, np.nan)
        df["STOCH_K"] = 100 * (df["Close"] - low14) / rng
        df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

        # ATR
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - df["Close"].shift()).abs()
        tr3 = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()
        df["ATR_PCT"] = (df["ATR"] / df["Close"]) * 100

        # MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

        return df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    except Exception:
        return None

# -------------------- CHARTS --------------------
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
            "EMA200": "#9E9E3E",
        }
        for ema in ["EMA9", "EMA21", "EMA50", "EMA200"]:
            if ema in d.columns:
                fig.add_trace(
                    go.Scatter(
                        x=d.index,
                        y=d[ema],
                        name=ema,
                        line=dict(color=colors.get(ema, "#FFFFFF"), width=1.5),
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
        fig.add_hline(
            y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1
        )
        fig.add_hline(
            y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1
        )
        fig.add_hline(
            y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1
        )

        fig.update_layout(
            height=700,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            template="plotly_dark",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#333")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#333")
        return fig
    except Exception:
        return None

# -------------------- SCORING HELPERS --------------------
def ema_alignment_score(r: pd.Series):
    pts = (
        int(r["EMA9"] > r["EMA21"]) +
        int(r["EMA21"] > r["EMA50"]) +
        int(r["EMA50"] > r["EMA200"]) +
        int(r["Close"] > r["EMA9"])
    )
    label = {
        4: "🟢 Very Strong",
        3: "🟡 Strong",
        2: "🟠 Moderate",
        1: "🔴 Weak",
        0: "🔴 Weak",
    }[pts]
    return pts, label

def grade_from_score(score: int):
    if score >= 90:
        return "A+", 90
    if score >= 80:
        return "A", 80
    if score >= 70:
        return "B+", 75
    if score >= 60:
        return "B", 70
    if score >= 50:
        return "C", 60
    return "D", max(score, 0)

# -------------------- PHASE 1: LIQUIDITY FILTER --------------------
def apply_liquidity_filter(df: pd.DataFrame, mode: str = "default"):
    """
    Filter anti saham tidur:
    - Harga minimal 50
    - Avg volume 20 hari cukup
    - Turnover cukup
    - Hari aktif (volume > 0) minimal
    - Volume hari ini minimal
    Mode "gorilla" sedikit lebih ketat.
    """
    try:
        r = df.iloc[-1]
        price = float(r["Close"])
        vol20 = float(df["Volume"].tail(20).mean())
        vol_today = float(r["Volume"])
        turnover = price * vol20
        active_days = int((df["Volume"].tail(20) > 0).sum())

        # base threshold
        min_price = 50
        min_vol20 = 500_000
        min_turnover = 50_000_000
        min_active_days = 15
        min_today_vol = 200_000

        if mode == "gorilla":
            min_vol20 = 1_000_000
            min_turnover = 100_000_000
            min_today_vol = 400_000

        if price < min_price:
            return False, f"Price < {min_price}"
        if vol20 < min_vol20:
            return False, f"Avg 20D vol < {min_vol20:,}"
        if turnover < min_turnover:
            return False, f"Turnover < {min_turnover:,}"
        if active_days < min_active_days:
            return False, "Too many zero-volume days"
        if vol_today < min_today_vol:
            return False, f"Today vol < {min_today_vol:,}"

        return True, "Passed"
    except Exception:
        return False, "Error"

# -------------------- STRATEGY SCORING --------------------
def score_general(df: pd.DataFrame):
    """SPEED Trader (1-2 hari)"""
    try:
        ok, reason = apply_liquidity_filter(df, mode="default")
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"
        r = df.iloc[-1]

        if r["Close"] < r["EMA50"]:
            return 0, {"Rejected": "Below EMA50 (down/side)"}, 0, "F"

        score, details = 0, {}

        pts, label = ema_alignment_score(r)
        score += {4: 45, 3: 30, 2: 15}.get(pts, 0)
        details["Trend_EMA"] = label

        mom20 = float(r["MOM_20D"])
        if mom20 < -8:
            return 0, {"Rejected": f"Strong negative momentum ({mom20:.1f}%)"}, 0, "F"
        if mom20 > 12:
            score += 15
            details["MOM 20D"] = f"🟢 {mom20:.1f}%"
        elif mom20 > 5:
            score += 10
            details["MOM 20D"] = f"🟡 {mom20:.1f}%"
        elif mom20 > 0:
            score += 5
            details["MOM 20D"] = f"⚪ {mom20:.1f}%"
        else:
            details["MOM 20D"] = f"🟠 {mom20:.1f}%"

        rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50.0
        if 50 <= rsi <= 65:
            score += 20
            details["RSI"] = f"🟢 {rsi:.0f}"
        elif 45 <= rsi < 50:
            score += 12
            details["RSI"] = f"🟡 {rsi:.0f}"
        elif 40 <= rsi < 45:
            score += 5
            details["RSI"] = f"🟠 {rsi:.0f}"
        elif rsi > 70:
            details["RSI"] = f"🔴 Overbought {rsi:.0f}"
        else:
            details["RSI"] = f"⚪ {rsi:.0f}"

        volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
        if volr > 2.0:
            score += 15
            details["Volume"] = f"🟢 {volr:.1f}x"
        elif volr > 1.3:
            score += 10
            details["Volume"] = f"🟡 {volr:.1f}x"
        elif volr > 1.0:
            score += 5
            details["Volume"] = f"⚪ {volr:.1f}x"
        else:
            details["Volume"] = f"🟠 {volr:.1f}x"

        m5 = float(r["MOM_5D"])
        m10 = float(r["MOM_10D"])
        if m5 > 3 and m10 > 5:
            score += 15
            details["ST Momentum"] = f"🟢 +{m5:.1f}% (5D)"
        elif m5 > 1 and m10 > 2:
            score += 10
            details["ST Momentum"] = f"🟡 +{m5:.1f}% (5D)"
        elif m5 > 0:
            score += 5
            details["ST Momentum"] = f"⚪ +{m5:.1f}%"
        else:
            details["ST Momentum"] = f"🟠 {m5:.1f}%"

        grade, conf = grade_from_score(score)
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"

def score_bpjs(df: pd.DataFrame):
    """Short-term momentum / pseudo-daytrading (based on daily data)"""
    try:
        ok, reason = apply_liquidity_filter(df, mode="default")
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"
        r = df.iloc[-1]

        if r["Close"] < r["EMA21"]:
            return 0, {"Rejected": "Below EMA21"}, 0, "F"

        score, details = 0, {}

        pts, label = ema_alignment_score(r)
        score += {4: 35, 3: 25, 2: 10}.get(pts, 0)
        details["Trend_EMA"] = label

        volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 0.0
        if volr < 1.0:
            details["Volume"] = f"🟠 {volr:.1f}x"
        elif volr > 3.0:
            score += 30
            details["Volume"] = f"🟢 {volr:.1f}x"
        elif volr > 2.0:
            score += 20
            details["Volume"] = f"🟡 {volr:.1f}x"
        else:
            score += 10
            details["Volume"] = f"⚪ {volr:.1f}x"

        atr = float(r["ATR_PCT"]) if not np.isnan(r["ATR_PCT"]) else 0.0
        if 2.0 <= atr <= 6.0:
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
            details["RSI"] = f"⚪ {rsi:.0f}"

        grade, conf = grade_from_score(score)
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"

def score_bsjp(df: pd.DataFrame):
    """Overnight swing pendek"""
    try:
        ok, reason = apply_liquidity_filter(df, mode="default")
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"
        r = df.iloc[-1]

        if r["Close"] < r["EMA50"]:
            return 0, {"Rejected": "Below EMA50"}, 0, "F"

        score, details = 0, {}

        pts, label = ema_alignment_score(r)
        score += {4: 40, 3: 30, 2: 15}.get(pts, 0)
        details["Trend_EMA"] = label

        if float(r["MACD_HIST"]) > 0:
            score += 15
            details["MACD"] = "🟢 Histogram > 0"
        else:
            details["MACD"] = "🟠 Lemah"

        volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
        if volr > 1.5:
            score += 15
            details["Volume"] = f"🟡 {volr:.1f}x"
        elif volr > 1.0:
            score += 8
            details["Volume"] = f"⚪ {volr:.1f}x"
        else:
            details["Volume"] = f"🟠 {volr:.1f}x"

        rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50
        if 50 <= rsi <= 68:
            score += 15
            details["RSI"] = f"🟢 {rsi:.0f}"
        elif rsi > 72:
            details["RSI"] = f"🔴 Overbought {rsi:.0f}"
        else:
            details["RSI"] = f"⚪ {rsi:.0f}"

        grade, conf = grade_from_score(score)
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"

def score_bandar(df: pd.DataFrame):
    """Bandarmology basic via OBV + Volume + BB"""
    try:
        ok, reason = apply_liquidity_filter(df, mode="default")
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"
        r = df.iloc[-1]
        score, details = 0, {}

        if r["OBV"] > r["OBV_EMA"]:
            score += 30
            details["OBV"] = "🟢 Above OBV-EMA (accumulation)"
        else:
            details["OBV"] = "🟠 Below OBV-EMA"

        width = float(r["BB_WIDTH"]) if not np.isnan(r["BB_WIDTH"]) else 0.0
        recent = df["BB_WIDTH"].tail(60).dropna()
        p25 = float(np.percentile(recent, 25)) if len(recent) else 0.0
        if width < p25:
            score += 15
            details["BB Squeeze"] = f"🟢 {width:.2f}% (low volatility)"

        if r["Close"] > r["BB_UPPER"]:
            score += 20
            details["BB Breakout"] = "🟢 Close > Upper band"

        pts, label = ema_alignment_score(r)
        score += {4: 25, 3: 15, 2: 5}.get(pts, 0)
        details["Trend_EMA"] = label

        volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
        if volr > 2.0:
            score += 20
            details["Volume"] = f"🟢 {volr:.1f}x"
        elif volr > 1.2:
            score += 10
            details["Volume"] = f"🟡 {volr:.1f}x"
        else:
            details["Volume"] = f"⚪ {volr:.1f}x"

        grade, conf = grade_from_score(score)
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"

def score_swing(df: pd.DataFrame):
    """Swing 3–5 hari"""
    try:
        ok, reason = apply_liquidity_filter(df, mode="default")
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"
        r = df.iloc[-1]

        if r["Close"] < r["EMA50"]:
            return 0, {"Rejected": "Below EMA50"}, 0, "F"

        score, details = 0, {}

        pts, label = ema_alignment_score(r)
        score += {4: 45, 3: 30, 2: 15}.get(pts, 0)
        details["Trend_EMA"] = label

        dist = abs((r["Close"] - r["EMA21"]) / r["Close"]) * 100
        if dist <= 3:
            score += 15
            details["Pullback"] = f"🟢 EMA21 {dist:.1f}%"
        elif dist <= 5:
            score += 8
            details["Pullback"] = f"🟡 {dist:.1f}%"
        else:
            details["Pullback"] = f"⚪ {dist:.1f}%"

        rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50
        if 50 <= rsi <= 65:
            score += 20
            details["RSI"] = f"🟢 {rsi:.0f}"
        elif 45 <= rsi < 50:
            score += 10
            details["RSI"] = f"🟡 {rsi:.0f}"
        elif rsi > 70:
            details["RSI"] = f"🔴 Overbought {rsi:.0f}"
        else:
            details["RSI"] = f"⚪ {rsi:.0f}"

        volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
        if volr > 1.5:
            score += 15
            details["Volume"] = f"🟢 {volr:.1f}x"
        elif volr > 1.0:
            score += 8
            details["Volume"] = f"⚪ {volr:.1f}x"
        else:
            details["Volume"] = f"🟠 {volr:.1f}x"

        grade, conf = grade_from_score(score)
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"

def score_value(df: pd.DataFrame):
    """Value / Discount / Rebound (teknikal)"""
    try:
        ok, reason = apply_liquidity_filter(df, mode="default")
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"
        r = df.iloc[-1]
        score, details = 0, {}

        if r["Close"] >= r["EMA200"]:
            score += 25
            details["Trend_EMA"] = "🟢 >= EMA200 (uptrend/value)"
        else:
            score += 10
            details["Trend_EMA"] = "🟡 < EMA200 (mean reversion)"

        rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 50
        if 35 <= rsi <= 50:
            score += 25
            details["RSI"] = f"🟢 {rsi:.0f} (value zone)"
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
        else:
            details["BB"] = "⚪ Near/above mid"

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

def score_gorilla(df: pd.DataFrame):
    """GORILLA MODE v2 – agresif, cari yang mau meledak"""
    try:
        ok, reason = apply_liquidity_filter(df, mode="gorilla")
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"
        r = df.iloc[-1]
        score, details = 0, {}

        if r["Close"] > r["EMA9"] > r["EMA21"] > r["EMA50"]:
            score += 40
            details["Trend_EMA"] = "🦍 Gorilla Uptrend"
        else:
            return 0, {"Rejected": "Trend kurang kuat (butuh Close>EMA9>EMA21>EMA50)"}, 0, "F"

        m5 = float(r["MOM_5D"])
        m20 = float(r["MOM_20D"])
        if m5 > 7 and m20 > 15:
            score += 30
            details["Momentum"] = f"🟢 {m5:.1f}% (5D) / {m20:.1f}% (20D)"
        elif m5 > 5 and m20 > 10:
            score += 20
            details["Momentum"] = f"🟡 {m5:.1f}% / {m20:.1f}%"
        else:
            return 0, {"Rejected": "Momentum kurang agresif"}, 0, "F"

        volr = float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 1.0
        if volr >= 3.0:
            score += 20
            details["Volume"] = f"🟢 {volr:.1f}x"
        elif volr >= 2.0:
            score += 10
            details["Volume"] = f"🟡 {volr:.1f}x"
        else:
            return 0, {"Rejected": f"VOL_RATIO {volr:.1f}x < 2"}, 0, "F"

        atr = float(r["ATR_PCT"]) if not np.isnan(r["ATR_PCT"]) else 0.0
        if 3.0 <= atr <= 10.0:
            score += 10
            details["ATR"] = f"🟢 {atr:.1f}%"
        else:
            details["ATR"] = f"⚪ {atr:.1f}%"

        rsi = float(r["RSI"]) if not np.isnan(r["RSI"]) else 55.0
        if 55 <= rsi <= 78:
            score += 10
            details["RSI"] = f"🟢 {rsi:.0f}"
        elif rsi > 80:
            details["RSI"] = f"🔴 Overbought {rsi:.0f}"
        else:
            details["RSI"] = f"⚪ {rsi:.0f}"

        grade, conf = grade_from_score(score)
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"

# -------------------- TREND, SIGNAL & TRADE PLAN --------------------
def detect_trend(r: pd.Series) -> str:
    """Deteksi tren yang sedikit lebih ramah pullback."""
    price = float(r["Close"])
    ema9 = float(r["EMA9"])
    ema21 = float(r["EMA21"])
    ema50 = float(r["EMA50"])
    ema200 = float(r["EMA200"])

    if price <= 0:
        return "Sideways"

    # Strong uptrend: EMA50 > EMA200, EMA21 > EMA50, harga di atas EMA21
    if ema50 > ema200 and ema21 > ema50 and price >= ema21:
        return "Strong Uptrend"

    # Uptrend: EMA50 > EMA200, harga tidak lebih dari 3% di bawah EMA50
    if ema50 > ema200 and price >= ema50 * 0.97:
        return "Uptrend"

    # Sideways: harga sekitar EMA50 (±5%)
    if abs(price - ema50) / price <= 0.05:
        return "Sideways"

    return "Downtrend"

def classify_signal(r: pd.Series, score: int, grade: str, trend: str) -> str:
    """Klasifikasi sinyal netral: Strong Bullish / Bullish / Neutral / Bearish."""
    rsi = float(r.get("RSI", 50.0)) if pd.notna(r.get("RSI", np.nan)) else 50.0
    volr = float(r.get("VOL_RATIO", 1.0)) if pd.notna(r.get("VOL_RATIO", np.nan)) else 1.0
    m5 = float(r.get("MOM_5D", 0.0)) if pd.notna(r.get("MOM_5D", np.nan)) else 0.0
    m20 = float(r.get("MOM_20D", 0.0)) if pd.notna(r.get("MOM_20D", np.nan)) else 0.0

    # Kriteria paling kuat
    if (
        trend in ["Strong Uptrend", "Uptrend"]
        and grade in ["A+", "A"]
        and volr > 1.5
        and 50 <= rsi <= 70
        and m5 > 0
        and m20 > 0
    ):
        return "Strong Bullish"

    # Masih uptrend dan kualitas bagus
    if (
        trend in ["Strong Uptrend", "Uptrend"]
        and grade in ["A+", "A", "B+"]
        and volr > 1.0
        and 45 <= rsi <= 75
    ):
        return "Bullish"

    # Trend oke tapi kualitas sedang
    if trend in ["Strong Uptrend", "Uptrend"] and grade in ["A+", "A", "B+", "B", "C"]:
        return "Neutral"

    # Selain itu anggap Bearish / lemah
    return "Bearish"

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
    elif strategy == "Gorilla":
        entry = round_to_tick(price * 0.995)
        tp1 = round_to_tick(entry * 1.06)
        tp2 = round_to_tick(entry * 1.12)
        tp3 = round_to_tick(entry * 1.18)
        sl = round_to_tick(entry * 0.96)
    else:
        # General / BPJS / BSJP / Bandar
        entry = round_to_tick(price * 0.995)
        tp1 = round_to_tick(entry * 1.04)
        tp2 = round_to_tick(entry * 1.07)
        tp3 = None
        sl = round_to_tick(entry * 0.97)

    # Adjust entry dekat EMA21 kalau tren bagus
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

# -------------------- ORCHESTRATION --------------------
def analyze_ticker(ticker: str, strategy: str, period: str) -> dict | None:
    df = fetch_data(ticker, period)
    if df is None or df.empty:
        return None
    r = df.iloc[-1]

    # 1) Scoring per strategi
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
    elif strategy == "Gorilla":
        score, details, conf, grade = score_gorilla(df)
    else:
        score, details, conf, grade = score_general(df)

    if grade not in ["A+", "A", "B+", "B", "C"]:
        return None

    # 2) Trend & Signal
    trend = detect_trend(r)
    signal = classify_signal(r, score, grade, trend)

    # 3) Hard filter untuk screener multi-stocks:
    bullish_strategies = {"General", "BPJS", "BSJP", "Bandar", "Swing", "Gorilla"}
    if strategy in bullish_strategies:
        if trend not in ["Strong Uptrend", "Uptrend"]:
            return None
        if signal not in ["Strong Bullish", "Bullish"]:
            return None

    # 4) Trade plan
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
    return res

def save_scan_to_session(df2: pd.DataFrame, df1: pd.DataFrame, strategy: str):
    st.session_state.last_scan_results = (df2, df1)
    st.session_state.last_scan_time = datetime.now()
    st.session_state.last_scan_strategy = strategy
    st.session_state.scan_count = (st.session_state.scan_count or 0) + 1

def display_last_scan_info() -> bool:
    if st.session_state.last_scan_results:
        df2, _ = st.session_state.last_scan_results
        mins = int(
            (datetime.now() - st.session_state.last_scan_time).total_seconds() / 60
        )
        st.markdown(
            f"""
        <div style='background:linear-gradient(135deg,#064e3b 0%,#065f46 100%);
                    padding:12px;border-radius:8px;margin-bottom:15px;
                    border-left:4px solid #10b981;'>
          <p style='margin:0;color:white;font-weight:bold;'>📂 LAST SCAN RESULTS</p>
          <p style='margin:5px 0 0 0;color:#d1fae5;font-size:0.9em;'>
            Strategy: {st.session_state.last_scan_strategy} |
            Time: {st.session_state.last_scan_time.strftime('%H:%M:%S')} ({mins} min ago) |
            Found: {len(df2)} Grade A+/A/B+/B stocks
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return True
    return False

def create_csv_download(df: pd.DataFrame, strategy_label: str, button_key: str):
    if df is None or df.empty:
        return
    export = df.copy()
    csv = export.to_csv(index=False).encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "💾 Download Results (CSV)",
        data=csv,
        file_name=f"IDX_{strategy_label}_scan_{ts}.csv",
        mime="text/csv",
        key=button_key,
    )

def process_ticker(t: str, strategy: str, period: str):
    try:
        return analyze_ticker(t, strategy, period)
    except Exception:
        return None

def scan_stocks(
    tickers: list[str], strategy: str, period: str, limit1: int, limit2: int
):
    st.info(f"🔍 **STAGE 1**: Scanning {len(tickers)} stocks for {strategy}...")
    results = []
    progress = st.progress(0)
    status = st.empty()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(process_ticker, t, strategy, period): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            done += 1
            progress.progress(done / len(tickers))
            status.text(f"📊 {done}/{len(tickers)} | Collected: {len(results)} bullish candidates")
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
        .head(limit1)
    )
    st.success(
        f"✅ Stage 1: Found {len(df1)} candidates (Avg score: {df1['Score'].mean():.0f})"
    )

    df2 = df1[df1["Grade"].isin(["A+", "A", "B+"])]  # lebih elite
    df2 = df2.head(limit2)
    if not df2.empty:
        st.success(
            f"🏆 Stage 2: {len(df2)} elite picks (Avg conf: {df2['Confidence'].mean():.0f}%)"
        )
    else:
        st.warning("Stage 2: Tidak ada yang memenuhi kriteria elite (A+/A/B+).")

    save_scan_to_session(df2, df1, strategy)
    return df1, df2

# -------------------- UI HELPERS --------------------
def show_table(df: pd.DataFrame, title: str, strategy_label: str):
    if df is None or df.empty:
        st.warning("Tidak ada hasil.")
        return
    st.subheader(title)
    show = df.copy()

    # Bulatkan ke tick IDX
    if "Price" in show:
        show["Price"] = show["Price"].apply(lambda x: round_to_tick(x))

    # Rename SL -> CL (Cut Loss) untuk tampilan
    if "SL" in show.columns:
        show = show.rename(columns={"SL": "CL"})

    order = [
        c
        for c in [
            "Ticker",
            "Price",
            "Entry",             # Ideal entry
            "Entry_Aggressive",  # Agresif entry
            "TP1",
            "TP2",
            "TP3",
            "CL",                # dari SL
            "Signal",
            "Trend",
            "Score",
            "Confidence",
            "Grade",
        ]
        if c in show.columns
    ]

    st.dataframe(show[order], use_container_width=True, hide_index=True)
    create_csv_download(
        show[order],
        strategy_label,
        button_key=f"download_{strategy_label}_{title.replace(' ', '_')}",
    )

    with st.expander("🔎 Detail per saham"):
        for _, row in show.iterrows():
            st.markdown(
                f"**{row['Ticker']}** | Grade **{row['Grade']}** | Trend **{row['Trend']}** | Signal **{row['Signal']}**"
            )
            tp3_line = (
                f" | TP3 {format_idr(row['TP3'])}"
                if "TP3" in row and pd.notna(row["TP3"])
                else ""
            )
            st.caption(
                f"Price {format_idr(row['Price'])} | "
                f"Ideal Entry {format_idr(row['Entry'])} "
                f"(Agresif {format_idr(row['Entry_Aggressive'])}) | "
                f"TP1 {format_idr(row['TP1'])} | TP2 {format_idr(row['TP2'])}"
                f"{tp3_line} | CL {format_idr(row['CL'])}"
            )
            details = row.get("Details", {})
            if isinstance(details, dict) and details:
                st.write({k: str(v) for k, v in details.items()})
            st.markdown("---")

# ================== MAIN UI ==================
st.title("🚀 IDX Power Screener v6 – Educational Only")
st.caption(
    "BPJS / BSJP / Bandar / Swing / Value / Gorilla + IHSG + Session + Tick-size | "
    "Tujuan: bantu belajar & riset, bukan rekomendasi. "
    + DISCLAIMER
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
            "⚡ SPEED Trader (1-3d)",
            "🎯 SWING Trader (3-5d)",
            "💎 VALUE Plays (Technical Discount)",
            "⚡ BPJS (Short-Term Momentum)",
            "🌙 BSJP (Overnight)",
            "🔮 Bandar Tracking",
            "🦍 Gorilla Mode (Very Aggressive)",
            "🔍 Single Stock",
        ],
    )
    st.markdown("---")

    if "Single Stock" not in menu:
        period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
        st.markdown("### 🎯 Internal Filtering (fixed by code)")
        c1, c2 = st.columns(2)
        with c1:
            limit1 = st.slider("Stage 1: Top N", 50, 200, 100, 10)
        with c2:
            limit2 = st.slider("Stage 2: Elite", 10, 50, 20, 5)
        sublist_input = st.text_input(
            "Filter tickers (comma separated, optional)",
            placeholder="e.g. BBRI, BBCA, TLKM",
        )
        run_scan = st.button("🚀 RUN SCAN", type="primary", use_container_width=True)
    st.caption("v6 – Fokus bullish & uptrend, anti saham tidur, Gorilla v2")

display_last_scan_info()

# ================== ROUTING ==================
if "Single Stock" in menu:
    st.markdown("### 🔍 Single Stock Analysis")
    default_symbol = tickers[0].replace(".JK", "") if tickers else "BBRI"
    selected = st.text_input(
        "Symbol (tanpa .JK)", value=default_symbol
    ).strip().upper()
    strategy_single = st.selectbox(
        "Strategy", ["General", "BPJS", "BSJP", "Bandar", "Swing", "Value", "Gorilla"]
    )
    period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)

    st.markdown(
        "- **Strong Bullish**: tren & momentum sangat kuat  \n"
        "- **Bullish**: tren naik cukup sehat  \n"
        "- **Neutral**: bisa sideways / butuh konfirmasi  \n"
        "- **Bearish**: tren lemah / cenderung turun"
    )

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
                    st.error(
                        "❌ Analysis gagal / tidak memenuhi filter bullish. "
                        "Untuk belajar, boleh cek chart & indikator manual."
                    )
                else:
                    st.markdown(f"## 💎 {res['Ticker']}")
                    st.markdown(
                        f"### Grade: **{res['Grade']}** | Trend: **{res['Trend']}** | Signal: **{res['Signal']}**"
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Price", format_idr(res["Price"]))
                    c2.metric("Score", f"{res['Score']}/100")
                    c3.metric("Confidence", f"{res['Confidence']}%")

                    label_map = {
                        "General": "SPEED PLAN (1–3 Hari)",
                        "BPJS": "BPJS PLAN (Short-Term)",
                        "BSJP": "BSJP PLAN (Overnight)",
                        "Bandar": "BANDAR PLAN (Momentum)",
                        "Swing": "SWING PLAN (3–5 Hari)",
                        "Value": "VALUE PLAN (Rebound)",
                        "Gorilla": "GORILLA PLAN (Very Aggressive)",
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
                        f"• **Cut Loss (CL):** {format_idr(res['SL'])}"
                    )

                    st.markdown("**Technical Notes:**")
                    for k, v in res["Details"].items():
                        st.caption(f"• **{k}**: {v}")
else:
    strategy_map = {
        "⚡ SPEED Trader (1-3d)": "General",
        "🎯 SWING Trader (3-5d)": "Swing",
        "💎 VALUE Plays (Technical Discount)": "Value",
        "⚡ BPJS (Short-Term Momentum)": "BPJS",
        "🌙 BSJP (Overnight)": "BSJP",
        "🔮 Bandar Tracking": "Bandar",
        "🦍 Gorilla Mode (Very Aggressive)": "Gorilla",
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
            df1, df2 = scan_stocks(run_list, strategy, period, limit1, limit2)
            if df1.empty and df2.empty:
                st.error("❌ No stocks matched filters.")
            else:
                show_table(df1, "Stage 1 – Candidates (sorted by Score)", strategy)
                show_table(
                    df2,
                    "Stage 2 – Elite Picks (Uptrend + Strong/Bullish + Best Scores)",
                    strategy,
                )
    else:
        st.info("Tekan **RUN SCAN** untuk mulai pemindaian saham.")

st.markdown("---")
st.caption(
    "🚀 IDX Power Screener v6 – Educational purposes only | Tick-size aware | "
    "Dibuat untuk bantu belajar, bukan memberi sinyal pasti."
)