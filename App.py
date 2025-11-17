#!/usr/bin/env python3
import os
import json
import time
import math
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================== BASIC CONFIG ==================
st.set_page_config(
    page_title="IDX Power Screener – EXTREME BUILD",
    page_icon="📈",
    layout="wide",
)

# ================== BLUE DARK THEME ==================
st.markdown(
    """
<style>
body, .stApp {
  background-color: #020617;
  color: #e5e7eb;
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
}
section[data-testid="stSidebar"] {
  background: #020617;
}
.big-title {
  font-size: 30px;
  font-weight: 700;
  color: #60a5fa;
}
.subtitle {
  font-size: 13px;
  color: #9ca3af;
}
.stage-header {
  padding: 0.6rem 0.9rem;
  border-radius: 0.9rem;
  background: rgba(37,99,235,0.15);
  border: 1px solid rgba(59,130,246,0.65);
  color: #bfdbfe;
  font-weight: 600;
}
.badge-blue {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(59,130,246,0.18);
  color: #bfdbfe;
  font-size: 11px;
  font-weight: 600;
}
.badge-gray {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(148,163,184,0.3);
  color: #e5e7eb;
  font-size: 11px;
}
.block-container {
  padding-top: 1.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============= SESSION STATE =============
if "last_scan_results" not in st.session_state:
    st.session_state.last_scan_results = None
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "last_scan_strategy" not in st.session_state:
    st.session_state.last_scan_strategy = None
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

# ============= TIME HELPER (WIB) =============
def get_jakarta_time():
    return datetime.now(timezone(timedelta(hours=7)))

def is_bpjs_time():
    h = get_jakarta_time().hour
    return 9 <= h < 10

def is_bsjp_time():
    h = get_jakarta_time().hour
    return 14 <= h < 16

# ============= IHSG MARKET WIDGET =============
@st.cache_data(ttl=180)
def fetch_ihsg_data():
    try:
        import yfinance as yf
        ihsg = yf.Ticker("^JKSE")
        hist = ihsg.history(period="1d")
        if not hist.empty:
            current = hist["Close"].iloc[-1]
            open_price = hist["Open"].iloc[-1]
            high = hist["High"].iloc[-1]
            low = hist["Low"].iloc[-1]
            change = current - open_price
            change_pct = (change / open_price) * 100
            return {
                "price": float(current),
                "change": float(change),
                "change_pct": float(change_pct),
                "high": float(high),
                "low": float(low),
                "status": "up" if change >= 0 else "down",
            }
    except Exception:
        pass
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
        guidance = "✅ Good for most strategies"
    elif ihsg["change_pct"] > -0.5:
        condition = "➡️ Sideways - Mixed conditions"
        guidance = "⚠️ Be selective, gunakan tight stop"
    elif ihsg["change_pct"] > -1.5:
        condition = "📉 Moderate downtrend - Caution"
        guidance = "⚠️ Fokus ke VALUE / defensif"
    else:
        condition = "🔻 Strong downtrend - High risk"
        guidance = "❌ Kurangi agresif, lebih banyak cash"

    st.markdown(
        f"""
    <div style='background: linear-gradient(135deg, #0b1120 0%, #1d4ed8 100%);
                padding: 14px; border-radius: 10px; margin-bottom: 16px;
                border-left: 5px solid {"#22c55e" if ihsg["status"]=="up" else "#ef4444"}'>
      <div style='display:flex;justify-content:space-between;align-items:center;gap:1rem;'>
        <div>
          <h3 style='margin:0;color:white;'>📊 MARKET OVERVIEW – IHSG</h3>
          <p style='margin:4px 0;color:#e0e7ff;font-size:0.9em;'>
            Jakarta Composite Index • Status: <b>{status_text}</b>
          </p>
          <p style='margin:4px 0;color:#bfdbfe;font-size:0.85em;'> {condition} </p>
          <p style='margin:2px 0;color:#c7d2fe;font-size:0.8em;'> {guidance} </p>
        </div>
        <div style='text-align:right;min-width:160px;'>
          <h2 style='margin:0;color:white;'>
            {status_emoji} {ihsg['price']:,.2f}
          </h2>
          <p style='margin:4px 0;color:{"#22c55e" if ihsg["status"]=="up" else "#ef4444"};
                    font-size:1.05em;font-weight:bold;'>
            {ihsg['change']:+,.2f} ({ihsg['change_pct']:+.2f}%)
          </p>
          <p style='margin:2px 0;color:#9ca3af;font-size:0.75em;'>
            High: {ihsg['high']:,.2f} • Low: {ihsg['low']:,.2f}<br/>
            ⏰ {datetime.now().strftime('%H:%M:%S')} WIB
          </p>
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ============= LOAD TICKERS =============
def load_tickers():
    """
    Cari file idx_stocks.json kalau ada:
    {
      "tickers": ["BBRI","BBCA","BUMI",...]
    }
    Kalau tidak ada, pakai fallback pendek (bisa kamu edit).
    """
    try:
        if os.path.exists("idx_stocks.json"):
            with open("idx_stocks.json", "r") as f:
                data = json.load(f)
            raw = data.get("tickers", [])
            return [t if t.endswith(".JK") else f"{t}.JK" for t in raw]
    except Exception:
        pass

    return [
        "BBRI.JK","BBCA.JK","BMRI.JK","BBNI.JK","BRPT.JK","BREN.JK","BUMI.JK",
        "CUAN.JK","INET.JK","RAJA.JK","RAKU.JK","PGAS.JK","PTBA.JK","INCO.JK",
        "INDF.JK","ITMG.JK","TLKM.JK","TOWR.JK","UNTR.JK","UNVR.JK","MDKA.JK",
        "GOTO.JK","ANTM.JK","HRUM.JK","MEDC.JK","CPIN.JK","MIKA.JK","PTRO.JK",
        "DADA.JK","DCII.JK","RAJA.JK","RAKU.JK",
    ]

# ============= CHART =============
def create_chart(df, ticker, period_days=60):
    try:
        df_chart = df.tail(period_days).copy()

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{ticker} – Price & EMAs", "Volume", "RSI"),
        )

        fig.add_trace(
            go.Candlestick(
                x=df_chart.index,
                open=df_chart["Open"],
                high=df_chart["High"],
                low=df_chart["Low"],
                close=df_chart["Close"],
                name="Price",
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
            ),
            row=1, col=1,
        )

        colors = {
            "EMA9": "#60a5fa",
            "EMA21": "#f97316",
            "EMA50": "#ef4444",
            "EMA200": "#9ca3af",
        }
        for ema in ["EMA9", "EMA21", "EMA50", "EMA200"]:
            if ema in df_chart.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_chart.index,
                        y=df_chart[ema],
                        name=ema,
                        line=dict(color=colors[ema], width=1.4),
                    ),
                    row=1, col=1,
                )

        colors_vol = [
            "#ef4444" if df_chart["Close"].iloc[i] < df_chart["Open"].iloc[i] else "#22c55e"
            for i in range(len(df_chart))
        ]
        fig.add_trace(
            go.Bar(
                x=df_chart.index,
                y=df_chart["Volume"],
                name="Volume",
                marker_color=colors_vol,
                showlegend=False,
            ),
            row=2, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["RSI"],
                name="RSI",
                line=dict(color="#a855f7", width=2),
            ),
            row=3, col=1,
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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1f2933")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1f2933")

        return fig
    except Exception:
        return None

# ============= FETCH DATA YAHOO =============
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, period="6mo"):
    try:
        end = int(datetime.now().timestamp())
        days = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}.get(period, 180)
        start = end - days * 86400

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        r = requests.get(
            url,
            params={"period1": start, "period2": end, "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if r.status_code != 200:
            return None

        data = r.json()
        result = data["chart"]["result"][0]
        q = result["indicators"]["quote"][0]

        df = pd.DataFrame(
            {
                "Open": q["open"],
                "High": q["high"],
                "Low": q["low"],
                "Close": q["close"],
                "Volume": q["volume"],
            },
            index=pd.to_datetime(result["timestamp"], unit="s"),
        )
        df = df.dropna()
        if len(df) < 50:
            return None

        # EMA
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA200"] = (
            df["Close"].ewm(span=200, adjust=False).mean()
            if len(df) >= 200
            else df["Close"].ewm(span=len(df), adjust=False).mean()
        )

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Volume stats
        df["VOL_SMA20"] = df["Volume"].rolling(20).mean()
        df["VOL_RATIO"] = df["Volume"] / df["VOL_SMA20"]

        # Momentum
        df["MOM_5D"] = (df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5) * 100
        df["MOM_10D"] = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10) * 100
        df["MOM_20D"] = (df["Close"] - df["Close"].shift(20)) / df["Close"].shift(20) * 100

        # ATR
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - df["Close"].shift()).abs()
        tr3 = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()
        df["ATR_PCT"] = df["ATR"] / df["Close"] * 100

        # MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

        return df
    except Exception:
        return None

# ============= PRE-FILTERS =============
def apply_liquidity_filter(df):
    try:
        r = df.iloc[-1]
        price = float(r["Close"])
        vol_avg = df["Volume"].tail(20).mean()

        if price < 50:
            return False, "Price too low (<50)"
        if vol_avg < 500_000:
            return False, "Volume too low (<500K)"
        turnover = price * vol_avg
        if turnover < 100_000_000:
            return False, "Turnover too low (<100M)"
        return True, "Passed"
    except Exception:
        return False, "Error"

# ============= BASE SCORING =============
def score_base(df, *, allow_weaker_momentum=False, require_strong_volume=True):
    try:
        r = df.iloc[-1]
        score = 0
        details = {}

        passed, reason = apply_liquidity_filter(df)
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"

        close = float(r["Close"])
        ema21 = float(r["EMA21"])
        ema50 = float(r["EMA50"])
        ema200 = float(r["EMA200"])

        # Hard downtrend
        if close < ema21 < ema50 < ema200:
            return 0, {"Rejected": "Strong downtrend (EMA alignment)"}, 0, "F"

        if close < ema50:
            return 0, {"Rejected": "Price below EMA50"}, 0, "F"

        mom_20d = float(r["MOM_20D"])
        vol_ratio = float(r["VOL_RATIO"]) if not pd.isna(r["VOL_RATIO"]) else 1.0

        if not allow_weaker_momentum and mom_20d < -5:
            return (
                0,
                {"Rejected": f"Strong negative momentum 20D ({mom_20d:.1f}%)"},
                0,
                "F",
            )

        if require_strong_volume and vol_ratio < 1.0:
            return (
                0,
                {"Rejected": f"Insufficient volume (VOL_RATIO {vol_ratio:.1f}x < 1.0x)"},
                0,
                "F",
            )

        # Momentum penalty
        if mom_20d < -8:
            penalty = 0.5
            details["⚠ Momentum"] = f"Very weak 20D {mom_20d:.1f}%"
        elif -8 <= mom_20d < -5:
            penalty = 0.7
            details["⚠ Momentum"] = f"Weak 20D {mom_20d:.1f}%"
        elif -5 <= mom_20d < 0:
            penalty = 0.85
            details["⚠ Momentum"] = f"Slight negative 20D {mom_20d:.1f}%"
        else:
            penalty = 1.0

        # EMA alignment
        ema9 = float(r["EMA9"])
        ema_alignment = 0
        if ema9 > ema21:
            ema_alignment += 1
        if ema21 > ema50:
            ema_alignment += 1
        if ema50 > ema200:
            ema_alignment += 1
        if close > ema9:
            ema_alignment += 1

        if ema_alignment == 4:
            score += 40
            details["Trend"] = "🟢 Perfect EMA alignment"
        elif ema_alignment == 3:
            score += 28
            details["Trend"] = "🟡 Strong trend"
        elif ema_alignment == 2:
            score += 14
            details["Trend"] = "🟠 Moderate trend"
        else:
            details["Trend"] = "🔴 Weak trend"

        # RSI
        rsi = float(r["RSI"])
        if 50 <= rsi <= 65:
            score += 25
            details["RSI"] = f"🟢 Ideal {rsi:.0f}"
        elif 45 <= rsi < 50:
            score += 20
            details["RSI"] = f"🟡 Good {rsi:.0f}"
        elif 40 <= rsi < 45:
            score += 10
            details["RSI"] = f"🟠 OK {rsi:.0f}"
        elif rsi > 70:
            details["RSI"] = f"🔴 Overbought {rsi:.0f}"
        elif rsi < 35:
            details["RSI"] = f"🔴 Oversold {rsi:.0f}"
        else:
            score += 5
            details["RSI"] = f"⚪ Neutral {rsi:.0f}"

        # Volume
        if vol_ratio > 2.0:
            score += 20
            details["Volume"] = f"🟢 Surge {vol_ratio:.1f}x"
        elif vol_ratio > 1.5:
            score += 15
            details["Volume"] = f"🟡 Strong {vol_ratio:.1f}x"
        elif vol_ratio > 1.0:
            score += 7
            details["Volume"] = f"🟠 Normal {vol_ratio:.1f}x"
        else:
            score += 3
            details["Volume"] = f"⚪ Low {vol_ratio:.1f}x"

        # Short-term momentum
        mom_5d = float(r["MOM_5D"])
        mom_10d = float(r["MOM_10D"])

        if mom_5d > 4 and mom_10d > 6:
            score += 18
            details["Short-term"] = f"🟢 Strong +{mom_5d:.1f}% (5D)"
        elif mom_5d > 2 and mom_10d > 3:
            score += 12
            details["Short-term"] = f"🟡 Good +{mom_5d:.1f}% (5D)"
        elif mom_5d > 0:
            score += 6
            details["Short-term"] = f"🟠 Positive +{mom_5d:.1f}% (5D)"
        elif mom_20d > 5:
            score += 8
            details["Short-term"] = f"🟡 20D momentum +{mom_20d:.1f}%"

        score = int(score * penalty)

        if score >= 90:
            grade, conf = "A+", 90
        elif score >= 80:
            grade, conf = "A", 80
        elif score >= 70:
            grade, conf = "B+", 70
        elif score >= 60:
            grade, conf = "B", 60
        elif score >= 50:
            grade, conf = "C", 50
        else:
            grade, conf = "D", max(score, 0)

        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"

# ============= STRATEGY-SPECIFIC SCORING =============
def score_speed(df):
    return score_base(df, allow_weaker_momentum=False, require_strong_volume=True)

def score_swing(df):
    return score_base(df, allow_weaker_momentum=True, require_strong_volume=True)

def score_value(df):
    return score_base(df, allow_weaker_momentum=True, require_strong_volume=False)

def score_bpjs(df):
    # day trade lebih ketat momentum
    return score_base(df, allow_weaker_momentum=False, require_strong_volume=True)

def score_bsjp(df):
    # overnight boleh momentum sedikit lemah
    return score_base(df, allow_weaker_momentum=True, require_strong_volume=True)

def score_bandar(df):
    # sementara pakai base, bisa ditambah broker data nanti
    return score_base(df, allow_weaker_momentum=True, require_strong_volume=True)

# ============= TREND, SIGNAL, TRADE PLAN =============
def detect_trend(last_row):
    price = float(last_row["Close"])
    ema9 = float(last_row["EMA9"])
    ema21 = float(last_row["EMA21"])
    ema50 = float(last_row["EMA50"])
    ema200 = float(last_row["EMA200"])

    if price > ema9 > ema21 > ema50 > ema200:
        return "Strong Uptrend"
    elif price > ema50 and ema9 > ema21 > ema50:
        return "Uptrend"
    elif abs(price - ema50) / price < 0.03:
        return "Sideways"
    else:
        return "Downtrend"

def classify_signal(last_row, score, grade, trend):
    rsi = float(last_row["RSI"]) if not pd.isna(last_row["RSI"]) else 50.0
    vol_ratio = float(last_row["VOL_RATIO"]) if not pd.isna(last_row["VOL_RATIO"]) else 1.0
    mom_5d = float(last_row["MOM_5D"]) if not pd.isna(last_row["MOM_5D"]) else 0.0
    mom_20d = float(last_row["MOM_20D"]) if not pd.isna(last_row["MOM_20D"]) else 0.0

    if (
        trend in ["Strong Uptrend", "Uptrend"]
        and grade in ["A+", "A"]
        and vol_ratio > 1.5
        and 45 <= rsi <= 70
        and mom_5d > 0
        and mom_20d > 0
    ):
        return "Strong Buy"

    if (
        trend in ["Strong Uptrend", "Uptrend"]
        and grade in ["A+", "A", "B+"]
        and vol_ratio > 1.0
        and 40 <= rsi <= 75
    ):
        return "Buy"

    if trend in ["Strong Uptrend", "Uptrend"] and grade in ["A+", "A", "B+", "B"]:
        return "Hold"

    if trend == "Sideways" and grade in ["B+", "B", "C"]:
        return "Hold"

    return "Sell"

def compute_trade_plan(df, strategy, trend):
    r = df.iloc[-1]
    price = float(r["Close"])
    ema21 = float(r["EMA21"])

    s = strategy.upper()
    if s in ["SWING"]:
        entry_ideal = round(price * 0.99, 0)
        tp1 = round(entry_ideal * 1.06, 0)
        tp2 = round(entry_ideal * 1.10, 0)
        tp3 = round(entry_ideal * 1.15, 0)
        sl = round(entry_ideal * 0.95, 0)
    elif s in ["VALUE"]:
        entry_ideal = round(price * 0.98, 0)
        tp1 = round(entry_ideal * 1.15, 0)
        tp2 = round(entry_ideal * 1.25, 0)
        tp3 = round(entry_ideal * 1.35, 0)
        sl = round(entry_ideal * 0.93, 0)
    else:
        # SPEED / BPJS / BSJP / BANDAR / GENERAL
        entry_ideal = round(price * 0.995, 0)
        tp1 = round(entry_ideal * 1.04, 0)
        tp2 = round(entry_ideal * 1.07, 0)
        tp3 = None
        sl = round(entry_ideal * 0.97, 0)

    if trend in ["Strong Uptrend", "Uptrend"] and ema21 < price:
        ema_entry = round(ema21 * 1.01, 0)
        if price * 0.9 < ema_entry < price:
            entry_ideal = ema_entry

    if trend == "Downtrend":
        sl = round(entry_ideal * 0.96, 0)

    entry_aggressive = round(price, 0)
    return {
        "entry_ideal": entry_ideal,
        "entry_aggressive": entry_aggressive,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
    }

# ============= PROCESS TICKER =============
def process_ticker(ticker, strategy, period):
    try:
        df = fetch_data(ticker, period)
        if df is None:
            return None

        last_row = df.iloc[-1]
        price = float(last_row["Close"])
        strat = strategy.upper()

        if strat == "SPEED":
            score, details, conf, grade = score_speed(df)
        elif strat == "SWING":
            score, details, conf, grade = score_swing(df)
        elif strat == "VALUE":
            score, details, conf, grade = score_value(df)
        elif strat == "BPJS":
            score, details, conf, grade = score_bpjs(df)
        elif strat == "BSJP":
            score, details, conf, grade = score_bsjp(df)
        elif strat == "BANDAR":
            score, details, conf, grade = score_bandar(df)
        else:
            score, details, conf, grade = score_speed(df)

        if grade not in ["A+", "A", "B+", "B", "C"]:
            return None

        trend = detect_trend(last_row)
        signal = classify_signal(last_row, score, grade, trend)
        plan = compute_trade_plan(df, strat, trend)

        result = {
            "Ticker": ticker.replace(".JK", ""),
            "Price": price,
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
            result["TP3"] = plan["tp3"]

        return result
    except Exception:
        return None

# ============= SESSION HELPERS =============
def save_scan_to_session(df2, df1, strategy):
    st.session_state.last_scan_results = (df2, df1)
    st.session_state.last_scan_time = datetime.now()
    st.session_state.last_scan_strategy = strategy
    st.session_state.scan_count += 1

def display_last_scan_info():
    if st.session_state.last_scan_results:
        df2, df1 = st.session_state.last_scan_results
        delta = datetime.now() - st.session_state.last_scan_time
        mins = int(delta.total_seconds() / 60)
        st.markdown(
            f"""
        <div style='background:linear-gradient(135deg,#022c57 0%,#0f766e 100%);
                    padding:10px;border-radius:8px;margin-bottom:10px;
                    border-left:4px solid #38bdf8;'>
          <p style='margin:0;color:white;font-weight:600;'>📂 LAST SCAN SAVED</p>
          <p style='margin:4px 0 0 0;color:#e0f2fe;font-size:0.85em;'>
            Strategy: {st.session_state.last_scan_strategy} •
            Time: {st.session_state.last_scan_time.strftime('%H:%M:%S')} ({mins} min ago) •
            Elite: {len(df2)} • Stage1: {len(df1)}
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return True
    return False

def create_csv_download(df, strategy, label="Stage2_Elite"):
    if not df.empty:
        export_df = df.copy()
        if "Details" in export_df.columns:
            export_df = export_df.drop(columns=["Details"])
        csv = export_df.to_csv(index=False)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"IDX_{strategy}_{label}_{ts}.csv"
        st.download_button(
            label="💾 Download Elite Picks (CSV)",
            data=csv,
            file_name=fname,
            mime="text/csv",
        )

# ============= EXTREME SCAN (STAGE1 & STAGE2) =============
def scan_stocks_extreme(tickers, strategy, period, limit1, limit2):
    st.info(
        f"🔍 Scanning {len(tickers)} stocks for **{strategy}** – this may take a moment..."
    )

    results = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ticker, t, strategy, period): t for t in tickers}
        done = 0
        total = len(futures)
        for future in as_completed(futures):
            done += 1
            progress.progress(done / total)
            status.text(f"📊 {done}/{total} | Collected: {len(results)}")
            res = future.result()
            if res:
                results.append(res)
            time.sleep(0.03)

    progress.empty()
    status.empty()

    if not results:
        st.warning("Tidak ada saham yang lolos filter dasar untuk strategi ini.")
        return pd.DataFrame(), pd.DataFrame()

    df_all = pd.DataFrame(results)

    # ---------- Stage 1 filter ----------
    mask_signal = df_all["Signal"].isin(["Strong Buy", "Buy"])
    # buang Downtrend
    mask_no_down = ~df_all["Trend"].fillna("").str.contains("down", case=False)

    df_stage1_raw = df_all[mask_signal & mask_no_down].copy()
    if df_stage1_raw.empty:
        st.warning("Semua kandidat Strong Buy/Buy saat ini dalam tren lemah / downtrend.")
        return df_stage1_raw, df_stage1_raw

    df_stage1_raw = df_stage1_raw.sort_values("Score", ascending=False)
    limit1 = min(limit1, len(df_stage1_raw))
    df1 = df_stage1_raw.head(limit1).reset_index(drop=True)

    st.success(
        f"✅ Stage 1: {len(df1)} kandidat (Strong Buy/Buy & NO Downtrend) – "
        f"Avg Score: {df1['Score'].mean():.0f}"
    )

    # ---------- Stage 2 elite ----------
    # Step 1: Grade A+/A & Strong Buy
    elite = df1[(df1["Grade"].isin(["A+", "A"])) & (df1["Signal"] == "Strong Buy")].copy()

    # fallback: kalau kosong, longgarkan jadi A+/A & (Strong Buy/Buy)
    if elite.empty:
        elite = df1[(df1["Grade"].isin(["A+", "A"])) & (df1["Signal"].isin(["Strong Buy", "Buy"]))].copy()

    # fallback kedua: kalau masih kosong, ambil B+ Strong Buy
    if elite.empty:
        elite = df1[(df1["Grade"].isin(["A+", "A", "B+"])) & (df1["Signal"] == "Strong Buy")].copy()

    if elite.empty:
        st.warning("Belum ada Elite Picks (A+/A Strong Buy) untuk setting ini.")
        save_scan_to_session(pd.DataFrame(), df1, strategy)
        return df1, pd.DataFrame()

    elite = elite.sort_values("Score", ascending=False)
    limit2 = min(limit2, len(elite))
    df2 = elite.head(limit2).reset_index(drop=True)

    st.success(
        f"🏆 Stage 2: {len(df2)} Elite Picks (Grade A+/A fokus Strong Buy) – "
        f"Avg Score: {df2['Score'].mean():.0f}"
    )

    save_scan_to_session(df2, df1, strategy)
    return df1, df2

# ============= TABLE RENDERING =============
def pretty_table(df, index_col="Ticker"):
    if df.empty:
        st.info("Tidak ada data untuk filter ini.")
        return
    show_df = df.copy()
    preferred = [
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
    cols = [c for c in preferred if c in show_df.columns] + [
        c for c in show_df.columns if c not in preferred
    ]
    show_df = show_df[cols]
    if index_col in show_df.columns:
        show_df = show_df.set_index(index_col)
    st.dataframe(show_df, use_container_width=True)

# ============= UI LAYOUT =============
st.markdown('<div class="big-title">IDX Power Screener – EXTREME BUILD</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Stage 1: filter 100–200 saham bagus • Stage 2: Elite A+/A & Strong Buy • '
    'Tools untuk bantu perbaiki portofolio secara disiplin'
    '</div>',
    unsafe_allow_html=True,
)
st.write("")

display_ihsg_widget()
tickers = load_tickers()

# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Settings")

    st.metric("Total tickers", len(tickers))

    menu = st.radio(
        "📋 Mode",
        [
            "⚡ SPEED Trader (1–2D)",
            "🎯 SWING Trader (3–10D)",
            "💎 VALUE Plays",
            "⚡ BPJS (Day Trade)",
            "🌙 BSJP (Overnight)",
            "🔮 Bandar Tracking",
            "🔍 Single Stock",
        ],
    )

    jkt = get_jakarta_time()
    st.caption(f"🕒 Jakarta time: {jkt.strftime('%H:%M WIB')}")

    st.markdown("---")

    if "Single" not in menu:
        period = st.selectbox("Period (data history)", ["3mo", "6mo", "1y"], index=1)

        stage1_limit = st.slider(
            "Stage 1 – Top N (Strong Buy/Buy)",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
        )

        max_stage2 = min(50, stage1_limit)
        stage2_limit = st.slider(
            "Stage 2 – Elite Picks (A+/A & Strong Buy)",
            min_value=5,
            max_value=max_stage2,
            value=min(20, max_stage2),
            step=5,
        )

        st.markdown("---")
        if "BPJS" in menu:
            st.caption(
                "⏰ BPJS ideal: 09.00–10.00 WIB\n"
                + ("✅ Sekarang JAM BPJS" if is_bpjs_time() else "ℹ Di luar jam BPJS")
            )
        if "BSJP" in menu:
            st.caption(
                "⏰ BSJP ideal: 14.00–16.00 WIB\n"
                + ("✅ Sekarang JAM BSJP" if is_bsjp_time() else "ℹ Di luar jam BSJP")
            )

    st.markdown("---")
    st.caption(
        "💡 Screener ini alat bantu edukasi & riset. Bukan rekomendasi beli/jual.\n"
        "Tetap gunakan trading plan & money management sendiri."
    )

# ============= MAIN =============

# SINGLE STOCK MODE
if "Single Stock" in menu:
    st.markdown("### 🔍 Single Stock Analyzer")

    selected = st.selectbox(
        "Pilih saham", sorted([t.replace(".JK", "") for t in tickers])
    )
    strat_single = st.selectbox(
        "Strategi",
        ["SPEED", "SWING", "VALUE", "BPJS", "BSJP", "BANDAR"],
        index=0,
    )
    period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)

    if st.button("🔍 ANALYZE", type="primary"):
        full = selected if selected.endswith(".JK") else f"{selected}.JK"
        with st.spinner(f"Menganalisa {selected} ({strat_single}) ..."):
            df = fetch_data(full, period)
            if df is None:
                st.error("❌ Gagal mengambil data dari Yahoo.")
            else:
                res = process_ticker(full, strat_single, period)
                if res is None:
                    st.error("❌ Ditolak oleh filter (grade/volume/trend).")
                    chart = create_chart(df, selected)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.markdown("#### 📊 Chart & Signal")
                    chart = create_chart(df, selected)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)

                    st.markdown(
                        f"### 💎 {res['Ticker']} – Grade **{res['Grade']}** | "
                        f"Trend **{res['Trend']}** | Signal **{res['Signal']}**"
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Price", f"Rp {res['Price']:,.0f}")
                    c2.metric("Score", f"{res['Score']}/100")
                    c3.metric("Confidence", f"{res['Confidence']}%")

                    st.success(
                        f"""
**Template Trade Plan ({strat_single})**

• Entry Ideal: **Rp {res['Entry']:,.0f}**  
• Entry Agresif (harga sekarang): **Rp {res['Entry_Aggressive']:,.0f}**

• TP1: **Rp {res['TP1']:,.0f}**  
• TP2: **Rp {res['TP2']:,.0f}**

• Stop Loss: **Rp {res['SL']:,.0f}**

_Sesuaikan lot dan risk per trade dengan modal & mental (Gatot Kaca)_ 💪
"""
                    )

                    st.markdown("**Catatan Teknis:**")
                    for k, v in res["Details"].items():
                        st.caption(f"• **{k}**: {v}")

else:
    # MULTI SCAN
    st.markdown("### 🧮 Multi-Stock Scanner – Extreme Mode")

    has_last = display_last_scan_info()

    if has_last:
        with st.expander("📁 Lihat hasil scan terakhir"):
            df2_last, df1_last = st.session_state.last_scan_results
            if not df2_last.empty:
                st.markdown("**Stage 2 – Elite terakhir:**")
                pretty_table(df2_last)
            if not df1_last.empty:
                st.markdown("**Stage 1 – Candidates terakhir:**")
                pretty_table(df1_last)

    if st.button("🚀 START NEW SCAN", type="primary"):
        if "SPEED" in menu:
            strat = "SPEED"
        elif "SWING" in menu:
            strat = "SWING"
        elif "VALUE" in menu:
            strat = "VALUE"
        elif "BPJS" in menu:
            strat = "BPJS"
        elif "BSJP" in menu:
            strat = "BSJP"
        elif "Bandar" in menu:
            strat = "BANDAR"
        else:
            strat = "SPEED"

        with st.spinner(f"Scanning {len(tickers)} saham untuk strategi {strat}..."):
            df1, df2 = scan_stocks_extreme(
                tickers, strat, period, stage1_limit, stage2_limit
            )

        # Stage 2 – Elite Picks
        st.markdown(
            f'<div class="stage-header">🏆 Stage 2 – Elite Picks '
            f'(A+/A & Strong Buy) – {len(df2)} ticker</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if df2.empty:
            st.info(
                "Belum ada kandidat A+/A & Strong Buy. Coba ganti strategi, period, atau perbesar Stage 1."
            )
        else:
            pretty_table(df2)
            create_csv_download(df2, strat, label="Stage2_Elite")

        st.write("")
        st.markdown("---")
        st.write("")

        # Stage 1 – Top Candidates
        st.markdown(
            f'<div class="stage-header">📊 Stage 1 – Top Candidates '
            f'(Strong Buy / Buy, NO Downtrend) – {len(df1)} ticker</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        if df1.empty:
            st.info("Stage 1 kosong untuk setting ini.")
        else:
            rows_per_page = st.selectbox(
                "Rows per page (Stage 1)", [20, 40, 60, 80, 100], index=1
            )
            total_rows = len(df1)
            total_pages = max(1, math.ceil(total_rows / rows_per_page))

            if total_pages > 1:
                col_page, col_info = st.columns([1, 2])
                with col_page:
                    page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        step=1,
                        key="stage1_page",
                    )
                with col_info:
                    st.markdown(
                        f"<span class='badge-blue'>Menampilkan {rows_per_page} baris / halaman "
                        f"(total {total_rows})</span>",
                        unsafe_allow_html=True,
                    )
                start = (page - 1) * rows_per_page
                end = start + rows_per_page
                df_page = df1.iloc[start:end].reset_index(drop=True)
                pretty_table(df_page)
            else:
                pretty_table(df1)

st.markdown("---")
st.caption(
    "🚀 IDX Power Screener - EXTREME BUILD • "
    "$$$, $$$. $$$"
)

