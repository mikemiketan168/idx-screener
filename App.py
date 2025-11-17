#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from math import ceil
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================== BASIC CONFIG ==================
st.set_page_config(
    page_title="IDX Power Screener v5.0 ELITE",
    page_icon="🚀",
    layout="wide",
)

# ============= SESSION STATE (LOCK SCREEN SAFE) =============
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
    jkt_hour = get_jakarta_time().hour
    return 9 <= jkt_hour < 10


def is_bsjp_time():
    jkt_hour = get_jakarta_time().hour
    return 14 <= jkt_hour < 16


# ============= IHSG MARKET WIDGET =============
@st.cache_data(ttl=180)
def fetch_ihsg_data():
    """Fetch IHSG data via yfinance (fallback: None)"""
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

    st.markdown(
        f"""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                padding: 15px; border-radius: 10px; margin-bottom: 20px;
                border-left: 5px solid {"#22c55e" if ihsg['status']=="up" else "#ef4444"}'>
      <div style='display:flex;justify-content:space-between;align-items:center;'>
        <div>
          <h3 style='margin:0;color:white;'>📊 MARKET OVERVIEW</h3>
          <p style='margin:5px 0;color:#e0e7ff;font-size:0.9em;'>
            Jakarta Composite Index
          </p>
        </div>
        <div style='text-align:right;'>
          <h2 style='margin:0;color:white;'>
            {status_emoji} {ihsg['price']:,.2f}
          </h2>
          <p style='margin:5px 0;color:{"#22c55e" if ihsg['status']=="up" else "#ef4444"};
                    font-size:1.1em;font-weight:bold;'>
            {ihsg['change']:+,.2f} ({ihsg['change_pct']:+.2f}%)
          </p>
        </div>
      </div>
      <div style='margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.2);'>
        <p style='margin:3px 0;color:#e0e7ff;font-size:0.85em;'>
          📊 High: {ihsg['high']:,.2f} | Low: {ihsg['low']:,.2f}
          | Status: <strong>{status_text}</strong>
        </p>
        <p style='margin:3px 0;color:#fbbf24;font-size:0.9em;'> {condition} </p>
        <p style='margin:3px 0;color:#a5b4fc;font-size:0.85em;'> {guidance} </p>
        <p style='margin:5px 0 0 0;color:#94a3b8;font-size:0.75em;'>
          ⏰ Last update: {datetime.now().strftime('%H:%M:%S')} WIB | 🔄 Auto-refresh: 3 min
        </p>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ============= LOAD TICKERS =============
def load_tickers():
    """Load IDX tickers from json if exists, else fallback list."""
    try:
        if os.path.exists("idx_stocks.json"):
            with open("idx_stocks.json", "r") as f:
                data = json.load(f)
            tickers = data.get("tickers", [])
            return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except Exception:
        pass

    # Fallback (boleh kamu ganti dengan list lengkapmu)
    return [
        "AALI.JK",
        "ABBA.JK",
        "ABMM.JK",
        "ACES.JK",
        "ADRO.JK",
        "AGII.JK",
        "ANTM.JK",
        "ASII.JK",
        "BBCA.JK",
        "BBNI.JK",
        "BBRI.JK",
        "BMRI.JK",
        "BRPT.JK",
        "BREN.JK",
        "BUMI.JK",
        "BUVA.JK",
        "CPIN.JK",
        "DADA.JK",
        "DCII.JK",
        "ELSA.JK",
        "GOTO.JK",
        "HRUM.JK",
        "ICBP.JK",
        "INCO.JK",
        "INDF.JK",
        "INKP.JK",
        "INTP.JK",
        "ITMG.JK",
        "KLBF.JK",
        "MEDC.JK",
        "MDKA.JK",
        "MIKA.JK",
        "PGAS.JK",
        "PGEO.JK",
        "PTBA.JK",
        "PTPP.JK",
        "PTRO.JK",
        "RAJA.JK",
        "SMGR.JK",
        "SMRA.JK",
        "TBIG.JK",
        "TLKM.JK",
        "TOWR.JK",
        "UNTR.JK",
        "UNVR.JK",
        "WSBP.JK",
        "WSKT.JK",
        "YELO.JK",
        "ZINC.JK",
    ]


# ============= CHART VISUALIZATION =============
def create_chart(df, ticker, period_days=60):
    """Create interactive chart with technical indicators"""
    try:
        df_chart = df.tail(period_days).copy()

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
                x=df_chart.index,
                open=df_chart["Open"],
                high=df_chart["High"],
                low=df_chart["Low"],
                close=df_chart["Close"],
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
            if ema in df_chart.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_chart.index,
                        y=df_chart[ema],
                        name=ema,
                        line=dict(color=colors[ema], width=1.5),
                    ),
                    row=1,
                    col=1,
                )

        # Volume
        colors_vol = [
            "#ef5350"
            if df_chart["Close"].iloc[i] < df_chart["Open"].iloc[i]
            else "#26a69a"
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
            row=2,
            col=1,
        )

        # RSI
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["RSI"],
                name="RSI",
                line=dict(color="#9C27B0", width=2),
            ),
            row=3,
            col=1,
        )

        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            row=3,
            col=1,
        )
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            row=3,
            col=1,
        )
        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="gray",
            opacity=0.3,
            row=3,
            col=1,
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


# ============= FETCH DATA =============
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, period="6mo"):
    try:
        end = int(datetime.now().timestamp())
        days = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}.get(
            period, 180
        )
        start = end - (days * 86400)

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
        df["VOL_SMA50"] = df["Volume"].rolling(50).mean()
        df["VOL_RATIO"] = df["Volume"] / df["VOL_SMA20"]

        # Momentum
        df["MOM_5D"] = (
            (df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5)
        ) * 100
        df["MOM_10D"] = (
            (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)
        ) * 100
        df["MOM_20D"] = (
            (df["Close"] - df["Close"].shift(20)) / df["Close"].shift(20)
        ) * 100

        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                obv.append(obv[-1] + df["Volume"].iloc[i])
            elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
                obv.append(obv[-1] - df["Volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["OBV"] = obv
        df["OBV_EMA"] = pd.Series(df["OBV"]).ewm(span=10, adjust=False).mean()

        # Bollinger Bands
        df["BB_MID"] = df["Close"].rolling(20).mean()
        df["BB_STD"] = df["Close"].rolling(20).std()
        df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
        df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]
        df["BB_WIDTH"] = ((df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"]) * 100

        # Stochastic
        low14 = df["Low"].rolling(14).min()
        high14 = df["High"].rolling(14).max()
        df["STOCH_K"] = 100 * (df["Close"] - low14) / (high14 - low14)
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

        return df
    except Exception:
        return None


# ============= PRE-FILTERS =============
def apply_liquidity_filter(df):
    """Strict liquidity requirements."""
    try:
        r = df.iloc[-1]
        price = float(r["Close"])
        vol_avg = df["Volume"].tail(20).mean()

        if price < 50:
            return False, "Price too low"

        if vol_avg < 500_000:
            return False, "Volume too low"

        turnover = price * vol_avg
        if turnover < 100_000_000:  # 100M
            return False, "Turnover too low"

        return True, "Passed"
    except Exception:
        return False, "Error"


# ============= BASE SCORING =============
def score_base(df, *, allow_weaker_momentum=False, require_strong_volume=True):
    """
    Core scoring engine.
    Digunakan semua strategi, hanya beda parameter.
    """
    try:
        r = df.iloc[-1]
        score = 0
        details = {}

        passed, reason = apply_liquidity_filter(df)
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"

        # HARD REJECT: strong downtrend
        if r["Close"] < r["EMA21"] < r["EMA50"] < r["EMA200"]:
            return 0, {"Rejected": "Strong downtrend"}, 0, "F"

        if r["Close"] < r["EMA50"]:
            return 0, {"Rejected": "Below EMA50"}, 0, "F"

        mom_20d = float(r["MOM_20D"])
        vol_ratio = float(r["VOL_RATIO"])

        # Momentum filter
        if not allow_weaker_momentum and mom_20d < -5:
            return (
                0,
                {"Rejected": f"Strong negative momentum ({mom_20d:.1f}%)"},
                0,
                "F",
            )

        # Volume filter
        if require_strong_volume and vol_ratio < 1.0:
            return (
                0,
                {"Rejected": f"Insufficient volume ({vol_ratio:.1f}x)"},
                0,
                "F",
            )

        # Momentum penalty
        if mom_20d < -8:
            momentum_penalty = 0.5
            details["⚠️ Warning"] = f"Very weak momentum {mom_20d:.1f}%"
        elif -8 <= mom_20d < -5:
            momentum_penalty = 0.7
            details["⚠️ Warning"] = f"Weak momentum {mom_20d:.1f}%"
        elif -5 <= mom_20d < 0:
            momentum_penalty = 0.85
            details["⚠️ Warning"] = f"Slight negative momentum {mom_20d:.1f}%"
        else:
            momentum_penalty = 1.0

        # EMA alignment
        ema_alignment = 0
        if r["EMA9"] > r["EMA21"]:
            ema_alignment += 1
        if r["EMA21"] > r["EMA50"]:
            ema_alignment += 1
        if r["EMA50"] > r["EMA200"]:
            ema_alignment += 1
        if r["Close"] > r["EMA9"]:
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
            details["Momentum"] = f"🟢 Strong short-term +{mom_5d:.1f}% (5D)"
        elif mom_5d > 2 and mom_10d > 3:
            score += 12
            details["Momentum"] = f"🟡 Good +{mom_5d:.1f}% (5D)"
        elif mom_5d > 0:
            score += 6
            details["Momentum"] = f"🟠 Positive +{mom_5d:.1f}% (5D)"
        elif mom_20d > 5:
            score += 8
            details["Momentum"] = f"🟡 20D momentum +{mom_20d:.1f}%"

        score = int(score * momentum_penalty)

        # Grade mapping
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
def score_general(df):
    # SPEED / default
    return score_base(df, allow_weaker_momentum=False, require_strong_volume=True)


def score_swing(df):
    # sedikit lebih lembut terhadap momentum
    return score_base(df, allow_weaker_momentum=True, require_strong_volume=True)


def score_value(df):
    # boleh momentum lemah, volume tidak harus ekstrem
    return score_base(df, allow_weaker_momentum=True, require_strong_volume=False)


def score_bpjs(df):
    # day trade: butuh volume & momentum agresif
    return score_base(df, allow_weaker_momentum=False, require_strong_volume=True)


def score_bsjp(df):
    # overnight: boleh momentum sedikit lemah tapi volume OK
    return score_base(df, allow_weaker_momentum=True, require_strong_volume=True)


def score_bandar(df):
    # proxy: pakai base, nanti kamu bisa tambahkan analisa bandarnya
    return score_base(df, allow_weaker_momentum=True, require_strong_volume=True)


# ============= TREND, SIGNAL & TRADE PLAN =============
def detect_trend(last_row):
    """Deteksi tren utama berdasarkan EMA"""
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
    """Label Strong Buy / Buy / Hold / Sell"""
    rsi = float(last_row["RSI"]) if not pd.isna(last_row["RSI"]) else 50.0
    vol_ratio = (
        float(last_row["VOL_RATIO"])
        if not pd.isna(last_row["VOL_RATIO"])
        else 1.0
    )
    mom_5d = (
        float(last_row["MOM_5D"]) if not pd.isna(last_row["MOM_5D"]) else 0.0
    )
    mom_20d = (
        float(last_row["MOM_20D"]) if not pd.isna(last_row["MOM_20D"]) else 0.0
    )

    # Strong Buy
    if (
        trend in ["Strong Uptrend", "Uptrend"]
        and grade in ["A+", "A"]
        and vol_ratio > 1.5
        and 45 <= rsi <= 70
        and mom_5d > 0
        and mom_20d > 0
    ):
        return "Strong Buy"

    # Buy
    if (
        trend in ["Strong Uptrend", "Uptrend"]
        and grade in ["A+", "A", "B+"]
        and vol_ratio > 1.0
        and 40 <= rsi <= 75
    ):
        return "Buy"

    # Trend bagus tapi kurang syarat → Hold
    if trend in ["Strong Uptrend", "Uptrend"] and grade in [
        "A+",
        "A",
        "B+",
        "B",
    ]:
        return "Hold"

    # Sideways tapi masih OK
    if trend == "Sideways" and grade in ["B+", "B", "C"]:
        return "Hold"

    return "Sell"


def compute_trade_plan(df, strategy, trend):
    """
    Hitung Entry Ideal, Entry Agresif, TP1–TP3, SL.
    """
    r = df.iloc[-1]
    price = float(r["Close"])
    ema21 = float(r["EMA21"])

    if strategy in ["Swing", "SWING"]:
        entry_ideal = round(price * 0.99, 0)
        tp1 = round(entry_ideal * 1.06, 0)
        tp2 = round(entry_ideal * 1.10, 0)
        tp3 = round(entry_ideal * 1.15, 0)
        sl = round(entry_ideal * 0.95, 0)
    elif strategy in ["Value", "VALUE"]:
        entry_ideal = round(price * 0.98, 0)
        tp1 = round(entry_ideal * 1.15, 0)
        tp2 = round(entry_ideal * 1.25, 0)
        tp3 = round(entry_ideal * 1.35, 0)
        sl = round(entry_ideal * 0.93, 0)
    else:
        # SPEED / BPJS / BSJP / Bandar / General
        entry_ideal = round(price * 0.995, 0)
        tp1 = round(entry_ideal * 1.04, 0)
        tp2 = round(entry_ideal * 1.07, 0)
        tp3 = None
        sl = round(entry_ideal * 0.97, 0)

    # Uptrend → entry di sekitar EMA21 (pullback sehat)
    if trend in ["Strong Uptrend", "Uptrend"] and ema21 < price:
        ema_entry = round(ema21 * 1.01, 0)
        if price * 0.9 < ema_entry < price:
            entry_ideal = ema_entry

    # Downtrend → SL dipersempit
    if trend == "Downtrend":
        sl = round(entry_ideal * 0.96, 0)

    entry_aggressive = round(price, 0)  # harga sekarang

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
    """
    Wrapper utama:
    - ambil data
    - hitung score/grade
    - deteksi trend + signal
    - hitung trade plan (Entry/TP/SL)
    """
    try:
        df = fetch_data(ticker, period)
        if df is None:
            return None

        last_row = df.iloc[-1]
        price = float(last_row["Close"])

        strategy_upper = strategy.upper()

        if strategy_upper == "BPJS":
            score, details, conf, grade = score_bpjs(df)
        elif strategy_upper == "BSJP":
            score, details, conf, grade = score_bsjp(df)
        elif strategy_upper == "BANDAR":
            score, details, conf, grade = score_bandar(df)
        elif strategy_upper == "SWING":
            score, details, conf, grade = score_swing(df)
        elif strategy_upper == "VALUE":
            score, details, conf, grade = score_value(df)
        else:  # SPEED / General
            score, details, conf, grade = score_general(df)

        if grade not in ["A+", "A", "B+", "B", "C"]:
            return None

        trend = detect_trend(last_row)
        signal = classify_signal(last_row, score, grade, trend)
        plan = compute_trade_plan(df, strategy_upper, trend)

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


# ======= SESSION / SCAN HELPERS =======
def save_scan_to_session(df2, df1, strategy):
    st.session_state.last_scan_results = (df2, df1)
    st.session_state.last_scan_time = datetime.now()
    st.session_state.last_scan_strategy = strategy
    st.session_state.scan_count += 1


def display_last_scan_info():
    if st.session_state.last_scan_results:
        df2, df1 = st.session_state.last_scan_results
        time_ago = datetime.now() - st.session_state.last_scan_time
        mins_ago = int(time_ago.total_seconds() / 60)
        st.markdown(
            f"""
        <div style='background:linear-gradient(135deg,#064e3b 0%,#065f46 100%);
                    padding:12px;border-radius:8px;margin-bottom:15px;
                    border-left:4px solid #10b981;'>
          <p style='margin:0;color:white;font-weight:bold;'>📂 LAST SCAN RESULTS</p>
          <p style='margin:5px 0 0 0;color:#d1fae5;font-size:0.9em;'>
            Strategy: {st.session_state.last_scan_strategy} |
            Time: {st.session_state.last_scan_time.strftime('%H:%M:%S')} ({mins_ago} min ago) |
            Found: {len(df2)} Elite picks (Strong Buy)
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return True
    return False


def create_csv_download(df, strategy):
    if not df.empty:
        export_df = df.copy()
        if "Details" in export_df.columns:
            export_df = export_df.drop("Details", axis=1)
        csv = export_df.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"IDX_{strategy}_scan_{timestamp}.csv"
        st.download_button(
            label="💾 Download Results (CSV)",
            data=csv,
            file_name=filename,
            mime="text/csv",
        )


def scan_stocks(tickers, strategy, period, limit1):
    st.info(
        f"🔍 **STAGE 1**: Scanning {len(tickers)} stocks for strategy **{strategy}**..."
    )

    results = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_ticker, t, strategy, period): t
            for t in tickers
        }
        completed = 0

        for future in as_completed(futures):
            completed += 1
            progress.progress(completed / len(tickers))
            status.text(f"📊 {completed}/{len(tickers)} | Found: {len(results)}")
            result = future.result()
            if result:
                results.append(result)
            time.sleep(0.03)

    progress.empty()
    status.empty()

    if not results:
        st.warning("Tidak ada saham yang lolos filter dasar.")
        return pd.DataFrame(), pd.DataFrame()

    df_all = pd.DataFrame(results).sort_values("Score", ascending=False)
    df1 = df_all.head(limit1)

    st.success(
        f"✅ Stage 1: Found {len(df1)} candidates (Avg score: {df1['Score'].mean():.0f})"
    )

    # Stage 2: elite picks
    elite = df1[
        (df1["Grade"].isin(["A+", "A", "B+"])) & (df1["Signal"] == "Strong Buy")
    ].copy()

    # Kalau Strong Buy kosong, pakai semua A+/A
    if elite.empty:
        elite = df1[df1["Grade"].isin(["A+", "A"])].copy()

    st.success(
        f"🏆 Stage 2: {len(elite)} elite picks (A+/A/B+ & Strong Buy priority)"
    )

    save_scan_to_session(elite, df1, strategy)
    return df1, elite


# ============= UI LAYOUT =============
st.title("🚀 IDX Power Screener v5.0 ELITE")
st.caption(
    "3 Trading Styles + BPJS/BSJP + Bandar Tracking + IHSG Dashboard | Lock Screen Safe!"
)

display_ihsg_widget()
tickers = load_tickers()

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.info(f"📊 Total stocks: {len(tickers)}")

    jkt_time = get_jakarta_time()
    st.caption(f"🕐 Jakarta: {jkt_time.strftime('%H:%M WIB')}")

    st.markdown("---")

    menu = st.radio(
        "📋 Mode",
        [
            "⚡ SPEED Trader (1-2d)",
            "🎯 SWING Trader (3-10d)",
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
        limit1 = st.slider("Stage 1: Top N (by Score)", 20, 200, 80, 10)
        st.caption(
            f"Scan {len(tickers)} → ambil Top {limit1} by Score → filter ke Elite Strong Buy"
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
        "v5.0 ELITE – Strong Buy Focus | Educational only, **No financial advice.**"
    )

# ============= MAIN CONTENT =============

# --- SINGLE STOCK ---
if "Single Stock" in menu:
    st.markdown("### 🔍 Single Stock Analysis")

    selected = st.selectbox(
        "Select Stock", sorted([t.replace(".JK", "") for t in tickers])
    )
    strategy_single = st.selectbox(
        "Strategy", ["General", "BPJS", "BSJP", "Bandar", "Swing", "Value"]
    )
    period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)

    if st.button("🔍 ANALYZE", type="primary"):
        ticker_full = (
            selected if selected.endswith(".JK") else f"{selected}.JK"
        )

        with st.spinner(f"Analyzing {selected}..."):
            df = fetch_data(ticker_full, period)

            if df is None:
                st.error("❌ Failed to fetch data")
            else:
                strat_map = {
                    "General": "SPEED",
                    "BPJS": "BPJS",
                    "BSJP": "BSJP",
                    "Bandar": "BANDAR",
                    "Swing": "SWING",
                    "Value": "VALUE",
                }
                result = process_ticker(
                    ticker_full, strat_map[strategy_single], period
                )

                if result is None:
                    st.error(
                        "❌ Analysis failed or stock rejected by filters (grade / volume / trend)"
                    )
                    st.markdown("### 📊 Chart (Reference)")
                    chart = create_chart(df, selected)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.markdown("### 📊 Interactive Chart")
                    chart = create_chart(df, selected)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)

                    st.markdown(f"## 💎 {result['Ticker']}")
                    st.markdown(
                        f"### Grade: **{result['Grade']}** | Trend: **{result['Trend']}** | Signal: **{result['Signal']}**"
                    )

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price", f"Rp {result['Price']:,.0f}")
                    col2.metric("Score", f"{result['Score']}/100")
                    col3.metric("Confidence", f"{result['Confidence']}%")

                    st.success(
                        f"""
                    🧠 **Template Trade Plan**

                    • **Entry Ideal:** Rp {result['Entry']:,.0f}  
                    • **Entry Agresif (harga sekarang):** Rp {result['Entry_Aggressive']:,.0f}

                    • **TP1:** Rp {result['TP1']:,.0f}  
                    • **TP2:** Rp {result['TP2']:,.0f}  

                    • **Stop Loss:** Rp {result['SL']:,.0f}  

                    ⏰ Sesuaikan dengan style: SPEED / SWING / VALUE.
                    """
                    )

                    st.markdown("**Technical Notes:**")
                    for k, v in result["Details"].items():
                        st.caption(f"• **{k}**: {v}")

else:
    # --- MULTI SCAN MODES ---
    st.markdown("### 🧮 Multi-Stock Scanner")

    has_last = display_last_scan_info()

    if has_last:
        with st.expander("📁 Show last scan tables"):
            df2_last, df1_last = st.session_state.last_scan_results
            st.write("Stage 1 – Top candidates:")
            st.dataframe(df1_last.drop(columns=["Details"]), use_container_width=True)
            st.write("Stage 2 – Elite picks:")
            st.dataframe(df2_last.drop(columns=["Details"]), use_container_width=True)

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

        with st.spinner(f"Scanning {len(tickers)} stocks for {strat}..."):
            df1, df2 = scan_stocks(tickers, strat, period, limit1)

        if not df1.empty:
            st.markdown("## 📊 Stage 1 – Top Candidates (by Score)")
            st.dataframe(
                df1.drop(columns=["Details"]).reset_index(drop=True),
                use_container_width=True,
            )

        if not df2.empty:
            st.markdown("## 🏆 Stage 2 – Elite Picks (Strong Buy Focus)")
            # Pagination aman
            rows_per_page = st.selectbox(
                "Rows per page", [10, 20, 40, 80, 100], index=2
            )
            num_pages = max(1, ceil(len(df2) / rows_per_page))

            if num_pages > 1:
                page = st.slider("Page", 1, num_pages, 1)
            else:
                page = 1

            start = (page - 1) * rows_per_page
            end = start + rows_per_page

            df2_view = df2.drop(columns=["Details"]).reset_index(drop=True)
            st.dataframe(
                df2_view.iloc[start:end],
                use_container_width=True,
            )

            create_csv_download(df2_view, strat)
        else:
            st.warning("Belum ada elite picks (Strong Buy) untuk setting ini.")

st.markdown("---")
st.caption(
    "🚀 IDX Power Screener v5.0 ELITE | Strong Buy Focus | For education & backtest only."
)
