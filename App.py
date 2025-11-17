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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================== BASIC CONFIG ==================
st.set_page_config(
    page_title="IDX Power Screener – EXTREME BUILD",
    page_icon="⚡",
    layout="wide"
)

# ============= SESSION STATE (LOCK SCREEN SAFE) =============
if "last_scan_df" not in st.session_state:
    st.session_state.last_scan_df = None
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "last_scan_strategy" not in st.session_state:
    st.session_state.last_scan_strategy = None
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

# ============= IHSG MARKET WIDGET =============
@st.cache_data(ttl=180)
def fetch_ihsg_data():
    """Fetch IHSG (Jakarta Composite Index) data."""
    try:
        import yfinance as yf

        ihsg = yf.Ticker("^JKSE")
        hist = ihsg.history(period="1d")
        if hist.empty:
            return None

        current = float(hist["Close"].iloc[-1])
        open_price = float(hist["Open"].iloc[-1])
        high = float(hist["High"].iloc[-1])
        low = float(hist["Low"].iloc[-1])
        change = current - open_price
        change_pct = (change / open_price) * 100 if open_price != 0 else 0

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
        condition = "🔥 Strong uptrend - Good for momentum!"
        guidance = "✅ Excellent for SPEED / SWING trades"
    elif ihsg["change_pct"] > 0.5:
        condition = "📈 Moderate uptrend - Good conditions"
        guidance = "✅ Good for all strategies"
    elif ihsg["change_pct"] > -0.5:
        condition = "➡️ Sideways - Mixed conditions"
        guidance = "⚠️ Be selective, gunakan stop ketat"
    elif ihsg["change_pct"] > -1.5:
        condition = "📉 Moderate downtrend - Caution"
        guidance = "⚠️ Fokus VALUE, kurangi SPEED"
    else:
        condition = "🔻 Strong downtrend - High risk"
        guidance = "❌ Lebih baik tunggu atau super selektif"

    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
                    padding: 16px; border-radius: 12px; margin-bottom: 18px;
                    border-left: 5px solid {"#22c55e" if ihsg['status']=="up" else "#ef4444"}'>
          <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div>
              <h3 style='margin:0;color:white;'>📊 MARKET OVERVIEW – IHSG</h3>
              <p style='margin:4px 0;color:#e0e7ff;font-size:0.9em;'>
                Jakarta Composite Index
              </p>
            </div>
            <div style='text-align:right;'>
              <h2 style='margin:0;color:white;'>
                {status_emoji} {ihsg['price']:,.2f}
              </h2>
              <p style='margin:4px 0;color:{"#22c55e" if ihsg['status']=="up" else "#ef4444"};
                        font-size:1.05em;font-weight:bold;'>
                {ihsg['change']:+,.2f} ({ihsg['change_pct']:+.2f}%)
              </p>
            </div>
          </div>
          <div style='margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.2);'>
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
    """Load daftar saham IDX dari json atau fallback list."""
    try:
        if os.path.exists("idx_stocks.json"):
            with open("idx_stocks.json", "r") as f:
                data = json.load(f)
            tickers = data.get("tickers", [])
            if tickers:
                return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except Exception:
        pass

    # Fallback pendek (silakan ganti dengan list 799 saham Chef sendiri)
    return [
        "AALI.JK","ABBA.JK","ABMM.JK","ACES.JK","ADRO.JK","ANTM.JK","ASII.JK","BBCA.JK",
        "BBNI.JK","BBRI.JK","BMRI.JK","BRIS.JK","BRMS.JK","BRPT.JK","BREN.JK","BUKA.JK",
        "BUMI.JK","BUVA.JK","CPIN.JK","DADA.JK","DCII.JK","DOID.JK","ELSA.JK","ERAA.JK",
        "EXCL.JK","GGRM.JK","GOTO.JK","HRUM.JK","ICBP.JK","INCO.JK","INDF.JK","INKP.JK",
        "INTP.JK","ISAT.JK","ITMG.JK","JSMR.JK","KLBF.JK","MEDC.JK","MDKA.JK","MIKA.JK",
        "PGAS.JK","PGEO.JK","PTBA.JK","PTPP.JK","PTRO.JK","RAJA.JK","SMGR.JK","SMRA.JK",
        "TBIG.JK","TINS.JK","TKIM.JK","TLKM.JK","TOWR.JK","UNTR.JK","UNVR.JK","WSKT.JK",
        "WSBP.JK","WTON.JK","YELO.JK","ZINC.JK"
    ]

def get_jakarta_time():
    return datetime.now(timezone(timedelta(hours=7)))

# ============= CHART VISUALIZATION =============
def create_chart(df, ticker, period_days=60):
    """Create interactive chart with technical indicators."""
    try:
        df_chart = df.tail(period_days).copy()

        fig = make_subplots(
            rows=3,
            cols=1,
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
            row=1,
            col=1,
        )

        colors = {
            "EMA9": "#3b82f6",
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
                        line=dict(color=colors[ema], width=1.5),
                    ),
                    row=1,
                    col=1,
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
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["RSI"],
                name="RSI",
                line=dict(color="#a855f7", width=2),
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
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1f2933")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1f2933")

        return fig
    except Exception:
        return None

# ============= FETCH DATA =============
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, period="6mo"):
    """Fetch OHLCV + indikator teknikal untuk 1 saham."""
    try:
        end = int(datetime.now().timestamp())
        days_map = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
        days = days_map.get(period, 180)
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
        ).dropna()

        if len(df) < 50:
            return None

        # EMAs
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

        # Volume metrics
        df["VOL_SMA20"] = df["Volume"].rolling(20).mean()
        df["VOL_SMA50"] = df["Volume"].rolling(50).mean()
        df["VOL_RATIO"] = df["Volume"] / df["VOL_SMA20"]

        # Momentum
        df["MOM_5D"] = (df["Close"] / df["Close"].shift(5) - 1) * 100
        df["MOM_10D"] = (df["Close"] / df["Close"].shift(10) - 1) * 100
        df["MOM_20D"] = (df["Close"] / df["Close"].shift(20) - 1) * 100

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
    """Filter saham tuyul: harga terlalu murah / volume tipis dibuang."""
    try:
        r = df.iloc[-1]
        price = float(r["Close"])
        vol_avg = float(df["Volume"].tail(20).mean())

        if price < 50:
            return False, "Price < 50 (tuyul)"
        if vol_avg < 500_000:
            return False, "Vol < 500k (tipis)"
        turnover = price * vol_avg
        if turnover < 150_000_000:  # 150M
            return False, "Value < 150M"

        return True, "OK"
    except Exception:
        return False, "Error"

# ============= SCORING ENGINE =============
def score_general(df):
    """Skor utama – gaya swing cepat 1–3 hari, super ketat."""
    try:
        ok, reason = apply_liquidity_filter(df)
        if not ok:
            return 0, {"Rejected": reason}, 0, "F"

        r = df.iloc[-1]
        price = float(r["Close"])
        ema9 = float(r["EMA9"])
        ema21 = float(r["EMA21"])
        ema50 = float(r["EMA50"])
        ema200 = float(r["EMA200"])
        rsi = float(r["RSI"])
        vol_ratio = float(r["VOL_RATIO"])
        mom5 = float(r["MOM_5D"])
        mom20 = float(r["MOM_20D"])
        macd = float(r["MACD"])
        macd_hist = float(r["MACD_HIST"])
        atr_pct = float(r["ATR_PCT"])

        details = {}
        score = 0

        # Trend dasar
        if price < ema50:
            return 0, {"Rejected": "Price < EMA50 (down / lemah)"}, 0, "F"
        if price < ema21 < ema50 < ema200:
            return 0, {"Rejected": "EMA berurutan downtrend"}, 0, "F"

        # EMA alignment
        ema_alignment = 0
        if ema9 > ema21:
            ema_alignment += 1
        if ema21 > ema50:
            ema_alignment += 1
        if ema50 > ema200:
            ema_alignment += 1
        if price > ema9:
            ema_alignment += 1

        if ema_alignment == 4:
            score += 40
            details["Trend"] = "🟢 Perfect uptrend"
        elif ema_alignment == 3:
            score += 28
            details["Trend"] = "🟡 Strong uptrend"
        elif ema_alignment == 2:
            score += 15
            details["Trend"] = "🟠 OK up / sideways"
        else:
            details["Trend"] = "🔴 Lemah"

        # RSI
        if 50 <= rsi <= 65:
            score += 20
            details["RSI"] = f"🟢 Ideal {rsi:.0f}"
        elif 45 <= rsi < 50:
            score += 15
            details["RSI"] = f"🟡 Cukup {rsi:.0f}"
        elif 40 <= rsi < 45:
            score += 8
            details["RSI"] = f"🟠 Rapuh {rsi:.0f}"
        elif rsi > 70:
            details["RSI"] = f"🔴 Overbought {rsi:.0f}"
        elif rsi < 35:
            details["RSI"] = f"🔴 Oversold {rsi:.0f}"
        else:
            score += 5
            details["RSI"] = f"⚪ Netral {rsi:.0f}"

        # Volume
        if vol_ratio > 2.5:
            score += 18
            details["Volume"] = f"🟢 Big money {vol_ratio:.1f}x"
        elif vol_ratio > 1.5:
            score += 12
            details["Volume"] = f"🟡 Bagus {vol_ratio:.1f}x"
        elif vol_ratio > 1.0:
            score += 6
            details["Volume"] = f"🟠 Oke {vol_ratio:.1f}x"
        else:
            details["Volume"] = f"🔴 Tipis {vol_ratio:.1f}x"

        # Momentum
        if mom5 > 3 and mom20 > 5:
            score += 15
            details["Momentum"] = f"🟢 Kencang +{mom5:.1f}% (5D)"
        elif mom5 > 1 and mom20 > 2:
            score += 10
            details["Momentum"] = f"🟡 Naik +{mom5:.1f}% (5D)"
        elif mom5 > 0 and mom20 > 0:
            score += 6
            details["Momentum"] = f"🟠 Positif +{mom5:.1f}%"
        elif mom20 < -8:
            return 0, {"Rejected": f"Momentum 20D jelek {mom20:.1f}%"}, 0, "F"

        # MACD
        if macd > 0 and macd_hist > 0:
            score += 10
            details["MACD"] = "🟢 Bullish"
        elif macd > 0 > macd_hist:
            score += 5
            details["MACD"] = "🟡 Koreksi sehat"
        elif macd < 0 and macd_hist < 0:
            details["MACD"] = "🔴 Bearish"
        else:
            details["MACD"] = "⚪ Netral"

        # ATR – volatilitas sehat untuk trader cepat
        if 2 <= atr_pct <= 8:
            score += 7
            details["ATR"] = f"🟢 Vol {atr_pct:.1f}% (sehat)"
        elif atr_pct < 1.5:
            details["ATR"] = f"⚪ Terlalu kalem {atr_pct:.1f}%"
        elif atr_pct > 12:
            details["ATR"] = f"🔴 Gila {atr_pct:.1f}%"

        # Grade mapping
        if score >= 85:
            grade, conf = "A+", 90
        elif score >= 75:
            grade, conf = "A", 80
        elif score >= 65:
            grade, conf = "B+", 70
        elif score >= 55:
            grade, conf = "B", 60
        elif score >= 45:
            grade, conf = "C", 50
        else:
            grade, conf = "D", max(score, 30)

        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"

# Strategi lain (sementara pakai base general – nanti bisa di-tune)
def score_bpjs(df):
    # Day trade – lebih agresif sedikit, tapi tetap pakai base general
    return score_general(df)

def score_bsjp(df):
    # Overnight – butuh trend & volume bagus
    return score_general(df)

def score_bandar(df):
    # Bandarmology – nanti bisa ditambah logika akumulasi; sementara pakai umum
    return score_general(df)

def score_swing(df):
    # Swing trader 3–5 hari – mirip general
    return score_general(df)

def score_value(df):
    # Value / masih murah – nanti bisa pakai PBV/PE; sementara pakai general dahulu
    return score_general(df)

# ============= TREND, SIGNAL & TRADE PLAN =============
def detect_trend(last_row):
    """Deteksi tren utama berdasarkan EMA."""
    price = float(last_row["Close"])
    ema9 = float(last_row["EMA9"])
    ema21 = float(last_row["EMA21"])
    ema50 = float(last_row["EMA50"])
    ema200 = float(last_row["EMA200"])

    if price > ema9 > ema21 > ema50 > ema200:
        return "Strong Uptrend"
    if price > ema50 and ema9 > ema21 > ema50:
        return "Uptrend"
    if abs(price - ema50) / price < 0.03:
        return "Sideways"
    return "Downtrend"


def classify_signal(last_row, score, grade, trend):
    """Label Strong Buy / Buy / Hold / Sell."""
    rsi = float(last_row["RSI"]) if not pd.isna(last_row["RSI"]) else 50.0
    vol_ratio = float(last_row["VOL_RATIO"]) if not pd.isna(last_row["VOL_RATIO"]) else 1.0
    mom5 = float(last_row["MOM_5D"]) if not pd.isna(last_row["MOM_5D"]) else 0.0
    mom20 = float(last_row["MOM_20D"]) if not pd.isna(last_row["MOM_20D"]) else 0.0

    if (
        trend in ["Strong Uptrend", "Uptrend"]
        and grade in ["A+", "A"]
        and vol_ratio > 1.5
        and 45 <= rsi <= 70
        and mom5 > 0
        and mom20 > 0
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
    """Hitung Entry / TP / CL – gaya trader cepat 1–3 hari."""
    r = df.iloc[-1]
    price = float(r["Close"])
    ema21 = float(r["EMA21"])

    if strategy in ["BPJS", "BSJP", "Speed"]:
        entry_ideal = round(price * 0.995, 0)
        tp1 = round(entry_ideal * 1.03, 0)
        tp2 = round(entry_ideal * 1.06, 0)
        tp3 = round(entry_ideal * 1.09, 0)
        sl = round(entry_ideal * 0.97, 0)
    elif strategy in ["Swing", "Bandar"]:
        entry_ideal = round(price * 0.99, 0)
        tp1 = round(entry_ideal * 1.05, 0)
        tp2 = round(entry_ideal * 1.10, 0)
        tp3 = round(entry_ideal * 1.15, 0)
        sl = round(entry_ideal * 0.95, 0)
    elif strategy == "Value":
        entry_ideal = round(price * 0.98, 0)
        tp1 = round(entry_ideal * 1.12, 0)
        tp2 = round(entry_ideal * 1.25, 0)
        tp3 = round(entry_ideal * 1.40, 0)
        sl = round(entry_ideal * 0.93, 0)
    else:
        entry_ideal = round(price * 0.995, 0)
        tp1 = round(entry_ideal * 1.04, 0)
        tp2 = round(entry_ideal * 1.07, 0)
        tp3 = round(entry_ideal * 1.10, 0)
        sl = round(entry_ideal * 0.97, 0)

    # Uptrend: ideal entry dekat EMA21 (pullback sehat)
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

# ============= CORE PROCESS =============
def process_ticker(ticker, strategy, period):
    """Ambil data, hitung skor, trend, signal, trade plan."""
    try:
        df = fetch_data(ticker, period)
        if df is None:
            return None

        # Scoring per strategi
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

        last_row = df.iloc[-1]
        trend = detect_trend(last_row)
        signal = classify_signal(last_row, score, grade, trend)
        plan = compute_trade_plan(df, strategy, trend)

        result = {
            "Ticker": ticker.replace(".JK", ""),
            "Price": float(last_row["Close"]),
            "Score": score,
            "Confidence": conf,
            "Grade": grade,
            "Trend": trend,
            "Signal": signal,
            "Entry": plan["entry_ideal"],
            "Entry_Aggressive": plan["entry_aggressive"],
            "TP1": plan["tp1"],
            "TP2": plan["tp2"],
            "TP3": plan["tp3"],
            "CL": plan["sl"],
            "Details": details,
        }
        return result
    except Exception:
        return None

def scan_universe(tickers, strategy, period):
    """Scan semua saham – return DataFrame hasil lengkap (tanpa Stage filter)."""
    results = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ticker, t, strategy, period): t for t in tickers}
        total = len(futures)
        done = 0

        for future in as_completed(futures):
            done += 1
            progress.progress(done / total)
            status.text(f"📊 Scan {done}/{total} | Found: {len(results)}")
            res = future.result()
            if res:
                results.append(res)
            time.sleep(0.02)

    progress.empty()
    status.empty()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    st.session_state.last_scan_df = df
    st.session_state.last_scan_time = datetime.now()
    st.session_state.last_scan_strategy = strategy
    st.session_state.scan_count += 1
    return df

def display_last_scan_info():
    if st.session_state.last_scan_df is None:
        return
    df = st.session_state.last_scan_df
    t = st.session_state.last_scan_time
    strat = st.session_state.last_scan_strategy
    mins_ago = int((datetime.now() - t).total_seconds() / 60)
    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,#022c22 0%,#065f46 100%);
                    padding:12px;border-radius:8px;margin-bottom:12px;
                    border-left:4px solid #22c55e;'>
          <p style='margin:0;color:white;font-weight:bold;'>📂 LAST SCAN</p>
          <p style='margin:4px 0 0 0;color:#d1fae5;font-size:0.9em;'>
            Strategy: {strat} | Time: {t.strftime('%H:%M:%S')} ({mins_ago} min ago) |
            Candidates: {len(df)}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============= UI =============
st.title("⚡ IDX Power Screener – EXTREME BUILD")
st.caption("Gaya trader cepat 1–3 hari • Anti saham tuyul • Fokus A+/A & Strong Buy")

display_ihsg_widget()
tickers = load_tickers()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.info(f"📊 Total stocks loaded: **{len(tickers)}**")

    jkt = get_jakarta_time()
    st.caption(f"🕒 Jakarta time: {jkt.strftime('%d %b %Y • %H:%M WIB')}")

    st.markdown("---")

    menu = st.radio(
        "📋 Menu Utama",
        [
            "🔎 Screen ALL IDX",
            "🔍 Single Stock",
            "🌙 BSJP (Beli sore jual pagi)",
            "⚡ BPJS (Beli pagi jual sore)",
            "💎 Saham Masih Murah (Value)",
            "🔮 Bandarmology",
        ],
    )

    st.markdown("---")

    if menu != "🔍 Single Stock":
        period = st.selectbox("Period data", ["3mo", "6mo", "1y"], index=1)

        st.markdown("### 🎯 Stage Filter")
        stage1_limit = st.selectbox(
            "Stage 1 – Top N kandidat",
            options=[50, 100, 150, 200],
            index=1,
            help="Berapa banyak saham terbaik dari 799 yang mau diambil sebagai kandidat awal."
        )
        stage2_limit = st.selectbox(
            "Stage 2 – Elite Picks",
            options=[10, 20, 30, 40, 50],
            index=1,
            help="Disaring lagi dari Stage 1 menjadi saham paling bagus-bagus."
        )

        st.markdown("### 📄 Tabel Tampilan")
        rows_per_page = st.selectbox(
            "Rows per page (Stage 1 table)",
            options=[20, 40, 60, 80, 100],
            index=1,
        )

    st.markdown("---")
    st.caption("v6 – Extreme Build • Edukasi saja, bukan rekomendasi beli / jual.")

# ------- SCREEN ALL IDX -------
if menu == "🔎 Screen ALL IDX":
    st.markdown("## 🔎 Screen ALL IDX – Full Universe")

    display_last_scan_info()

    strategy_scan = st.selectbox(
        "Pilih gaya strategi scan:",
        [
            "Speed (1–2 hari cepat)",
            "Swing (3–5 hari)",
            "Value (murah & masa depan bagus)",
            "BPJS (beli pagi jual sore)",
            "BSJP (beli sore jual pagi)",
            "Bandar (bandarmology)",
        ],
    )

    strat_map = {
        "Speed (1–2 hari cepat)": "Speed",
        "Swing (3–5 hari)": "Swing",
        "Value (murah & masa depan bagus)": "Value",
        "BPJS (beli pagi jual sore)": "BPJS",
        "BSJP (beli sore jual pagi)": "BSJP",
        "Bandar (bandarmology)": "Bandar",
    }
    strat_key = strat_map[strategy_scan]

    if st.button("🚀 RUN SCAN", type="primary"):
        with st.spinner(f"Scanning {len(tickers)} saham untuk strategi {strategy_scan}..."):
            df_all = scan_universe(tickers, strat_key, period)

        if df_all.empty:
            st.error("❌ Tidak ada saham yang lolos filter hari ini. Market mungkin lagi jelek sekali.")
        else:
            # Stage 1: filter hanya Strong Buy / Buy & bukan downtrend
            stage1_universe = df_all[
                (df_all["Signal"].isin(["Strong Buy", "Buy"])) &
                (df_all["Trend"] != "Downtrend")
            ].sort_values("Score", ascending=False)

            stage1_df = stage1_universe.head(stage1_limit)

            st.markdown("### 🥇 Stage 1 – Top Kandidat (Strong Buy / Buy • No Downtrend)")
            if stage1_df.empty:
                st.info("Belum ada kandidat yang memenuhi syarat Stage 1.")
            else:
                total_stage1 = len(stage1_df)
                num_pages = max((total_stage1 - 1) // rows_per_page + 1, 1)
                page = st.slider("Page Stage 1", 1, num_pages, 1)
                start = (page - 1) * rows_per_page
                end = start + rows_per_page

                st.dataframe(
                    stage1_df.iloc[start:end][
                        ["Ticker", "Price", "Score", "Confidence", "Grade",
                         "Trend", "Signal", "Entry", "Entry_Aggressive", "TP1", "TP2", "TP3", "CL"]
                    ],
                    use_container_width=True,
                    height=420,
                )

            # Stage 2: Elite picks
            st.markdown("### 🏆 Stage 2 – Elite Picks (A+/A & Strong Buy Only)")
            stage2_universe = stage1_df[
                (stage1_df["Signal"] == "Strong Buy") &
                (stage1_df["Grade"].isin(["A+", "A"]))
            ].sort_values("Score", ascending=False)
            elite_df = stage2_universe.head(stage2_limit)

            if elite_df.empty:
                st.info(
                    "Belum ada saham dengan kombinasi **Strong Buy + Grade A+/A** "
                    "dari Stage 1. Wajar kalau market lagi sepi / lemah."
                )
            else:
                st.dataframe(
                    elite_df[
                        ["Ticker", "Price", "Score", "Confidence", "Grade",
                         "Trend", "Signal", "Entry", "Entry_Aggressive", "TP1", "TP2", "TP3", "CL"]
                    ],
                    use_container_width=True,
                    height=380,
                )

                csv = elite_df.to_csv(index=False)
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                st.download_button(
                    "💾 Download Elite Picks (CSV)",
                    data=csv,
                    file_name=f"IDX_elite_{strat_key}_{ts}.csv",
                    mime="text/csv",
                )

# ------- SINGLE STOCK ANALYSIS -------
elif menu == "🔍 Single Stock":
    st.markdown("## 🔍 Single Stock Analysis")

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected = st.selectbox("Pilih saham", [t.replace(".JK", "") for t in tickers])
    with col_sel2:
        period = st.selectbox("Period data", ["3mo", "6mo", "1y"], index=1)

    strat_single = st.selectbox(
        "Pilih gaya trading:",
        ["Speed", "BPJS", "BSJP", "Swing", "Value", "Bandar"],
    )

    if st.button("🔍 ANALYZE", type="primary"):
        ticker_full = selected if selected.endswith(".JK") else f"{selected}.JK"
        with st.spinner(f"Menganalisa {selected}..."):
            df = fetch_data(ticker_full, period)

        if df is None:
            st.error("❌ Gagal mengambil data dari Yahoo Finance.")
        else:
            # pakai core engine
            result = process_ticker(ticker_full, strat_single, period)
            last_row = df.iloc[-1]

            st.markdown("### 📊 Chart & Price Action")
            chart = create_chart(df, selected)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

            if result is None:
                st.warning("Saham ini dibuang oleh filter (grade / volume / trend).")
            else:
                st.markdown(f"## 💎 {result['Ticker']} – {result['Signal']} ({result['Grade']})")
                colm1, colm2, colm3, colm4 = st.columns(4)
                colm1.metric("Price", f"Rp {result['Price']:,.0f}")
                colm2.metric("Score", f"{result['Score']}/100")
                colm3.metric("Confidence", f"{result['Confidence']}%")
                colm4.metric("Trend", result["Trend"])

                st.success(
                    f"""
                    **TRADE PLAN ({strat_single}) – gaya cepat 1–3 hari**

                    • Entry Ideal: **Rp {result['Entry']:,.0f}**  
                    • Entry Agresif (chase): **Rp {result['Entry_Aggressive']:,.0f}**  

                    • TP1: **Rp {result['TP1']:,.0f}**  
                    • TP2: **Rp {result['TP2']:,.0f}**  
                    • TP3: **Rp {result['TP3']:,.0f}**  

                    • Cut Loss: **Rp {result['CL']:,.0f}**  

                    ⏰ Saran: maksimal hold 1–3 hari, disiplin pada plan & CL.
                    """
                )

                st.markdown("### 📌 Technical Notes")
                for k, v in result["Details"].items():
                    st.caption(f"• **{k}**: {v}")

# ------- BSJP SHORTCUT -------
elif menu == "🌙 BSJP (Beli sore jual pagi)":
    st.markdown("## 🌙 BSJP – Beli Sore Jual Pagi")
    st.info("Scan fokus saham yang cocok dipegang overnight, gaya cepat 1–2 hari.")

    if st.button("🚀 Scan BSJP Now", type="primary"):
        with st.spinner("Scanning untuk BSJP..."):
            df_all = scan_universe(tickers, "BSJP", period)

        if df_all.empty:
            st.error("Tidak ada kandidat hari ini.")
        else:
            stage1 = df_all[
                (df_all["Signal"].isin(["Strong Buy", "Buy"])) &
                (df_all["Trend"].isin(["Strong Uptrend", "Uptrend", "Sideways"]))
            ].head(stage1_limit)

            st.markdown("### 🥇 BSJP – Stage 1 Kandidat")
            st.dataframe(
                stage1[
                    ["Ticker", "Price", "Score", "Grade", "Trend", "Signal",
                     "Entry", "Entry_Aggressive", "TP1", "TP2", "TP3", "CL"]
                ],
                use_container_width=True,
            )

            stage2 = stage1[
                (stage1["Signal"] == "Strong Buy") &
                (stage1["Grade"].isin(["A+", "A"]))
            ].head(stage2_limit)

            st.markdown("### 🏆 BSJP – Elite Picks")
            st.dataframe(
                stage2[
                    ["Ticker", "Price", "Score", "Grade", "Trend", "Signal",
                     "Entry", "Entry_Aggressive", "TP1", "TP2", "TP3", "CL"]
                ],
                use_container_width=True,
            )

# ------- BPJS SHORTCUT -------
elif menu == "⚡ BPJS (Beli pagi jual sore)":
    st.markdown("## ⚡ BPJS – Beli Pagi Jual Sore")
    st.info("Scan fokus saham untuk daytrade, keluar di hari yang sama.")

    if st.button("🚀 Scan BPJS Now", type="primary"):
        with st.spinner("Scanning untuk BPJS..."):
            df_all = scan_universe(tickers, "BPJS", period)

        if df_all.empty:
            st.error("Tidak ada kandidat hari ini.")
        else:
            stage1 = df_all[
                (df_all["Signal"].isin(["Strong Buy", "Buy"])) &
                (df_all["Trend"].isin(["Strong Uptrend", "Uptrend"]))
            ].head(stage1_limit)

            st.markdown("### 🥇 BPJS – Stage 1 Kandidat")
            st.dataframe(
                stage1[
                    ["Ticker", "Price", "Score", "Grade", "Trend", "Signal",
                     "Entry", "Entry_Aggressive", "TP1", "TP2", "TP3", "CL"]
                ],
                use_container_width=True,
            )

            stage2 = stage1[
                (stage1["Signal"] == "Strong Buy") &
                (stage1["Grade"].isin(["A+", "A"]))
            ].head(stage2_limit)

            st.markdown("### 🏆 BPJS – Elite Picks")
            st.dataframe(
                stage2[
                    ["Ticker", "Price", "Score", "Grade", "Trend", "Signal",
                     "Entry", "Entry_Aggressive", "TP1", "TP2", "TP3", "CL"]
                ],
                use_container_width=True,
            )

# ------- VALUE / MASIH MURAH -------
elif menu == "💎 Saham Masih Murah (Value)":
    st.markdown("## 💎 Saham Masih Murah & Punya Masa Depan")
    st.info("Untuk sementara pakai engine teknikal, nanti bisa ditambah FA (PBV/PE).")

    if st.button("🚀 Scan Value Picks", type="primary"):
        with st.spinner("Scanning untuk Value / Murah..."):
            df_all = scan_universe(tickers, "Value", period)

        if df_all.empty:
            st.error("Tidak ada kandidat hari ini.")
        else:
            stage1 = df_all[
                (df_all["Signal"].isin(["Strong Buy", "Buy", "Hold"])) &
                (df_all["Trend"].isin(["Strong Uptrend", "Uptrend", "Sideways"]))
            ].head(stage1_limit)

            st.markdown("### 🥇 Value – Stage 1 Kandidat")
            st.dataframe(
                stage1[
                    ["Ticker", "Price", "Score", "Grade", "Trend", "Signal",
                     "Entry", "Entry_Aggressive", "TP1", "TP2", "TP3", "CL"]
                ],
                use_container_width=True,
            )

            stage2 = stage1[
                (stage1["Signal"].isin(["Strong Buy", "Buy"])) &
                (stage1["Grade"].isin(["A+", "A", "B+"]))
            ].head(stage2_limit)

            st.markdown("### 🏆 Value – Elite Picks")
            st.dataframe(
                stage2[
                    ["Ticker", "Price", "Score", "Grade", "Trend", "Signal",
                     "Entry", "Entry_Aggressive", "TP1", "TP2", "TP3", "CL"]
                ],
                use_container_width=True,
            )

# ------- BANDARMOLOGY -------
elif menu == "🔮 Bandarmology":
    st.markdown("## 🔮 Bandarmology – Nyari Jejak Uang Besar")
    st.info(
        "Versi awal: pakai kombinasi volume, trend, momentum untuk cari saham yang terlihat ada pergerakan 'bandar'. "
        "Nanti bisa ditambah data broker summary kalau tersedia."
    )

    if st.button("🚀 Scan Bandarmology", type="primary"):
        with st.spinner("Scanning untuk Bandarmology..."):
            df_all = scan_universe(tickers, "Bandar", period)

        if df_all.empty:
            st.error("Tidak ada kandidat hari ini.")
        else:
            stage1 = df_all[
                (df_all["Signal"].isin(["Strong Buy", "Buy", "Hold"])) &
                (df_all["Trend"].isin(["Strong Uptrend", "Uptrend", "Sideways"]))
            ].head(stage1_limit)

            st.markdown("### 🥇 Bandarmology – Stage 1 Kandidat")
            st.dataframe(
                stage1[
                    ["Ticker", "Price", "Score", "Grade", "Trend", "Signal",
                     "Entry", "Entry_Aggressive", "TP1", "TP2", "TP3", "CL"]
                ],
                use_container_width=True,
            )

            stage2 = stage1[
                (stage1["Signal"].isin(["Strong Buy"])) &
                (stage1["Grade"].isin(["A+", "A"]))
            ].head(stage2_limit)

            st.markdown("### 🏆 Bandarmology – Elite Picks")
            st.dataframe(
                stage2[
                    ["Ticker", "Price", "Score", "Grade", "Trend", "Signal",
                     "Entry", "Entry_Aggressive", "TP1", "TP2", "TP3", "CL"]
                ],
                use_container_width=True,
            )

st.markdown("---")
st.caption(
    "⚡ IDX Power Screener – EXTREME BUILD • Fokus trader cepat 1–3 hari • Edukasi saja, bukan ajakan beli/jual."
)