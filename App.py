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
import plotly.express as px
from io import BytesIO
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except:
    EXCEL_AVAILABLE = False
from scipy import stats
from scipy.signal import find_peaks, argrelextrema

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

# ============= WATCHLIST & PORTFOLIO STATE =============
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []  # {ticker, buy_price, qty, buy_date}
if "alerts" not in st.session_state:
    st.session_state.alerts = []  # {ticker, type, value, condition}
if "trade_journal" not in st.session_state:
    st.session_state.trade_journal = []  # {ticker, entry, exit, profit, date, strategy}
if "comparison_list" not in st.session_state:
    st.session_state.comparison_list = []

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

# ============= SECTOR MAPPING =============
SECTOR_MAP = {
    # Basic Materials
    "ADRO": "Mining", "ANTM": "Mining", "PTBA": "Mining", "ITMG": "Mining", "INCO": "Mining",
    "TINS": "Mining", "MDKA": "Mining", "MEDC": "Mining", "BUMI": "Mining",
    # Energy
    "PGAS": "Energy", "ELSA": "Energy", "PGEO": "Energy",
    # Financials
    "BBCA": "Banking", "BBRI": "Banking", "BMRI": "Banking", "BBNI": "Banking", "BRIS": "Banking",
    "BBTN": "Banking", "BJBR": "Banking", "MEGA": "Banking", "BNGA": "Banking",
    # Consumer
    "UNVR": "Consumer", "ICBP": "Consumer", "INDF": "Consumer", "GGRM": "Consumer", "HMSP": "Consumer",
    "KLBF": "Consumer", "CPIN": "Consumer", "JPFA": "Consumer", "SIDO": "Consumer",
    # Infrastructure
    "TLKM": "Telco", "EXCL": "Telco", "ISAT": "Telco", "TOWR": "Infrastructure",
    "JSMR": "Infrastructure", "PTPP": "Infrastructure", "WIKA": "Infrastructure", "WSKT": "Infrastructure",
    # Property
    "ASRI": "Property", "BSDE": "Property", "CTRA": "Property", "SMRA": "Property", "PWON": "Property",
    # Technology
    "GOTO": "Technology", "BUKA": "Technology", "EMTK": "Technology", "MAPI": "Technology",
    # Transportation
    "BIRD": "Transportation", "GIAA": "Transportation", "WEHA": "Transportation",
    # Automotive
    "ASII": "Automotive", "AUTO": "Automotive", "UNTR": "Automotive",
}

# ============= ADVANCED TECHNICAL FUNCTIONS =============
def detect_support_resistance(df, lookback=20):
    """Detect support and resistance levels using local minima/maxima."""
    try:
        highs = df['High'].values
        lows = df['Low'].values

        # Find local maxima (resistance)
        resistance_idx = argrelextrema(highs, np.greater, order=lookback)[0]
        resistance_levels = highs[resistance_idx]

        # Find local minima (support)
        support_idx = argrelextrema(lows, np.less, order=lookback)[0]
        support_levels = lows[support_idx]

        # Get the most recent and significant levels
        resistances = sorted(resistance_levels[-5:], reverse=True) if len(resistance_levels) > 0 else []
        supports = sorted(support_levels[-5:], reverse=True) if len(support_levels) > 0 else []

        return {
            'resistances': resistances[:3],  # Top 3
            'supports': supports[:3]  # Top 3
        }
    except:
        return {'resistances': [], 'supports': []}

def calculate_fibonacci_levels(df, period=50):
    """Calculate Fibonacci retracement levels."""
    try:
        recent = df.tail(period)
        high = recent['High'].max()
        low = recent['Low'].min()
        diff = high - low

        levels = {
            '0.0%': high,
            '23.6%': high - (diff * 0.236),
            '38.2%': high - (diff * 0.382),
            '50.0%': high - (diff * 0.5),
            '61.8%': high - (diff * 0.618),
            '78.6%': high - (diff * 0.786),
            '100.0%': low,
        }
        return levels
    except:
        return {}

def detect_chart_patterns(df, lookback=30):
    """Simple pattern detection (double top, double bottom, head & shoulders)."""
    patterns = []
    try:
        recent = df.tail(lookback)

        # Find peaks and troughs
        peaks_idx, _ = find_peaks(recent['High'].values, distance=5)
        troughs_idx, _ = find_peaks(-recent['Low'].values, distance=5)

        # Double Top
        if len(peaks_idx) >= 2:
            last_two_peaks = recent['High'].iloc[peaks_idx[-2:]]
            if abs(last_two_peaks.iloc[0] - last_two_peaks.iloc[1]) / last_two_peaks.iloc[0] < 0.02:
                patterns.append("⚠️ Double Top (Bearish)")

        # Double Bottom
        if len(troughs_idx) >= 2:
            last_two_troughs = recent['Low'].iloc[troughs_idx[-2:]]
            if abs(last_two_troughs.iloc[0] - last_two_troughs.iloc[1]) / last_two_troughs.iloc[0] < 0.02:
                patterns.append("✅ Double Bottom (Bullish)")

        # Head & Shoulders (simplified)
        if len(peaks_idx) >= 3:
            last_three_peaks = recent['High'].iloc[peaks_idx[-3:]]
            if last_three_peaks.iloc[1] > last_three_peaks.iloc[0] and last_three_peaks.iloc[1] > last_three_peaks.iloc[2]:
                if abs(last_three_peaks.iloc[0] - last_three_peaks.iloc[2]) / last_three_peaks.iloc[0] < 0.03:
                    patterns.append("⚠️ Head & Shoulders (Bearish)")

        return patterns if patterns else ["No clear pattern detected"]
    except:
        return ["Pattern detection failed"]

def calculate_risk_metrics(entry_price, stop_loss, take_profit, capital, risk_pct=2):
    """Calculate position sizing and risk/reward metrics."""
    try:
        risk_per_trade = capital * (risk_pct / 100)
        risk_per_share = entry_price - stop_loss

        if risk_per_share <= 0:
            return None

        position_size = int(risk_per_trade / risk_per_share)
        total_investment = position_size * entry_price

        potential_profit = (take_profit - entry_price) * position_size
        potential_loss = risk_per_share * position_size

        rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0

        return {
            'position_size': position_size,
            'total_investment': total_investment,
            'potential_profit': potential_profit,
            'potential_loss': potential_loss,
            'rr_ratio': rr_ratio
        }
    except:
        return None

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
def create_chart(df, ticker, period_days=60, show_sr=True, show_fib=False):
    """Create interactive chart with technical indicators, S/R, and Fibonacci."""
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

        # Add Support & Resistance
        if show_sr:
            sr_levels = detect_support_resistance(df_chart)
            for i, res in enumerate(sr_levels['resistances']):
                fig.add_hline(y=res, line_dash="dash", line_color="#ef4444",
                             opacity=0.5, annotation_text=f"R{i+1}", row=1, col=1)
            for i, sup in enumerate(sr_levels['supports']):
                fig.add_hline(y=sup, line_dash="dash", line_color="#22c55e",
                             opacity=0.5, annotation_text=f"S{i+1}", row=1, col=1)

        # Add Fibonacci Levels
        if show_fib:
            fib_levels = calculate_fibonacci_levels(df_chart)
            fib_colors = ['#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#c4b5fd', '#a78bfa', '#8b5cf6']
            for idx, (level_name, level_value) in enumerate(fib_levels.items()):
                fig.add_hline(y=level_value, line_dash="dot",
                             line_color=fib_colors[idx % len(fib_colors)],
                             opacity=0.4, annotation_text=level_name, row=1, col=1)

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

# ============= EXPORT FUNCTIONS =============
def export_to_excel(df, filename="idx_screener_results.xlsx"):
    """Export dataframe to Excel with formatting."""
    if not EXCEL_AVAILABLE:
        return None

    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)

            workbook = writer.book
            worksheet = writer.sheets['Results']

            # Auto-adjust column width
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                ) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 30)

        return output.getvalue()
    except:
        return None

# ============= MULTI-TIMEFRAME ANALYSIS =============
def fetch_multi_timeframe(ticker):
    """Fetch data for multiple timeframes."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)

        # Daily, Weekly, Monthly
        daily = stock.history(period="3mo", interval="1d")
        weekly = stock.history(period="1y", interval="1wk")
        monthly = stock.history(period="5y", interval="1mo")

        results = {}
        for name, data in [("Daily", daily), ("Weekly", weekly), ("Monthly", monthly)]:
            if not data.empty and len(data) > 20:
                # Calculate basic trend
                close = data['Close']
                ema20 = close.ewm(span=20, adjust=False).mean()
                ema50 = close.ewm(span=50, adjust=False).mean() if len(data) >= 50 else ema20

                last_close = close.iloc[-1]
                last_ema20 = ema20.iloc[-1]
                last_ema50 = ema50.iloc[-1]

                trend = "Uptrend" if last_close > last_ema20 > last_ema50 else \
                        "Sideways" if abs(last_close - last_ema20) / last_close < 0.03 else "Downtrend"

                results[name] = {
                    "trend": trend,
                    "price": last_close,
                    "ema20": last_ema20,
                    "ema50": last_ema50,
                    "change_pct": ((last_close - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100)
                }

        return results
    except:
        return {}

# ============= SECTOR ANALYSIS =============
def analyze_sectors(df_results):
    """Analyze performance by sector."""
    sector_data = []

    for _, row in df_results.iterrows():
        ticker = row['Ticker']
        sector = SECTOR_MAP.get(ticker, "Other")
        sector_data.append({
            'Ticker': ticker,
            'Sector': sector,
            'Score': row.get('Score', 0),
            'Grade': row.get('Grade', 'N/A'),
            'Signal': row.get('Signal', 'N/A')
        })

    sector_df = pd.DataFrame(sector_data)
    sector_summary = sector_df.groupby('Sector').agg({
        'Score': 'mean',
        'Ticker': 'count'
    }).rename(columns={'Ticker': 'Count'}).sort_values('Score', ascending=False)

    return sector_summary

def create_sector_heatmap(df_results):
    """Create sector performance heatmap."""
    sector_data = []
    for _, row in df_results.iterrows():
        ticker = row['Ticker']
        sector = SECTOR_MAP.get(ticker, "Other")
        sector_data.append({
            'Ticker': ticker,
            'Sector': sector,
            'Score': row.get('Score', 0)
        })

    if not sector_data:
        return None

    df_sector = pd.DataFrame(sector_data)
    sector_avg = df_sector.groupby('Sector')['Score'].mean().reset_index()
    sector_avg['Count'] = df_sector.groupby('Sector').size().values

    fig = px.treemap(
        sector_avg,
        path=['Sector'],
        values='Count',
        color='Score',
        color_continuous_scale='RdYlGn',
        title='Sector Performance Heatmap',
        labels={'Score': 'Avg Score', 'Count': 'Stock Count'}
    )

    fig.update_layout(height=500, template="plotly_dark")
    return fig

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
            "---",
            "📝 Watchlist & Portfolio",
            "🔬 Stock Comparison",
            "🗺️ Sector Analysis",
            "💰 Risk Calculator",
            "📓 Trade Journal",
            "⏰ Multi-Timeframe",
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

                # Download options
                col_dl1, col_dl2 = st.columns(2)

                with col_dl1:
                    csv = elite_df.to_csv(index=False)
                    ts = datetime.now().strftime("%Y%m%d_%H%M")
                    st.download_button(
                        "💾 Download Elite Picks (CSV)",
                        data=csv,
                        file_name=f"IDX_elite_{strat_key}_{ts}.csv",
                        mime="text/csv",
                    )

                with col_dl2:
                    if EXCEL_AVAILABLE:
                        excel_data = export_to_excel(elite_df, f"IDX_elite_{strat_key}_{ts}.xlsx")
                        if excel_data:
                            st.download_button(
                                "📊 Download Excel (Formatted)",
                                data=excel_data,
                                file_name=f"IDX_elite_{strat_key}_{ts}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                    else:
                        st.info("Excel export not available")

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

    # Chart options
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        show_sr = st.checkbox("Show Support/Resistance", value=True)
    with col_opt2:
        show_fib = st.checkbox("Show Fibonacci Levels", value=False)

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
            chart = create_chart(df, selected, show_sr=show_sr, show_fib=show_fib)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

            # Advanced Analysis
            col_adv1, col_adv2 = st.columns(2)

            with col_adv1:
                st.markdown("### 🎯 Support & Resistance")
                sr_levels = detect_support_resistance(df)
                if sr_levels['resistances']:
                    st.write("**Resistance Levels:**")
                    for i, r in enumerate(sr_levels['resistances']):
                        st.caption(f"R{i+1}: Rp {r:,.0f}")
                if sr_levels['supports']:
                    st.write("**Support Levels:**")
                    for i, s in enumerate(sr_levels['supports']):
                        st.caption(f"S{i+1}: Rp {s:,.0f}")

            with col_adv2:
                st.markdown("### 📐 Fibonacci Levels")
                fib_levels = calculate_fibonacci_levels(df)
                if fib_levels:
                    for level, value in fib_levels.items():
                        st.caption(f"{level}: Rp {value:,.0f}")

            # Pattern Detection
            st.markdown("### 🔍 Chart Pattern Detection")
            patterns = detect_chart_patterns(df)
            for pattern in patterns:
                if "Bullish" in pattern:
                    st.success(pattern)
                elif "Bearish" in pattern:
                    st.warning(pattern)
                else:
                    st.info(pattern)

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

                # Quick add to watchlist
                col_w1, col_w2 = st.columns([3, 1])
                with col_w2:
                    if st.button(f"📝 Add {selected} to Watchlist", key="quick_watchlist"):
                        if selected not in st.session_state.watchlist:
                            st.session_state.watchlist.append(selected)
                            st.success(f"Added to watchlist!")

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

# ------- WATCHLIST & PORTFOLIO -------
elif menu == "📝 Watchlist & Portfolio":
    st.markdown("## 📝 Watchlist & Portfolio Management")

    tab1, tab2 = st.tabs(["📋 Watchlist", "💼 Portfolio"])

    with tab1:
        st.markdown("### 📋 Your Watchlist")

        # Add to watchlist
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_to_add = st.selectbox("Add stock to watchlist",
                                        [t.replace(".JK", "") for t in tickers],
                                        key="watchlist_add")
        with col2:
            st.write("")
            st.write("")
            if st.button("➕ Add", key="add_watchlist"):
                if ticker_to_add not in st.session_state.watchlist:
                    st.session_state.watchlist.append(ticker_to_add)
                    st.success(f"Added {ticker_to_add} to watchlist!")
                else:
                    st.warning("Already in watchlist!")

        # Display watchlist
        if st.session_state.watchlist:
            st.markdown(f"**Tracking {len(st.session_state.watchlist)} stocks:**")

            watchlist_data = []
            for ticker in st.session_state.watchlist:
                ticker_full = f"{ticker}.JK"
                df = fetch_data(ticker_full, "1mo")
                if df is not None:
                    last = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else last
                    change_pct = ((last['Close'] - prev['Close']) / prev['Close'] * 100)

                    watchlist_data.append({
                        'Ticker': ticker,
                        'Price': f"Rp {last['Close']:,.0f}",
                        'Change': f"{change_pct:+.2f}%",
                        'RSI': f"{last['RSI']:.1f}",
                        'Volume': f"{last['Volume']:,.0f}"
                    })

            if watchlist_data:
                wl_df = pd.DataFrame(watchlist_data)
                st.dataframe(wl_df, use_container_width=True, height=300)

            # Remove from watchlist
            to_remove = st.selectbox("Remove from watchlist", st.session_state.watchlist, key="remove_wl")
            if st.button("🗑️ Remove", key="remove_watchlist"):
                st.session_state.watchlist.remove(to_remove)
                st.success(f"Removed {to_remove}!")
                st.rerun()
        else:
            st.info("Your watchlist is empty. Add stocks to track them!")

    with tab2:
        st.markdown("### 💼 Your Portfolio")

        # Add position
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            port_ticker = st.selectbox("Ticker", [t.replace(".JK", "") for t in tickers], key="port_ticker")
        with col2:
            buy_price = st.number_input("Buy Price", min_value=1, value=1000, step=10, key="buy_price")
        with col3:
            qty = st.number_input("Quantity", min_value=1, value=100, step=100, key="qty")
        with col4:
            st.write("")
            st.write("")
            if st.button("➕ Add Position", key="add_port"):
                st.session_state.portfolio.append({
                    'ticker': port_ticker,
                    'buy_price': buy_price,
                    'qty': qty,
                    'buy_date': datetime.now().strftime('%Y-%m-%d')
                })
                st.success(f"Added {qty} shares of {port_ticker}!")

        # Display portfolio
        if st.session_state.portfolio:
            portfolio_data = []
            total_value = 0
            total_cost = 0

            for pos in st.session_state.portfolio:
                ticker_full = f"{pos['ticker']}.JK"
                df = fetch_data(ticker_full, "1mo")
                if df is not None:
                    current_price = df.iloc[-1]['Close']
                    cost = pos['buy_price'] * pos['qty']
                    value = current_price * pos['qty']
                    profit = value - cost
                    profit_pct = (profit / cost * 100)

                    total_value += value
                    total_cost += cost

                    portfolio_data.append({
                        'Ticker': pos['ticker'],
                        'Buy Price': f"Rp {pos['buy_price']:,.0f}",
                        'Current': f"Rp {current_price:,.0f}",
                        'Qty': pos['qty'],
                        'Cost': f"Rp {cost:,.0f}",
                        'Value': f"Rp {value:,.0f}",
                        'P/L': f"Rp {profit:,.0f}",
                        'P/L %': f"{profit_pct:+.2f}%",
                        'Date': pos['buy_date']
                    })

            if portfolio_data:
                port_df = pd.DataFrame(portfolio_data)
                st.dataframe(port_df, use_container_width=True, height=300)

                total_profit = total_value - total_cost
                total_profit_pct = (total_profit / total_cost * 100) if total_cost > 0 else 0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Cost", f"Rp {total_cost:,.0f}")
                col2.metric("Current Value", f"Rp {total_value:,.0f}")
                col3.metric("Total P/L", f"Rp {total_profit:,.0f}", delta=f"{total_profit_pct:.2f}%")
                col4.metric("Positions", len(st.session_state.portfolio))

            if st.button("🗑️ Clear All Positions", key="clear_port"):
                st.session_state.portfolio = []
                st.success("Portfolio cleared!")
                st.rerun()
        else:
            st.info("Your portfolio is empty. Add positions to track P/L!")

# ------- STOCK COMPARISON -------
elif menu == "🔬 Stock Comparison":
    st.markdown("## 🔬 Stock Comparison Tool")

    st.info("Compare up to 5 stocks side-by-side")

    # Select stocks to compare
    col1, col2 = st.columns([4, 1])
    with col1:
        selected_stock = st.selectbox("Select stock to add", [t.replace(".JK", "") for t in tickers], key="comp_select")
    with col2:
        st.write("")
        st.write("")
        if st.button("➕ Add", key="add_comparison"):
            if len(st.session_state.comparison_list) < 5:
                if selected_stock not in st.session_state.comparison_list:
                    st.session_state.comparison_list.append(selected_stock)
                    st.success(f"Added {selected_stock}!")
                else:
                    st.warning("Already in comparison!")
            else:
                st.error("Maximum 5 stocks!")

    if st.session_state.comparison_list:
        st.markdown(f"**Comparing {len(st.session_state.comparison_list)} stocks:** {', '.join(st.session_state.comparison_list)}")

        if st.button("🗑️ Clear All", key="clear_comp"):
            st.session_state.comparison_list = []
            st.rerun()

        # Fetch and compare
        comparison_data = []
        for ticker in st.session_state.comparison_list:
            ticker_full = f"{ticker}.JK"
            df = fetch_data(ticker_full, "3mo")
            if df is not None:
                result = process_ticker(ticker_full, "Speed", "3mo")
                if result:
                    comparison_data.append(result)

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df[['Ticker', 'Price', 'Score', 'Grade', 'Trend', 'Signal',
                                   'Entry', 'TP1', 'TP2', 'CL']], use_container_width=True)

            # Price comparison chart
            st.markdown("### 📊 Price Comparison (Normalized)")
            fig = go.Figure()
            for ticker in st.session_state.comparison_list:
                ticker_full = f"{ticker}.JK"
                df = fetch_data(ticker_full, "3mo")
                if df is not None:
                    # Normalize to 100
                    normalized = (df['Close'] / df['Close'].iloc[0]) * 100
                    fig.add_trace(go.Scatter(x=df.index, y=normalized, name=ticker, mode='lines'))

            fig.update_layout(
                height=400,
                template="plotly_dark",
                yaxis_title="Normalized Price (Base=100)",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

# ------- SECTOR ANALYSIS -------
elif menu == "🗺️ Sector Analysis":
    st.markdown("## 🗺️ Sector Analysis")

    if st.session_state.last_scan_df is not None and not st.session_state.last_scan_df.empty:
        df_results = st.session_state.last_scan_df

        st.markdown("### 📊 Sector Performance")
        sector_summary = analyze_sectors(df_results)
        st.dataframe(sector_summary, use_container_width=True)

        st.markdown("### 🗺️ Sector Heatmap")
        heatmap = create_sector_heatmap(df_results)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)

        # Best stocks by sector
        st.markdown("### 🏆 Top Stocks by Sector")
        for sector in sector_summary.index[:5]:  # Top 5 sectors
            sector_stocks = df_results[df_results['Ticker'].map(lambda x: SECTOR_MAP.get(x, "Other") == sector)]
            if not sector_stocks.empty:
                top_3 = sector_stocks.nlargest(3, 'Score')
                with st.expander(f"🏢 {sector} - Top 3 Stocks"):
                    st.dataframe(top_3[['Ticker', 'Price', 'Score', 'Grade', 'Signal', 'Entry', 'TP1']],
                               use_container_width=True)
    else:
        st.info("⚠️ Run a scan first to see sector analysis!")

# ------- RISK CALCULATOR -------
elif menu == "💰 Risk Calculator":
    st.markdown("## 💰 Risk Management Calculator")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Input Parameters")
        capital = st.number_input("Total Capital (Rp)", min_value=1000000, value=10000000, step=1000000)
        risk_pct = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        entry_price = st.number_input("Entry Price (Rp)", min_value=1, value=1000, step=10)
        stop_loss = st.number_input("Stop Loss (Rp)", min_value=1, value=950, step=10)
        take_profit = st.number_input("Take Profit (Rp)", min_value=1, value=1100, step=10)

    with col2:
        st.markdown("### 📈 Calculation Results")

        if stop_loss >= entry_price:
            st.error("⚠️ Stop Loss must be below Entry Price!")
        elif take_profit <= entry_price:
            st.error("⚠️ Take Profit must be above Entry Price!")
        else:
            metrics = calculate_risk_metrics(entry_price, stop_loss, take_profit, capital, risk_pct)

            if metrics:
                st.success(f"**Position Size:** {metrics['position_size']:,} shares")
                st.info(f"**Total Investment:** Rp {metrics['total_investment']:,.0f}")

                col_a, col_b = st.columns(2)
                col_a.metric("Potential Profit", f"Rp {metrics['potential_profit']:,.0f}",
                           delta=f"+{(metrics['potential_profit']/metrics['total_investment']*100):.1f}%")
                col_b.metric("Potential Loss", f"Rp {metrics['potential_loss']:,.0f}",
                           delta=f"-{(metrics['potential_loss']/metrics['total_investment']*100):.1f}%")

                st.metric("Risk/Reward Ratio", f"1 : {metrics['rr_ratio']:.2f}")

                if metrics['rr_ratio'] < 1.5:
                    st.warning("⚠️ R:R ratio kurang dari 1.5 - Trade kurang optimal!")
                elif metrics['rr_ratio'] < 2:
                    st.info("✅ R:R ratio acceptable (1.5-2)")
                else:
                    st.success(f"🎯 Excellent R:R ratio! ({metrics['rr_ratio']:.2f}:1)")

    st.markdown("---")
    st.markdown("### 💡 Risk Management Tips")
    st.markdown("""
    - **Never risk more than 2-3% of your capital** on a single trade
    - **Aim for R:R ratio of at least 1.5:1**, ideally 2:1 or higher
    - **Always use stop loss** - no exceptions!
    - **Position sizing is key** - don't over-leverage
    - **Diversify** - don't put all eggs in one basket
    """)

# ------- TRADE JOURNAL -------
elif menu == "📓 Trade Journal":
    st.markdown("## 📓 Trade Journal & Performance Tracking")

    tab1, tab2 = st.tabs(["📝 Add Trade", "📊 View Journal"])

    with tab1:
        st.markdown("### 📝 Log New Trade")

        col1, col2, col3 = st.columns(3)
        with col1:
            journal_ticker = st.selectbox("Ticker", [t.replace(".JK", "") for t in tickers], key="journal_ticker")
            entry = st.number_input("Entry Price", min_value=1, value=1000, step=10, key="journal_entry")
            strategy = st.selectbox("Strategy", ["BPJS", "BSJP", "Swing", "Value", "Bandar", "Speed"], key="journal_strat")
        with col2:
            exit_price = st.number_input("Exit Price", min_value=1, value=1050, step=10, key="journal_exit")
            qty = st.number_input("Quantity", min_value=1, value=100, step=100, key="journal_qty")
        with col3:
            trade_date = st.date_input("Trade Date", key="journal_date")

        if st.button("💾 Save Trade", key="save_trade"):
            profit = (exit_price - entry) * qty
            profit_pct = ((exit_price - entry) / entry * 100)

            st.session_state.trade_journal.append({
                'ticker': journal_ticker,
                'entry': entry,
                'exit': exit_price,
                'qty': qty,
                'profit': profit,
                'profit_pct': profit_pct,
                'strategy': strategy,
                'date': str(trade_date)
            })
            st.success(f"Trade logged! P/L: Rp {profit:,.0f} ({profit_pct:+.2f}%)")

    with tab2:
        st.markdown("### 📊 Trade History")

        if st.session_state.trade_journal:
            journal_data = []
            for trade in st.session_state.trade_journal:
                journal_data.append({
                    'Date': trade['date'],
                    'Ticker': trade['ticker'],
                    'Strategy': trade['strategy'],
                    'Entry': f"Rp {trade['entry']:,.0f}",
                    'Exit': f"Rp {trade['exit']:,.0f}",
                    'Qty': trade['qty'],
                    'P/L': f"Rp {trade['profit']:,.0f}",
                    'P/L %': f"{trade['profit_pct']:+.2f}%"
                })

            journal_df = pd.DataFrame(journal_data)
            st.dataframe(journal_df, use_container_width=True, height=400)

            # Statistics
            st.markdown("### 📈 Performance Statistics")
            total_trades = len(st.session_state.trade_journal)
            winning_trades = sum(1 for t in st.session_state.trade_journal if t['profit'] > 0)
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            total_profit = sum(t['profit'] for t in st.session_state.trade_journal)
            avg_profit = total_profit / total_trades if total_trades > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", total_trades)
            col2.metric("Win Rate", f"{win_rate:.1f}%")
            col3.metric("Total P/L", f"Rp {total_profit:,.0f}")
            col4.metric("Avg P/L", f"Rp {avg_profit:,.0f}")

            # P/L Chart
            if len(st.session_state.trade_journal) > 1:
                st.markdown("### 📊 Cumulative P/L")
                cumulative_pl = []
                running_total = 0
                for trade in st.session_state.trade_journal:
                    running_total += trade['profit']
                    cumulative_pl.append(running_total)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=cumulative_pl,
                    mode='lines+markers',
                    name='Cumulative P/L',
                    line=dict(color='#22c55e' if cumulative_pl[-1] > 0 else '#ef4444', width=3)
                ))
                fig.update_layout(height=300, template="plotly_dark", yaxis_title="P/L (Rp)")
                st.plotly_chart(fig, use_container_width=True)

            if st.button("🗑️ Clear Journal", key="clear_journal"):
                st.session_state.trade_journal = []
                st.success("Journal cleared!")
                st.rerun()
        else:
            st.info("No trades logged yet. Start logging your trades to track performance!")

# ------- MULTI-TIMEFRAME ANALYSIS -------
elif menu == "⏰ Multi-Timeframe":
    st.markdown("## ⏰ Multi-Timeframe Analysis")

    selected = st.selectbox("Select stock for MTF analysis", [t.replace(".JK", "") for t in tickers])

    if st.button("🔍 Analyze", type="primary"):
        ticker_full = f"{selected}.JK"

        with st.spinner(f"Analyzing {selected} across multiple timeframes..."):
            mtf_data = fetch_multi_timeframe(ticker_full)

        if mtf_data:
            st.markdown(f"### 📊 Multi-Timeframe Analysis: {selected}")

            # Create comparison table
            mtf_rows = []
            for tf, data in mtf_data.items():
                mtf_rows.append({
                    'Timeframe': tf,
                    'Trend': data['trend'],
                    'Price': f"Rp {data['price']:,.0f}",
                    'EMA20': f"Rp {data['ema20']:,.0f}",
                    'EMA50': f"Rp {data['ema50']:,.0f}",
                    'Change %': f"{data['change_pct']:+.2f}%"
                })

            mtf_df = pd.DataFrame(mtf_rows)
            st.dataframe(mtf_df, use_container_width=True)

            # Trend alignment check
            trends = [data['trend'] for data in mtf_data.values()]
            uptrends = sum(1 for t in trends if t == "Uptrend")

            st.markdown("### 🎯 Timeframe Alignment")
            if uptrends == 3:
                st.success("✅ All timeframes aligned UPTREND - Strong bullish signal!")
            elif uptrends == 2:
                st.info("🟡 2 out of 3 timeframes uptrend - Moderate bullish")
            elif uptrends == 1:
                st.warning("⚠️ Only 1 timeframe uptrend - Mixed signals")
            else:
                st.error("🔴 No uptrends - Bearish condition")

            st.markdown("### 💡 Trading Recommendation")
            if uptrends >= 2:
                st.success(f"""
                **GOOD SETUP for {selected}**
                - Multiple timeframes showing strength
                - Consider entry on pullback to support
                - Use higher timeframe trend as confirmation
                """)
            else:
                st.warning(f"""
                **WAIT for better setup on {selected}**
                - Timeframes not aligned
                - High risk of reversal
                - Better opportunities elsewhere
                """)
        else:
            st.error("Failed to fetch multi-timeframe data.")

st.markdown("---")
st.caption(
    "⚡ IDX Power Screener – EXTREME BUILD • Fokus trader cepat 1–3 hari • Edukasi saja, bukan ajakan beli/jual."
)