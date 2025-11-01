#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="IDX Power Screener", page_icon="🚀", layout="wide")

# ---------- UI THEME ----------
st.markdown("""
<style>
.big {font-size:2rem;font-weight:700;color:#1f77b4}
.subtitle {font-size:1.2rem;color:#666;margin-bottom:2rem}
.tag {display:inline-block;padding:.3rem .8rem;border-radius:.5rem;background:#eee;margin:.2rem;font-weight:600}
.buy {background:#d7ffd9;color:#0a5f0a}
.sell{background:#ffd7d7;color:#7f0000}
.warn{background:#fff2cc;color:#7f6000}
.bandar {background:#e3d7ff;color:#4a0a7f}
.value {background:#d7f5ff;color:#0a4f7f}
.metric-card {background:#f8f9fa;padding:1rem;border-radius:.5rem;border-left:4px solid #1f77b4}
</style>
""", unsafe_allow_html=True)

# ---------- GLOBAL CACHE ----------
@st.cache_resource
def get_session():
    """Persistent session for API calls"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': '*/*',
        'Connection': 'keep-alive'
    })
    return session

# ---------- DATA FETCHER ----------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_yahoo_api(ticker: str, period="6mo", interval="1d"):
    """Fetch data dari Yahoo Finance API"""
    try:
        end = int(datetime.now().timestamp())
        period_map = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
        days = period_map.get(period, 180)
        start = end - (days * 24 * 3600)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"period1": start, "period2": end, "interval": interval, "events": "history"}
        
        headers = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)'}
        response = requests.get(url, params=params, headers=headers, timeout=15, verify=False)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': quotes['open'],
            'High': quotes['high'],
            'Low': quotes['low'],
            'Close': quotes['close'],
            'Volume': quotes['volume']
        }, index=pd.to_datetime(timestamps, unit='s'))
        
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if len(df) < 10:
            return None
        
        # Technical Indicators
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = df["Close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain, index=df.index).rolling(14, min_periods=14).mean()
        avg_loss = pd.Series(loss, index=df.index).rolling(14, min_periods=14).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["SIGNAL"]
        
        # Bollinger Bands
        df["BB_MID"] = df["Close"].rolling(20).mean()
        df["BB_STD"] = df["Close"].rolling(20).std()
        df["BB_UPPER"] = df["BB_MID"] + (2 * df["BB_STD"])
        df["BB_LOWER"] = df["BB_MID"] - (2 * df["BB_STD"])
        
        # Stochastic
        low14 = df["Low"].rolling(14).min()
        high14 = df["High"].rolling(14).max()
        df["STOCH_K"] = 100 * (df["Close"] - low14) / (high14 - low14)
        df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()
        
        # ATR (Average True Range)
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["ATR"] = true_range.rolling(14).mean()
        
        return df
    except Exception as e:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_with_yfinance(ticker: str, period="6mo", interval="1d"):
    """Fallback yfinance"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, auto_adjust=False)
        
        if df is None or df.empty or len(df) < 10:
            return None
        
        df.columns = [col.capitalize() if isinstance(col, str) else col for col in df.columns]
        
        # Apply same indicators
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        
        delta = df["Close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain, index=df.index).rolling(14, min_periods=14).mean()
        avg_loss = pd.Series(loss, index=df.index).rolling(14, min_periods=14).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        df["RSI"] = 100 - (100 / (1 + rs))
        
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["SIGNAL"]
        
        df["BB_MID"] = df["Close"].rolling(20).mean()
        df["BB_STD"] = df["Close"].rolling(20).std()
        df["BB_UPPER"] = df["BB_MID"] + (2 * df["BB_STD"])
        df["BB_LOWER"] = df["BB_MID"] - (2 * df["BB_STD"])
        
        low14 = df["Low"].rolling(14).min()
        high14 = df["High"].rolling(14).max()
        df["STOCH_K"] = 100 * (df["Close"] - low14) / (high14 - low14)
        df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()
        
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["ATR"] = true_range.rolling(14).mean()
        
        return df
    except:
        return None

def fetch_data(ticker: str, period="6mo", interval="1d", use_api=True):
    """Smart fetch dengan fallback"""
    if use_api:
        df = fetch_yahoo_api(ticker, period, interval)
        if df is not None:
            return df
        time.sleep(0.5)
    return fetch_with_yfinance(ticker, period, interval)

# ---------- SCORING SYSTEMS ----------

def score_full_screener(df):
    """Full technical analysis scoring"""
    try:
        row = df.iloc[-1]
        score = 0
        details = {}
        
        # Trend (40 points)
        if row["Close"] > row["EMA9"] > row["EMA21"] > row["EMA50"]:
            score += 40
            details["Trend"] = "✅ Strong Uptrend (+40)"
        elif row["Close"] > row["EMA9"]:
            score += 20
            details["Trend"] = "⚠️ Short-term Up (+20)"
        else:
            details["Trend"] = "❌ Downtrend (+0)"
        
        # RSI (20 points)
        rsi = row["RSI"]
        if 40 <= rsi <= 70:
            score += 20
            details["RSI"] = f"✅ {rsi:.1f} Healthy (+20)"
        elif 30 < rsi < 40 or 70 < rsi < 80:
            score += 10
            details["RSI"] = f"⚠️ {rsi:.1f} Warning (+10)"
        else:
            details["RSI"] = f"❌ {rsi:.1f} Extreme (+0)"
        
        # MACD (20 points)
        if row["MACD"] > row["SIGNAL"] and row["MACD_HIST"] > 0:
            score += 20
            details["MACD"] = "✅ Bullish (+20)"
        elif row["MACD"] > row["SIGNAL"]:
            score += 10
            details["MACD"] = "⚠️ Weakening (+10)"
        else:
            details["MACD"] = "❌ Bearish (+0)"
        
        # Volume (10 points)
        vol_avg = df["Volume"].rolling(20).mean().iloc[-1]
        if row["Volume"] > vol_avg * 1.5:
            score += 10
            details["Volume"] = "✅ Surge (+10)"
        elif row["Volume"] > vol_avg:
            score += 5
            details["Volume"] = "⚠️ Above avg (+5)"
        else:
            details["Volume"] = "❌ Low (+0)"
        
        # Momentum (10 points)
        price_5d_ago = df["Close"].iloc[-5]
        momentum = (row["Close"] - price_5d_ago) / price_5d_ago * 100
        if 2 <= momentum <= 15:
            score += 10
            details["Momentum"] = f"✅ {momentum:.1f}% (+10)"
        elif momentum > 0:
            score += 5
            details["Momentum"] = f"⚠️ {momentum:.1f}% (+5)"
        else:
            details["Momentum"] = f"❌ {momentum:.1f}% (+0)"
        
        return score, details
    except:
        return 0, {}

def score_bpjs(df):
    """BPJS - Beli Pagi Jual Sore"""
    try:
        score = 0
        details = {}
        
        # Volatility (30 pts)
        atr = df["ATR"].iloc[-1]
        avg_price = df["Close"].iloc[-1]
        volatility_pct = (atr / avg_price) * 100
        
        if 2 < volatility_pct < 5:
            score += 30
            details["Volatility"] = f"✅ {volatility_pct:.2f}% Perfect (+30)"
        elif 1.5 < volatility_pct < 6:
            score += 15
            details["Volatility"] = f"⚠️ {volatility_pct:.2f}% OK (+15)"
        else:
            details["Volatility"] = f"❌ {volatility_pct:.2f}% Bad (+0)"
        
        # Volume Surge (25 pts)
        vol_avg = df["Volume"].rolling(20).mean().iloc[-1]
        recent_vol = df["Volume"].tail(3).mean()
        vol_ratio = recent_vol / vol_avg
        
        if vol_ratio > 1.5:
            score += 25
            details["Volume"] = f"✅ {vol_ratio:.2f}x (+25)"
        elif vol_ratio > 1.2:
            score += 12
            details["Volume"] = f"⚠️ {vol_ratio:.2f}x (+12)"
        else:
            details["Volume"] = f"❌ {vol_ratio:.2f}x (+0)"
        
        # RSI Entry Zone (25 pts)
        rsi = df["RSI"].iloc[-1]
        if 30 < rsi < 50:
            score += 25
            details["RSI"] = f"✅ {rsi:.1f} Entry zone (+25)"
        elif 25 < rsi < 55:
            score += 12
            details["RSI"] = f"⚠️ {rsi:.1f} OK (+12)"
        else:
            details["RSI"] = f"❌ {rsi:.1f} Bad (+0)"
        
        # Stochastic (20 pts)
        stoch = df["STOCH_K"].iloc[-1]
        if stoch < 30:
            score += 20
            details["Stochastic"] = f"✅ {stoch:.1f} Oversold (+20)"
        elif stoch < 50:
            score += 10
            details["Stochastic"] = f"⚠️ {stoch:.1f} OK (+10)"
        else:
            details["Stochastic"] = f"❌ {stoch:.1f} High (+0)"
        
        return score, details
    except:
        return 0, {}

def score_bsjp(df):
    """BSJP - Beli Sore Jual Pagi"""
    try:
        score = 0
        details = {}
        
        # BB Position (30 pts)
        bb_lower = df["BB_LOWER"].iloc[-1]
        bb_upper = df["BB_UPPER"].iloc[-1]
        bb_mid = df["BB_MID"].iloc[-1]
        price = df["Close"].iloc[-1]
        
        bb_position = (price - bb_lower) / (bb_upper - bb_lower) * 100
        
        if bb_position < 20:
            score += 30
            details["BB Position"] = f"✅ {bb_position:.1f}% Near support (+30)"
        elif bb_position < 40:
            score += 15
            details["BB Position"] = f"⚠️ {bb_position:.1f}% OK (+15)"
        else:
            details["BB Position"] = f"❌ {bb_position:.1f}% High (+0)"
        
        # Evening Dip (25 pts)
        prev_close = df["Close"].iloc[-2]
        gap = (price - prev_close) / prev_close * 100
        
        if -1.5 < gap < 0:
            score += 25
            details["Gap"] = f"✅ {gap:.2f}% Perfect dip (+25)"
        elif -2.5 < gap < 0.5:
            score += 12
            details["Gap"] = f"⚠️ {gap:.2f}% OK (+12)"
        else:
            details["Gap"] = f"❌ {gap:.2f}% Bad (+0)"
        
        # RSI (25 pts)
        rsi = df["RSI"].iloc[-1]
        if 35 < rsi < 55:
            score += 25
            details["RSI"] = f"✅ {rsi:.1f} Perfect (+25)"
        elif 30 < rsi < 60:
            score += 12
            details["RSI"] = f"⚠️ {rsi:.1f} OK (+12)"
        else:
            details["RSI"] = f"❌ {rsi:.1f} Bad (+0)"
        
        # Overnight History (20 pts)
        overnight_gains = []
        for i in range(-5, -1):
            try:
                prev = df["Close"].iloc[i]
                next_open = df["Open"].iloc[i+1]
                gain = (next_open - prev) / prev * 100
                overnight_gains.append(gain)
            except:
                pass
        
        if overnight_gains:
            avg = np.mean(overnight_gains)
            if avg > 0.3:
                score += 20
                details["History"] = f"✅ Avg +{avg:.2f}% (+20)"
            elif avg > 0:
                score += 10
                details["History"] = f"⚠️ Avg +{avg:.2f}% (+10)"
            else:
                details["History"] = f"❌ Avg {avg:.2f}% (+0)"
        
        return score, details
    except:
        return 0, {}

def score_bandar(df):
    """Bandar Tracking - Deteksi akumulasi"""
    try:
        score = 0
        details = {}
        phase = "UNKNOWN"
        
        # OBV Calculation
        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
                obv.append(obv[-1] + df["Volume"].iloc[i])
            elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
                obv.append(obv[-1] - df["Volume"].iloc[i])
            else:
                obv.append(obv[-1])
        
        df["OBV"] = obv
        
        # Volume Analysis
        vol_ma20 = df["Volume"].rolling(20).mean().iloc[-1]
        recent_vol = df["Volume"].tail(5).mean()
        vol_ratio = recent_vol / vol_ma20
        
        # Price Change
        price_change = (df["Close"].iloc[-1] - df["Close"].iloc[-20]) / df["Close"].iloc[-20] * 100
        
        # OBV Trend
        obv_ma = pd.Series(df["OBV"]).rolling(20).mean()
        obv_trend = (df["OBV"].iloc[-1] - obv_ma.iloc[-20]) / abs(obv_ma.iloc[-20])
        
        # Phase Detection
        if vol_ratio > 1.3 and price_change > -2 and obv_trend > 0.1:
            phase = "🟢 AKUMULASI"
            score = 85
            details["Phase"] = "✅ AKUMULASI (Bandar ngumpul)"
            details["Volume"] = f"✅ {vol_ratio:.2f}x avg"
            details["Price"] = f"✅ {price_change:+.2f}%"
            details["OBV"] = "✅ Trending UP"
            details["Action"] = "🚀 BUY - Ikut bandar!"
            
        elif vol_ratio > 1.3 and price_change < -3 and obv_trend < -0.1:
            phase = "🔴 DISTRIBUSI"
            score = 15
            details["Phase"] = "⚠️ DISTRIBUSI (Bandar buang)"
            details["Volume"] = f"⚠️ {vol_ratio:.2f}x avg"
            details["Price"] = f"❌ {price_change:+.2f}%"
            details["OBV"] = "❌ Trending DOWN"
            details["Action"] = "🛑 AVOID - Bandar lagi jual!"
            
        elif price_change > 5 and vol_ratio < 1.5:
            phase = "🚀 MARKUP"
            score = 90
            details["Phase"] = "✅ MARKUP (Harga naik)"
            details["Price"] = f"✅ {price_change:+.2f}%"
            details["Volume"] = f"Normal ({vol_ratio:.2f}x)"
            details["Action"] = "🎯 HOLD/BUY - Ride the wave!"
            
        elif price_change < -5:
            phase = "📉 MARKDOWN"
            score = 10
            details["Phase"] = "❌ MARKDOWN (Harga turun)"
            details["Price"] = f"❌ {price_change:+.2f}%"
            details["Action"] = "⏸️ WAIT - Tunggu bottom"
            
        else:
            phase = "⚪ SIDEWAYS"
            score = 50
            details["Phase"] = "⚠️ SIDEWAYS (Konsolidasi)"
            details["Volume"] = f"{vol_ratio:.2f}x avg"
            details["Price"] = f"{price_change:+.2f}%"
            details["Action"] = "⏸️ MONITOR - Tunggu breakout"
        
        return score, details, phase
    except:
        return 0, {}, "UNKNOWN"

def score_value_hunting(df, ticker):
    """Saham Murah Fundamental Bagus"""
    try:
        score = 0
        details = {}
        
        price = df["Close"].iloc[-1]
        
        # 52-Week Position (30 pts)
        high_52w = df["High"].tail(252).max() if len(df) > 252 else df["High"].max()
        low_52w = df["Low"].tail(252).min() if len(df) > 252 else df["Low"].min()
        position_52w = (price - low_52w) / (high_52w - low_52w) * 100
        
        if position_52w < 25:
            score += 30
            details["52W Position"] = f"✅ {position_52w:.1f}% SANGAT MURAH (+30)"
        elif position_52w < 40:
            score += 20
            details["52W Position"] = f"✅ {position_52w:.1f}% MURAH (+20)"
        elif position_52w < 50:
            score += 10
            details["52W Position"] = f"⚠️ {position_52w:.1f}% Fair (+10)"
        else:
            details["52W Position"] = f"❌ {position_52w:.1f}% Mahal (+0)"
        
        # RSI Reversal (25 pts)
        rsi = df["RSI"].iloc[-1]
        if 25 < rsi < 40:
            score += 25
            details["RSI"] = f"✅ {rsi:.1f} Recovery zone (+25)"
        elif 20 < rsi < 45:
            score += 15
            details["RSI"] = f"⚠️ {rsi:.1f} OK (+15)"
        else:
            details["RSI"] = f"❌ {rsi:.1f} Bad (+0)"
        
        # Volume Interest (20 pts)
        vol_ma20 = df["Volume"].rolling(20).mean().iloc[-1]
        recent_vol = df["Volume"].tail(5).mean()
        vol_ratio = recent_vol / vol_ma20
        
        if vol_ratio > 1.5:
            score += 20
            details["Volume"] = f"✅ {vol_ratio:.2f}x Interest muncul (+20)"
        elif vol_ratio > 1.2:
            score += 10
            details["Volume"] = f"⚠️ {vol_ratio:.2f}x OK (+10)"
        else:
            details["Volume"] = f"❌ {vol_ratio:.2f}x Low (+0)"
        
        # Price Trend (15 pts)
        sma20 = df["SMA20"].iloc[-1]
        if price > sma20:
            score += 15
            details["Trend"] = "✅ Above SMA20 (+15)"
        elif price > sma20 * 0.98:
            score += 7
            details["Trend"] = "⚠️ Near SMA20 (+7)"
        else:
            details["Trend"] = "❌ Below SMA20 (+0)"
        
        # Stability (10 pts)
        volatility = df["Close"].pct_change().std() * 100
        if volatility < 2.5:
            score += 10
            details["Volatility"] = f"✅ {volatility:.2f}% Stable (+10)"
        elif volatility < 4:
            score += 5
            details["Volatility"] = f"⚠️ {volatility:.2f}% OK (+5)"
        else:
            details["Volatility"] = f"❌ {volatility:.2f}% High (+0)"
        
        # Potential Gain
        target = low_52w + (high_52w - low_52w) * 0.6
        potential = (target - price) / price * 100
        details["Potential"] = f"🎯 Target +{potential:.1f}% (ke 60% range)"
        
        return score, details
    except:
        return 0, {}

# ---------- BATCH SCANNER ----------
def batch_scan(tickers, strategy, period, limit, use_api):
    """Batch scan dengan strategy"""
    results = []
    if limit:
        tickers = tickers[:limit]
    
    progress = st.progress(0)
    status = st.empty()
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / total)
        status.text(f"📊 {i+1}/{total}: {ticker}")
        
        try:
            df = fetch_data(ticker, period, use_api=use_api)
            if df is None or len(df) < 50:
                continue
            
            price = float(df["Close"].iloc[-1])
            
            if strategy == "Full Screener":
                score, details = score_full_screener(df)
                if score >= 70:
                    signal = "STRONG BUY"
                elif score >= 60:
                    signal = "BUY"
                elif score >= 40:
                    signal = "NEUTRAL"
                else:
                    signal = "SELL"
                phase = None
                
            elif strategy == "BPJS":
                score, details = score_bpjs(df)
                signal = "BUY PAGI" if score >= 70 else "WAIT"
                phase = None
                
            elif strategy == "BSJP":
                score, details = score_bsjp(df)
                signal = "BUY SORE" if score >= 70 else "WAIT"
                phase = None
                
            elif strategy == "Bandar":
                score, details, phase = score_bandar(df)
                signal = phase
                
            elif strategy == "Value":
                score, details = score_value_hunting(df, ticker)
                signal = "VALUE BUY" if score >= 70 else "MONITOR"
                phase = None
            
            results.append({
                "Ticker": ticker,
                "Price": price,
                "Score": score,
                "Signal": signal,
                "Phase": phase,
                "Details": details
            })
            
        except:
            pass
        
        time.sleep(0.4)
    
    progress.empty()
    status.empty()
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results).sort_values("Score", ascending=False)

# ---------- HELPER ----------
def load_tickers(path="idx_stocks.json"):
    """Load ticker list"""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        tickers = data.get("tickers", [])
        return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except:
        # Default fallback
        return ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]

# ---------- UI ----------
st.markdown('<div class="big">🚀 IDX Power Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Professional Stock Screening System</div>', unsafe_allow_html=True)

tickers = load_tickers()
st.info(f"📊 **{len(tickers)} tickers** loaded from IDX")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    use_api = st.checkbox("🔧 Direct API Mode", value=True, 
                          help="Recommended for mobile")
    
    menu = st.radio("📋 Menu", [
        "1️⃣ Full Screener",
        "2️⃣ Single Stock Analysis",
        "3️⃣ BPJS (Day Trading)",
        "4️⃣ BSJP (Overnight)",
        "5️⃣ Bandar Tracking",
        "6️⃣ Value Hunting"
    ])
    
    st.markdown("---")
    
    if "​​​​​​​​​​​​​​​​
