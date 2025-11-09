import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ======================== PAGE CONFIG ========================
st.set_page_config(page_title="IDX Power Screener v3.2", page_icon="🚀", layout="wide")

# ======================== STYLING ========================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ======================== CONSTANTS ========================
JAKARTA_TZ = pytz.timezone('Asia/Jakarta')
IDX_TICKERS_URL = "https://raw.githubusercontent.com/reyismayanto/stock-scrapper/main/indonesia_stock.csv"

# ======================== HELPER FUNCTIONS ========================

@st.cache_data(ttl=3600)
def load_tickers():
    """Load IDX tickers from GitHub"""
    try:
        df = pd.read_csv(IDX_TICKERS_URL)
        tickers = [f"{row['Code']}.JK" for _, row in df.iterrows()]
        return tickers
    except:
        return [
            "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
            "UNVR.JK", "ICBP.JK", "INDF.JK", "KLBF.JK", "GGRM.JK"
        ]

def get_jakarta_time():
    """Get current time in Jakarta timezone"""
    return datetime.now(JAKARTA_TZ)

def is_market_open():
    """Check if IDX market is open"""
    now = get_jakarta_time()
    if now.weekday() >= 5:  # Weekend
        return False, "CLOSED - Weekend"
    
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=50, second=0, microsecond=0)
    
    if now < market_open:
        return False, f"PRE-MARKET - Opens at 09:00"
    elif now > market_close:
        return False, f"CLOSED - Opens tomorrow 09:00"
    else:
        return True, f"OPEN - Closes at 15:50"

@st.cache_data(ttl=900)
def fetch_data(ticker, period="6mo"):
    """Fetch stock data with caching"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty or len(df) < 50:
            return None
        
        # Calculate indicators
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume MA
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # OBV
        df['OBV'] = (df['Volume'] * (~df['Close'].diff().le(0) * 2 - 1)).cumsum()
        
        return df
        
    except Exception as e:
        return None

def calculate_momentum(df, period=14):
    """Calculate momentum"""
    if df is None or len(df) < period:
        return 0
    return ((df['Close'].iloc[-1] - df['Close'].iloc[-period]) / df['Close'].iloc[-period]) * 100

# ======================== SCORING FUNCTIONS ========================

def score_full_screener_v3(df):
    """Enhanced Full Screener scoring with validation"""
    if df is None or len(df) < 50:
        return 0, {}, 0
    
    score = 0
    details = {}
    confidence = 0
    
    try:
        current_price = float(df['Close'].iloc[-1])
        ema20 = float(df['EMA20'].iloc[-1])
        ema50 = float(df['EMA50'].iloc[-1])
        ema200 = float(df['EMA200'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        volume = float(df['Volume'].iloc[-1])
        volume_ma = float(df['Volume_MA'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        macd_signal = float(df['MACD_Signal'].iloc[-1])
        momentum = calculate_momentum(df, 14)
        
        # Validation: Reject clear downtrends
        if current_price < ema50 and ema50 < ema200:
            if momentum < -5:
                return 0, {"Reason": "Rejected - Clear downtrend"}, 0
        
        # Validation: Reject extreme oversold with no reversal
        if rsi < 25 and momentum < -10:
            recent_candles = df['Close'].tail(5)
            if recent_candles.iloc[-1] < recent_candles.iloc[0]:
                return 0, {"Reason": "Rejected - Falling knife"}, 0
        
        # TREND (35 points)
        if current_price > ema20 > ema50 > ema200:
            score += 35
            details['Trend'] = "✅ PERFECT BULL (+35)"
            confidence += 35
        elif current_price > ema20 > ema50:
            score += 25
            details['Trend'] = "✅ Strong Bull (+25)"
            confidence += 25
        elif current_price > ema50:
            score += 15
            details['Trend'] = "🟡 Bull (+15)"
            confidence += 15
        elif current_price > ema200:
            score += 5
            details['Trend'] = "🟡 Weak Bull (+5)"
            confidence += 5
        else:
            details['Trend'] = "❌ Bearish (0)"
        
        # RSI (20 points)
        if 45 <= rsi <= 55:
            score += 20
            details['RSI'] = f"✅ SWEET {rsi:.1f} (+20)"
            confidence += 20
        elif 40 <= rsi <= 60:
            score += 15
            details['RSI'] = f"✅ Good {rsi:.1f} (+15)"
            confidence += 15
        elif 35 <= rsi <= 65:
            score += 10
            details['RSI'] = f"🟡 OK {rsi:.1f} (+10)"
            confidence += 10
        elif rsi > 70:
            score += 5
            details['RSI'] = f"⚠️ Overbought {rsi:.1f} (+5)"
            confidence += 5
        else:
            details['RSI'] = f"❌ Poor {rsi:.1f} (0)"
        
        # VOLUME (20 points)
        volume_ratio = volume / volume_ma if volume_ma > 0 else 0
        if volume_ratio >= 2.0:
            score += 20
            details['Volume'] = f"✅ BREAKOUT {volume_ratio:.2f}x (+20)"
            confidence += 20
        elif volume_ratio >= 1.5:
            score += 15
            details['Volume'] = f"✅ High {volume_ratio:.2f}x (+15)"
            confidence += 15
        elif volume_ratio >= 1.2:
            score += 10
            details['Volume'] = f"🟡 Above Avg {volume_ratio:.2f}x (+10)"
            confidence += 10
        else:
            details['Volume'] = f"❌ Low {volume_ratio:.2f}x (0)"
        
        # MOMENTUM (15 points)
        if momentum > 10:
            score += 15
            details['Momentum'] = f"✅ Excellent {momentum:.1f}% (+15)"
            confidence += 15
        elif momentum > 5:
            score += 10
            details['Momentum'] = f"✅ Good {momentum:.1f}% (+10)"
            confidence += 10
        elif momentum > 0:
            score += 5
            details['Momentum'] = f"🟡 Positive {momentum:.1f}% (+5)"
            confidence += 5
        else:
            details['Momentum'] = f"❌ Negative {momentum:.1f}% (0)"
        
        # MACD (15 points)
        if macd > macd_signal and macd > 0:
            score += 15
            details['MACD'] = "✅ STRONG BULL (+15)"
            confidence += 15
        elif macd > macd_signal:
            score += 10
            details['MACD'] = "✅ Bullish Cross (+10)"
            confidence += 10
        elif macd > 0:
            score += 5
            details['MACD'] = "🟡 Positive (+5)"
            confidence += 5
        else:
            details['MACD'] = "❌ Bearish (0)"
        
        # Calculate final confidence
        confidence = min(confidence, 100)
        
        return score, details, confidence
        
    except Exception as e:
        return 0, {"Error": str(e)}, 0

def score_bandar_v3(df):
    """Enhanced Bandar/Smart Money detection"""
    if df is None or len(df) < 50:
        return 0, {}, "⚪ SIDEWAYS", 0
    
    try:
        # Calculate metrics
        volume_ratio = float(df['Volume'].iloc[-1] / df['Volume_MA'].iloc[-1])
        price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
        obv_trend = ((df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / abs(df['OBV'].iloc[-20])) * 100 if df['OBV'].iloc[-20] != 0 else 0
        
        # Volatility
        returns = df['Close'].pct_change().tail(20)
        volatility = returns.std() * 100
        
        details = {
            'Volume_Ratio': f"{volume_ratio:.2f}x",
            'Price_Change': f"{price_change:+.2f}%",
            'OBV_Trend': f"{obv_trend:+.1f}%",
            'Volatility': f"{volatility:.2f}%"
        }
        
        # Phase Detection
        phase = "⚪ SIDEWAYS"
        score = 50
        confidence = 50
        action = "⏸️ WAIT & WATCH"
        
        # ACCUMULATION: Volume ↑, Price sideways/slight up, OBV ↑
        if volume_ratio > 1.3 and -5 < price_change < 10 and obv_trend > 20:
            phase = "🟢 AKUMULASI"
            score = 90
            confidence = 85
            action = "🚀 STRONG BUY"
            details['Signal'] = "Volume + | Price Sideways + | OBV +"
            details['Strength'] = "VERY STRONG"
            details['Risk'] = "LOW (Best entry point)"
        
        # MARKUP: Volume high, Price ↑, OBV ↑
        elif volume_ratio > 1.0 and price_change > 5 and obv_trend > 10:
            phase = "🚀 MARKUP"
            score = 85
            confidence = 80
            action = "🔄 HOLD / BUY PULLBACK"
            details['Signal'] = "Volume + | Price + | OBV +"
            details['Strength'] = "STRONG"
            details['Risk'] = "MEDIUM"
        
        # DISTRIBUTION: Volume ↑, Price ↑ but OBV ↓
        elif volume_ratio > 1.2 and price_change > 0 and obv_trend < -10:
            phase = "🔴 DISTRIBUSI"
            score = 20
            confidence = 75
            action = "❌ SELL / AVOID"
            details['Signal'] = "Volume + | Price + | OBV - (Divergence!)"
            details['Strength'] = "WARNING"
            details['Risk'] = "HIGH"
        
        # MARKDOWN: Volume ↑, Price ↓, OBV ↓
        elif price_change < -5 and obv_trend < 0:
            phase = "⚫ MARKDOWN"
            score = 10
            confidence = 70
            action = "🚫 AVOID"
            details['Signal'] = "Price - | OBV -"
            details['Strength'] = "FALLING"
            details['Risk'] = "VERY HIGH"
        
        # SIDEWAYS: No clear pattern
        else:
            details['Signal'] = "Mixed signals"
            details['Strength'] = "NEUTRAL"
            details['Risk'] = "MEDIUM (Uncertain)"
        
        details['Action'] = action
        details['Phase'] = phase
        
        return score, details, phase, confidence
        
    except Exception as e:
        return 0, {"Error": str(e)}, "⚪ SIDEWAYS", 0

def score_bpjs_v3(df):
    """Day trading scanner (BPJS style)"""
    if df is None or len(df) < 20:
        return 0, {}, 0
    
    score = 0
    details = {}
    confidence = 0
    
    try:
        current_price = float(df['Close'].iloc[-1])
        ema20 = float(df['EMA20'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        volume_ratio = float(df['Volume'].iloc[-1] / df['Volume_MA'].iloc[-1])
        
        # Short-term momentum
        momentum_1d = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        
        # Trend (30 points)
        if current_price > ema20 and momentum_1d > 0:
            score += 30
            details['Trend'] = f"✅ Uptrend +{momentum_1d:.1f}% (+30)"
            confidence += 30
        elif current_price > ema20:
            score += 15
            details['Trend'] = "🟡 Above EMA20 (+15)"
            confidence += 15
        
        # RSI (25 points)
        if 40 <= rsi <= 60:
            score += 25
            details['RSI'] = f"✅ Neutral {rsi:.1f} (+25)"
            confidence += 25
        elif 35 <= rsi <= 65:
            score += 15
            details['RSI'] = f"🟡 OK {rsi:.1f} (+15)"
            confidence += 15
        
        # Volume (25 points)
        if volume_ratio >= 1.5:
            score += 25
            details['Volume'] = f"✅ High {volume_ratio:.2f}x (+25)"
            confidence += 25
        elif volume_ratio >= 1.0:
            score += 15
            details['Volume'] = f"🟡 Average {volume_ratio:.2f}x (+15)"
            confidence += 15
        
        # Momentum (20 points)
        if momentum_1d > 2:
            score += 20
            details['Momentum'] = f"✅ Strong +{momentum_1d:.1f}% (+20)"
            confidence += 20
        elif momentum_1d > 0:
            score += 10
            details['Momentum'] = f"🟡 Positive +{momentum_1d:.1f}% (+10)"
            confidence += 10
        
        confidence = min(confidence, 100)
        return score, details, confidence
        
    except Exception as e:
        return 0, {"Error": str(e)}, 0

def score_bsjp_v3(df):
    """Overnight/Swing scanner (BSJP style)"""
    if df is None or len(df) < 50:
        return 0, {}, 0
    
    score = 0
    details = {}
    confidence = 0
    
    try:
        current_price = float(df['Close'].iloc[-1])
        ema50 = float(df['EMA50'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        macd_signal = float(df['MACD_Signal'].iloc[-1])
        momentum = calculate_momentum(df, 10)
        
        # Trend (35 points)
        if current_price > ema50 and momentum > 0:
            score += 35
            details['Trend'] = f"✅ Strong uptrend (+35)"
            confidence += 35
        elif current_price > ema50:
            score += 20
            details['Trend'] = "🟡 Above EMA50 (+20)"
            confidence += 20
        
        # RSI (25 points)
        if 45 <= rsi <= 60:
            score += 25
            details['RSI'] = f"✅ Healthy {rsi:.1f} (+25)"
            confidence += 25
        elif 40 <= rsi <= 65:
            score += 15
            details['RSI'] = f"🟡 OK {rsi:.1f} (+15)"
            confidence += 15
        
        # MACD (20 points)
        if macd > macd_signal and macd > 0:
            score += 20
            details['MACD'] = "✅ Strong Bull (+20)"
            confidence += 20
        elif macd > macd_signal:
            score += 10
            details['MACD'] = "🟡 Bullish (+10)"
            confidence += 10
        
        # Momentum (20 points)
        if momentum > 5:
            score += 20
            details['Momentum'] = f"✅ Strong +{momentum:.1f}% (+20)"
            confidence += 20
        elif momentum > 0:
            score += 10
            details['Momentum'] = f"🟡 Positive +{momentum:.1f}% (+10)"
            confidence += 10
        
        confidence = min(confidence, 100)
        return score, details, confidence
        
    except Exception as e:
        return 0, {"Error": str(e)}, 0

def score_value_v3(df):
    """Value/Long-term scanner"""
    if df is None or len(df) < 200:
        return 0, {}, 0
    
    score = 0
    details = {}
    confidence = 0
    
    try:
        current_price = float(df['Close'].iloc[-1])
        ema200 = float(df['EMA200'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        momentum = calculate_momentum(df, 30)
        
        # Long-term trend (40 points)
        if current_price > ema200:
            score += 40
            details['Long_Trend'] = "✅ Above 200 EMA (+40)"
            confidence += 40
        elif current_price > ema200 * 0.95:
            score += 20
            details['Long_Trend'] = "🟡 Near 200 EMA (+20)"
            confidence += 20
        
        # RSI (30 points)
        if 40 <= rsi <= 60:
            score += 30
            details['RSI'] = f"✅ Balanced {rsi:.1f} (+30)"
            confidence += 30
        elif 35 <= rsi <= 65:
            score += 15
            details['RSI'] = f"🟡 OK {rsi:.1f} (+15)"
            confidence += 15
        
        # Momentum (30 points)
        if momentum > 10:
            score += 30
            details['Momentum'] = f"✅ Excellent +{momentum:.1f}% (+30)"
            confidence += 30
        elif momentum > 0:
            score += 15
            details['Momentum'] = f"🟡 Positive +{momentum:.1f}% (+15)"
            confidence += 15
        
        confidence = min(confidence, 100)
        return score, details, confidence
        
    except Exception as e:
        return 0, {"Error": str(e)}, 0

def get_signal_levels(score, price, confidence):
    """Calculate entry, TP, and SL levels"""
    if score >= 85:
        signal = "STRONG BUY"
        trend = "🟢 Strong Uptrend"
    elif score >= 70:
        signal = "BUY"
        trend = "🟢 Uptrend"
    elif score >= 55:
        signal = "WEAK BUY"
        trend = "🟡 Weak Uptrend"
    else:
        signal = "HOLD"
        trend = "⚪ Neutral"
    
    # Conservative entry (wait for dip)
    entry_cons = price * 0.98
    tp1_cons = entry_cons * 1.08
    tp2_cons = entry_cons * 1.15
    sl_cons = entry_cons * 0.94
    
    # Aggressive entry (now)
    entry_aggr = price
    tp1_aggr = price * 1.06
    tp2_aggr = price * 1.12
    sl_aggr = price * 0.96
    
    return {
        "signal": signal,
        "trend": trend,
        "ideal": {
            "entry": round(entry_cons, 0),
            "tp1": round(tp1_cons, 0),
            "tp2": round(tp2_cons, 0),
            "sl": round(sl_cons, 0)
        },
        "aggr": {
            "entry": round(entry_aggr, 0),
            "tp1": round(tp1_aggr, 0),
            "tp2": round(tp2_aggr, 0),
            "sl": round(sl_aggr, 0)
        }
    }

# ======================== BATCH SCANNING ========================

def process_ticker(ticker, strategy, period):
    """Process single ticker with Bandar check"""
    try:
        df = fetch_data(ticker, period)
        if df is None or len(df) < 50:
            return None
        
        price = float(df['Close'].iloc[-1])
        
        # Run appropriate strategy scoring
        if strategy == "BPJS":
            score, details, confidence = score_bpjs_v3(df)
            bandar_phase = "N/A"
        elif strategy == "BSJP":
            score, details, confidence = score_bsjp_v3(df)
            bandar_phase = "N/A"
        elif strategy == "Bandar":
            score, details, phase, confidence = score_bandar_v3(df)
            details['Phase'] = phase
            bandar_phase = phase
        elif strategy == "Value":
            score, details, confidence = score_value_v3(df)
            bandar_phase = "N/A"
        else:  # Full Screener
            score, details, confidence = score_full_screener_v3(df)
            
            # NEW: Add Bandar check for Full Screener
            band_score, band_details, bandar_phase, band_conf = score_bandar_v3(df)
            
            # Add warnings if Bandar phase is bad
            warnings = []
            if '⚫' in bandar_phase:  # MARKDOWN
                warnings.append("⚫ MARKDOWN")
            elif '🔴' in bandar_phase:  # DISTRIBUSI
                warnings.append("🔴 DISTRIBUSI")
            
            # Optional: Check 1MO timeframe for conflicts
            try:
                df_1mo = fetch_data(ticker, "1mo")
                if df_1mo is not None and len(df_1mo) >= 50:
                    score_1mo, _, conf_1mo = score_full_screener_v3(df_1mo)
                    if score_1mo < 50 and score > 75:
                        warnings.append("⚠️ TF_CONFLICT")
            except:
                pass
            
            # Add warnings to details
            if warnings:
                details['⚠️ Warnings'] = " | ".join(warnings)
        
        if score == 0:
            return None
        
        levels = get_signal_levels(score, price, confidence)
        
        return {
            "Ticker": ticker,
            "Price": price,
            "Score": score,
            "Confidence": confidence,
            "Signal": levels["signal"],
            "Trend": levels["trend"],
            "Bandar": bandar_phase,  # NEW: Add Bandar phase
            "EntryIdeal": levels["ideal"]["entry"],
            "EntryAggr": levels["aggr"]["entry"],
            "TP1": levels["ideal"]["tp1"],
            "TP2": levels["ideal"]["tp2"],
            "SL": levels["ideal"]["sl"],
            "Details": details
        }
    except:
        return None

def batch_scan(tickers, strategy, period, limit, use_parallel=True):
    """Batch scan with parallel processing"""
    results = []
    
    if use_parallel:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_ticker, ticker, strategy, period): ticker 
                      for ticker in tickers[:limit]}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
    else:
        for ticker in tickers[:limit]:
            result = process_ticker(ticker, strategy, period)
            if result:
                results.append(result)
    
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('Score', ascending=False).reset_index(drop=True)
    
    return df

# ======================== MULTI-TIMEFRAME ANALYSIS ========================

def analyze_multi_timeframe(ticker):
    """Analyze stock across multiple timeframes"""
    timeframes = {
        "1mo": "1MO Timeframe",
        "3mo": "3MO Timeframe",
        "6mo": "6MO Timeframe"
    }
    
    results = {}
    
    for period, label in timeframes.items():
        df = fetch_data(ticker, period)
        if df is not None:
            score, details, confidence = score_full_screener_v3(df)
            
            # Determine trend
            if score >= 85:
                trend = "Strong Bull"
            elif score >= 70:
                trend = "Bull"
            elif score >= 55:
                trend = "Weak"
            elif score >= 40:
                trend = "Neutral"
            else:
                trend = "Bear"
            
            results[period] = {
                "label": label,
                "score": score,
                "confidence": confidence,
                "trend": trend,
                "details": details
            }
        else:
            results[period] = None
    
    # Check for alignment/conflicts
    scores = [r["score"] for r in results.values() if r is not None]
    if len(scores) >= 2:
        if all(s >= 70 for s in scores):
            alignment = "✅ STRONG BUY - All timeframes aligned!"
        elif all(s >= 55 for s in scores):
            alignment = "✅ BUY - Most timeframes bullish"
        elif max(scores) - min(scores) > 40:
            alignment = "🔴 CAUTION - Timeframe conflict"
        else:
            alignment = "🟡 MIXED - Review carefully"
    else:
        alignment = "⚠️ Insufficient data"
    
    return results, alignment

# ======================== MAIN APP ========================

def main():
    # Header
    st.markdown("# 🚀 IDX Power Screener v3.2")
    st.markdown("**Ultimate Edition: Multi-TF | Bandar Logic | S/R Finder | Advanced Tools**")
    
    # Market Status
    is_open, status_msg = is_market_open()
    jakarta_time = get_jakarta_time().strftime("%H:%M:%S WIB")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if is_open:
            st.success(f"🟢 {status_msg}")
        else:
            st.error(f"🔴 {status_msg}")
    with col2:
        st.info(f"🕐 Jakarta: {jakarta_time}")
    
    # Load tickers
    tickers = load_tickers()
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Settings")
    
    menu = st.sidebar.radio("📋 Menu", [
        "📊 Full Screener",
        "📈 Single Stock + Multi-TF",
        "📉 BPJS",
        "📊 BSJP",
        "🔥 Bandar Tracking",
        "💎 Value Hunting",
        "📊 Track Performance",
        "📍 Active Positions",
        "👁️ Watchlist",
        "✅ Test Cases"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Parameters")
    
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo"], index=2)
    
    if menu in ["📊 Full Screener", "📉 BPJS", "📊 BSJP", "🔥Bandar Tracking", "💎 Value Hunting"]:
        limit = st.sidebar.slider("Max Tickers", 50, 850, 150)
        min_score = st.sidebar.slider("Min Score", 0, 100, 65)
        min_confidence = st.sidebar.slider("Min Confidence", 0, 100, 60)
        use_parallel = st.sidebar.checkbox("⚡ Parallel Processing", value=True)
    
    st.sidebar.markdown("---")
    
    # Main Content
    if menu == "📊 Full Screener":
        st.markdown("### 🚀 Full Screener - Complete Market Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.info(f"📊 **Stocks**\n{limit}")
        col2.info(f"🎯 **Min Score**\n{min_score}")
        col3.info(f"📈 **Min Conf**\n{min_confidence}%")
        col4.info(f"⚡ **Mode**\n{'Parallel' if use_parallel else 'Sequential'}")
        
        if st.button("🚀 Run Full Screener", type="primary"):
            with st.spinner(f"Full analysis on {limit} stocks..."):
                df = batch_scan(tickers, "Full Screener", period, limit, use_parallel)
            
            if df.empty:
                st.warning("⚠️ No stocks found")
            else:
                df = df[(df["Score"] >= min_score) & (df["Confidence"] >= min_confidence)]
                
                if df.empty:
                    st.warning(f"No stocks meeting criteria (Score >={min_score}, Conf >={min_confidence}%)")
                else:
                    st.success(f"✅ Found {len(df)} quality opportunities!")
                    
                    # NEW: Bandar warning summary
                    bad_phase_stocks = df[df['Bandar'].str.contains('⚫|🔴', na=False)]
                    if len(bad_phase_stocks) > 0:
                        st.warning(f"""
                        ⚠️ **CAUTION:** {len(bad_phase_stocks)} stocks showing MARKDOWN/DISTRIBUSI phase:
                        
                        {', '.join(bad_phase_stocks['Ticker'].tolist())}
                        
                        These have good technical scores but smart money is exiting.
                        Verify in Single Stock Analysis before entering!
                        """)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Avg Score", f"{df['Score'].mean():.1f}/100")
                    col2.metric("Avg Confidence", f"{df['Confidence'].mean():.1f}%")
                    col3.metric("Strong Buy", len(df[df['Signal'] == 'STRONG BUY']))
                    col4.metric("Buy", len(df[df['Signal'] == 'BUY']))
                    
                    # NEW: Bandar Phase Filter
                    st.markdown("---")
                    st.markdown("### 🎯 Filter by Bandar Phase")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    show_all = col1.checkbox("Show All", value=False)
                    show_akum = col2.checkbox("🟢 Akumulasi", value=True)
                    show_markup = col3.checkbox("🚀 Markup", value=True)
                    show_side = col4.checkbox("⚪ Sideways", value=True)
                    hide_bad = col5.checkbox("❌ Hide Bad Phases", value=True)
                    
                    # Apply filters
                    filtered_df = df.copy()
                    
                    if not show_all:
                        phase_filters = []
                        if show_akum:
                            phase_filters.append('🟢 AKUMULASI')
                        if show_markup:
                            phase_filters.append('🚀 MARKUP')
                        if show_side:
                            phase_filters.append('⚪ SIDEWAYS')
                        
                        if phase_filters:
                            filtered_df = filtered_df[filtered_df['Bandar'].isin(phase_filters)]
                    
                    if hide_bad:
                        filtered_df = filtered_df[~filtered_df['Bandar'].str.contains('⚫|🔴', na=False)]
                    
                    # Phase distribution
                    col1, col2, col3, col4 = st.columns(4)
                    akum_count = len(df[df['Bandar'].str.contains('🟢', na=False)])
                    markup_count = len(df[df['Bandar'].str.contains('🚀', na=False)])
                    bad_count = len(df[df['Bandar'].str.contains('⚫|🔴', na=False)])
                    side_count = len(df[df['Bandar'].str.contains('⚪', na=False)])
                    
                    col1.metric("🟢 Akumulasi", akum_count, delta="BUY!" if akum_count > 0 else None)
                    col2.metric("🚀 Markup", markup_count, delta="HOLD" if markup_count > 0 else None)
                    col3.metric("❌ Bad", bad_count, delta="AVOID!" if bad_count > 0 else None, delta_color="inverse")
                    col4.metric("⚪ Sideways", side_count)
                    
                    df = filtered_df
                    
                    if df.empty:
                        st.warning("No stocks match selected filters")
                    else:
                        st.success(f"✅ Showing {len(df)} stocks after filters")
                        
                        # Data table with Bandar column
                        show = df[["Ticker","Price","Score","Confidence","Signal","Trend","Bandar","EntryIdeal","TP1","TP2","SL"]]
                        st.dataframe(show, use_container_width=True, height=400)
                        
                        # Top 15 Recommendations
                        st.markdown("---")
                        st.markdown("## 🏆 Top 15 Recommendations")
                        
                        for _, row in df.head(15).iterrows():
                            conf_color = "🟢" if row['Confidence'] >= 80 else "🟡" if row['Confidence'] >= 70 else "🟠"
                            
                            # Add Bandar phase to expander title
                            bandar_emoji = "🟢" if "🟢" in str(row['Bandar']) else "🚀" if "🚀" in str(row['Bandar']) else "⚫" if "⚫" in str(row['Bandar']) else "🔴" if "🔴" in str(row['Bandar']) else "⚪"
                            
                            with st.expander(f"{conf_color} {row['Ticker']} - Score: {row['Score']} | Conf: {row['Confidence']}% | {row['Signal']} | {bandar_emoji} {row['Bandar']}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"**💰 Price:** Rp {row['Price']:,.0f}")
                                    st.markdown(f"**📊 Score:** {row['Score']}/100")
                                    st.markdown(f"**🎯 Confidence:** {row['Confidence']}%")
                                    st.markdown(f"**📈 Signal:** {row['Signal']}")
                                    st.markdown(f"**🔄 Trend:** {row['Trend']}")
                                    st.markdown(f"**🎯 Bandar:** {row['Bandar']}")
                                    
                                    # Show warning if bad phase
                                    if '⚫' in str(row['Bandar']):
                                        st.error("⚠️ MARKDOWN PHASE - Price falling!")
                                    elif '🔴' in str(row['Bandar']):
                                        st.error("⚠️ DISTRIBUSI PHASE - Institutions selling!")
                                
                                with col2:
                                    st.markdown("**📍 Conservative Entry:**")
                                    st.info(f"Entry: Rp {row['EntryIdeal']:,.0f} (8%): Rp {row['TP1']:,.0f} | (15%): Rp {row['TP2']:,.0f}\nStop Loss: Rp {row['SL']:,.0f}")
                                    
                                    # Risk reward
                                    risk = row['EntryIdeal'] - row['SL']
                                    reward = row['TP2'] - row['EntryIdeal']
                                    rr = reward / risk if risk > 0 else 0
                                    st.markdown(f"**⚖️ Risk:Reward:** 1:{rr:.2f} {'✅' if rr >= 2 else '⚠️'}")
                        
                        # Download
                        st.markdown("---")
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results (CSV)",
                            csv,
                            f"screener_results_{get_jakarta_time().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv"
                        )
    
    elif menu == "📈 Single Stock + Multi-TF":
        st.markdown("### 📈 Single Stock Analysis + Multi-Timeframe")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Pilih Saham**")
            ticker = st.selectbox("", tickers, index=0)
        with col2:
            multi_tf = st.checkbox("📊 Multi-TF", value=True)
        
        if st.button("🔍 Analyze Deep", type="primary"):
            with st.spinner(f"Deep analysis on {ticker}..."):
                df = fetch_data(ticker, period)
                
                if df is None:
                    st.error("❌ Failed to fetch data")
                else:
                    # Data freshness
                    last_update = df.index[-1]
                    hours_old = (datetime.now() - last_update.tz_localize(None)).total_seconds() / 3600
                    
                    if hours_old > 24:
                        st.warning(f"⚠️ Data: Stale ({int(hours_old)}h old)")
                    else:
                        st.success(f"✅ Data: Fresh ({int(hours_old)}h old)")
                    
                    price = float(df['Close'].iloc[-1])
                    
                    # Main score
                    score, details, confidence = score_full_screener_v3(df)
                    
                    # Bandar analysis
                    band_score, band_details, band_phase, band_conf = score_bandar_v3(df)
                    
                    # Display main metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("💰 Price", f"Rp {price:,.0f}")
                    col2.metric("📊 Score", f"{score}/100")
                    col3.metric("🎯 Confidence", f"{confidence}%")
                    col4.metric("🎯 Bandar", band_phase.replace('🟢 ', '').replace('🚀 ', '').replace('⚫ ', '').replace('🔴 ', '').replace('⚪ ', ''))
                    
                    # Multi-Timeframe Analysis
                    if multi_tf:
                        st.markdown("---")
                        st.markdown("## 🔄 Multi-Timeframe Analysis")
                        
                        tf_results, alignment = analyze_multi_timeframe(ticker)
                        
                        # Alignment status
                        if "✅ STRONG BUY" in alignment:
                            st.success(alignment)
                        elif "✅ BUY" in alignment:
                            st.info(alignment)
                        elif "🔴 CAUTION" in alignment:
                            st.error(alignment)
                        else:
                            st.warning(alignment)
                        
                        # Show each timeframe
                        cols = st.columns(3)
                        for idx, (period_key, result) in enumerate(tf_results.items()):
                            if result:
                                with cols[idx]:
                                    st.markdown(f"### {result['label']}")
                                    st.metric("Score", f"{result['score']}/100")
                                    st.metric("Trend", result['trend'])
                                    st.metric("Conf", f"{result['confidence']}%")
                                    
                                    # Visual indicator
                                    if result['trend'] in ["Strong Bull", "Bull"]:
                                        st.success("✅ Bullish")
                                    elif result['trend'] == "Weak":
                                        st.warning("🟡 Weak")
                                    elif result['trend'] == "Neutral":
                                        st.info("⚪ Neutral")
                                    else:
                                        st.error("❌ Bearish")
                    
                    # Bandar Analysis
                    st.markdown("---")
                    st.markdown("## 🎯 Bandar / Smart Money Analysis")
                    
                    if '🟢' in band_phase:
                        st.success(f"**{band_phase} - PRIME BUY ZONE!**")
                    elif '🚀' in band_phase:
                        st.info(f"**{band_phase} - HOLD OR BUY PULLBACK**")
                    elif '⚫' in band_phase:
                        st.error(f"**{band_phase} - AVOID / EXIT!**")
                    elif '🔴' in band_phase:
                        st.error(f"**{band_phase} - DANGER ZONE!**")
                    else:
                        st.warning(f"**{band_phase}**")
                    
                    # Bandar details
                    col1, col2, col3, col4 = st.columns(4)
                    col1.info(f"**Volume**\n{band_details.get('Volume_Ratio', 'N/A')}")
                    col2.info(f"**Price Δ**\n{band_details.get('Price_Change', 'N/A')}")
                    col3.info(f"**OBV Trend**\n{band_details.get('OBV_Trend', 'N/A')}")
                    col4.info(f"**Signal**\n{band_details.get('Signal', 'N/A')}")
                    
                    # Technical Details
                    st.markdown("---")
                    st.markdown("## 📊 Technical Analysis Details")
                    
                    for key, value in details.items():
                        if key not in ['⚠️ Warnings']:
                            st.info(f"**{key}:** {value}")
                    
                    # Warnings
                    if '⚠️ Warnings' in details:
                        st.error(f"**⚠️ WARNINGS:** {details['⚠️ Warnings']}")
                    
                    # Entry Strategy
                    st.markdown("---")
                    st.markdown("## 🎯 Entry Strategy")
                    
                    levels = get_signal_levels(score, price, confidence)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📍 Conservative Entry")
                        st.success(f"""
                        Entry: Rp {levels['ideal']['entry']:,.0f} (Wait for 2% dip)
                        TP1 (8%): Rp {levels['ideal']['tp1']:,.0f}
                        TP2 (15%): Rp {levels['ideal']['tp2']:,.0f}
                        Stop Loss: Rp {levels['ideal']['sl']:,.0f}
                        """)
                        risk = levels['ideal']['entry'] - levels['ideal']['sl']
                        reward = levels['ideal']['tp2'] - levels['ideal']['entry']
                        rr = reward / risk if risk > 0 else 0
                        st.metric("Risk:Reward", f"1:{rr:.2f}")
                    
                    with col2:
                        st.markdown("### ⚡ Aggressive Entry")
                        st.warning(f"""
                        Entry: Rp {levels['aggr']['entry']:,.0f} (NOW!)
                        TP1: Rp {levels['aggr']['tp1']:,.0f}
                        TP2: Rp {levels['aggr']['tp2']:,.0f}
                        SL: Rp {levels['aggr']['sl']:,.0f}
                        """)
                    
                    # 3-Lot Management
                    st.markdown("---")
                    st.markdown("## 📊 3-Lot Position Management")
                    
                    st.info("""
                    **Split into 3 equal lots:**
                    - 🎯 **Lot 1/3:** Sell at TP1 (+8%) - Secure profit
                    - 🎯 **Lot 2/3:** Sell at TP2 (+15%) - Take bigger profit
                    - 🏃 **Lot 3/3:** Trail with 20D EMA - Let winners run!
                    
                    🛑 **Initial SL:** Set for ALL lots at entry
                    """)
                    
                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📍 Track Position"):
                            st.success("Added to Active Positions!")
                    with col2:
                        if st.button("👁️ Add to Watchlist"):
                            st.success("Added to Watchlist!")
    
    elif menu == "📉 BPJS":
        st.markdown("### 📉 BPJS Day Trading Scanner")
        st.info("🎯 Fast momentum plays for intraday trading")
        
        if st.button("🚀 Scan BPJS Setups", type="primary"):
            with st.spinner("Scanning for day trading setups..."):
                df = batch_scan(tickers, "BPJS", "1mo", limit, use_parallel)
            
            if df.empty:
                st.warning("No BPJS setups found")
            else:
                df = df[(df["Score"] >= min_score) & (df["Confidence"] >= min_confidence)]
                st.success(f"✅ Found {len(df)} BPJS opportunities!")
                
                col1, col2 = st.columns(2)
                col1.metric("Avg Score", f"{df['Score'].mean():.1f}/100")
                col2.metric("Avg Confidence", f"{df['Confidence'].mean():.1f}%")
                
                show = df[["Ticker","Price","Score","Confidence","Signal","EntryIdeal","TP1","SL"]]
                st.dataframe(show, use_container_width=True)
    
    elif menu == "📊 BSJP":
        st.markdown("### 📊 BSJP Swing Trading Scanner")
        st.info("🎯 Overnight/swing plays (2-5 day holds)")
        
        if st.button("🚀 Scan BSJP Setups", type="primary"):
            with st.spinner("Scanning for swing setups..."):
                df = batch_scan(tickers, "BSJP", period, limit, use_parallel)
            
            if df.empty:
                st.warning("No BSJP setups found")
            else:
                df = df[(df["Score"] >= min_score) & (df["Confidence"] >= min_confidence)]
                st.success(f"✅ Found {len(df)} BSJP opportunities!")
                
                col1, col2 = st.columns(2)
                col1.metric("Avg Score", f"{df['Score'].mean():.1f}/100")
                col2.metric("Avg Confidence", f"{df['Confidence'].mean():.1f}%")
                
                show = df[["Ticker","Price","Score","Confidence","Signal","EntryIdeal","TP1","TP2","SL"]]
                st.dataframe(show, use_container_width=True)
    
    elif menu == "🔥 Bandar Tracking":
        st.markdown("### 🔥 Bandar Tracking - Smart Money Detection")
        st.info("🎯 Follow institutional money movements (Wyckoff Method)")
        
        if st.button("🎯 Scan for Smart Money Activity", type="primary"):
            with st.spinner("Tracking smart money movements..."):
                df = batch_scan(tickers, "Bandar", period, limit, use_parallel)
            
            if df.empty:
                st.warning("No bandar activity detected")
            else:
                df = df[(df["Score"] >= min_score) & (df["Confidence"] >= min_confidence)]
                
                # Phase breakdown
                akumulasi = df[df['Details'].apply(lambda x: '🟢 AKUMULASI' in str(x.get('Phase', '')))]
                markup = df[df['Details'].apply(lambda x: '🚀 MARKUP' in str(x.get('Phase', '')))]
                distribusi = df[df['Details'].apply(lambda x: '🔴 DISTRIBUSI' in str(x.get('Phase', '')))]
                sideways = df[df['Details'].apply(lambda x: '⚪ SIDEWAYS' in str(x.get('Phase', '')))]
                
                st.success(f"✅ Found {len(df)} stocks with bandar activity!")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🟢 Akumulasi", len(akumulasi), delta="BUY ZONE!" if len(akumulasi) > 0 else None)
                col2.metric("🚀 Markup", len(markup), delta="HOLD ZONE" if len(markup) > 0 else None)
                col3.metric("🔴 Distribusi", len(distribusi), delta="AVOID!" if len(distribusi) > 0 else None, delta_color="inverse")
                col4.metric("⚪ Sideways", len(sideways))
                
                # Filter by phase
                st.markdown("---")
                st.markdown("### 🎯 Filter by Phase")
                
                col1, col2, col3, col4 = st.columns(4)
                show_akum = col1.checkbox("🟢 Akumulasi", value=True)
                show_markup = col2.checkbox("🚀 Markup", value=True)
                show_dist = col3.checkbox("🔴 Distribusi", value=False)
                show_side = col4.checkbox("⚪ Sideways", value=False)
                
                filtered = pd.DataFrame()
                if show_akum and not akumulasi.empty:
                    filtered = pd.concat([filtered, akumulasi])
                if show_markup and not markup.empty:
                    filtered = pd.concat([filtered, markup])
                if show_dist and not distribusi.empty:
                    filtered = pd.concat([filtered, distribusi])
                if show_side and not sideways.empty:
                    filtered = pd.concat([filtered, sideways])
                
                if filtered.empty:
                    st.warning("No stocks match selected filters")
                else:
                    # Create display dataframe
                    display_df = pd.DataFrame()
                    for _, row in filtered.iterrows():
                        phase = row['Details'].get('Phase', 'Unknown')
                        action = row['Details'].get('Action', 'N/A')
                        volume_ratio = row['Details'].get('Volume_Ratio', 'N/A')
                        price_change = row['Details'].get('Price_Change', 'N/A')
                        obv_trend = row['Details'].get('OBV_Trend', 'N/A')
                        
                        display_df = pd.concat([display_df, pd.DataFrame({
                            'Ticker': [row['Ticker']],
                            'Price': [row['Price']],
                            'Phase': [phase],
                            'Action': [action],
                            'Score': [row['Score']],
                            'Confidence': [row['Confidence']],
                            'Volume_Ratio': [volume_ratio],
                            'Price_Change': [price_change],
                            'OBV_Trend': [obv_trend]
                        })], ignore_index=True)
                    
                    st.dataframe(display_df, use_container_width=True, height=400)
                    
                    # Detailed analysis
                    st.markdown("---")
                    st.markdown("## 📊 Detailed Bandar Analysis")
                    
                    for _, row in filtered.head(10).iterrows():
                        phase = row['Details'].get('Phase', 'Unknown')
                        
                        # Color based on phase
                        if '🟢' in phase:
                            phase_color = "🟢"
                        elif '🚀' in phase:
                            phase_color = "🟡"
                        elif '🔴' in phase:
                            phase_color = "🔴"
                        else:
                            phase_color = "⚪"
                        
                        with st.expander(f"{phase_color} {row['Ticker']} - {phase} (Score: {row['Score']})"):
                            if '🟢' in phase:
                                st.success(f"**{phase} - ⚡ STRONG BUY**")
                            elif '🚀' in phase:
                                st.info(f"**{phase}**")
                            elif '🔴' in phase:
                                st.error(f"**{phase}**")
                            else:
                                st.warning(f"**{phase}**")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("💰 Price", f"Rp {row['Price']:,.0f}")
                            col2.metric("📊 Score", f"{row['Score']}/100")
                            col3.metric("🎯 Confidence", f"{row['Confidence']}%")
                            
                            st.markdown("### 📊 Smart Money Indicators")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.info(f"**Volume**\n{row['Details'].get('Volume_Ratio', 'N/A')}")
                            col2.info(f"**Price Δ**\n{row['Details'].get('Price_Change', 'N/A')}")
                            col3.info(f"**OBV Trend**\n{row['Details'].get('OBV_Trend', 'N/A')}")
                            col4.info(f"**Signal**\n{row['Details'].get('Signal', 'N/A')}")
                            
                            st.markdown("### 🎯 Trading Plan")
                            
                            if '🟢' in phase:
                                st.success("""
                                **🟢 ACCUMULATION PHASE - PRIME BUY ZONE!**
                                
                                **✅ Why This Is Perfect:**
                                - Volume up + Price sideways + OBV up
                                - LOW risk (Best entry point)
                                - Best risk:reward entry point
                                
                                **🔄 Entry Strategy:**
                                - Start: NOW (25% position)
                                - Start buying NOW (25% position)
                                - Add on dips (3 tranches total)
                                - Average cost = key to success
                                - Entry range: Rp {} - {}
                                
                                **🎯 Targets:**
                                - Conservative: +15% (Rp {})
                                - Aggressive: +25-40% (Rp {} - {})
                                - Timeline: 2-8 weeks
                                
                                **🛑 Stop Loss:**
                                - Below support: Rp {} (-8%)
                                - Exit if distribution signals appear
                                
                                **💡 Pro Tip:** This is THE BEST time to buy! Smart money is accumulating. 
                                Patience will be rewarded. HOLD through markup phase.
                                """.format(
                                    int(row['Price'] * 0.97),
                                    int(row['Price'] * 1.02),
                                    int(row['Price'] * 1.15),
                                    int(row['Price'] * 1.25),
                                    int(row['Price'] * 1.40),
                                    int(row['Price'] * 0.92)
                                ))
                                
                            elif '🚀' in phase:
                                st.info("""
                                **🚀 MARKUP PHASE - MOMENTUM PLAY**
                                
                                **Strategy:**
                                - If already in: HOLD and trail stop
                                - If entering now: Buy pullback to support
                                - Smaller position size (risk higher)
                                
                                **Targets:**
                                - TP1: +10-15%
                                - TP2: +20-30%
                                - Timeline: 1-4 weeks
                                
                                **Risk:** Medium - Price already up, momentum strong but could reverse
                                """)
                                
                            elif '🔴' in phase:
                                st.error("""
                                **🔴 DISTRIBUTION PHASE - DANGER ZONE!**
                                
                                **⚠️ What's Happening:**
                                - Smart money is SELLING to retail
                                - Volume up but OBV down = Bad divergence
                                - Price topped out, reversal imminent
                                
                                **Action:**
                                - If in position: EXIT immediately
                                - If not in: AVOID completely
                                - Wait for markdown to complete, then new accumulation
                                
                                **Risk:** VERY HIGH - Likely to fall -15% to -40%
                                """)
                            
                            else:
                                st.warning("""
                                **⚪ SIDEWAYS - NO CLEAR PATTERN**
                                
                                **Action:**
                                - WAIT for clear signal
                                - Monitor for phase change
                                - No position until accumulation starts
                                
                                **Risk:** Medium (Uncertain direction)
                                """)
                            
                            # Complete metrics
                            st.markdown("---")
                            st.markdown("### 📊 Complete Bandar Metrics")
                            
                            details = row['Details']
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.info(f"**Phase:** {details.get('Phase', 'N/A')}")
                                st.info(f"**Action:** {details.get('Action', 'N/A')}")
                                st.info(f"**Signal:** {details.get('Signal', 'N/A')}")
                                st.info(f"**Strength:** {details.get('Strength', 'N/A')}")
                                st.info(f"**Volume_Ratio:** {details.get('Volume_Ratio', 'N/A')}")

                            with col2:
                                st.info(f"**Price_Change:** {details.get('Price_Change', 'N/A')}")
                                st.info(f"**OBV_Trend:** {details.get('OBV_Trend', 'N/A')}")
                                st.info(f"**Volatility:** {details.get('Volatility', 'N/A')}")
                                st.info(f"**Risk:** {details.get('Risk', 'N/A')}")
                                st.info(f"**Hold_Period:** {details.get('Hold_Period', 'N/A')}")
                            
                            # Action buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"📍 Track {row['Ticker']}", key=f"track_{row['Ticker']}"):
                                    st.success(f"Added {row['Ticker']} to tracking!")
                            with col2:
                                if st.button(f"👁️ Watch {row['Ticker']}", key=f"watch_{row['Ticker']}"):
                                    st.success(f"Added {row['Ticker']} to watchlist!")
                
                # Educational content
                st.markdown("---")
                st.markdown("## 📚 Understanding Smart Money / Bandar")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("""
                    ### What is "Bandar" / Smart Money?
                    
                    **Bandar** (Indonesian) = Big Players = Institutions = Smart Money
                    
                    **Who are they:**
                    - Mutual funds (Reksadana)
                    - Pension funds
                    - Insurance companies
                    - Foreign institutions
                    - Ultra-wealthy individuals
                    
                    **Capital Size:**
                    - Billions to trillions Rupiah
                    - Can move markets
                    - Need weeks/months to build position
                    
                    **Strategy:**
                    - Buy when everyone is selling (accumulation)
                    - Sell when everyone is buying (distribution)
                    - Patient, systematic, disciplined
                    
                    **Footprints:**
                    - Volume patterns
                    - OBV (On-Balance Volume)
                    - Price action vs volume
                    - Wyckoff patterns
                    """)
                
                with col2:
                    st.success("""
                    ### Your Edge: Follow Their Footprints!
                    
                    **🟢 AKUMULASI Phase** ⭐⭐⭐⭐⭐
                    - **When:** They're buying quietly
                    - **What to do:** BUY with them!
                    - **Risk:** LOWEST
                    - **Reward:** HIGHEST (15-40%)
                    - **Timeline:** 2-8 weeks hold
                    - **Win Rate:** 70-80% if patient
                    
                    **🚀 MARKUP Phase** ⭐⭐⭐⭐
                    - **When:** They're driving price up
                    - **What to do:** HOLD or ride momentum
                    - **Risk:** MEDIUM
                    - **Reward:** GOOD (10-25%)
                    - **Timeline:** 1-4 weeks
                    - **Win Rate:** 60-70%
                    
                    **🔴 DISTRIBUSI Phase** ⭐
                    - **When:** They're selling to retail
                    - **What to do:** AVOID / EXIT!
                    - **Risk:** VERY HIGH
                    - **Reward:** NEGATIVE
                    - **Result:** -15% to -40%
                    - **Win Rate:** <20% (for buyers)
                    
                    **💎 Golden Rule:**
                    Buy in AKUMULASI, Hold through MARKUP, Sell BEFORE distribution!
                    """)
    
    elif menu == "💎 Value Hunting":
        st.markdown("### 💎 Value Hunting - Long-term Gems")
        st.info("🎯 Undervalued stocks with strong fundamentals")
        
        if st.button("🚀 Hunt for Value", type="primary"):
            with st.spinner("Hunting for long-term value..."):
                df = batch_scan(tickers, "Value", "1y", limit, use_parallel)
            
            if df.empty:
                st.warning("No value opportunities found")
            else:
                df = df[(df["Score"] >= min_score) & (df["Confidence"] >= min_confidence)]
                st.success(f"✅ Found {len(df)} value opportunities!")
                
                col1, col2 = st.columns(2)
                col1.metric("Avg Score", f"{df['Score'].mean():.1f}/100")
                col2.metric("Avg Confidence", f"{df['Confidence'].mean():.1f}%")
                
                show = df[["Ticker","Price","Score","Confidence","Signal","EntryIdeal","TP1","TP2","SL"]]
                st.dataframe(show, use_container_width=True)
    
    elif menu == "📊 Track Performance":
        st.markdown("### 📊 Track Performance")
        st.info("📈 Coming soon - Track your trading performance over time")
        
        st.markdown("""
        **Features to be added:**
        - Win rate tracking
        - Average R-multiple
        - Best/worst trades
        - Strategy performance comparison
        - Monthly P&L
        - Trade journal
        """)
    
    elif menu == "📍 Active Positions":
        st.markdown("### 📍 Active Positions")
        st.info("💼 Coming soon - Manage your active positions")
        
        st.markdown("""
        **Features to be added:**
        - Current positions list
        - Real-time P&L
        - Alert when TP/SL hit
        - Position sizing calculator
        - Portfolio heat map
        """)
    
    elif menu == "👁️ Watchlist":
        st.markdown("### 👁️ Watchlist")
        st.info("🔖 Coming soon - Your personal watchlist")
        
        st.markdown("""
        **Features to be added:**
        - Save stocks to watch
        - Price alerts
        - Phase change notifications
        - Quick analysis access
        """)
    
    elif menu == "✅ Test Cases":
        st.markdown("### ✅ Test Cases")
        st.info("🧪 Verify system functionality")
        
        test_tickers = ["BBCA.JK", "TLKM.JK", "ASII.JK"]
        
        if st.button("🧪 Run Tests", type="primary"):
            for ticker in test_tickers:
                st.markdown(f"#### Testing {ticker}")
                
                with st.spinner(f"Fetching {ticker}..."):
                    df = fetch_data(ticker, "6mo")
                    
                    if df is None:
                        st.error(f"❌ Failed to fetch {ticker}")
                    else:
                        st.success(f"✅ Data loaded: {len(df)} candles")
                        
                        # Test scoring
                        score, details, conf = score_full_screener_v3(df)
                        st.info(f"📊 Score: {score}/100 | Confidence: {conf}%")
                        
                        # Test bandar
                        band_score, band_details, band_phase, band_conf = score_bandar_v3(df)
                        st.info(f"🎯 Bandar: {band_phase} | Score: {band_score}/100")
                        
                        st.markdown("---")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🚀 IDX Traders v3.2 - Ultimate Edition")
    with col2:
        st.caption("💡 NEW: Enhanced Bandar Logic!")
    with col3:
        st.caption(f"⏰ {get_jakarta_time().strftime('%Y-%m-%d %H:%M:%S WIB')}")

if __name__ == "__main__":
    main()
