#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="IDX Power Screener v4.1", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom
st.markdown("""
<style>
    .big-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1e40af;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .signal-box {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        font-weight: 700;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .strong-buy {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    .buy {
        background: linear-gradient(135deg, #34d399, #10b981);
        color: white;
    }
    .hold {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        color: white;
    }
    .cl {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
    }
    .stage-box {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stock-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    .error-box {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        color: #92400e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============= KONFIGURASI =============
class Config:
    MAX_RETRIES = 3
    TIMEOUT = 25
    WORKERS = 6
    CACHE_TTL = 300

def init_db():
    conn = sqlite3.connect('screener.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scan_results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, ticker TEXT, 
                  strategy TEXT, score INTEGER, signal TEXT, price REAL,
                  entry_ideal REAL, entry_agresif REAL, tp1 REAL, tp2 REAL, 
                  cut_loss REAL, trend TEXT, volume_ratio REAL, rsi REAL,
                  momentum_5d REAL, ema_alignment TEXT)''')
    conn.commit()
    conn.close()

# ============= UTILITIES =============
def get_jakarta_time():
    jkt_tz = timezone(timedelta(hours=7))
    return datetime.now(jkt_tz)

def check_market_hours():
    jkt_time = get_jakarta_time()
    hour = jkt_time.hour
    return {
        'bpjs_time': 9 <= hour < 10,
        'bsjp_time': 14 <= hour < 16,
        'market_open': 9 <= hour < 16 and jkt_time.weekday() < 5
    }

# ============= DATA FETCHING =============
@st.cache_data(ttl=Config.CACHE_TTL)
def fetch_yahoo_data(ticker, period="3mo"):
    try:
        end = int(datetime.now().timestamp())
        period_sec = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}.get(period, 90) * 86400
        start = end - period_sec
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        response = requests.get(
            url,
            params={"period1": start, "period2": end, "interval": "1d"},
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=Config.TIMEOUT,
            verify=False
        )
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        
        # Check if data is available
        if not data['chart']['result']:
            return None
            
        result = data['chart']['result'][0]
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': quote['open'], 'High': quote['high'], 'Low': quote['low'],
            'Close': quote['close'], 'Volume': quote['volume']
        }, index=pd.to_datetime(result['timestamp'], unit='s'))
        
        df = df.dropna()
        if len(df) < 20:
            return None
            
        return calculate_technical_indicators(df)
        
    except Exception as e:
        return None

def calculate_technical_indicators(df):
    """Calculate all technical indicators"""
    try:
        # Moving Averages
        df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Volume Indicators
        df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']
        
        # Replace infinite values and NaN in Volume_Ratio
        df['Volume_Ratio'] = df['Volume_Ratio'].replace([np.inf, -np.inf], 1.0)
        df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1.0)
        
        # Momentum
        df['Momentum_5D'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        df['Momentum_10D'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        df['Momentum_20D'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20) * 100
        
        return df
        
    except Exception as e:
        return df

def fetch_with_retry(ticker, period="3mo"):
    for attempt in range(Config.MAX_RETRIES):
        df = fetch_yahoo_data(ticker, period)
        if df is not None and len(df) > 20:
            return df
        time.sleep(2 ** attempt)
    return None

# ============= IMPROVED SCORING SYSTEM =============
def calculate_advanced_score(df):
    """Advanced scoring system"""
    try:
        current = df.iloc[-1]
        score = 0
        details = []
        
        # 1. TREND STRENGTH (40 points max)
        trend_score = 0
        
        # EMA Alignment (20 points)
        if current['Close'] > current['EMA9'] > current['EMA21'] > current['EMA50']:
            trend_score += 20
            details.append("Perfect EMA Bullish")
        elif current['Close'] > current['EMA9'] > current['EMA21']:
            trend_score += 15
            details.append("Strong EMA Bullish")
        elif current['Close'] > current['EMA9']:
            trend_score += 10
            details.append("Short-term Bullish")
        else:
            trend_score -= 5
            details.append("EMA Bearish")
        
        # Price vs MAs (10 points)
        if current['Close'] > current['SMA20'] > current['SMA50']:
            trend_score += 10
            details.append("SMA Bullish")
        elif current['Close'] > current['SMA20']:
            trend_score += 5
            details.append("Above SMA20")
        
        # Trend Momentum (10 points)
        if current['Momentum_5D'] > 3:
            trend_score += 10
            details.append("Strong Momentum")
        elif current['Momentum_5D'] > 0:
            trend_score += 5
            details.append("Positive Momentum")
        
        score += max(0, trend_score)
        
        # 2. MOMENTUM & STRENGTH (35 points max)
        momentum_score = 0
        
        # RSI Strength (15 points)
        if 45 <= current['RSI'] <= 65:
            momentum_score += 15
            details.append("RSI Optimal")
        elif 40 <= current['RSI'] <= 70:
            momentum_score += 10
            details.append("RSI Good")
        elif 30 <= current['RSI'] <= 75:
            momentum_score += 5
            details.append("RSI Acceptable")
        else:
            momentum_score -= 5
            details.append("RSI Extreme")
        
        # MACD Strength (10 points)
        if current['MACD'] > current['MACD_Signal'] and current['MACD_Histogram'] > 0:
            momentum_score += 10
            details.append("MACD Bullish")
        elif current['MACD'] > current['MACD_Signal']:
            momentum_score += 5
            details.append("MACD Turning")
        
        # Recent Performance (10 points)
        if current['Momentum_10D'] > 5:
            momentum_score += 10
            details.append("Strong 10D Gain")
        elif current['Momentum_10D'] > 0:
            momentum_score += 5
            details.append("Positive 10D")
        
        score += max(0, momentum_score)
        
        # 3. VOLUME & PARTICIPATION (25 points max)
        volume_score = 0
        
        # Volume Ratio (15 points)
        if current['Volume_Ratio'] > 2.0:
            volume_score += 15
            details.append("Very High Volume")
        elif current['Volume_Ratio'] > 1.5:
            volume_score += 10
            details.append("High Volume")
        elif current['Volume_Ratio'] > 1.0:
            volume_score += 5
            details.append("Average Volume")
        
        # Volume Trend (10 points)
        recent_volume_avg = df['Volume_Ratio'].tail(5).mean()
        if recent_volume_avg > 1.2:
            volume_score += 10
            details.append("Volume Uptrend")
        elif recent_volume_avg > 0.8:
            volume_score += 5
            details.append("Stable Volume")
        
        score += max(0, volume_score)
        
        # BONUS: Strong patterns (10 points)
        bonus = 0
        if (current['Close'] > current['EMA9'] > current['EMA21'] > current['EMA50'] and 
            current['Volume_Ratio'] > 1.5 and 
            40 <= current['RSI'] <= 65):
            bonus += 10
            details.append("Perfect Setup")
        
        score += bonus
        
        # Ensure score is within 0-100
        final_score = max(0, min(100, score))
        
        return final_score, " | ".join(details)
        
    except Exception as e:
        return 0, f"Scoring Error: {str(e)}"

# ============= IMPROVED TRADING LEVELS =============
def calculate_trading_levels(price, score, trend_info):
    if score >= 80:
        signal = "STRONG BUY"
        signal_class = "strong-buy"
        entry_ideal = round(price * 0.97, 2)
        entry_agresif = round(price * 0.995, 2)
        tp1 = round(price * 1.08, 2)
        tp2 = round(price * 1.15, 2)
        cut_loss = round(price * 0.92, 2)
        
    elif score >= 65:
        signal = "BUY"
        signal_class = "buy"
        entry_ideal = round(price * 0.975, 2)
        entry_agresif = round(price, 2)
        tp1 = round(price * 1.06, 2)
        tp2 = round(price * 1.12, 2)
        cut_loss = round(price * 0.94, 2)
        
    elif score >= 50:
        signal = "HOLD"
        signal_class = "hold"
        entry_ideal = round(price * 0.95, 2)
        entry_agresif = None
        tp1 = round(price * 1.05, 2)
        tp2 = round(price * 1.10, 2)
        cut_loss = round(price * 0.90, 2)
        
    else:
        signal = "CL"
        signal_class = "cl"
        entry_ideal = None
        entry_agresif = None
        tp1 = None
        tp2 = None
        cut_loss = None
    
    # Determine trend
    if "Bullish" in trend_info or "Uptrend" in trend_info:
        trend = "🟢 UPTREND"
    elif "Bearish" in trend_info or "Downtrend" in trend_info:
        trend = "🔴 DOWNTREND"
    else:
        trend = "🟡 SIDEWAYS"
    
    return {
        'signal': signal,
        'signal_class': signal_class,
        'entry_ideal': entry_ideal,
        'entry_agresif': entry_agresif,
        'tp1': tp1,
        'tp2': tp2,
        'cut_loss': cut_loss,
        'trend': trend
    }

# ============= LOAD TICKERS =============
def load_all_tickers():
    """Load Indonesian stocks"""
    try:
        with open("idx_stocks.json", "r") as f:
            data = json.load(f)
        tickers = data.get("tickers", [])
        return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except:
        base_stocks = [
            "BBCA", "BBRI", "BMRI", "BBNI", "BBTN", "BRIS", "BREN", "TLKM", 
            "ASII", "UNVR", "ICBP", "INDF", "ADRO", "ANTM", "PTBA", "PGAS",
            "AKRA", "WSKT", "EXCL", "FREN", "JSMR", "TPIA", "MDKA", "ITMG",
            "SMBR", "SSIA", "TINS", "TOWR", "EMTK", "BUMI", "BYAN"
        ]
        return [f"{ticker}.JK" for ticker in base_stocks]

# ============= IMPROVED PROCESS TICKERS =============
def process_ticker_advanced(ticker, strategy, period):
    """UNIFIED processing for both single and batch"""
    try:
        df = fetch_with_retry(ticker, period)
        if df is None or len(df) < 20:
            return None
            
        current_price = df['Close'].iloc[-1]
        current_volume_ratio = df['Volume_Ratio'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_momentum = df['Momentum_5D'].iloc[-1]
        
        # Use the SAME scoring system for both single and batch
        score, trend_info = calculate_advanced_score(df)
        
        # LOWER threshold to catch more good stocks
        if score < 30:
            return None
            
        levels = calculate_trading_levels(current_price, score, trend_info)
        
        # Additional technical info
        ema_alignment = "Bullish" if current_price > df['EMA9'].iloc[-1] > df['EMA21'].iloc[-1] else "Mixed"
        
        return {
            'Ticker': ticker,
            'Price': current_price,
            'Score': score,
            'Signal': levels['signal'],
            'SignalClass': levels['signal_class'],
            'Entry Ideal': levels['entry_ideal'],
            'Entry Agresif': levels['entry_agresif'],
            'TP1': levels['tp1'],
            'TP2': levels['tp2'],
            'Cut Loss': levels['cut_loss'],
            'Trend': levels['trend'],
            'Volume Ratio': round(current_volume_ratio, 2),
            'RSI': round(current_rsi, 1),
            'Momentum 5D': round(current_momentum, 2),
            'Trend Info': trend_info,
            'EMA Alignment': ema_alignment
        }
        
    except Exception as e:
        return None

# ============= IMPROVED STREAMLIT UI =============
def main():
    init_db()
    
    st.markdown('<div class="big-title">🚀 IDX Power Screener v4.1</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #64748b; margin-bottom: 2rem;">Advanced Technical Analysis | Real-time Alerts | Portfolio Management</div>', unsafe_allow_html=True)
    
    all_tickers = load_all_tickers()
    market_hours = check_market_hours()
    
    with st.sidebar:
        st.markdown("## ⚙️ Menu")
        
        if market_hours['market_open']:
            st.success("🟢 MARKET OPEN")
        else:
            st.warning("🔴 MARKET CLOSED")
        
        st.markdown("---")
        
        menu = st.radio("Pilih Menu:", [
            "1. Full Screener",
            "2. Single Analysis", 
            "3. Bandar",
            "4. BPJS", 
            "5. BSJP"
        ])
        
        st.markdown("---")
        
        if menu == "1. Full Screener":
            stage = st.radio("Screening Stage:", [
                "Stage 1: Screening to 60 Stocks",
                "Stage 2: Screening to 15 Stocks"
            ])
            
            min_score = st.slider("Minimum Score:", 30, 90, 50)
            use_fast_mode = st.checkbox("⚡ Fast Mode", value=True)
        
        period = st.selectbox("Period:", ["1mo", "3mo", "6mo"], index=1)
        
        st.markdown("---")
        st.caption(f"Total Tickers: {len(all_tickers)}")
    
    # Main content
    if menu == "1. Full Screener":
        show_full_screener_improved(all_tickers, stage, min_score, period, use_fast_mode)
    elif menu == "2. Single Analysis":
        show_single_analysis_improved(all_tickers, period)
    else:
        st.info("Fitur dalam pengembangan")

def show_full_screener_improved(tickers, stage, min_score, period, use_fast_mode):
    st.markdown("## 📊 Full Market Screener")
    
    if st.button("🚀 Run Advanced Screening", type="primary", use_container_width=True):
        with st.spinner("Running advanced screening..."):
            workers = 6 if use_fast_mode else 3
            results = []
            
            for ticker in tickers[:100]:  # Test with 100 stocks first
                result = process_ticker_advanced(ticker, "Full Screener", period)
                if result:
                    results.append(result)
            
            if not results:
                st.error("❌ No stocks found. Please check internet connection.")
                return
            
            results_df = pd.DataFrame(results)
            
            # Filter by minimum score
            filtered_results = results_df[results_df['Score'] >= min_score]
            
            if "Stage 1" in stage:
                target_count = 60
                if len(filtered_results) > target_count:
                    final_results = filtered_results.nlargest(target_count, 'Score')
                else:
                    final_results = filtered_results
                
                st.success(f"✅ Stage 1 Complete: Found {len(final_results)} qualifying stocks!")
                display_results_improved(final_results, f"Stage 1 Results - Top {len(final_results)} Stocks")
            else:
                target_count = 15
                if len(filtered_results) > target_count:
                    final_results = filtered_results.nlargest(target_count, 'Score')
                else:
                    final_results = filtered_results
                
                st.success(f"✅ Stage 2 Complete: Found {len(final_results)} elite stocks!")
                display_results_improved(final_results, f"Stage 2 Results - Top {len(final_results)} Stocks")

def show_single_analysis_improved(tickers, period):
    st.markdown("## 🔍 Single Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_ticker = st.selectbox("Pilih Saham:", tickers, index=0)
    
    with col2:
        if st.button("🔍 Analyze Stock", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {selected_ticker}..."):
                # Use the SAME function as batch processing
                result = process_ticker_advanced(selected_ticker, "Single Analysis", period)
                
                if result is None:
                    st.error("""
                    ❌ Failed to analyze stock. Possible reasons:
                    - No data available from Yahoo Finance
                    - Stock is delisted or suspended
                    - Internet connection issue
                    - API rate limiting
                    """)
                    return
                
                display_single_stock_detailed(result)

def display_single_stock_detailed(stock_data):
    """Display detailed single stock analysis - FIXED ERROR HANDLING"""
    
    # Header with ticker and basic info
    st.markdown(f'<div class="stock-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💰 Current Price", f"Rp {stock_data['Price']:,.0f}")
        st.metric("📊 Score", f"{stock_data['Score']}/100")
    
    with col2:
        st.metric("📈 Volume Ratio", f"{stock_data['Volume Ratio']}x")
        st.metric("🎯 RSI", f"{stock_data['RSI']}")
    
    with col3:
        st.metric("🚀 Momentum 5D", f"{stock_data['Momentum 5D']}%")
        st.metric("📊 EMA Alignment", stock_data['EMA Alignment'])
    
    # Signal Box
    st.markdown(f'<div class="signal-box {stock_data["SignalClass"]}">{stock_data["Signal"]} - {stock_data["Trend"]}</div>', unsafe_allow_html=True)
    
    # Trading Levels - FIXED: Handle None values
    st.markdown("### 🎯 Trading Levels")
    
    if stock_data['Entry Ideal'] is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.success(f"**Entry Ideal**\nRp {stock_data['Entry Ideal']:,.0f}")
        
        with col2:
            # FIX: Check if Entry Agresif is not None before formatting
            if stock_data['Entry Agresif'] is not None:
                st.warning(f"**Entry Agresif**\nRp {stock_data['Entry Agresif']:,.0f}")
            else:
                st.warning("**Entry Agresif**\nN/A")
        
        with col3:
            st.info(f"**TP1**\nRp {stock_data['TP1']:,.0f}")
        
        with col4:
            st.info(f"**TP2**\nRp {stock_data['TP2']:,.0f}")
        
        with col5:
            st.error(f"**Cut Loss**\nRp {stock_data['Cut Loss']:,.0f}")
    else:
        st.markdown('<div class="error-box">⚠️ Trading levels not available for this signal. Consider waiting for better entry points.</div>', unsafe_allow_html=True)
    
    # Technical Details
    st.markdown("### 📊 Technical Details")
    st.write(f"**Trend Analysis:** {stock_data['Trend Info']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_results_improved(results_df, title):
    st.markdown(f"### {title}")
    
    if results_df.empty:
        st.warning("No results to display")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Score", f"{results_df['Score'].mean():.1f}")
    col2.metric("Strong Buy", len(results_df[results_df['Signal'] == 'STRONG BUY']))
    col3.metric("Buy", len(results_df[results_df['Signal'] == 'BUY']))
    col4.metric("Avg RSI", f"{results_df['RSI'].mean():.1f}")
    
    # Display table
    display_cols = ['Ticker', 'Price', 'Score', 'Signal', 'Trend', 'Entry Ideal', 'TP1', 'TP2', 'Cut Loss']
    
    # Ensure all columns exist
    available_cols = [col for col in display_cols if col in results_df.columns]
    display_df = results_df[available_cols].copy()
    
    # Format numbers safely
    def safe_format(x):
        if pd.isna(x) or x is None:
            return "N/A"
        try:
            return f"{x:,.0f}"
        except:
            return str(x)
    
    # Apply formatting only to numeric columns
    format_cols = ['Price', 'Entry Ideal', 'TP1', 'TP2', 'Cut Loss']
    for col in format_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(safe_format)
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Detailed view
    st.markdown("### 📋 Detailed Analysis")
    for _, row in results_df.iterrows():
        with st.expander(f"{row['Ticker']} - {row['Signal']} (Score: {row['Score']})"):
            display_single_stock_detailed(row)

if __name__ == "__main__":
    main()
