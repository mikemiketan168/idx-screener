#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime, timedelta, timezone
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="IDX Power Screener v4.2", 
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
    .stock-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============= KONFIGURASI =============
class Config:
    MAX_RETRIES = 2
    TIMEOUT = 15
    CACHE_TTL = 600  # 10 minutes

def init_db():
    try:
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
    except:
        pass  # Skip DB errors for now

# ============= SIMPLE DATA FETCHING =============
@st.cache_data(ttl=Config.CACHE_TTL)
def fetch_stock_data_simple(ticker, period="1mo"):
    """
    Simple and reliable data fetching with fallbacks
    """
    try:
        # Map period to days
        period_days = {
            "1mo": 30,
            "3mo": 90, 
            "6mo": 180
        }.get(period, 30)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Format dates for Yahoo Finance
        period1 = int(start_date.timestamp())
        period2 = int(end_date.timestamp())
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        
        # Simple request with short timeout
        response = requests.get(
            url,
            params={
                "period1": period1,
                "period2": period2, 
                "interval": "1d"
            },
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            timeout=Config.TIMEOUT,
            verify=False
        )
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        
        # Check if we have valid data
        if not data.get('chart', {}).get('result'):
            return None
            
        result = data['chart']['result'][0]
        quote = result['indicators']['quote'][0]
        timestamps = result['timestamp']
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': quote['open'],
            'High': quote['high'],
            'Low': quote['low'],
            'Close': quote['close'],
            'Volume': quote['volume']
        }, index=pd.to_datetime(timestamps, unit='s'))
        
        # Clean data
        df = df.dropna()
        
        if len(df) < 10:  # Reduced minimum requirement
            return None
            
        # Calculate basic indicators
        df = calculate_basic_indicators(df)
        return df
        
    except Exception as e:
        print(f"Error fetching {ticker}: {str(e)}")
        return None

def calculate_basic_indicators(df):
    """Calculate only essential indicators"""
    try:
        # Basic moving averages
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume
        df['Volume_SMA20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']
        
        # Handle volume ratio errors
        df['Volume_Ratio'] = df['Volume_Ratio'].replace([np.inf, -np.inf], 1.0)
        df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1.0)
        
        # Simple momentum
        df['Momentum_5D'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        
        return df
        
    except Exception as e:
        return df

# ============= SIMPLE SCORING SYSTEM =============
def calculate_simple_score(df):
    """Fast and reliable scoring"""
    try:
        if df is None or len(df) < 10:
            return 0, "Insufficient Data"
            
        current = df.iloc[-1]
        score = 50  # Start from neutral
        
        details = []
        
        # 1. Price vs EMA (30 points)
        if current['Close'] > current['EMA9'] > current['EMA21']:
            score += 25
            details.append("Strong Uptrend")
        elif current['Close'] > current['EMA9']:
            score += 15
            details.append("Short-term Bullish")
        elif current['Close'] < current['EMA9']:
            score -= 10
            details.append("Short-term Bearish")
        
        # 2. RSI Score (20 points)
        if 40 <= current['RSI'] <= 60:
            score += 20
            details.append("RSI Optimal")
        elif 30 <= current['RSI'] <= 70:
            score += 10
            details.append("RSI Good")
        elif current['RSI'] > 70:
            score -= 5
            details.append("RSI Overbought")
        elif current['RSI'] < 30:
            score -= 5
            details.append("RSI Oversold")
        
        # 3. Volume Score (20 points)
        if current['Volume_Ratio'] > 1.5:
            score += 15
            details.append("High Volume")
        elif current['Volume_Ratio'] > 1.0:
            score += 5
            details.append("Average Volume")
        
        # 4. Momentum (10 points)
        if current['Momentum_5D'] > 2:
            score += 10
            details.append("Positive Momentum")
        elif current['Momentum_5D'] < -2:
            score -= 5
            details.append("Negative Momentum")
        
        # Cap score between 0-100
        final_score = max(0, min(100, score))
        
        return final_score, " | ".join(details)
        
    except Exception as e:
        return 0, f"Error: {str(e)}"

# ============= TRADING LEVELS =============
def calculate_trading_levels(price, score, trend_info):
    if score >= 75:
        signal = "STRONG BUY"
        signal_class = "strong-buy"
        entry_ideal = round(price * 0.98, 2)
        entry_agresif = round(price * 1.00, 2)
        tp1 = round(price * 1.08, 2)
        tp2 = round(price * 1.15, 2)
        cut_loss = round(price * 0.92, 2)
        
    elif score >= 60:
        signal = "BUY"
        signal_class = "buy"
        entry_ideal = round(price * 0.98, 2)
        entry_agresif = round(price * 1.00, 2)
        tp1 = round(price * 1.06, 2)
        tp2 = round(price * 1.12, 2)
        cut_loss = round(price * 0.94, 2)
        
    elif score >= 45:
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
    if "Uptrend" in trend_info or "Bullish" in trend_info:
        trend = "🟢 UPTREND"
    elif "Bearish" in trend_info:
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

# ============= TICKER LIST =============
def load_tickers():
    """Reliable ticker list"""
    reliable_tickers = [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BREN.JK",
        "TLKM.JK", "ASII.JK", "UNVR.JK", "ICBP.JK", "INDF.JK",
        "ADRO.JK", "ANTM.JK", "PTBA.JK", "PGAS.JK", "AKRA.JK",
        "WSKT.JK", "EXCL.JK", "JSMR.JK", "TPIA.JK", "MDKA.JK",
        "ITMG.JK", "SMBR.JK", "SSIA.JK", "TINS.JK", "TOWR.JK"
    ]
    return reliable_tickers

# ============= SINGLE STOCK ANALYSIS =============
def analyze_single_stock(ticker, period):
    """Fast single stock analysis"""
    try:
        # Show immediate loading state
        progress_text = st.empty()
        progress_text.markdown("<div class='loading-spinner'>🔄 Fetching stock data...</div>", unsafe_allow_html=True)
        
        # Fetch data
        df = fetch_stock_data_simple(ticker, period)
        
        if df is None:
            progress_text.empty()
            return None
            
        progress_text.markdown("<div class='loading-spinner'>📊 Analyzing technical indicators...</div>", unsafe_allow_html=True)
        
        current_price = df['Close'].iloc[-1]
        current_volume = df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df else 1.0
        current_rsi = df['RSI'].iloc[-1] if 'RSI' in df else 50
        current_momentum = df['Momentum_5D'].iloc[-1] if 'Momentum_5D' in df else 0
        
        # Calculate score
        score, trend_info = calculate_simple_score(df)
        
        # Get trading levels
        levels = calculate_trading_levels(current_price, score, trend_info)
        
        progress_text.empty()
        
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
            'Volume Ratio': round(current_volume, 2),
            'RSI': round(current_rsi, 1),
            'Momentum 5D': round(current_momentum, 2),
            'Trend Info': trend_info
        }
        
    except Exception as e:
        return None

# ============= DISPLAY FUNCTIONS =============
def display_stock_analysis(stock_data):
    """Display stock analysis results"""
    if stock_data is None:
        st.error("""
        ❌ Tidak dapat menganalisis saham ini. Kemungkinan penyebab:
        - Data tidak tersedia di Yahoo Finance
        - Saham tidak aktif atau delisted
        - Koneksi internet bermasalah
        - Terlalu banyak request (coba tunggu beberapa saat)
        """)
        return
    
    st.markdown(f'<div class="stock-card">', unsafe_allow_html=True)
    
    # Header
    st.markdown(f"### 📈 {stock_data['Ticker']} Analysis")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💰 Current Price", f"Rp {stock_data['Price']:,.0f}")
        st.metric("📊 Score", f"{stock_data['Score']}/100")
    
    with col2:
        st.metric("📈 Volume", f"{stock_data['Volume Ratio']}x")
        st.metric("🎯 RSI", f"{stock_data['RSI']}")
    
    with col3:
        st.metric("🚀 Momentum 5D", f"{stock_data['Momentum 5D']}%")
        st.metric("📈 Trend", stock_data['Trend'])
    
    # Signal box
    st.markdown(f'<div class="signal-box {stock_data["SignalClass"]}">{stock_data["Signal"]} - Confidence: {stock_data["Score"]}%</div>', unsafe_allow_html=True)
    
    # Trading levels
    st.markdown("### 🎯 Trading Levels")
    
    if stock_data['Entry Ideal']:
        cols = st.columns(5)
        
        with cols[0]:
            st.success(f"**Entry Ideal**\nRp {stock_data['Entry Ideal']:,.0f}")
        with cols[1]:
            if stock_data['Entry Agresif']:
                st.warning(f"**Entry Agresif**\nRp {stock_data['Entry Agresif']:,.0f}")
            else:
                st.warning("**Entry Agresif**\nN/A")
        with cols[2]:
            st.info(f"**TP1**\nRp {stock_data['TP1']:,.0f}")
        with cols[3]:
            st.info(f"**TP2**\nRp {stock_data['TP2']:,.0f}")
        with cols[4]:
            st.error(f"**Cut Loss**\nRp {stock_data['Cut Loss']:,.0f}")
    else:
        st.info("📝 Tidak ada rekomendasi trading untuk sinyal ini")
    
    # Technical details
    st.markdown("### 📊 Technical Analysis")
    st.write(f"**Analysis:** {stock_data['Trend Info']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============= MAIN APP =============
def main():
    # Initialize
    init_db()
    
    st.markdown('<div class="big-title">🚀 IDX Power Screener v4.2</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #64748b; margin-bottom: 2rem;">Fast & Reliable Stock Analysis</div>', unsafe_allow_html=True)
    
    # Load tickers
    tickers = load_tickers()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Menu")
        
        # Simple market status
        jkt_time = get_jakarta_time()
        hour = jkt_time.hour
        if 9 <= hour < 16 and jkt_time.weekday() < 5:
            st.success("🟢 MARKET OPEN")
        else:
            st.warning("🔴 MARKET CLOSED")
        
        st.markdown("---")
        
        menu = st.radio("Pilih Analisis:", [
            "Single Stock Analysis",
            "Quick Screener"
        ])
        
        st.markdown("---")
        
        period = st.selectbox("Data Period:", ["1mo", "3mo"], index=0)
        
        st.markdown("---")
        st.caption(f"🔄 {jkt_time.strftime('%H:%M WIB')} | {len(tickers)} Stocks")
    
    # Main content
    if menu == "Single Stock Analysis":
        show_single_analysis(tickers, period)
    else:
        show_quick_screener(tickers, period)

def get_jakarta_time():
    jkt_tz = timezone(timedelta(hours=7))
    return datetime.now(jkt_tz)

def show_single_analysis(tickers, period):
    st.markdown("## 🔍 Single Stock Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_ticker = st.selectbox(
            "Pilih Saham:",
            options=tickers,
            index=0,
            key="stock_selector"
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("🚀 Analyze Now", type="primary", use_container_width=True):
            # Use session state to track analysis
            st.session_state.analyze_ticker = selected_ticker
            st.session_state.analyze_period = period
    
    # Perform analysis if triggered
    if hasattr(st.session_state, 'analyze_ticker'):
        with st.spinner(f"Analyzing {st.session_state.analyze_ticker}..."):
            result = analyze_single_stock(st.session_state.analyze_ticker, st.session_state.analyze_period)
            display_stock_analysis(result)

def show_quick_screener(tickers, period):
    st.markdown("## ⚡ Quick Screener")
    st.info("Scanning top 10 stocks for quick opportunities...")
    
    if st.button("🔍 Run Quick Scan", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Scan only 10 stocks for speed
        for i, ticker in enumerate(tickers[:10]):
            status_text.text(f"Analyzing {ticker}...")
            progress_bar.progress((i + 1) / 10)
            
            result = analyze_single_stock(ticker, period)
            if result and result['Score'] >= 60:  # Only show good opportunities
                results.append(result)
            
            time.sleep(0.5)  # Slow down to avoid rate limiting
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            st.success(f"🎯 Found {len(results)} opportunities!")
            
            # Display results
            for result in results:
                with st.expander(f"{result['Ticker']} - {result['Signal']} (Score: {result['Score']})"):
                    display_stock_analysis(result)
        else:
            st.warning("No strong opportunities found in top 10 stocks.")

if __name__ == "__main__":
    main()
