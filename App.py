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
    page_title="IDX Power Screener v4.0", 
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
    .metric-card {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============= KONFIGURASI =============
class Config:
    MAX_RETRIES = 3
    TIMEOUT = 20
    WORKERS = 5
    CACHE_TTL = 300

def init_db():
    conn = sqlite3.connect('screener.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scan_results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, ticker TEXT, 
                  strategy TEXT, score INTEGER, signal TEXT, price REAL,
                  entry_ideal REAL, entry_agresif REAL, tp1 REAL, tp2 REAL, 
                  cut_loss REAL, trend TEXT, volume_ratio REAL, rsi REAL)''')
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
            timeout=Config.TIMEOUT
        )
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        result = data['chart']['result'][0]
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': quote['open'], 'High': quote['high'], 'Low': quote['low'],
            'Close': quote['close'], 'Volume': quote['volume']
        }, index=pd.to_datetime(result['timestamp'], unit='s'))
        
        df = df.dropna()
        if len(df) < 20:
            return None
            
        # Calculate indicators
        df['EMA9'] = df['Close'].ewm(span=9).mean()
        df['EMA21'] = df['Close'].ewm(span=21).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # Volume
        df['Volume_SMA20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']
        
        # Momentum
        df['Momentum_5D'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        
        return df
        
    except Exception as e:
        return None

def fetch_with_retry(ticker, period="3mo"):
    for attempt in range(Config.MAX_RETRIES):
        df = fetch_yahoo_data(ticker, period)
        if df is not None:
            return df
        time.sleep(2 ** attempt)
    return None

# ============= SCORING SYSTEMS =============
def calculate_trend_strength(df):
    current = df.iloc[-1]
    trend_score = 0
    trend_info = []
    
    # EMA Alignment
    if current['Close'] > current['EMA9'] > current['EMA21'] > current['EMA50']:
        trend_score += 40
        trend_info.append("EMA Bullish Alignment")
    elif current['Close'] > current['EMA9'] > current['EMA21']:
        trend_score += 25
        trend_info.append("Short-term Bullish")
    elif current['Close'] > current['EMA50']:
        trend_score += 15
        trend_info.append("Above EMA50")
    else:
        trend_score -= 10
        trend_info.append("Downtrend")
    
    # Price vs SMA20
    if current['Close'] > current['SMA20']:
        trend_score += 10
        trend_info.append("Above SMA20")
    
    # Momentum
    if current['Momentum_5D'] > 2:
        trend_score += 15
        trend_info.append("Positive Momentum")
    
    return trend_score, " | ".join(trend_info)

def score_full_screener(df):
    try:
        trend_score, trend_info = calculate_trend_strength(df)
        current = df.iloc[-1]
        
        total_score = trend_score
        
        # Volume scoring
        if current['Volume_Ratio'] > 2.0:
            total_score += 25
        elif current['Volume_Ratio'] > 1.5:
            total_score += 15
        elif current['Volume_Ratio'] > 1.0:
            total_score += 5
            
        # RSI scoring
        if 40 <= current['RSI'] <= 60:
            total_score += 20
        elif 30 <= current['RSI'] <= 70:
            total_score += 10
            
        # Price position
        if current['Close'] > current['EMA9']:
            total_score += 10
            
        return max(0, min(100, total_score)), trend_info
        
    except:
        return 0, "Error in scoring"

def score_bpjs(df):
    try:
        current = df.iloc[-1]
        score = 0
        reasons = []
        
        # BPJS criteria - morning momentum
        if current['Volume_Ratio'] > 2.5:
            score += 40
            reasons.append("High Volume")
        elif current['Volume_Ratio'] > 1.8:
            score += 25
            reasons.append("Good Volume")
            
        if 30 <= current['RSI'] <= 50:
            score += 30
            reasons.append("Oversold Bounce")
            
        if current['Momentum_5D'] > 1:
            score += 20
            reasons.append("Positive Momentum")
            
        if current['Close'] > current['EMA9']:
            score += 10
            reasons.append("Above EMA9")
            
        return max(0, min(100, score)), " | ".join(reasons)
        
    except:
        return 0, "Error in BPJS scoring"

def score_bsjp(df):
    try:
        current = df.iloc[-1]
        score = 0
        reasons = []
        
        # BSJP criteria - afternoon recovery
        if current['RSI'] < 35:
            score += 40
            reasons.append("Oversold")
        elif current['RSI'] < 45:
            score += 25
            reasons.append("Near Oversold")
            
        if current['Volume_Ratio'] > 1.5:
            score += 30
            reasons.append("Volume Support")
            
        # Price near day low but showing support
        day_range = (current['High'] - current['Low']) / current['Low'] * 100
        if day_range > 2 and current['Close'] < (current['High'] + current['Low']) / 2:
            score += 20
            reasons.append("Potential Reversal")
            
        if current['Close'] > current['EMA21']:
            score += 10
            reasons.append("Above EMA21")
            
        return max(0, min(100, score)), " | ".join(reasons)
        
    except:
        return 0, "Error in BSJP scoring"

def score_bandar(df):
    try:
        current = df.iloc[-1]
        score = 0
        reasons = []
        
        # Bandar criteria - accumulation patterns
        if current['Volume_Ratio'] > 3.0:
            score += 40
            reasons.append("Very High Volume")
        elif current['Volume_Ratio'] > 2.0:
            score += 25
            reasons.append("High Volume")
            
        if current['RSI'] < 60:
            score += 20
            reasons.append("Not Overbought")
            
        # Price consolidation
        volatility = df['Close'].pct_change().std() * 100
        if volatility < 3:
            score += 20
            reasons.append("Low Volatility")
            
        if current['Close'] > current['EMA50']:
            score += 20
            reasons.append("Above EMA50")
            
        return max(0, min(100, score)), " | ".join(reasons)
        
    except:
        return 0, "Error in Bandar scoring"

# ============= SIGNAL & LEVELS CALCULATION =============
def calculate_trading_levels(price, score, trend_info):
    if score >= 80:
        signal = "STRONG BUY"
        signal_class = "strong-buy"
        entry_ideal = price * 0.97  # 3% below for better entry
        entry_agresif = price
        tp1 = price * 1.08  # 8% target
        tp2 = price * 1.15  # 15% target
        cut_loss = price * 0.92  # 8% stop loss
        
    elif score >= 65:
        signal = "BUY"
        signal_class = "buy"
        entry_ideal = price * 0.98
        entry_agresif = price
        tp1 = price * 1.06
        tp2 = price * 1.12
        cut_loss = price * 0.94
        
    elif score >= 50:
        signal = "HOLD"
        signal_class = "hold"
        entry_ideal = price * 0.95
        entry_agresif = None
        tp1 = price * 1.05
        tp2 = price * 1.10
        cut_loss = price * 0.90
        
    else:
        signal = "CL"
        signal_class = "cl"
        entry_ideal = None
        entry_agresif = None
        tp1 = None
        tp2 = None
        cut_loss = None
    
    # Determine trend
    if "Bullish" in trend_info:
        trend = "🟢 UPTREND"
    elif "Downtrend" in trend_info:
        trend = "🔴 DOWNTREND"
    else:
        trend = "🟡 SIDEWAYS"
    
    return {
        'signal': signal,
        'signal_class': signal_class,
        'entry_ideal': round(entry_ideal, 2) if entry_ideal else None,
        'entry_agresif': round(entry_agresif, 2) if entry_agresif else None,
        'tp1': round(tp1, 2) if tp1 else None,
        'tp2': round(tp2, 2) if tp2 else None,
        'cut_loss': round(cut_loss, 2) if cut_loss else None,
        'trend': trend
    }

# ============= LOAD TICKERS =============
def load_all_tickers():
    try:
        with open("idx_stocks.json", "r") as f:
            data = json.load(f)
        tickers = data.get("tickers", [])
        return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except:
        return [f"{ticker}.JK" for ticker in [
            "BBCA", "BBRI", "BMRI", "BBNI", "BBTN", "BRIS", "BJBR", "BJTM",
            "TLKM", "EXCL", "FREN", "ISAT", "TELK", "TKIM",
            "ASII", "AUTO", "BRPT", "GJTL", "HMSP", "ICBP", "INDF", "JPFA",
            "KLBF", "MBSS", "MLBI", "MYOR", "ROTI", "SCMA", "STTP", "ULTJ",
            "ADRO", "AKRA", "ANTM", "BUMI", "BYAN", "DOID", "ELSA", "EMTK",
            "ENRG", "HRUM", "ITMG", "MDKA", "PGAS", "PTBA", "PTPP", "SMBR",
            "SSIA", "TINS", "TOWR", "AKSI", "ARTO", "ASRM", "BOLT", "BRAM",
            "CASS", "CLEO", "DMMX", "EDGE", "FAST", "GDST", "HOKI", "ICON",
            "IGAR", "IKAI", "IMAS", "INKP", "IPCC", "JAST", "JAYA", "JSMR",
            "KBLI", "KBLM", "KIJA", "LION", "LPCK", "MAPI", "MCAS", "MIKA",
            "MTDL", "PANI", "PBSA", "PCAR", "POLY", "POWR", "PRIM", "PSDN",
            "PSSI", "RALS", "RICY", "SAME", "SAPX", "SDMU", "SEMA", "SIDO",
            "SILO", "SIMP", "SMMA", "SMSM", "SOCL", "SONA", "SOSS", "SULI",
            "TARA", "TCID", "TCPI", "TFCO", "TGRA", "TOTO", "TOYS", "TRST",
            "TSPC", "ULTJ", "UNIC", "UNTR", "WEGE", "WICO", "WIIM", "WSBP",
            "WTON", "YELO", "ZBRA", "ZYRX"
        ]]

# ============= PROCESS TICKERS =============
def process_ticker(ticker, strategy, period):
    try:
        df = fetch_with_retry(ticker, period)
        if df is None or len(df) < 20:
            return None
            
        current_price = df['Close'].iloc[-1]
        current_volume_ratio = df['Volume_Ratio'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        
        # Select scoring function based on strategy
        if strategy == "Full Screener":
            score, trend_info = score_full_screener(df)
        elif strategy == "BPJS":
            score, trend_info = score_bpjs(df)
        elif strategy == "BSJP":
            score, trend_info = score_bsjp(df)
        elif strategy == "Bandar":
            score, trend_info = score_bandar(df)
        else:
            score, trend_info = score_full_screener(df)
        
        if score < 40:  # Minimum threshold
            return None
            
        levels = calculate_trading_levels(current_price, score, trend_info)
        
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
            'Volume Ratio': current_volume_ratio,
            'RSI': current_rsi,
            'Trend Info': trend_info
        }
        
    except Exception as e:
        return None

def batch_process(tickers, strategy, period, max_workers=5):
    results = []
    total = len(tickers)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_ticker, ticker, strategy, period): ticker 
            for ticker in tickers
        }
        
        completed = 0
        for future in as_completed(future_to_ticker):
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"📊 Processing {completed}/{total} stocks...")
            
            result = future.result()
            if result:
                results.append(result)
            
            time.sleep(0.1)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    return results

# ============= STREAMLIT UI =============
def main():
    init_db()
    
    st.markdown('<div class="big-title">🚀 IDX Power Screener v4.0</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #64748b; margin-bottom: 2rem;">Advanced Technical Analysis | Real-time Alerts | Portfolio Management</div>', unsafe_allow_html=True)
    
    all_tickers = load_all_tickers()
    market_hours = check_market_hours()
    
    with st.sidebar:
        st.markdown("## ⚙️ Menu")
        
        # Market status
        if market_hours['market_open']:
            st.success("🟢 MARKET OPEN")
        else:
            st.warning("🔴 MARKET CLOSED")
            
        if market_hours['bpjs_time']:
            st.info("⏰ BPJS Time (09:00-10:00)")
        if market_hours['bsjp_time']:
            st.info("⏰ BSJP Time (14:00-16:00)")
        
        st.markdown("---")
        
        menu = st.radio("Pilih Strategi:", [
            "1. Full Screener",
            "2. Single Analysis", 
            "3. Bandar",
            "4. BPJS",
            "5. BSJP"
        ])
        
        st.markdown("---")
        
        if menu != "2. Single Analysis":
            stages = st.radio("Screening Stages:", [
                "Stage 1: 800 → 60 Stocks",
                "Stage 2: 60 → 15 Stocks"
            ])
            
            min_score = st.slider("Minimum Score:", 40, 90, 60)
            use_fast_mode = st.checkbox("⚡ Fast Mode", value=True)
        
        period = st.selectbox("Period:", ["1mo", "3mo", "6mo"], index=1)
        
        st.markdown("---")
        st.caption(f"Total Tickers: {len(all_tickers)}")
    
    # Main content
    if menu == "1. Full Screener":
        show_full_screener(all_tickers, stages, min_score, period, use_fast_mode)
    elif menu == "2. Single Analysis":
        show_single_analysis(all_tickers, period)
    elif menu == "3. Bandar":
        show_strategy_screener(all_tickers, "Bandar", stages, min_score, period, use_fast_mode)
    elif menu == "4. BPJS":
        show_strategy_screener(all_tickers, "BPJS", stages, min_score, period, use_fast_mode)
    elif menu == "5. BSJP":
        show_strategy_screener(all_tickers, "BSJP", stages, min_score, period, use_fast_mode)

def show_full_screener(tickers, stages, min_score, period, use_fast_mode):
    st.markdown("## 📊 Full Market Screener")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Stocks", len(tickers))
    col2.metric("Target", "60 → 15" if "Stage 2" in stages else "800 → 60")
    col3.metric("Min Score", min_score)
    
    if st.button("🚀 Run Full Screening", type="primary", use_container_width=True):
        with st.spinner("Stage 1: Screening 800 stocks to 60..."):
            # Stage 1: Quick screening to 60 stocks
            stage1_tickers = tickers[:800]  # Take first 800 stocks
            workers = 10 if use_fast_mode else 3
            
            results = batch_process(stage1_tickers, "Full Screener", period, workers)
            results_df = pd.DataFrame(results)
            
            if results_df.empty:
                st.error("No stocks found meeting criteria")
                return
                
            # Sort and take top 60
            stage1_results = results_df.nlargest(60, 'Score')
            
            if "Stage 1" in stages:
                display_results(stage1_results, "Stage 1 Results - Top 60 Stocks")
                return
            
            # Stage 2: Detailed analysis of 60 stocks to 15
            st.info("Stage 2: Detailed analysis of 60 stocks to 15...")
            
            detailed_results = []
            progress_bar = st.progress(0)
            
            for i, (_, row) in enumerate(stage1_results.iterrows()):
                progress_bar.progress((i + 1) / len(stage1_results))
                detailed_df = fetch_with_retry(row['Ticker'], "6mo")  # Longer period for detailed analysis
                
                if detailed_df is not None:
                    # Re-score with more data
                    score, trend_info = score_full_screener(detailed_df)
                    if score >= min_score:
                        levels = calculate_trading_levels(row['Price'], score, trend_info)
                        detailed_results.append({
                            'Ticker': row['Ticker'],
                            'Price': row['Price'],
                            'Score': score,
                            'Signal': levels['signal'],
                            'SignalClass': levels['signal_class'],
                            'Entry Ideal': levels['entry_ideal'],
                            'Entry Agresif': levels['entry_agresif'],
                            'TP1': levels['tp1'],
                            'TP2': levels['tp2'],
                            'Cut Loss': levels['cut_loss'],
                            'Trend': levels['trend'],
                            'Volume Ratio': row['Volume Ratio'],
                            'RSI': row['RSI'],
                            'Trend Info': trend_info
                        })
            
            progress_bar.empty()
            
            # Take top 15
            final_results = pd.DataFrame(detailed_results).nlargest(15, 'Score')
            display_results(final_results, "Stage 2 Results - Top 15 Stocks")

def show_strategy_screener(tickers, strategy, stages, min_score, period, use_fast_mode):
    st.markdown(f"## 📊 {strategy} Screener")
    
    # Strategy-specific info
    if strategy == "BPJS":
        st.info("**BPJS Strategy**: Beli Pagi Jual Sore - Momentum trading di pagi hari")
    elif strategy == "BSJP":
        st.info("**BSJP Strategy**: Beli Sore Jual Pagi - Swing trading overnight")
    elif strategy == "Bandar":
        st.info("**Bandar Strategy**: Tracking smart money accumulation")
    
    if st.button(f"🚀 Run {strategy} Screening", type="primary", use_container_width=True):
        with st.spinner(f"Running {strategy} screening..."):
            workers = 10 if use_fast_mode else 3
            results = batch_process(tickers[:800], strategy, period, workers)
            results_df = pd.DataFrame(results)
            
            if results_df.empty:
                st.error("No stocks found meeting criteria")
                return
            
            if "Stage 1" in stages:
                filtered_results = results_df[results_df['Score'] >= min_score].nlargest(60, 'Score')
                display_results(filtered_results, f"{strategy} - Top 60 Stocks")
            else:
                filtered_results = results_df[results_df['Score'] >= min_score].nlargest(15, 'Score')
                display_results(filtered_results, f"{strategy} - Top 15 Stocks")

def show_single_analysis(tickers, period):
    st.markdown("## 🔍 Single Stock Analysis")
    
    selected_ticker = st.selectbox("Pilih Saham:", tickers)
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy = st.selectbox("Strategy:", ["Full Analysis", "BPJS", "BSJP", "Bandar"])
    
    with col2:
        if st.button("Analyze Stock", type="primary"):
            with st.spinner(f"Analyzing {selected_ticker}..."):
                df = fetch_with_retry(selected_ticker, period)
                
                if df is None:
                    st.error("Failed to fetch data")
                    return
                
                # Calculate scores for all strategies
                current_price = df['Close'].iloc[-1]
                current_volume = df['Volume_Ratio'].iloc[-1]
                current_rsi = df['RSI'].iloc[-1]
                
                strategies = {
                    "Full Analysis": score_full_screener,
                    "BPJS": score_bpjs,
                    "BSJP": score_bsjp,
                    "Bandar": score_bandar
                }
                
                selected_strategy = strategy if strategy != "Full Analysis" else "Full Screener"
                score_func = strategies[strategy]
                score, trend_info = score_func(df)
                levels = calculate_trading_levels(current_price, score, trend_info)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("💰 Current Price", f"Rp {current_price:,.0f}")
                    st.metric("📊 Score", f"{score}/100")
                
                with col2:
                    st.metric("🎯 Signal", levels['signal'])
                    st.metric("📈 Trend", levels['trend'])
                
                with col3:
                    st.metric("📊 RSI", f"{current_rsi:.1f}")
                    st.metric("📈 Volume Ratio", f"{current_volume:.2f}x")
                
                # Signal box
                st.markdown(f'<div class="signal-box {levels["signal_class"]}">{levels["signal"]} - Score: {score}/100</div>', unsafe_allow_html=True)
                
                # Trading levels
                st.markdown("### 🎯 Trading Levels")
                if levels['entry_ideal']:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    col1.success(f"**Entry Ideal**\nRp {levels['entry_ideal']:,.0f}")
                    col2.warning(f"**Entry Agresif**\nRp {levels['entry_agresif']:,.0f}")
                    col3.info(f"**TP1**\nRp {levels['tp1']:,.0f}")
                    col4.info(f"**TP2**\nRp {levels['tp2']:,.0f}")
                    col5.error(f"**Cut Loss**\nRp {levels['cut_loss']:,.0f}")
                
                # Technical details
                st.markdown("### 📊 Technical Details")
                st.write(f"**Trend Analysis:** {trend_info}")
                st.write(f"**Strategy:** {strategy}")
                
                # Price chart
                st.markdown("### 📈 Price Chart (Last 30 Days)")
                chart_data = df[['Close', 'EMA9', 'EMA21']].tail(30)
                st.line_chart(chart_data)

def display_results(results_df, title):
    st.markdown(f"### {title}")
    st.success(f"🎯 Found {len(results_df)} qualifying stocks!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Score", f"{results_df['Score'].mean():.1f}")
    col2.metric("Strong Buy", len(results_df[results_df['Signal'] == 'STRONG BUY']))
    col3.metric("Buy", len(results_df[results_df['Signal'] == 'BUY']))
    col4.metric("Avg RSI", f"{results_df['RSI'].mean():.1f}")
    
    # Display table
    display_cols = ['Ticker', 'Price', 'Score', 'Signal', 'Trend', 'Entry Ideal', 'TP1', 'TP2', 'Cut Loss']
    display_df = results_df[display_cols].copy()
    
    st.dataframe(
        display_df.style.format({
            'Price': '{:,.0f}',
            'Entry Ideal': '{:,.0f}',
            'TP1': '{:,.0f}',
            'TP2': '{:,.0f}',
            'Cut Loss': '{:,.0f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Detailed view
    st.markdown("### 📋 Detailed Analysis")
    for _, row in results_df.iterrows():
        with st.expander(f"{row['Ticker']} - {row['Signal']} (Score: {row['Score']})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Price", f"Rp {row['Price']:,.0f}")
                st.metric("RSI", f"{row['RSI']:.1f}")
                st.metric("Volume Ratio", f"{row['Volume Ratio']:.2f}x")
                
            with col2:
                st.metric("Trend", row['Trend'])
                st.metric("Entry Ideal", f"Rp {row['Entry Ideal']:,.0f}" if row['Entry Ideal'] else "N/A")
                st.metric("Cut Loss", f"Rp {row['Cut Loss']:,.0f}" if row['Cut Loss'] else "N/A")
            
            st.write(f"**Trend Info:** {row['Trend Info']}")

if __name__ == "__main__":
    main()
