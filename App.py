#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime, timedelta, timezone, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="IDX Power Screener v4.0", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom yang lebih baik
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
    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
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
    .neutral {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        color: white;
    }
    .sell {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ============= KONFIGURASI & SETUP =============
class Config:
    MAX_RETRIES = 3
    TIMEOUT = 20
    WORKERS = 5
    CACHE_TTL = 300  # 5 menit

def init_db():
    """Initialize database dengan tabel yang lebih komprehensif"""
    conn = sqlite3.connect('screener_tracking.db', check_same_thread=False)
    c = conn.cursor()
    
    # Tabel recommendations yang ditingkatkan
    c.execute('''CREATE TABLE IF NOT EXISTS recommendations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT, ticker TEXT, strategy TEXT, score INTEGER,
                  confidence INTEGER, entry_price REAL, current_price REAL,
                  signal TEXT, status TEXT DEFAULT 'ACTIVE', result TEXT,
                  profit_pct REAL, exit_price REAL, exit_date TEXT, notes TEXT,
                  position_size TEXT DEFAULT '3/3',
                  volume_ratio REAL, rsi REAL, trend_strength TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Tabel watchlist yang ditingkatkan
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date_added TEXT, ticker TEXT, strategy TEXT,
                  score INTEGER, confidence INTEGER, target_entry REAL,
                  current_price REAL, notes TEXT, status TEXT DEFAULT 'WATCHING',
                  volume_ratio REAL, rsi REAL, trend_strength TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

# ============= UTILITIES =============
def get_jakarta_time():
    """Get current Jakarta time (UTC+7)"""
    jkt_tz = timezone(timedelta(hours=7))
    return datetime.now(jkt_tz)

def check_idx_market_status():
    """Check if IDX market is open dengan informasi lebih detail"""
    jkt_time = get_jakarta_time()
    hour = jkt_time.hour
    minute = jkt_time.minute
    weekday = jkt_time.weekday()
    
    # IDX: Mon-Fri, 09:00-16:15
    if weekday >= 5:
        next_market = "Monday" if weekday == 6 else "Tomorrow"
        return f"🔴 CLOSED - Weekend (Opens {next_market})", False
    
    if hour < 9:
        open_in_minutes = (9 - hour) * 60 - minute
        return f"⏰ Opens in {open_in_minutes//60}h {open_in_minutes%60}m", False
    elif hour >= 16 or (hour == 16 and minute >= 15):
        return "🔴 CLOSED - After hours", False
    elif 12 <= hour < 13:
        close_in = (13 - hour) * 60 - minute
        return f"🟡 LUNCH BREAK (Resumes in {close_in}m)", False
    else:
        close_in = (16 - hour) * 60 + (15 - minute)
        return f"🟢 MARKET OPEN (Closes in {close_in//60}h {close_in%60}m)", True

# ============= DATA FETCHING YANG LEBIH ROBUST =============
@st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
def fetch_data_yahoo(ticker, period="6mo"):
    """Fetch data dari Yahoo Finance dengan error handling yang lebih baik"""
    try:
        # Mapping period ke detik
        period_map = {
            "5d": 5 * 86400,
            "1mo": 30 * 86400,
            "3mo": 90 * 86400,
            "6mo": 180 * 86400,
            "1y": 365 * 86400
        }
        
        end = int(datetime.now().timestamp())
        start = end - period_map.get(period, 180 * 86400)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "period1": start,
            "period2": end,
            "interval": "1d"
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(
            url, 
            params=params, 
            headers=headers, 
            timeout=Config.TIMEOUT,
            verify=False
        )
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        
        # Validasi struktur data
        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            return None
            
        result = data['chart']['result'][0]
        
        # Validasi data yang diperlukan
        if 'timestamp' not in result or 'indicators' not in result:
            return None
            
        timestamps = result['timestamp']
        quote_data = result['indicators']['quote'][0]
        
        # Buat DataFrame
        df = pd.DataFrame({
            'Open': quote_data['open'],
            'High': quote_data['high'],
            'Low': quote_data['low'],
            'Close': quote_data['close'],
            'Volume': quote_data['volume']
        }, index=pd.to_datetime(timestamps, unit='s'))
        
        # Hapus baris dengan data null
        df = df.dropna()
        
        if len(df) < 20:
            return None
            
        return calculate_technical_indicators(df)
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Hitung semua indikator teknikal"""
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
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']) * 100
        
        # Stochastic
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Volume Indicators
        df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']
        
        # Momentum
        df['Momentum_5D'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        df['Momentum_10D'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        df['Momentum_20D'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20) * 100
        
        return df
        
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        return df  # Return original df jika error

def fetch_data_with_retry(ticker, period="6mo", max_retries=Config.MAX_RETRIES):
    """Fetch data dengan retry mechanism"""
    for attempt in range(max_retries):
        try:
            df = fetch_data_yahoo(ticker, period)
            if df is not None and len(df) > 20:
                return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            continue
    return None

# ============= SCORING SYSTEM YANG LEBIH ADVANCE =============
class AdvancedScoring:
    def __init__(self):
        self.weights = {
            'trend': 0.25,
            'momentum': 0.20,
            'volume': 0.15,
            'rsi': 0.10,
            'macd': 0.10,
            'bollinger': 0.10,
            'support_resistance': 0.10
        }
    
    def calculate_trend_score(self, df):
        """Hitung skor trend dengan weighting"""
        try:
            current = df.iloc[-1]
            score = 0
            
            # EMA alignment (most important)
            if current['Close'] > current['EMA9'] > current['EMA21'] > current['EMA50'] > current['EMA200']:
                score += 40
            elif current['Close'] > current['EMA9'] > current['EMA21'] > current['EMA50']:
                score += 30
            elif current['Close'] > current['EMA9'] > current['EMA21']:
                score += 20
            elif current['Close'] > current['EMA9']:
                score += 10
                
            # SMA alignment
            if current['Close'] > current['SMA20'] > current['SMA50']:
                score += 20
            elif current['Close'] > current['SMA20']:
                score += 10
                
            return min(score, 50)
            
        except:
            return 0
    
    def calculate_momentum_score(self, df):
        """Hitung skor momentum"""
        try:
            current = df.iloc[-1]
            score = 0
            
            # RSI momentum
            if 45 <= current['RSI'] <= 55:
                score += 20  # Sweet spot
            elif 40 <= current['RSI'] <= 60:
                score += 15
            elif 35 <= current['RSI'] <= 65:
                score += 10
            else:
                score += 5
                
            # MACD momentum
            if current['MACD'] > current['MACD_Signal'] and current['MACD_Histogram'] > 0:
                score += 20
            elif current['MACD'] > current['MACD_Signal']:
                score += 10
                
            # Price momentum
            if current['Momentum_5D'] > 2:
                score += 15
            elif current['Momentum_5D'] > 0:
                score += 10
                
            return min(score, 50)
            
        except:
            return 0
    
    def calculate_volume_score(self, df):
        """Hitung skor volume"""
        try:
            current = df.iloc[-1]
            score = 0
            
            # Volume ratio
            if current['Volume_Ratio'] > 2.0:
                score += 30
            elif current['Volume_Ratio'] > 1.5:
                score += 20
            elif current['Volume_Ratio'] > 1.0:
                score += 10
                
            # Volume trend
            recent_volume = df['Volume_Ratio'].tail(5).mean()
            if recent_volume > 1.2:
                score += 20
            elif recent_volume > 0.8:
                score += 10
                
            return min(score, 40)
            
        except:
            return 0
    
    def calculate_technical_score(self, df):
        """Hitung skor teknikal komprehensif"""
        try:
            trend_score = self.calculate_trend_score(df)
            momentum_score = self.calculate_momentum_score(df) 
            volume_score = self.calculate_volume_score(df)
            
            # Additional technical scores
            current = df.iloc[-1]
            
            # RSI score
            rsi_score = 0
            if 40 <= current['RSI'] <= 60:
                rsi_score = 15
            elif 30 <= current['RSI'] <= 70:
                rsi_score = 10
            else:
                rsi_score = 5
                
            # Bollinger Band score
            bb_score = 0
            if current['BB_Position'] < 30:
                bb_score = 10  # Near lower band - potential bounce
            elif current['BB_Position'] > 70:
                bb_score = 5   # Near upper band - caution
            else:
                bb_score = 8   # Middle - neutral
                
            # Calculate weighted total
            total_score = (
                trend_score * self.weights['trend'] +
                momentum_score * self.weights['momentum'] + 
                volume_score * self.weights['volume'] +
                rsi_score * self.weights['rsi'] +
                bb_score * self.weights['bollinger']
            )
            
            # Confidence calculation
            confidence = min(int(total_score * 1.2), 100)
            
            details = {
                'Trend': f"{trend_score}/50",
                'Momentum': f"{momentum_score}/50", 
                'Volume': f"{volume_score}/40",
                'RSI': f"{rsi_score}/15",
                'Bollinger': f"{bb_score}/10"
            }
            
            return min(int(total_score), 100), details, confidence
            
        except Exception as e:
            print(f"Scoring error: {str(e)}")
            return 0, {}, 0

# ============= VISUALIZATION FUNCTIONS =============
def create_technical_chart(df, ticker):
    """Buat chart teknikal yang komprehensif"""
    try:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{ticker} - Price Chart', 
                'Volume', 
                'RSI',
                'MACD'
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price data
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ), row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA9'], line=dict(color='orange', width=1), name='EMA9'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA21'], line=dict(color='red', width=1), name='EMA21'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='purple', width=2), name='EMA50'),
            row=1, col=1
        )
        
        # Volume
        colors = ['red' if row['Close'] < df['Close'].iloc[i-1] else 'green' 
                 for i, row in df.iterrows()]
        colors[0] = 'green'  # First day
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], line=dict(color='blue', width=1), name='RSI'),
            row=3, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="grey", row=3, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1), name='MACD'),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='red', width=1), name='Signal'),
            row=4, col=1
        )
        
        # Histogram
        colors_macd = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram', marker_color=colors_macd),
            row=4, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
        
    except Exception as e:
        print(f"Chart error: {str(e)}")
        return None

# ============= FUNGSI UTAMA YANG SUDAH DIPERBAIKI =============
def process_ticker_advanced(ticker, strategy, period):
    """Process ticker dengan sistem yang lebih robust"""
    try:
        df = fetch_data_with_retry(ticker, period)
        if df is None or len(df) < 50:
            return None
        
        price = float(df['Close'].iloc[-1])
        
        # Gunakan advanced scoring
        scorer = AdvancedScoring()
        score, details, confidence = scorer.calculate_technical_score(df)
        
        if score == 0:
            return None
        
        # Determine signal
        if score >= 80 and confidence >= 75:
            signal = "STRONG BUY"
            signal_class = "strong-buy"
        elif score >= 65 and confidence >= 60:
            signal = "BUY" 
            signal_class = "buy"
        elif score >= 50:
            signal = "WATCH"
            signal_class = "neutral"
        else:
            signal = "PASS"
            signal_class = "sell"
        
        # Calculate levels
        if signal in ["STRONG BUY", "BUY"]:
            entry_ideal = round(price * 0.98, 0)
            tp1 = round(entry_ideal * 1.08, 0)
            tp2 = round(entry_ideal * 1.15, 0)
            sl = round(entry_ideal * 0.94, 0)
        else:
            entry_ideal = tp1 = tp2 = sl = None
        
        return {
            "Ticker": ticker, 
            "Price": price, 
            "Score": score, 
            "Confidence": confidence,
            "Signal": signal,
            "SignalClass": signal_class,
            "EntryIdeal": entry_ideal,
            "TP1": tp1,
            "TP2": tp2, 
            "SL": sl,
            "Details": details,
            "VolumeRatio": df['Volume_Ratio'].iloc[-1],
            "RSI": df['RSI'].iloc[-1]
        }
        
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None

# ============= LOAD TICKERS YANG LEBIH BAIK =============
def load_tickers():
    """Load tickers dari file JSON atau default"""
    try:
        # Coba load dari file lokal
        with open("idx_stocks.json", "r") as f:
            data = json.load(f)
        tickers = data.get("tickers", [])
        
        # Validasi dan format tickers
        formatted_tickers = []
        for ticker in tickers:
            if not ticker.endswith(".JK"):
                ticker = f"{ticker}.JK"
            formatted_tickers.append(ticker)
            
        return formatted_tickers
        
    except FileNotFoundError:
        st.warning("File idx_stocks.json tidak ditemukan, menggunakan default tickers")
        # Fallback ke default tickers
        return [
            "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
            "BREN.JK", "BRPT.JK", "RATU.JK", "RAJA.JK", "UNVR.JK",
            "ICBP.JK", "INDF.JK", "ADRO.JK", "ANTM.JK", "PTBA.JK",
            "PGAS.JK", "AKRA.JK", "WSKT.JK", "EXCL.JK", "FREN.JK"
        ]
    except Exception as e:
        st.error(f"Error loading tickers: {str(e)}")
        return ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]

# ============= SISTEM BATCH SCAN YANG LEBIH EFISIEN =============
def batch_scan_advanced(tickers, strategy, period, limit, use_parallel=True):
    """Batch scan dengan performance yang lebih baik"""
    results = []
    
    if limit and limit < len(tickers):
        tickers = tickers[:limit]
    
    total = len(tickers)
    
    if total == 0:
        return pd.DataFrame()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if use_parallel and total > 10:
        # Parallel processing untuk banyak tickers
        with ThreadPoolExecutor(max_workers=Config.WORKERS) as executor:
            futures = {
                executor.submit(process_ticker_advanced, ticker, strategy, period): ticker 
                for ticker in tickers
            }
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                progress_bar.progress(completed / total)
                status_text.text(f"📊 Processing {completed}/{total} stocks...")
                
                result = future.result()
                if result:
                    results.append(result)
                
                # Small delay untuk avoid rate limiting
                time.sleep(0.1)
    else:
        # Sequential processing untuk sedikit tickers
        for i, ticker in enumerate(tickers):
            progress_bar.progress((i + 1) / total)
            status_text.text(f"📊 Processing {i+1}/{total}: {ticker}")
            
            result = process_ticker_advanced(ticker, strategy, period)
            if result:
                results.append(result)
            
            # Rate limiting
            time.sleep(0.3)
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        return pd.DataFrame()
    
    # Convert ke DataFrame dan sort
    df = pd.DataFrame(results)
    df = df.sort_values(["Score", "Confidence"], ascending=[False, False])
    
    return df

def show_stock_detail(stock_data):
    """Tampilkan detail saham - SUDAH DIPERBAIKI"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💰 Price", f"Rp {stock_data['Price']:,.0f}")
        st.metric("📊 Score", f"{stock_data['Score']}/100")
    
    with col2:
        st.metric("🎯 Confidence", f"{stock_data['Confidence']}%")
        st.metric("📈 Signal", stock_data['Signal'])
    
    with col3:
        if stock_data['EntryIdeal']:
            st.metric("🎯 Ideal Entry", f"Rp {stock_data['EntryIdeal']:,.0f}")
            st.metric("🛑 Stop Loss", f"Rp {stock_data['SL']:,.0f}")
    
    # Signal box - BAGIAN YANG SUDAH DIPERBAIKI
    signal_class = stock_data.get("SignalClass", "neutral")
    st.markdown(
        f'<div class="signal-box {signal_class}">'
        f'{stock_data["Signal"]} - Confidence: {stock_data["Confidence"]}%</div>', 
        unsafe_allow_html=True
    )
    
    # Trading levels
    if stock_data['EntryIdeal']:
        st.markdown("### 🎯 Trading Levels")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.success(f"**Entry**\nRp {stock_data['EntryIdeal']:,.0f}")
        col2.info(f"**TP1 (+8%)**\nRp {stock_data['TP1']:,.0f}") 
        col3.info(f"**TP2 (+15%)**\nRp {stock_data['TP2']:,.0f}")
        col4.error(f"**SL (-6%)**\nRp {stock_data['SL']:,.0f}")
    
    # Technical details
    st.markdown("### 📊 Technical Breakdown")
    if 'Details' in stock_data:
        for indicator, value in stock_data['Details'].items():
            st.write(f"**{indicator}:** {value}")

# ============= FUNGSI UTAMA STREAMLIT =============
def main():
    # Initialize database
    init_db()
    
    # Header
    st.markdown('<div class="big-title">🚀 IDX Power Screener v4.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Advanced Technical Analysis | Real-time Alerts | Portfolio Management</div>', unsafe_allow_html=True)
    
    # Load tickers
    tickers = load_tickers()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Market status
        market_status, is_open = check_idx_market_status()
        if is_open:
            st.success(market_status)
        else:
            st.warning(market_status)
        
        jkt_time = get_jakarta_time()
        st.info(f"🕐 Jakarta: {jkt_time.strftime('%H:%M:%S WIB')}")
        
        st.markdown("---")
        
        # Menu
        menu = st.radio("📋 Menu", [
            "1️⃣ Dashboard", 
            "2️⃣ Full Screener", 
            "3️⃣ Single Analysis"
        ])
        
        st.markdown("---")
        
        # Settings berdasarkan menu
        if menu not in ["1️⃣ Dashboard"]:
            period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
            
            if menu in ["2️⃣ Full Screener"]:
                limit = st.slider("Max Tickers", 10, len(tickers), min(100, len(tickers)), step=10)
                min_score = st.slider("Min Score", 50, 100, 65, step=5)
                min_confidence = st.slider("Min Confidence", 40, 100, 60, step=5)
                use_parallel = st.checkbox("⚡ Fast Mode", value=True)
        
        # Position calculator
        with st.expander("💰 Position Calculator"):
            account = st.number_input("Account Size (Rp)", value=100_000_000, step=10_000_000, format="%d")
            risk_pct = st.slider("Risk per Trade (%)", 1.0, 5.0, 2.0, 0.5)
            
            st.caption(f"💵 Risk per trade: Rp {account * risk_pct / 100:,.0f}")
            st.caption("📊 Recommended: 2% per trade")
        
        st.markdown("---")
        st.caption("💡 IDX Power Screener v4.0 - Enhanced Edition")
    
    # Menu handlers
    if menu == "1️⃣ Dashboard":
        show_dashboard()
    elif menu == "2️⃣ Full Screener":
        show_full_screener(tickers, period, limit, min_score, min_confidence, use_parallel)
    elif menu == "3️⃣ Single Analysis":
        show_single_analysis(tickers, period)

def show_dashboard():
    st.markdown("## 📊 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">🔄<br>Market Status</div>', unsafe_allow_html=True)
        market_status, is_open = check_idx_market_status()
        if is_open:
            st.success("OPEN")
        else:
            st.warning("CLOSED")
    
    with col2:
        st.markdown('<div class="metric-card">📈<br>Total Stocks</div>', unsafe_allow_html=True)
        st.metric("Tracked", len(load_tickers()))
    
    with col3:
        st.markdown('<div class="metric-card">🎯<br>Active Alerts</div>', unsafe_allow_html=True)
        st.metric("Alerts", "0")
    
    with col4:
        st.markdown('<div class="metric-card">💰<br>Portfolio</div>', unsafe_allow_html=True)
        st.metric("Value", "Rp 0")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Run Full Scan", use_container_width=True):
            st.session_state.run_scan = True
    
    with col2:
        if st.button("📊 Check Portfolio", use_container_width=True):
            st.session_state.show_portfolio = True
    
    with col3:
        if st.button("🔔 View Alerts", use_container_width=True):
            st.session_state.show_alerts = True
    
    # Recent signals
    st.markdown("### 📈 Recent Signals")
    st.info("No recent signals. Run a scan to see recommendations.")

def show_full_screener(tickers, period, limit, min_score, min_confidence, use_parallel):
    st.markdown("## 🚀 Full Market Screener")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stocks", limit)
    col2.metric("Min Score", min_score)
    col3.metric("Min Confidence", f"{min_confidence}%")
    col4.metric("Mode", "Parallel" if use_parallel else "Sequential")
    
    if st.button("🚀 Run Comprehensive Scan", type="primary", use_container_width=True):
        with st.spinner(f"Scanning {limit} stocks with advanced analysis..."):
            results_df = batch_scan_advanced(tickers, "Full", period, limit, use_parallel)
        
        if results_df.empty:
            st.warning("No qualifying stocks found. Try adjusting your filters.")
        else:
            # Filter results
            filtered_df = results_df[
                (results_df["Score"] >= min_score) & 
                (results_df["Confidence"] >= min_confidence)
            ]
            
            if filtered_df.empty:
                st.warning(f"No stocks meet criteria: Score>={min_score}, Confidence>={min_confidence}")
            else:
                st.success(f"🎯 Found {len(filtered_df)} quality opportunities!")
                
                # Display results
                display_results = filtered_df[[
                    "Ticker", "Price", "Score", "Confidence", "Signal", 
                    "EntryIdeal", "TP1", "TP2", "SL"
                ]].copy()
                
                st.dataframe(
                    display_results.style.format({
                        "Price": "{:,.0f}",
                        "EntryIdeal": "{:,.0f}", 
                        "TP1": "{:,.0f}",
                        "TP2": "{:,.0f}",
                        "SL": "{:,.0f}"
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Detailed view
                st.markdown("### 📋 Detailed Analysis")
                for _, row in filtered_df.head(10).iterrows():
                    with st.expander(f"{row['Ticker']} - Score: {row['Score']} | Confidence: {row['Confidence']}%"):
                        show_stock_detail(row)

def show_single_analysis(tickers, period):
    st.markdown("## 📈 Single Stock Analysis")
    
    selected_ticker = st.selectbox("Select Stock", tickers)
    
    if st.button("🔍 Analyze Stock", type="primary"):
        with st.spinner(f"Analyzing {selected_ticker}..."):
            result = process_ticker_advanced(selected_ticker, "Single", period)
            
            if result is None:
                st.error(f"Failed to analyze {selected_ticker}. Please try again.")
            else:
                show_stock_detail(result)
                
                # Technical Chart
                st.markdown("### 📊 Technical Chart")
                df = fetch_data_with_retry(selected_ticker, period)
                if df is not None:
                    chart = create_technical_chart(df.tail(100), selected_ticker)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)

# ============= RUN APPLICATION =============
if __name__ == "__main__":
    main()
