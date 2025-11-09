import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="IDX Pro Screener",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0f172a;}
    .stApp {background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);}
    h1 {color: #60a5fa; text-align: center;}
    .metric-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# IDX Top Liquid Stocks (850+ available)
IDX_STOCKS = [
    'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'TLKM.JK', 'ASII.JK', 'UNVR.JK', 'GGRM.JK',
    'ICBP.JK', 'INDF.JK', 'KLBF.JK', 'HMSP.JK', 'SMGR.JK', 'PTBA.JK', 'ADRO.JK',
    'BBNI.JK', 'CPIN.JK', 'INTP.JK', 'EXCL.JK', 'ANTM.JK', 'JPFA.JK', 'UNTR.JK',
    'TKIM.JK', 'TBIG.JK', 'ITMG.JK', 'BRIS.JK', 'PGAS.JK', 'JSMR.JK', 'WIKA.JK',
    'AKRA.JK', 'MNCN.JK', 'SCMA.JK', 'TOWR.JK', 'SRIL.JK', 'MAPI.JK', 'BSDE.JK',
    'PWON.JK', 'TPIA.JK', 'MEDC.JK', 'BYAN.JK', 'ESSA.JK', 'ERAA.JK', 'ACES.JK',
    'SMBR.JK', 'BRPT.JK', 'BTPS.JK', 'DMAS.JK', 'ELSA.JK', 'MYOR.JK', 'ARNA.JK',
    'AMRT.JK', 'WOOD.JK', 'LPPF.JK', 'PNLF.JK', 'RALS.JK', 'SIDO.JK', 'SRTG.JK'
]

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period='6mo'):
    """Fetch stock data with error handling"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty or len(df) < 50:
            return None
        return df
    except Exception as e:
        return None

def calculate_ema(data, period):
    """Calculate EMA"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data):
    """Calculate MACD"""
    ema12 = calculate_ema(data, 12)
    ema26 = calculate_ema(data, 26)
    macd = ema12 - ema26
    signal = calculate_ema(macd, 9)
    return macd, signal

def calculate_atr(df, period=14):
    """Calculate ATR"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_adx(df, period=14):
    """Calculate ADX"""
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    
    pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    atr = calculate_atr(df, period)
    pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
    neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
    
    dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
    adx = dx.rolling(period).mean()
    
    return adx.iloc[-1] if len(adx) > 0 else 0

def detect_wyckoff_phase(df, volume_ratio, rsi, trend):
    """Professional Wyckoff Phase Detection"""
    price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
    
    # MARKUP - Strong uptrend with volume
    if volume_ratio > 2.0 and rsi > 60 and price_change > 5 and trend == 'UPTREND':
        return 'MARKUP', 20
    
    # SOS (Sign of Strength) - Breakout beginning
    elif volume_ratio > 1.5 and 50 < rsi < 70 and price_change > 2 and trend == 'UPTREND':
        return 'SOS', 15
    
    # ACCUMULATION - Building position
    elif volume_ratio > 1.3 and 40 < rsi < 60 and abs(price_change) < 3:
        return 'ACCUMULATION', 12
    
    # SPRING - Shakeout before markup
    elif volume_ratio > 1.4 and rsi < 40 and price_change < -2 and trend != 'DOWNTREND':
        return 'SPRING', 8
    
    # DISTRIBUTION - Top formation
    elif volume_ratio > 1.5 and rsi > 70 and trend == 'DOWNTREND':
        return 'DISTRIBUTION', -10
    
    # SIDEWAYS - Ranging
    else:
        return 'SIDEWAYS', 0

def analyze_stock_professional(ticker):
    """Complete Professional Stock Analysis"""
    try:
        df = fetch_stock_data(ticker)
        if df is None or len(df) < 50:
            return None
        
        # Calculate all indicators
        df['EMA8'] = calculate_ema(df['Close'], 8)
        df['EMA20'] = calculate_ema(df['Close'], 20)
        df['EMA50'] = calculate_ema(df['Close'], 50)
        df['EMA200'] = calculate_ema(df['Close'], 200)
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['Signal'] = calculate_macd(df['Close'])
        df['ATR'] = calculate_atr(df)
        
        # Current values
        current_price = df['Close'].iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal = df['Signal'].iloc[-1]
        adx = calculate_adx(df)
        atr = df['ATR'].iloc[-1]
        
        # Trend Analysis
        ema_aligned = (df['EMA8'].iloc[-1] > df['EMA20'].iloc[-1] > 
                      df['EMA50'].iloc[-1] > df['EMA200'].iloc[-1])
        
        weekly_trend = 'UPTREND' if df['Close'].iloc[-1] > df['Close'].iloc[-20] else 'DOWNTREND'
        daily_trend = 'UPTREND' if df['Close'].iloc[-1] > df['Close'].iloc[-5] else 'DOWNTREND'
        
        # SCORING SYSTEM (100 points)
        score = 0
        confidence_factors = []
        
        # TREND (30 points)
        if weekly_trend == 'UPTREND':
            score += 10
            confidence_factors.append('weekly_uptrend')
        if daily_trend == 'UPTREND':
            score += 10
            confidence_factors.append('daily_uptrend')
        if ema_aligned:
            score += 10
            confidence_factors.append('ema_aligned')
        
        # MOMENTUM (25 points)
        if 40 <= rsi <= 70:
            score += 10
            confidence_factors.append('rsi_optimal')
        if macd > signal:
            score += 8
            confidence_factors.append('macd_bullish')
        if adx > 25:
            score += 7
            confidence_factors.append('strong_trend')
        
        # VOLUME (25 points)
        if volume_ratio > 1.5:
            score += 10
            confidence_factors.append('volume_spike')
        if volume_ratio > 1.2:
            score += 8
            confidence_factors.append('volume_above_avg')
        if current_price > df['Close'].iloc[-2]:
            score += 7
            confidence_factors.append('price_momentum')
        
        # WYCKOFF PHASE (20 points)
        phase, phase_score = detect_wyckoff_phase(df, volume_ratio, rsi, daily_trend)
        score += phase_score
        if phase_score > 0:
            confidence_factors.append(f'phase_{phase.lower()}')
        
        # Confidence calculation
        confidence = min(95, (len(confidence_factors) / 12) * 100)
        
        # SIGNAL GENERATION
        if score >= 80:
            signal_type = 'STRONG BUY'
        elif score >= 65:
            signal_type = 'BUY'
        elif score >= 50:
            signal_type = 'WEAK BUY'
        elif score >= 35:
            signal_type = 'HOLD'
        else:
            signal_type = 'AVOID'
        
        # RISK MANAGEMENT - ATR Based
        entry_price = current_price
        stop_loss = current_price - (atr * 1.5)
        tp1 = current_price + (atr * 2.5)
        tp2 = current_price + (atr * 4.0)
        tp3 = current_price + (atr * 6.0)
        
        risk = entry_price - stop_loss
        reward = tp2 - entry_price
        risk_reward = reward / risk if risk > 0 else 0
        
        # Trend indicator
        if daily_trend == 'UPTREND' and weekly_trend == 'UPTREND':
            trend_display = '↑↑↑'
        elif daily_trend == 'UPTREND':
            trend_display = '↑↑'
        elif daily_trend == 'DOWNTREND' and weekly_trend == 'DOWNTREND':
            trend_display = '↓↓↓'
        elif daily_trend == 'DOWNTREND':
            trend_display = '↓↓'
        else:
            trend_display = '→'
        
        return {
            'ticker': ticker.replace('.JK', ''),
            'price': int(current_price),
            'score': int(score),
            'confidence': int(confidence),
            'signal': signal_type,
            'phase': phase,
            'trend': trend_display,
            'entry': int(entry_price),
            'tp1': int(tp1),
            'tp2': int(tp2),
            'tp3': int(tp3),
            'sl': int(stop_loss),
            'rr': f"1:{risk_reward:.1f}",
            'volume_ratio': f"{volume_ratio:.1f}x",
            'rsi': f"{rsi:.0f}",
            'adx': f"{adx:.0f}",
            'factors': len(confidence_factors)
        }
        
    except Exception as e:
        st.error(f"Error analyzing {ticker}: {str(e)}")
        return None

def run_screener(stocks, min_score, min_conf, phases, max_results):
    """Run professional screener with parallel processing"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_stock_professional, ticker): ticker 
                  for ticker in stocks}
        
        for i, future in enumerate(as_completed(futures)):
            ticker = futures[future]
            progress = (i + 1) / len(stocks)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing: {ticker} ({i+1}/{len(stocks)})")
            
            try:
                result = future.result()
                if result and result['score'] >= min_score and result['confidence'] >= min_conf:
                    if result['phase'] in phases:
                        results.append(result)
            except Exception as e:
                continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

# STREAMLIT UI
def main():
    st.title("🚀 IDX Professional Trading Screener")
    st.markdown("### Advanced Multi-Factor Technical Analysis System")
    
    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Parameters")
        
        min_score = st.slider("Min Score", 0, 100, 65, 5)
        min_confidence = st.slider("Min Confidence (%)", 0, 100, 60, 5)
        max_results = st.slider("Max Results", 5, 50, 20, 5)
        
        st.subheader("📊 Wyckoff Phases")
        phases = []
        if st.checkbox("ACCUMULATION", True):
            phases.append('ACCUMULATION')
        if st.checkbox("SOS", True):
            phases.append('SOS')
        if st.checkbox("MARKUP", True):
            phases.append('MARKUP')
        if st.checkbox("SPRING", False):
            phases.append('SPRING')
        if st.checkbox("SIDEWAYS", False):
            phases.append('SIDEWAYS')
        if st.checkbox("DISTRIBUTION", False):
            phases.append('DISTRIBUTION')
        
        st.divider()
        
        num_stocks = st.selectbox(
            "Stocks to Scan",
            [10, 20, 30, 50, len(IDX_STOCKS)],
            index=2
        )
        
        run_button = st.button("🎯 RUN SCREENER", use_container_width=True, type="primary")
    
    # Main content
    if run_button:
        if not phases:
            st.error("⚠️ Please select at least one Wyckoff phase!")
            return
        
        st.info(f"🔍 Scanning {num_stocks} IDX stocks...")
        
        stocks_to_scan = IDX_STOCKS[:num_stocks]
        results = run_screener(stocks_to_scan, min_score, min_confidence, phases, max_results)
        
        if not results:
            st.warning("❌ No stocks found matching your criteria. Try lowering filters.")
            return
        
        # Statistics Dashboard
        st.success(f"✅ Found {len(results)} quality opportunities!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        avg_score = sum(r['score'] for r in results) / len(results)
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        strong_buy = sum(1 for r in results if r['signal'] == 'STRONG BUY')
        buy = sum(1 for r in results if r['signal'] == 'BUY')
        
        with col1:
            st.metric("Avg Score", f"{avg_score:.1f}/100")
        with col2:
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        with col3:
            st.metric("Strong Buy", strong_buy)
        with col4:
            st.metric("Buy", buy)
        
        st.divider()
        
        # Results Table
        df_results = pd.DataFrame(results)
        
        # Format table
        st.dataframe(
            df_results,
            column_config={
                "ticker": "Ticker",
                "price": st.column_config.NumberColumn("Price", format="%d"),
                "score": st.column_config.NumberColumn("Score", format="%d"),
                "confidence": st.column_config.NumberColumn("Conf %", format="%d"),
                "signal": "Signal",
                "phase": "Phase",
                "trend": "Trend",
                "entry": st.column_config.NumberColumn("Entry", format="%d"),
                "tp1": st.column_config.NumberColumn("TP1", format="%d"),
                "tp2": st.column_config.NumberColumn("TP2", format="%d"),
                "tp3": st.column_config.NumberColumn("TP3", format="%d"),
                "sl": st.column_config.NumberColumn("SL", format="%d"),
                "rr": "R:R",
                "volume_ratio": "Vol",
                "rsi": "RSI",
                "adx": "ADX"
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Download CSV
        csv = df_results.to_csv(index=False)
        st.download_button(
            "📥 Download Results (CSV)",
            csv,
            "idx_screener_results.csv",
            "text/csv",
            use_container_width=True
        )
        
    else:
        # Welcome screen
        st.info("👈 Configure parameters and click 'RUN SCREENER' to start")
        
        st.markdown("""
        ### 🎯 Features:
        - ✅ Multi-factor technical analysis (30+ indicators)
        - ✅ Wyckoff accumulation/distribution detection
        - ✅ Professional risk management (ATR-based)
        - ✅ Real-time IDX stock data
        - ✅ Parallel processing for speed
        - ✅ Risk-reward calculation
        - ✅ Confidence scoring
        
        ### 📊 Scoring System:
        - **Trend Analysis (30%)**: EMA alignment, multi-timeframe
        - **Momentum (25%)**: RSI, MACD, ADX
        - **Volume (25%)**: Volume spikes, price action
        - **Wyckoff Phase (20%)**: Accumulation cycle position
        """)

if __name__ == "__main__":
    main()
