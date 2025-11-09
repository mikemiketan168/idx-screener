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
import os

st.set_page_config(page_title="IDX Power Screener v4.0", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.big-title {font-size:2.5rem;font-weight:800;color:#1e40af}
.subtitle {font-size:1.1rem;color:#64748b;margin-bottom:2rem}
.signal-box {padding:1rem;border-radius:0.5rem;margin:1rem 0;font-weight:700;text-align:center}
.strong-buy {background:#10b981;color:white}
.buy {background:#34d399;color:white}
.neutral {background:#fbbf24;color:white}
.sell {background:#ef4444;color:white}
.quality-badge {display:inline-block;padding:0.3rem 0.8rem;border-radius:0.3rem;font-weight:700;margin:0.2rem}
.grade-a {background:#10b981;color:white}
.grade-b {background:#3b82f6;color:white}
.grade-c {background:#f59e0b;color:white}
.grade-d {background:#ef4444;color:white}
</style>
""", unsafe_allow_html=True)

# ============= DATABASE SETUP =============
def init_db():
    """Initialize database with error handling"""
    try:
        conn = sqlite3.connect('screener_tracking.db', check_same_thread=False)
        c = conn.cursor()
        
        # Create recommendations table
        c.execute('''CREATE TABLE IF NOT EXISTS recommendations
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      date TEXT, ticker TEXT, strategy TEXT, score INTEGER,
                      confidence INTEGER, quality_grade TEXT, entry_price REAL, 
                      current_price REAL, signal TEXT, status TEXT DEFAULT 'ACTIVE', 
                      result TEXT, profit_pct REAL, exit_price REAL, exit_date TEXT, 
                      notes TEXT, position_size TEXT DEFAULT '3/3',
                      market_context TEXT, risk_level TEXT)''')
        
        # Create watchlist table
        c.execute('''CREATE TABLE IF NOT EXISTS watchlist
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      date_added TEXT, ticker TEXT, strategy TEXT,
                      score INTEGER, confidence INTEGER, quality_grade TEXT,
                      target_entry REAL, current_price REAL, notes TEXT, 
                      status TEXT DEFAULT 'WATCHING')''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False

# ============= HELPER FUNCTIONS =============
def get_jakarta_time():
    """Get current Jakarta time (UTC+7)"""
    jkt_tz = timezone(timedelta(hours=7))
    return datetime.now(jkt_tz)

def check_idx_market_status():
    """Check if IDX market is open"""
    jkt_time = get_jakarta_time()
    hour = jkt_time.hour
    minute = jkt_time.minute
    weekday = jkt_time.weekday()
    
    if weekday >= 5:
        return "🔴 CLOSED - Weekend", False
    
    if hour < 9:
        open_in_minutes = (9 - hour) * 60 - minute
        return f"⏰ Opens in {open_in_minutes//60}h {open_in_minutes%60}m", False
    elif hour >= 16 or (hour == 16 and minute >= 15):
        return "🔴 CLOSED - After hours", False
    elif 12 <= hour < 13:
        return "🟡 LUNCH BREAK", False
    else:
        return "🟢 MARKET OPEN", True

def display_quality_badge(grade):
    """Display quality grade badge"""
    colors = {"A": "grade-a", "B": "grade-b", "C": "grade-c", "D": "grade-d"}
    labels = {"A": "PREMIUM", "B": "GOOD", "C": "OK", "D": "POOR"}
    return f'<span class="quality-badge {colors.get(grade, "grade-d")}">Grade {grade} - {labels.get(grade, "POOR")}</span>'

def display_risk_badge(risk):
    """Display risk level badge"""
    if risk == "LOW":
        return "🟢 LOW RISK"
    elif risk == "MEDIUM":
        return "🟡 MEDIUM RISK"
    else:
        return "🔴 HIGH RISK"

def load_tickers():
    """Load tickers from JSON file or return defaults"""
    try:
        if os.path.exists("idx_stocks.json"):
            with open("idx_stocks.json", "r") as f:
                data = json.load(f)
            tickers = data.get("tickers", [])
            return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
        else:
            # Default tickers if file doesn't exist
            return ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
                    "BREN.JK", "BRPT.JK", "GOTO.JK", "AMMN.JK", "EMTK.JK"]
    except Exception as e:
        st.warning(f"Could not load idx_stocks.json: {str(e)}. Using defaults.")
        return ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]

# ============= DATA FETCHING =============
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, period="6mo"):
    """Fetch stock data from Yahoo Finance with error handling"""
    try:
        end = int(datetime.now().timestamp())
        days = {"5d":5,"1mo":30,"3mo":90,"6mo":180,"1y":365}.get(period,180)
        start = end - (days*86400)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {"period1": start, "period2": end, "interval": "1d"}
        
        r = requests.get(url, params=params, headers=headers, timeout=10)
        
        if r.status_code != 200:
            return None
            
        data = r.json()
        
        if 'chart' not in data or 'result' not in data['chart']:
            return None
            
        result = data['chart']['result'][0]
        q = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': q['open'],
            'High': q['high'],
            'Low': q['low'],
            'Close': q['close'],
            'Volume': q['volume']
        }, index=pd.to_datetime(result['timestamp'], unit='s'))
        
        df = df.dropna()
        
        if len(df) < 50:
            return None
        
        # Calculate indicators
        df['EMA9'] = df['Close'].ewm(span=9).mean()
        df['EMA21'] = df['Close'].ewm(span=21).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean() if len(df) >= 200 else df['Close'].ewm(span=len(df)).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume
        df['VOL_SMA20'] = df['Volume'].rolling(window=20).mean()
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA20']
        
        # Momentum
        df['MOM_5D'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['MOM_10D'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        df['MOM_20D'] = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100
        
        return df
        
    except Exception as e:
        return None

def fetch_data_with_retry(ticker, period="6mo", max_retries=2):
    """Fetch with retry logic"""
    for attempt in range(max_retries):
        try:
            time.sleep(0.2 * attempt)
            return fetch_data(ticker, period)
        except:
            if attempt == max_retries - 1:
                return None
    return None

# ============= VALIDATION =============
def validate_quality_gate(df):
    """Multi-layer quality validation"""
    try:
        r = df.iloc[-1]
        issues = []
        score = 100
        
        # Gate 1: Downtrend check
        if r['Close'] < r['EMA50'] and r['EMA50'] < r['EMA200']:
            issues.append("❌ Major downtrend")
            score -= 40
        
        if r['EMA9'] < r['EMA21'] < r['EMA50']:
            issues.append("❌ Death cross")
            score -= 30
        
        # Gate 2: Momentum
        mom_20d = df['MOM_20D'].iloc[-1]
        if pd.notna(mom_20d) and mom_20d < -10:
            issues.append(f"❌ Severe downward momentum")
            score -= 25
        
        # Gate 3: Overbought
        if r['RSI'] > 75:
            issues.append(f"❌ Overbought (RSI:{r['RSI']:.1f})")
            score -= 30
        
        # Grade assignment
        if score >= 80:
            grade = "A"
        elif score >= 65:
            grade = "B"
        elif score >= 50:
            grade = "C"
        else:
            grade = "D"
        
        # Reject if Grade D or critical issues
        if grade == "D" or len(issues) >= 2:
            return False, "D", issues, score
        
        return True, grade, issues, score
        
    except Exception as e:
        return False, "D", [f"Error: {str(e)}"], 0

# ============= SCORING =============
def score_advanced_v4(df):
    """Advanced multi-factor scoring"""
    try:
        # Quality gate first
        passed, grade, gate_issues, gate_score = validate_quality_gate(df)
        
        if not passed:
            return 0, {"⛔ REJECTED": gate_issues}, 0, "D", "HIGH"
        
        r = df.iloc[-1]
        score = 0
        details = {}
        confidence = 0
        risk_points = 0
        
        # Factor 1: Trend (35 pts)
        if r['Close'] > r['EMA9'] > r['EMA21'] > r['EMA50'] > r['EMA200']:
            score += 35
            details['Trend'] = '🟢 PERFECT (+35)'
            confidence += 35
        elif r['Close'] > r['EMA9'] > r['EMA21']:
            score += 18
            details['Trend'] = '🟡 Short-term (+18)'
            confidence += 20
        
        # Factor 2: Momentum (25 pts)
        mom_5d = df['MOM_5D'].iloc[-1]
        if pd.notna(mom_5d):
            if 2 <= mom_5d <= 8:
                score += 25
                details['Momentum'] = f'🟢 IDEAL {mom_5d:.1f}% (+25)'
                confidence += 25
            elif 0 < mom_5d <= 12:
                score += 15
                details['Momentum'] = f'🟢 Good {mom_5d:.1f}% (+15)'
                confidence += 15
        
        # Factor 3: RSI (20 pts)
        if 45 <= r['RSI'] <= 60:
            score += 20
            details['RSI'] = f'🟢 SWEET {r["RSI"]:.1f} (+20)'
            confidence += 20
        elif 40 <= r['RSI'] <= 65:
            score += 15
            details['RSI'] = f'🟢 Good {r["RSI"]:.1f} (+15)'
            confidence += 15
        
        # Factor 4: Volume (20 pts)
        vol_ratio = df['VOL_RATIO'].tail(5).mean()
        if pd.notna(vol_ratio):
            if vol_ratio > 1.5:
                score += 20
                details['Volume'] = f'🟢 STRONG {vol_ratio:.2f}x (+20)'
                confidence += 20
            elif vol_ratio > 1.0:
                score += 10
                details['Volume'] = f'🟡 Normal {vol_ratio:.2f}x (+10)'
                confidence += 10
        
        # Confidence calculation
        confidence = min(int(confidence * 0.85), 100)
        
        # Risk level
        risk_level = "LOW" if risk_points < 15 else "MEDIUM" if risk_points < 35 else "HIGH"
        
        # Minimum threshold
        if score < 35 or confidence < 45:
            return 0, details, 0, grade, "HIGH"
        
        return score, details, confidence, grade, risk_level
        
    except Exception as e:
        return 0, {"❌ Error": str(e)}, 0, "D", "HIGH"

# ============= SIGNAL GENERATION =============
def get_signal_levels_v4(score, price, confidence, grade, risk_level):
    """Generate trading signals"""
    if score >= 85 and confidence >= 75 and grade == "A":
        signal = "STRONG BUY"
        signal_class = "strong-buy"
        trend = "🟢 Excellent"
        entry_ideal = round(price * 0.98, 0)
        entry_aggr = round(price, 0)
    elif score >= 70 and confidence >= 60:
        signal = "BUY"
        signal_class = "buy"
        trend = "🟢 Good"
        entry_ideal = round(price * 0.97, 0)
        entry_aggr = round(price * 0.99, 0)
    elif score >= 55:
        signal = "WATCH"
        signal_class = "neutral"
        trend = "🟡 Monitor"
        entry_ideal = round(price * 0.95, 0)
        entry_aggr = None
    else:
        signal = "PASS"
        signal_class = "sell"
        trend = "⚪ Pass"
        entry_ideal = None
        entry_aggr = None
    
    if entry_ideal:
        tp1 = round(entry_ideal * 1.08, 0)
        tp2 = round(entry_ideal * 1.15, 0)
        sl = round(entry_ideal * 0.94, 0)
    else:
        tp1 = tp2 = sl = None
    
    return {
        "signal": signal,
        "signal_class": signal_class,
        "trend": trend,
        "ideal": {"entry": entry_ideal, "tp1": tp1, "tp2": tp2, "sl": sl},
        "aggr": {"entry": entry_aggr, "tp1": tp1, "tp2": tp2, "sl": sl}
    }

# ============= PROCESSING =============
def process_ticker_v4(ticker, strategy, period):
    """Process single ticker"""
    try:
        df = fetch_data_with_retry(ticker, period)
        if df is None or len(df) < 50:
            return None
        
        price = float(df['Close'].iloc[-1])
        score, details, confidence, grade, risk_level = score_advanced_v4(df)
        
        if score == 0:
            return None
        
        levels = get_signal_levels_v4(score, price, confidence, grade, risk_level)
        
        return {
            "Ticker": ticker,
            "Price": price,
            "Score": score,
            "Confidence": confidence,
            "Grade": grade,
            "Risk": risk_level,
            "Signal": levels["signal"],
            "Trend": levels["trend"],
            "EntryIdeal": levels["ideal"]["entry"],
            "TP1": levels["ideal"]["tp1"],
            "SL": levels["ideal"]["sl"],
            "Details": details
        }
    except Exception as e:
        return None

def batch_scan_v4(tickers, strategy, period, limit, min_score=65, min_conf=60):
    """Batch scan with progress tracking"""
    results = []
    if limit and limit < len(tickers):
        tickers = tickers[:limit]
    
    progress = st.progress(0)
    status = st.empty()
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / total)
        status.text(f"📊 Scanning {i+1}/{total}: {ticker}")
        
        result = process_ticker_v4(ticker, strategy, period)
        if result and result['Score'] >= min_score and result['Confidence'] >= min_conf:
            results.append(result)
        
        time.sleep(0.3)
    
    progress.empty()
    status.empty()
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results).sort_values(
        ["Grade", "Score", "Confidence"], 
        ascending=[True, False, False]
    )
    return df

# ============= MAIN APP =============
try:
    init_db()
except:
    st.warning("Database initialization skipped (read-only mode)")

st.markdown('<div class="big-title">🎯 IDX Power Screener v4.0</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Accuracy-First Edition</div>', unsafe_allow_html=True)

tickers = load_tickers()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    market_status, is_open = check_idx_market_status()
    if is_open:
        st.success(market_status)
    else:
        st.warning(market_status)
    
    jkt_time = get_jakarta_time()
    st.info(f"🕐 Jakarta: {jkt_time.strftime('%H:%M WIB')}")
    
    st.markdown("---")
    
    menu = st.radio("📋 Menu", [
        "🎯 Elite Screener",
        "📊 Full Screener",
        "🔍 Single Stock"
    ])
    
    st.markdown("---")
    
    period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
    limit = st.slider("Max Stocks", 10, min(100, len(tickers)), 30, step=10)
    min_score = st.slider("Min Score", 50, 100, 70, step=5)
    min_confidence = st.slider("Min Confidence", 40, 100, 60, step=5)
    
    st.markdown("---")
    st.caption("🎯 v4.0 - Accuracy First")

# ============= MENU HANDLERS =============

if "Elite" in menu:
    st.markdown("### 🏆 Elite Screener - Grade A Only")
    
    if st.button("🚀 Scan Elite Setups", type="primary"):
        with st.spinner(f"Scanning {limit} stocks..."):
            df = batch_scan_v4(tickers, "Elite", period, limit, min_score, min_confidence)
        
        if not df.empty:
            df = df[df['Grade'] == 'A']
        
        if df.empty:
            st.warning("⚠️ No Grade A setups found")
            st.info("Try: Lower min score, increase stocks scanned, or include Grade B")
        else:
            st.success(f"🏆 Found {len(df)} elite opportunities!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Elite Setups", len(df))
            col2.metric("Avg Score", f"{df['Score'].mean():.1f}")
            col3.metric("Avg Conf", f"{df['Confidence'].mean():.1f}%")
            
            st.dataframe(df[['Ticker', 'Price', 'Score', 'Confidence', 'Signal', 'EntryIdeal', 'TP1', 'SL']], 
                        use_container_width=True)

elif "Full" in menu:
    st.markdown("### 📊 Full Screener")
    
    if st.button("🚀 Run Full Scan", type="primary"):
        with st.spinner(f"Scanning {limit} stocks..."):
            df = batch_scan_v4(tickers, "Full", period, limit, min_score, min_confidence)
        
        if df.empty:
            st.warning("⚠️ No stocks found matching criteria")
        else:
            st.success(f"✅ Found {len(df)} opportunities!")
            st.dataframe(df[['Ticker', 'Price', 'Score', 'Confidence', 'Grade', 'Signal', 'EntryIdeal']], 
                        use_container_width=True)

else:  # Single Stock
    st.markdown("### 🔍 Single Stock Analysis")
    
    selected = st.selectbox("Select Stock", tickers)
    
    if st.button("🔍 Analyze", type="primary"):
        with st.spinner(f"Analyzing {selected}..."):
            result = process_ticker_v4(selected, "Single", period)
        
        if result is None:
            st.error("❌ Analysis failed or stock rejected")
        else:
            st.markdown(f"## {result['Ticker']}")
            st.markdown(display_quality_badge(result['Grade']), unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Price", f"Rp {result['Price']:,.0f}")
            col2.metric("Score", f"{result['Score']}/100")
            col3.metric("Confidence", f"{result['Confidence']}%")
            
            if result['EntryIdeal']:
                st.success(f"""
                **Entry:** Rp {result['EntryIdeal']:,.0f}
                **TP1:** Rp {result['TP1']:,.0f}
                **SL:** Rp {result['SL']:,.0f}
                """)
            
            for k, v in result['Details'].items():
                st.info(f"**{k}:** {v}")

st.markdown("---")
st.caption("🎯 IDX Power Screener v4.0 | Educational purposes only")
