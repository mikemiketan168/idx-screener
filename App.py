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
</style>
""", unsafe_allow_html=True)

# ============= DATABASE =============
def init_db():
    try:
        conn = sqlite3.connect('screener_tracking.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS recommendations
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      date TEXT, ticker TEXT, strategy TEXT, score INTEGER,
                      confidence INTEGER, quality_grade TEXT, entry_price REAL, 
                      current_price REAL, signal TEXT, status TEXT DEFAULT 'ACTIVE', 
                      result TEXT, profit_pct REAL, exit_price REAL, exit_date TEXT, 
                      notes TEXT, risk_level TEXT)''')
        conn.commit()
        conn.close()
        return True
    except:
        return False

# ============= HELPERS =============
def get_jakarta_time():
    jkt_tz = timezone(timedelta(hours=7))
    return datetime.now(jkt_tz)

def check_idx_market_status():
    jkt_time = get_jakarta_time()
    hour = jkt_time.hour
    weekday = jkt_time.weekday()
    
    if weekday >= 5:
        return "🔴 WEEKEND", False
    if hour < 9:
        return f"⏰ Opens in {9-hour}h", False
    elif hour >= 16:
        return "🔴 CLOSED", False
    elif 12 <= hour < 13:
        return "🟡 LUNCH", False
    else:
        return "🟢 OPEN", True

def is_valid_bpjs_time():
    """BPJS best 09:00-09:30"""
    return 9 <= get_jakarta_time().hour < 10

def is_valid_bsjp_time():
    """BSJP best 14:00-15:30"""
    return 14 <= get_jakarta_time().hour < 16

def display_quality_badge(grade):
    colors = {"A": "grade-a", "B": "grade-b", "C": "grade-c"}
    labels = {"A": "ELITE", "B": "GOOD", "C": "OK"}
    return f'<span class="quality-badge {colors.get(grade, "grade-c")}">Grade {grade} - {labels.get(grade, "OK")}</span>'

def load_tickers():
    """Load ALL 800+ IDX tickers"""
    try:
        if os.path.exists("idx_stocks.json"):
            with open("idx_stocks.json", "r") as f:
                data = json.load(f)
            tickers = data.get("tickers", [])
            return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
        else:
            # Extended default list (you should replace with full 800+ list)
            return ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK",
                    "TLKM.JK", "EXCL.JK", "ISAT.JK", "ASII.JK", "AUTO.JK",
                    "UNTR.JK", "PTBA.JK", "ADRO.JK", "ITMG.JK", "BREN.JK",
                    "BRPT.JK", "PGAS.JK", "INDF.JK", "ICBP.JK", "MYOR.JK",
                    "KLBF.JK", "KAEF.JK", "SIDO.JK", "CPIN.JK", "JPFA.JK",
                    "SMGR.JK", "WSBP.JK", "INTP.JK", "UNVR.JK", "HMSP.JK",
                    "GOTO.JK", "BUKA.JK", "EMTK.JK", "AMMN.JK", "ANTM.JK"]
    except:
        return ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]

# ============= DATA FETCHING =============
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, period="6mo"):
    try:
        end = int(datetime.now().timestamp())
        days = {"5d":5,"1mo":30,"3mo":90,"6mo":180,"1y":365}.get(period,180)
        start = end - (days*86400)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        r = requests.get(url, params={"period1":start,"period2":end,"interval":"1d"}, 
                        headers={'User-Agent':'Mozilla/5.0'}, timeout=10)
        
        if r.status_code != 200:
            return None
            
        data = r.json()
        if 'chart' not in data or 'result' not in data['chart']:
            return None
            
        result = data['chart']['result'][0]
        q = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open':q['open'],'High':q['high'],'Low':q['low'],
            'Close':q['close'],'Volume':q['volume']
        }, index=pd.to_datetime(result['timestamp'], unit='s'))
        
        df = df.dropna()
        if len(df) < 50:
            return None
        
        # Indicators
        df['EMA9'] = df['Close'].ewm(span=9).mean()
        df['EMA21'] = df['Close'].ewm(span=21).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean() if len(df)>=200 else df['Close'].ewm(span=len(df)).mean()
        
        delta = df['Close'].diff()
        gain = delta.where(delta>0,0).rolling(14).mean()
        loss = -delta.where(delta<0,0).rolling(14).mean()
        df['RSI'] = 100 - (100/(1+gain/loss))
        
        df['VOL_SMA20'] = df['Volume'].rolling(20).mean()
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA20']
        
        df['MOM_5D'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['MOM_10D'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        df['MOM_20D'] = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100
        
        # OBV for Bandar
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        
        # Bollinger Bands for BSJP
        df['BB_MID'] = df['Close'].rolling(20).mean()
        df['BB_STD'] = df['Close'].rolling(20).std()
        df['BB_LOWER'] = df['BB_MID'] - 2*df['BB_STD']
        
        # Stochastic for BPJS
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['STOCH_K'] = 100*(df['Close']-low14)/(high14-low14)
        
        return df
    except:
        return None

# ============= VALIDATION =============
def validate_quality_gate(df):
    try:
        r = df.iloc[-1]
        issues = []
        score = 100
        
        # Critical rejections
        if r['Close'] < r['EMA50'] and r['EMA50'] < r['EMA200']:
            issues.append("❌ Major downtrend")
            score -= 40
        
        if r['EMA9'] < r['EMA21'] < r['EMA50']:
            issues.append("❌ Death cross")
            score -= 30
        
        mom_20d = df['MOM_20D'].iloc[-1]
        if pd.notna(mom_20d) and mom_20d < -10:
            issues.append(f"❌ Severe down momentum")
            score -= 25
        
        if r['RSI'] > 75:
            issues.append(f"❌ Overbought")
            score -= 30
        
        # Grade
        if score >= 80:
            grade = "A"
        elif score >= 65:
            grade = "B"
        elif score >= 50:
            grade = "C"
        else:
            grade = "D"
        
        if grade == "D" or len(issues) >= 2:
            return False, "D", issues, score
        
        return True, grade, issues, score
    except:
        return False, "D", ["Error"], 0

# ============= SCORING STRATEGIES =============
def score_full_screener(df):
    """Main screener for general setups"""
    try:
        passed, grade, gate_issues, gate_score = validate_quality_gate(df)
        if not passed:
            return 0, {"⛔ REJECTED": gate_issues}, 0, "D", "HIGH"
        
        r = df.iloc[-1]
        score = 0
        details = {}
        confidence = 0
        
        # Trend (35 pts)
        if r['Close'] > r['EMA9'] > r['EMA21'] > r['EMA50'] > r['EMA200']:
            score += 35
            details['Trend'] = '🟢 PERFECT (+35)'
            confidence += 35
        elif r['Close'] > r['EMA9'] > r['EMA21']:
            score += 20
            details['Trend'] = '🟡 Short-term (+20)'
            confidence += 20
        
        # Momentum (25 pts)
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
        
        # RSI (20 pts)
        if 45 <= r['RSI'] <= 60:
            score += 20
            details['RSI'] = f'🟢 SWEET {r["RSI"]:.1f} (+20)'
            confidence += 20
        elif 40 <= r['RSI'] <= 65:
            score += 15
            details['RSI'] = f'🟢 Good {r["RSI"]:.1f} (+15)'
            confidence += 15
        
        # Volume (20 pts)
        vol_ratio = df['VOL_RATIO'].tail(5).mean()
        if pd.notna(vol_ratio) and vol_ratio > 1.5:
            score += 20
            details['Volume'] = f'🟢 STRONG {vol_ratio:.2f}x (+20)'
            confidence += 20
        
        confidence = min(int(confidence * 0.85), 100)
        risk_level = "LOW" if score >= 80 else "MEDIUM" if score >= 65 else "HIGH"
        
        if score < 40 or confidence < 50:
            return 0, details, 0, grade, "HIGH"
        
        return score, details, confidence, grade, risk_level
    except:
        return 0, {}, 0, "D", "HIGH"

def score_bpjs(df):
    """BPJS - Beli Pagi Jual Sore (Day Trading High Vol)"""
    try:
        passed, grade, _, _ = validate_quality_gate(df)
        if not passed:
            return 0, {"⛔ REJECTED": "Failed quality gate"}, 0, "D", "HIGH"
        
        r = df.iloc[-1]
        score = 0
        details = {}
        confidence = 0
        
        # Volatility (30 pts) - Need high intraday movement
        vol_pct = ((df['High']-df['Low'])/df['Low']*100).tail(5).mean()
        if 2 < vol_pct < 6:
            score += 30
            details['Volatility'] = f'🟢 IDEAL {vol_pct:.2f}% (+30)'
            confidence += 30
        
        # Volume spike (30 pts)
        if r['VOL_RATIO'] > 2.5:
            score += 30
            details['Volume'] = f'🟢 HUGE {r["VOL_RATIO"]:.2f}x (+30)'
            confidence += 30
        elif r['VOL_RATIO'] > 1.8:
            score += 20
            details['Volume'] = f'🟢 Strong {r["VOL_RATIO"]:.2f}x (+20)'
            confidence += 20
        
        # Oversold ready to bounce (25 pts)
        if 30 < r['RSI'] < 45:
            score += 25
            details['RSI'] = f"🟢 OVERSOLD {r['RSI']:.1f} (+25)"
            confidence += 25
        
        # Stochastic cross (15 pts)
        if pd.notna(r['STOCH_K']) and r['STOCH_K'] < 30:
            score += 15
            details['Stochastic'] = f"🟢 OVERSOLD {r['STOCH_K']:.1f} (+15)"
            confidence += 15
        
        confidence = min(int(confidence), 100)
        risk_level = "MEDIUM"  # BPJS always medium-high risk
        
        if score < 50:
            return 0, details, 0, grade, "HIGH"
        
        return score, details, confidence, grade, risk_level
    except:
        return 0, {}, 0, "D", "HIGH"

def score_bsjp(df):
    """BSJP - Beli Sore Jual Pagi (Gap Recovery)"""
    try:
        passed, grade, _, _ = validate_quality_gate(df)
        if not passed:
            return 0, {"⛔ REJECTED": "Failed quality gate"}, 0, "D", "HIGH"
        
        r = df.iloc[-1]
        score = 0
        details = {}
        confidence = 0
        
        # BB Position - need to be near lower band (30 pts)
        bb_pos = (r['Close']-r['BB_LOWER'])/(r['Close'])*100
        if bb_pos < 3:
            score += 30
            details['BB Position'] = f'🟢 EXTREME {bb_pos:.1f}% (+30)'
            confidence += 30
        
        # Gap down check (25 pts)
        gap = (r['Close']-df['Close'].iloc[-2])/df['Close'].iloc[-2]*100
        if -3 < gap < -0.5:
            score += 25
            details['Gap'] = f'🟢 IDEAL {gap:.2f}% (+25)'
            confidence += 25
        
        # Oversold (25 pts)
        if 30 < r['RSI'] < 50:
            score += 25
            details['RSI'] = f"🟢 OVERSOLD {r['RSI']:.1f} (+25)"
            confidence += 25
        
        # Volume confirmation (20 pts)
        if r['VOL_RATIO'] > 1.3:
            score += 20
            details['Volume'] = f'🟢 {r["VOL_RATIO"]:.2f}x (+20)'
            confidence += 20
        
        confidence = min(int(confidence), 100)
        risk_level = "MEDIUM"
        
        if score < 50:
            return 0, details, 0, grade, "HIGH"
        
        return score, details, confidence, grade, risk_level
    except:
        return 0, {}, 0, "D", "HIGH"

def score_bandar(df):
    """Bandar Tracking - Wyckoff Accumulation"""
    try:
        passed, grade, _, _ = validate_quality_gate(df)
        if not passed:
            return 0, {"⛔ REJECTED": "Failed quality gate"}, 0, "D", "HIGH"
        
        r = df.iloc[-1]
        details = {}
        
        # Volume & Price analysis
        vol_ratio = df['Volume'].tail(10).mean() / df['Volume'].rolling(30).mean().iloc[-1]
        price_chg = (r['Close'] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100
        
        # OBV trend
        obv_trend = (df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / abs(df['OBV'].iloc[-20]) if df['OBV'].iloc[-20] != 0 else 0
        obv_price_div = obv_trend > 0.1 and price_chg < 5
        
        # PHASE DETECTION
        if vol_ratio > 1.4 and -3 < price_chg < 5 and obv_price_div:
            # ACCUMULATION - Best for entry
            phase = "🟢 AKUMULASI KUAT"
            score = 95
            confidence = 90
            details['Phase'] = 'Wyckoff Accumulation'
            details['Action'] = '🚀 STRONG BUY'
            details['Signal'] = 'Smart money entering'
        elif price_chg > 5 and obv_trend > 0.1:
            # MARKUP - Hold phase
            phase = "🚀 MARKUP"
            score = 80
            confidence = 75
            details['Phase'] = 'Markup Phase'
            details['Action'] = '💪 HOLD'
            details['Signal'] = 'Uptrend active'
        elif vol_ratio > 1.5 and price_chg < -3:
            # DISTRIBUTION - Exit
            phase = "🔴 DISTRIBUSI"
            score = 10
            confidence = 15
            details['Phase'] = 'Distribution'
            details['Action'] = '🛑 AVOID'
            details['Signal'] = 'Smart money exiting'
        else:
            # SIDEWAYS
            phase = "⚪ SIDEWAYS"
            score = 50
            confidence = 50
            details['Phase'] = 'Consolidation'
            details['Action'] = '⏸️ WAIT'
            details['Signal'] = 'No clear direction'
        
        details['Volume'] = f'{vol_ratio:.2f}x'
        details['Price'] = f'{price_chg:+.2f}%'
        details['OBV Trend'] = '📈 Strong' if obv_trend > 0.1 else '📉 Weak'
        
        risk_level = "LOW" if "AKUMULASI" in phase else "MEDIUM" if "MARKUP" in phase else "HIGH"
        
        return score, details, confidence, grade, risk_level
    except:
        return 0, {}, 0, "D", "HIGH"

# ============= SIGNAL GENERATION =============
def get_signal_levels(score, price, confidence, grade, risk):
    if score >= 85 and confidence >= 75 and grade == "A":
        signal = "STRONG BUY"
        entry_ideal = round(price * 0.98, 0)
        entry_aggr = round(price, 0)
    elif score >= 70 and confidence >= 60:
        signal = "BUY"
        entry_ideal = round(price * 0.97, 0)
        entry_aggr = round(price * 0.99, 0)
    elif score >= 55:
        signal = "WATCH"
        entry_ideal = round(price * 0.95, 0)
        entry_aggr = None
    else:
        signal = "PASS"
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
        "ideal": {"entry": entry_ideal, "tp1": tp1, "tp2": tp2, "sl": sl},
        "aggr": {"entry": entry_aggr, "tp1": tp1, "tp2": tp2, "sl": sl}
    }

# ============= PROCESSING =============
def process_ticker(ticker, strategy, period):
    try:
        df = fetch_data(ticker, period)
        if df is None or len(df) < 50:
            return None
        
        price = float(df['Close'].iloc[-1])
        
        # Route to correct scoring function
        if strategy == "BPJS":
            score, details, confidence, grade, risk = score_bpjs(df)
        elif strategy == "BSJP":
            score, details, confidence, grade, risk = score_bsjp(df)
        elif strategy == "Bandar":
            score, details, confidence, grade, risk = score_bandar(df)
        else:
            score, details, confidence, grade, risk = score_full_screener(df)
        
        if score == 0:
            return None
        
        levels = get_signal_levels(score, price, confidence, grade, risk)
        
        return {
            "Ticker": ticker,
            "Price": price,
            "Score": score,
            "Confidence": confidence,
            "Grade": grade,
            "Risk": risk,
            "Signal": levels["signal"],
            "EntryIdeal": levels["ideal"]["entry"],
            "TP1": levels["ideal"]["tp1"],
            "SL": levels["ideal"]["sl"],
            "Details": details
        }
    except:
        return None

def two_stage_scan(tickers, strategy, period, stage1_limit=50, stage2_limit=10, use_parallel=True):
    """
    STAGE 1: Scan ALL tickers → Get Top 50
    STAGE 2: Deep analysis on Top 50 → Get Top 10 Elite
    """
    
    st.info(f"🔍 **STAGE 1**: Quick scan of {len(tickers)} stocks to find Top {stage1_limit}...")
    
    # STAGE 1: Fast scan with basic scoring
    stage1_results = []
    progress = st.progress(0)
    status = st.empty()
    
    if use_parallel and len(tickers) > 50:
        completed = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_ticker, t, strategy, period): t for t in tickers}
            for future in as_completed(futures):
                completed += 1
                progress.progress(completed / len(tickers))
                status.text(f"📊 Stage 1: {completed}/{len(tickers)} | Found: {len(stage1_results)}")
                result = future.result()
                if result and result['Score'] >= 50:  # Lower threshold for stage 1
                    stage1_results.append(result)
                time.sleep(0.05)
    else:
        for i, ticker in enumerate(tickers):
            progress.progress((i + 1) / len(tickers))
            status.text(f"📊 Stage 1: {i+1}/{len(tickers)} | Found: {len(stage1_results)}")
            result = process_ticker(ticker, strategy, period)
            if result and result['Score'] >= 50:
                stage1_results.append(result)
            time.sleep(0.2)
    
    progress.empty()
    status.empty()
    
    if not stage1_results:
        return pd.DataFrame(), pd.DataFrame()
    
    # Sort and get Top 50
    df_stage1 = pd.DataFrame(stage1_results).sort_values(
        ["Score", "Confidence"], ascending=[False, False]
    ).head(stage1_limit)
    
    st.success(f"✅ Stage 1 Complete: Found {len(df_stage1)} candidates from {len(tickers)} stocks")
    
    # STAGE 2: Deep analysis on Top 50
    st.info(f"🔬 **STAGE 2**: Deep analysis on Top {len(df_stage1)} → Selecting Top {stage2_limit} Elite...")
    
    # For stage 2, we already have the data, just apply stricter filters
    df_stage2 = df_stage1[
        (df_stage1['Score'] >= 70) & 
        (df_stage1['Confidence'] >= 65) &
        (df_stage1['Grade'].isin(['A', 'B']))
    ].head(stage2_limit)
    
    st.success(f"🏆 Stage 2 Complete: {len(df_stage2)} ELITE setups selected!")
    
    return df_stage1, df_stage2

# ============= MAIN APP =============
try:
    init_db()
except:
    pass

st.markdown('<div class="big-title">🎯 IDX Power Screener v4.0</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">2-Stage Filter | 800+ Stocks → Top 50 → Top 10 Elite</div>', unsafe_allow_html=True)

tickers = load_tickers()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    market_status, is_open = check_idx_market_status()
    if is_open:
        st.success(market_status)
    else:
        st.warning(market_status)
    
    jkt_time = get_jakarta_time()
    st.info(f"🕐 {jkt_time.strftime('%H:%M WIB')}")
    
    st.markdown("---")
    
    menu = st.radio("📋 Strategy", [
        "🎯 Elite Screener (General)",
        "⚡ BPJS (Beli Pagi Jual Sore)",
        "🌙 BSJP (Beli Sore Jual Pagi)",
        "🔮 Bandar Tracking",
        "🔍 Single Stock"
    ])
    
    st.markdown("---")
    
    if "Single" not in menu:
        period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
        
        st.markdown("### 🎯 2-Stage Filtering")
        st.info(f"**Total Stocks:** {len(tickers)}")
        
        stage1_limit = st.slider("Stage 1: Top N", 30, 100, 50, 5)
        st.caption(f"Quick scan {len(tickers)} → Top {stage1_limit}")
        
        stage2_limit = st.slider("Stage 2: Elite", 5, 30, 10, 5)
        st.caption(f"Deep filter {stage1_limit} → Top {stage2_limit}")
        
        use_parallel = st.checkbox("⚡ Parallel Scan", value=True)
        st.caption("Faster but uses more resources")
    
    st.markdown("---")
    st.caption(f"🎯 v4.0 | Stocks: {len(tickers)}")

# ============= MENU HANDLERS =============

if "Single" in menu:
    st.markdown("### 🔍 Single Stock Analysis")
    
    selected = st.selectbox("Select Stock", tickers)
    strategy_single = st.selectbox("Analysis Type", ["General", "BPJS", "BSJP", "Bandar"])
    period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
    
    if st.button("🔍 Analyze", type="primary"):
        with st.spinner(f"Analyzing {selected}..."):
            result = process_ticker(selected, strategy_single, period)
        
        if result is None:
            st.error("❌ Analysis failed or rejected by quality gates")
        else:
            st.markdown(f"## {result['Ticker']}")
            st.
