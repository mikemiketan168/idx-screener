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
    conn = sqlite3.connect('screener_tracking.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS recommendations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT, ticker TEXT, strategy TEXT, score INTEGER,
                  confidence INTEGER, quality_grade TEXT, entry_price REAL, 
                  current_price REAL, signal TEXT, status TEXT DEFAULT 'ACTIVE', 
                  result TEXT, profit_pct REAL, exit_price REAL, exit_date TEXT, 
                  notes TEXT, position_size TEXT DEFAULT '3/3',
                  market_context TEXT, risk_level TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date_added TEXT, ticker TEXT, strategy TEXT,
                  score INTEGER, confidence INTEGER, quality_grade TEXT,
                  target_entry REAL, current_price REAL, notes TEXT, 
                  status TEXT DEFAULT 'WATCHING')''')
    conn.commit()
    conn.close()

# ============= TIMEZONE & MARKET STATUS =============
def get_jakarta_time():
    jkt_tz = timezone(timedelta(hours=7))
    return datetime.now(jkt_tz)

def check_idx_market_status():
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
        return "🟡 LUNCH BREAK (12:00-13:00)", False
    else:
        return "🟢 MARKET OPEN", True

# ============= ENHANCED DATA FETCH WITH RETRY =============
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, period="6mo"):
    try:
        end = int(datetime.now().timestamp())
        days = {"5d":5,"1mo":30,"3mo":90,"6mo":180,"1y":365}.get(period,180)
        start = end - (days*86400)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        r = requests.get(url, params={"period1":start,"period2":end,"interval":"1d"}, 
                        headers={'User-Agent':'Mozilla/5.0'}, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()['chart']['result'][0]
        q = data['indicators']['quote'][0]
        df = pd.DataFrame({'Open':q['open'],'High':q['high'],'Low':q['low'],
                          'Close':q['close'],'Volume':q['volume']}, 
                         index=pd.to_datetime(data['timestamp'], unit='s'))
        df = df.dropna()
        if len(df) < 50:
            return None
        
        # ENHANCED INDICATORS
        df['EMA5'] = df['Close'].ewm(5).mean()
        df['EMA9'] = df['Close'].ewm(9).mean()
        df['EMA21'] = df['Close'].ewm(21).mean()
        df['EMA50'] = df['Close'].ewm(50).mean()
        df['EMA200'] = df['Close'].ewm(200).mean() if len(df) >= 200 else df['Close'].ewm(len(df)).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta>0,0)).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        df['RSI'] = 100 - (100/(1+gain/loss))
        
        # MACD
        exp1 = df['Close'].ewm(12).mean()
        exp2 = df['Close'].ewm(26).mean()
        df['MACD'] = exp1 - exp2
        df['SIGNAL'] = df['MACD'].ewm(9).mean()
        df['MACD_HIST'] = df['MACD'] - df['SIGNAL']
        
        # Bollinger Bands
        df['BB_MID'] = df['Close'].rolling(20).mean()
        df['BB_STD'] = df['Close'].rolling(20).std()
        df['BB_UPPER'] = df['BB_MID'] + 2*df['BB_STD']
        df['BB_LOWER'] = df['BB_MID'] - 2*df['BB_STD']
        
        # Stochastic
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['STOCH_K'] = 100*(df['Close']-low14)/(high14-low14)
        df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
        
        # Volume
        df['VOL_SMA20'] = df['Volume'].rolling(20).mean()
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA20']
        
        # ATR for volatility
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        df['ATR_PCT'] = (df['ATR'] / df['Close']) * 100
        
        # Momentum
        df['MOM_5D'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        df['MOM_10D'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        df['MOM_20D'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20) * 100
        
        # OBV for smart money tracking
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        df['OBV_EMA'] = pd.Series(obv).ewm(20).mean().values
        
        # Money Flow Index (MFI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # ADX for trend strength
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = true_range
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(14).mean()
        
        return df
    except:
        return None

def fetch_data_with_retry(ticker, period="6mo", max_retries=3):
    for attempt in range(max_retries):
        try:
            time.sleep(0.2 * attempt)  # Rate limiting
            return fetch_data(ticker, period)
        except:
            if attempt == max_retries - 1:
                return None
    return None

# ============= STRICT VALIDATION SYSTEM =============
def validate_quality_gate(df):
    """
    CRITICAL: Multi-layer quality gate
    Returns: (passed, grade, reasons)
    Grade: A (Excellent), B (Good), C (Acceptable), D (Reject)
    """
    try:
        r = df.iloc[-1]
        issues = []
        warnings = []
        score = 100
        
        # GATE 1: Downtrend Detection (CRITICAL)
        if r['Close'] < r['EMA50'] and r['EMA50'] < r['EMA200']:
            issues.append("❌ Major downtrend (Price < EMA50 < EMA200)")
            score -= 40
        
        if r['EMA9'] < r['EMA21'] < r['EMA50']:
            issues.append("❌ Death cross pattern")
            score -= 30
        
        # GATE 2: Momentum Check
        mom_20d = df['MOM_20D'].iloc[-1]
        if mom_20d < -10:
            issues.append(f"❌ Severe downward momentum: {mom_20d:.1f}%")
            score -= 25
        elif mom_20d < -5:
            warnings.append(f"⚠️ Negative momentum: {mom_20d:.1f}%")
            score -= 10
        
        # GATE 3: Volume Quality
        if r['VOL_RATIO'] > 5 and abs(df['MOM_5D'].iloc[-1]) < 1:
            issues.append("❌ Suspicious volume spike without price action")
            score -= 20
        
        recent_vol = df['VOL_RATIO'].tail(5).mean()
        if recent_vol < 0.5:
            warnings.append(f"⚠️ Low volume warning: {recent_vol:.2f}x")
            score -= 10
        
        # GATE 4: Overbought Check (STRICT)
        if r['RSI'] > 75 and r['STOCH_K'] > 85:
            issues.append(f"❌ Severely overbought (RSI:{r['RSI']:.1f}, Stoch:{r['STOCH_K']:.1f})")
            score -= 30
        elif r['RSI'] > 70:
            warnings.append(f"⚠️ Overbought territory (RSI:{r['RSI']:.1f})")
            score -= 15
        
        # GATE 5: ADX Trend Strength
        if not pd.isna(r['ADX']):
            if r['ADX'] < 20:
                warnings.append(f"⚠️ Weak trend (ADX:{r['ADX']:.1f})")
                score -= 10
        
        # GATE 6: Volatility Check
        if r['ATR_PCT'] > 8:
            warnings.append(f"⚠️ High volatility (ATR:{r['ATR_PCT']:.1f}%)")
            score -= 10
        elif r['ATR_PCT'] < 1:
            warnings.append(f"⚠️ Too low volatility (ATR:{r['ATR_PCT']:.1f}%)")
            score -= 5
        
        # GRADE ASSIGNMENT
        if score >= 80:
            grade = "A"
        elif score >= 65:
            grade = "B"
        elif score >= 50:
            grade = "C"
        else:
            grade = "D"
        
        # REJECT if Grade D or critical issues
        if grade == "D" or len(issues) >= 2:
            return False, "D", issues + warnings, score
        
        return True, grade, issues + warnings, score
        
    except Exception as e:
        return False, "D", [f"❌ Validation error: {str(e)}"], 0

# ============= WYCKOFF BANDAR DETECTION (ENHANCED) =============
def detect_wyckoff_phase(df):
    """
    Advanced Wyckoff accumulation/distribution detection
    Returns: phase, score, details, confidence
    """
    try:
        # Volume & Price Analysis
        vol_ratio = df['Volume'].tail(10).mean() / df['Volume'].rolling(30).mean().iloc[-1]
        price_chg_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100
        
        # OBV Trend
        obv_trend = (df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / abs(df['OBV'].iloc[-20]) if df['OBV'].iloc[-20] != 0 else 0
        obv_price_div = obv_trend > 0.1 and price_chg_20d < 3  # Divergence
        
        # Price range compression
        recent_range = (df['High'].tail(10).max() - df['Low'].tail(10).min()) / df['Low'].tail(10).min() * 100
        
        # MFI for money flow
        mfi = df['MFI'].iloc[-1]
        
        details = {}
        
        # PHASE 1: ACCUMULATION (Best for entry)
        if vol_ratio > 1.3 and -3 < price_chg_20d < 5 and obv_price_div and recent_range < 15:
            if mfi > 40 and mfi < 60:  # Smart money accumulating
                phase = "🟢 AKUMULASI KUAT"
                score = 95
                confidence = 90
                details['Signal'] = '🚀 STRONG BUY - Smart Money Entering'
                details['Phase'] = 'Wyckoff Accumulation - Phase C/D'
                details['Action'] = '💎 BUY NOW - Best Entry Zone'
            else:
                phase = "🟢 AKUMULASI"
                score = 85
                confidence = 75
                details['Signal'] = '✅ BUY - Accumulation Phase'
                details['Phase'] = 'Early Accumulation'
                details['Action'] = '👍 Good Entry Point'
        
        # PHASE 2: MARKUP (Hold zone)
        elif price_chg_20d > 5 and obv_trend > 0.15 and vol_ratio > 1.0:
            if price_chg_20d > 15:
                phase = "🚀 MARKUP LATE"
                score = 65
                confidence = 60
                details['Signal'] = '⚠️ HOLD - Late Stage'
                details['Phase'] = 'Markup - Consider Taking Profit'
                details['Action'] = '🎯 Scale Out'
            else:
                phase = "🚀 MARKUP"
                score = 80
                confidence = 75
                details['Signal'] = '✅ HOLD - Uptrend Strong'
                details['Phase'] = 'Markup Phase - Let It Run'
                details['Action'] = '💪 HOLD Position'
        
        # PHASE 3: DISTRIBUTION (Exit zone)
        elif vol_ratio > 1.5 and price_chg_20d < -3:
            phase = "🔴 DISTRIBUSI"
            score = 15
            confidence = 20
            details['Signal'] = '🛑 AVOID - Distribution'
            details['Phase'] = 'Smart Money Exiting'
            details['Action'] = '❌ DO NOT BUY'
        
        # PHASE 4: MARKDOWN (Avoid)
        elif price_chg_20d < -10 and obv_trend < -0.1:
            phase = "🔴 MARKDOWN"
            score = 5
            confidence = 10
            details['Signal'] = '🛑 AVOID - Downtrend'
            details['Phase'] = 'Active Selling Pressure'
            details['Action'] = '❌ STAY AWAY'
        
        # PHASE 5: SIDEWAYS (Wait)
        else:
            phase = "⚪ SIDEWAYS"
            score = 45
            confidence = 50
            details['Signal'] = '⏸️ WAIT - No Clear Direction'
            details['Phase'] = 'Consolidation/Ranging'
            details['Action'] = '👀 Watch & Wait'
        
        details['Volume Ratio'] = f'{vol_ratio:.2f}x'
        details['Price Change'] = f'{price_chg_20d:+.2f}%'
        details['OBV Trend'] = '📈 Strong' if obv_trend > 0.1 else '📉 Weak'
        details['MFI'] = f'{mfi:.1f}'
        
        return phase, score, details, confidence
        
    except:
        return "❓ UNKNOWN", 0, {}, 0

# ============= ENHANCED SCORING SYSTEM =============
def score_advanced_v4(df, strategy="Full"):
    """
    Multi-factor scoring with strict validation
    Returns: score, details, confidence, quality_grade, risk_level
    """
    try:
        # STEP 1: Quality Gate (MANDATORY)
        passed, grade, gate_issues, gate_score = validate_quality_gate(df)
        
        if not passed:
            return 0, {"⛔ REJECTED": gate_issues}, 0, "D", "HIGH"
        
        r = df.iloc[-1]
        score = 0
        details = {}
        confidence = 0
        risk_points = 0
        
        # Add quality gate feedback
        if grade in ["B", "C"]:
            details['⚠️ Quality Issues'] = gate_issues
        
        # FACTOR 1: TREND ALIGNMENT (35 points max)
        if r['Close'] > r['EMA9'] > r['EMA21'] > r['EMA50'] > r['EMA200']:
            score += 35
            details['Trend'] = '🟢 PERFECT ALIGNMENT (+35)'
            confidence += 35
        elif r['Close'] > r['EMA9'] > r['EMA21'] > r['EMA50']:
            score += 28
            details['Trend'] = '🟢 Strong Uptrend (+28)'
            confidence += 28
        elif r['Close'] > r['EMA9'] > r['EMA21']:
            score += 18
            details['Trend'] = '🟡 Short-term Up (+18)'
            confidence += 20
        else:
            score += 5
            details['Trend'] = '⚪ Weak Trend (+5)'
            confidence += 5
            risk_points += 15
        
        # FACTOR 2: MOMENTUM QUALITY (25 points max)
        mom_5d = df['MOM_5D'].iloc[-1]
        mom_10d = df['MOM_10D'].iloc[-1]
        mom_20d = df['MOM_20D'].iloc[-1]
        
        # Ideal: steady positive momentum across timeframes
        if 2 <= mom_5d <= 8 and mom_10d > 0 and mom_20d > 0:
            score += 25
            details['Momentum'] = f'🟢 IDEAL {mom_5d:.1f}% (+25)'
            confidence += 25
        elif 0 < mom_5d <= 12 and mom_10d > -3:
            score += 18
            details['Momentum'] = f'🟢 Good {mom_5d:.1f}% (+18)'
            confidence += 18
        elif mom_5d > 0:
            score += 10
            details['Momentum'] = f'🟡 Weak {mom_5d:.1f}% (+10)'
            confidence += 10
            risk_points += 10
        else:
            score += 0
            details['Momentum'] = f'🔴 Negative {mom_5d:.1f}% (+0)'
            risk_points += 20
        
        # FACTOR 3: RSI POSITION (20 points max)
        # Sweet spot: 45-60 (room to run, not overbought)
        if 45 <= r['RSI'] <= 60:
            score += 20
            details['RSI'] = f'🟢 SWEET SPOT {r["RSI"]:.1f} (+20)'
            confidence += 20
        elif 40 <= r['RSI'] <= 65:
            score += 15
            details['RSI'] = f'🟢 Good {r["RSI"]:.1f} (+15)'
            confidence += 15
        elif 30 <= r['RSI'] <= 70:
            score += 10
            details['RSI'] = f'🟡 OK {r["RSI"]:.1f} (+10)'
            confidence += 10
        else:
            score += 5
            details['RSI'] = f'⚠️ Extreme {r["RSI"]:.1f} (+5)'
            risk_points += 15
        
        # FACTOR 4: VOLUME CONFIRMATION (20 points max)
        vol_5d_avg = df['VOL_RATIO'].tail(5).mean()
        
        if vol_5d_avg > 1.8 and mom_5d > 2:
            score += 20
            details['Volume'] = f'🟢 BREAKOUT {vol_5d_avg:.2f}x (+20)'
            confidence += 20
        elif vol_5d_avg > 1.3 and mom_5d > 0:
            score += 15
            details['Volume'] = f'🟢 Strong {vol_5d_avg:.2f}x (+15)'
            confidence += 15
        elif vol_5d_avg > 0.8:
            score += 10
            details['Volume'] = f'🟡 Normal {vol_5d_avg:.2f}x (+10)'
            confidence += 10
        else:
            score += 5
            details['Volume'] = f'⚠️ Low {vol_5d_avg:.2f}x (+5)'
            risk_points += 10
        
        # FACTOR 5: SMART MONEY (OBV + MFI) (20 points max)
        obv_slope = (df['OBV'].iloc[-1] - df['OBV'].iloc[-10]) / abs(df['OBV'].iloc[-10]) if df['OBV'].iloc[-10] != 0 else 0
        mfi = df['MFI'].iloc[-1]
        
        if obv_slope > 0.1 and 50 <= mfi <= 70:
            score += 20
            details['Smart Money'] = f'🟢 ACCUMULATING (+20)'
            confidence += 20
        elif obv_slope > 0.05 and mfi > 40:
            score += 15
            details['Smart Money'] = f'🟢 Buying (+15)'
            confidence += 15
        elif obv_slope > 0:
            score += 10
            details['Smart Money'] = f'🟡 Mixed (+10)'
            confidence += 10
        else:
            score += 5
            details['Smart Money'] = f'🔴 Distributing (+5)'
            risk_points += 15
        
        # BONUS: ADX Trend Strength
        if not pd.isna(r['ADX']) and r['ADX'] > 25:
            score += 5
            details['Trend Strength'] = f'✅ Strong (ADX:{r["ADX"]:.1f}) (+5)'
            confidence += 5
        
        # PENALTY: High volatility
        if r['ATR_PCT'] > 6:
            score -= 10
            details['Volatility'] = f'⚠️ High Risk (ATR:{r["ATR_PCT"]:.1f}%) (-10)'
            risk_points += 20
        
        # Calculate confidence (capped at 100)
        confidence = min(int(confidence * 0.85), 100)
        
        # Risk Level Assessment
        if risk_points < 15:
            risk_level = "LOW"
        elif risk_points < 35:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # MINIMUM THRESHOLD
        if score < 35 or confidence < 45:
            return 0, details, 0, grade, "HIGH"
        
        return score, details, confidence, grade, risk_level
        
    except Exception as e:
        return 0, {"❌ Error": str(e)}, 0, "D", "HIGH"

# ============= SIGNAL GENERATION =============
def get_signal_levels_v4(score, price, confidence, grade, risk_level):
    """Generate trading signals based on multi-factor analysis"""
    
    # STRICT CRITERIA for signals
    if score >= 85 and confidence >= 75 and grade == "A" and risk_level == "LOW":
        signal = "STRONG BUY"
        signal_class = "strong-buy"
        trend = "🟢 Excellent Setup"
        entry_ideal = round(price * 0.98, 0)
        entry_aggr = round(price, 0)
    elif score >= 75 and confidence >= 65 and grade in ["A", "B"] and risk_level != "HIGH":
        signal = "BUY"
        signal_class = "buy"
        trend = "🟢 Good Setup"
        entry_ideal = round(price * 0.97, 0)
        entry_aggr = round(price * 0.99, 0)
    elif score >= 60 and confidence >= 55:
        signal = "WATCH"
        signal_class = "neutral"
        trend = "🟡 Monitor Closely"
        entry_ideal = round(price * 0.95, 0)
        entry_aggr = None
    else:
        signal = "PASS"
        signal_class = "sell"
        trend = "⚪ Not Ready"
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
    """Enhanced processing with quality gates"""
    try:
        df = fetch_data_with_retry(ticker, period)
        if df is None or len(df) < 50:
            return None
        
        price = float(df['Close'].iloc[-1])
        
        # Run advanced scoring
        score, details, confidence, grade, risk_level = score_advanced_v4(df, strategy)
        
        if score == 0:  # Rejected by quality gate
            return None
        
        # Get Wyckoff phase
        wyckoff_phase, wyckoff_score, wyckoff_details, wyckoff_conf = detect_wyckoff_phase(df)
        
        # Combine details
        all_details = {**details, **wyckoff_details}
        all_details['Wyckoff Phase'] = wyckoff_phase
        
        # Boost score if in accumulation phase
        if "AKUMULASI" in wyckoff_phase:
            score = min(score + 10, 100)
            confidence = min(confidence + 10, 100)
        
        # Generate signals
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
            "EntryAggr": levels["aggr"]["entry"],
            "TP1": levels["ideal"]["tp1"],
            "TP2": levels["ideal"]["tp2"],
            "SL": levels["ideal"]["sl"],
            "Details": all_details,
            "WyckoffPhase": wyckoff_phase,
            "WyckoffScore": wyckoff_score
        }
    except Exception as e:
        return None

def batch_scan_v4(tickers, strategy, period, limit, use_parallel=True, min_score=65, min_conf=60):
    """Enhanced batch scanning with progress tracking"""
    results = []
    if limit and limit < len(tickers):
        tickers = tickers[:limit]
    
    progress = st.progress(0)
    status = st.empty()
    total = len(tickers)
    
    if use_parallel and total > 20:
        completed = 0
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_ticker_v4, t, strategy, period): t for t in tickers}
            for future in as_completed(futures):
                completed += 1
                progress.progress(completed / total)
                status.text(f"📊 Scanning {completed}/{total} | Found: {len(results)}")
                result = future.result()
                if result and result['Score'] >= min_score and result['Confidence'] >= min_conf:
                    results.append(result)
                time.sleep(0.15)
    else:
        for i, ticker in enumerate(tickers):
            progress.progress((i + 1) / total)
            status.text(f"📊 {i+1}/{total}: {ticker} | Found: {len(results)}")
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

# ============= TRACKING & MANAGEMENT =============
def save_recommendation_v4(ticker, strategy, score, confidence, grade, entry_price, signal, risk_level):
    conn = sqlite3.connect('screener_tracking.db')
    c = conn.cursor()
    c.execute('''INSERT INTO recommendations 
                 (date, ticker, strategy, score, confidence, quality_grade, 
                  entry_price, current_price, signal, status, risk_level)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker, strategy, score, 
               confidence, grade, entry_price, entry_price, signal, 'ACTIVE', risk_level))
    conn.commit()
    conn.close()

def get_active_recommendations():
    conn = sqlite3.connect('screener_tracking.db')
    df = pd.read_sql("SELECT * FROM recommendations WHERE status='ACTIVE' ORDER BY quality_grade ASC, score DESC", conn)
    conn.close()
    return df

def get_performance_stats():
    conn = sqlite3.connect('screener_tracking.db')
    
    # Overall stats
    total = pd.read_sql("SELECT COUNT(*) as total FROM recommendations WHERE status='CLOSED'", conn)
    wins = pd.read_sql("SELECT COUNT(*) as wins FROM recommendations WHERE result='WIN'", conn)
    avg_profit = pd.read_sql("SELECT AVG(profit_pct) as avg FROM recommendations WHERE result='WIN'", conn)
    avg_loss = pd.read_sql("SELECT AVG(profit_pct) as avg FROM recommendations WHERE result='LOSS'", conn)
    
    # By quality grade
    by_grade = pd.read_sql("""SELECT quality_grade,
                                     COUNT(*) as total,
                                     SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                                     AVG(profit_pct) as avg_profit
                              FROM recommendations WHERE status='CLOSED'
                              GROUP BY quality_grade
                              ORDER BY quality_grade""", conn)
    
    # By risk level
    by_risk = pd.read_sql("""SELECT risk_level,
                                    COUNT(*) as total,
                                    SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                                    AVG(profit_pct) as avg_profit
                             FROM recommendations WHERE status='CLOSED'
                             GROUP BY risk_level""", conn)
    
    conn.close()
    
    return {
        'total': total['total'].iloc[0] if not total.empty else 0,
        'wins': wins['wins'].iloc[0] if not wins.empty else 0,
        'avg_profit': avg_profit['avg'].iloc[0] if not avg_profit.empty else 0,
        'avg_loss': avg_loss['avg'].iloc[0] if not avg_loss.empty else 0,
        'by_grade': by_grade,
        'by_risk': by_risk
    }

# ============= UI HELPERS =============
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
    try:
        with open("idx_stocks.json", "r") as f:
            data = json.load(f)
        tickers = data.get("tickers", [])
        return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except:
        return ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
                "BREN.JK", "BRPT.JK", "GOTO.JK", "AMMN.JK", "EMTK.JK"]

# ============= MAIN APP =============
init_db()

st.markdown('<div class="big-title">🎯 IDX Power Screener v4.0</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Accuracy-First | Multi-Layer Validation | Wyckoff Smart Money Tracking</div>', unsafe_allow_html=True)

tickers = load_tickers()

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
    
    menu = st.radio("📋 Menu", [
        "🎯 Elite Screener (Grade A Only)",
        "📊 Full Screener",
        "🔍 Single Stock Deep Analysis",
        "📈 Active Positions",
        "📊 Performance Analytics",
        "🧪 Test & Validation"
    ])
    
    st.markdown("---")
    
    if "Single" not in menu and "Active" not in menu and "Performance" not in menu:
        period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
        limit = st.slider("Max Stocks to Scan", 10, len(tickers), min(100, len(tickers)), step=10)
        
        st.markdown("### 🎯 Quality Filters")
        min_score = st.slider("Min Score", 50, 100, 70, step=5)
        min_confidence = st.slider("Min Confidence", 40, 100, 65, step=5)
        
        grade_filter = st.multiselect("Quality Grades", ["A", "B", "C"], default=["A", "B"])
        risk_filter = st.multiselect("Risk Levels", ["LOW", "MEDIUM", "HIGH"], default=["LOW", "MEDIUM"])
        
        use_parallel = st.checkbox("⚡ Fast Mode (Parallel)", value=True)
    
    st.markdown("---")
    st.markdown("### 💡 v4.0 Enhancements")
    st.success("""
    ✅ Multi-layer validation
    ✅ Wyckoff phase detection
    ✅ Quality grading (A-D)
    ✅ Smart money tracking
    ✅ Risk level assessment
    ✅ MFI + OBV + ADX
    """)
    
    st.caption("🎯 Precision over Quantity")

# ============= MENU HANDLERS =============

if "Elite" in menu:
    st.markdown("### 🏆 Elite Screener - Grade A Only")
    st.info("⭐ Only the highest quality setups with strict validation")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Quality Filter", "Grade A Only")
    col2.metric("Min Score", f"{min_score}/100")
    col3.metric("Min Confidence", f"{min_confidence}%")
    
    if st.button("🚀 Scan for Elite Setups", type="primary"):
        with st.spinner(f"Scanning {limit} stocks with strict quality gates..."):
            df = batch_scan_v4(tickers, "Elite", period, limit, use_parallel, min_score, min_confidence)
        
        # Filter for Grade A only
        if not df.empty:
            df = df[df['Grade'] == 'A']
        
        if df.empty:
            st.warning("⚠️ No Grade A setups found")
            st.info("""
            **Grade A criteria are very strict:**
            - Perfect trend alignment
            - Strong momentum across timeframes
            - Ideal RSI position (45-60)
            - Volume confirmation
            - Smart money accumulation
            - Low risk level
            
            💡 Try:
            - Increasing stocks scanned
            - Lowering min score to 65
            - Including Grade B in filters
            """)
        else:
            st.success(f"🏆 Found {len(df)} ELITE opportunities!")
            
            # Summary
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Elite Setups", len(df))
            col2.metric("Avg Score", f"{df['Score'].mean():.1f}")
            col3.metric("Avg Confidence", f"{df['Confidence'].mean():.1f}%")
            col4.metric("Avg Wyckoff", f"{df['WyckoffScore'].mean():.1f}")
            
            # Display results
            for _, row in df.iterrows():
                wyckoff_icon = "🟢" if "AKUMULASI" in row['WyckoffPhase'] else "🚀" if "MARKUP" in row['WyckoffPhase'] else "⚪"
                
                with st.expander(f"{wyckoff_icon} {row['Ticker']} | Score: {row['Score']} | Conf: {row['Confidence']}% | {row['WyckoffPhase']}", expanded=True):
                    
                    # Header
                    st.markdown(display_quality_badge(row['Grade']), unsafe_allow_html=True)
                    st.markdown(f"**{display_risk_badge(row['Risk'])}** | **{row['Signal']}** | {row['Trend']}")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("💰 Price", f"Rp {row['Price']:,.0f}")
                    col2.metric("📊 Score", f"{row['Score']}/100")
                    col3.metric("🎯 Confidence", f"{row['Confidence']}%")
                    col4.metric("🔮 Wyckoff", f"{row['WyckoffScore']}/100")
                    
                    # Entry levels
                    if row['EntryIdeal']:
                        st.markdown("### 🎯 Entry Strategy")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.success(f"""
                            **📍 IDEAL ENTRY (Wait for dip)**
                            Entry: Rp {row['EntryIdeal']:,.0f}
                            TP1 (8%): Rp {row['TP1']:,.0f}
                            TP2 (15%): Rp {row['TP2']:,.0f}
                            SL: Rp {row['SL']:,.0f}
                            """)
                        
                        with col2:
                            if row['EntryAggr']:
                                st.warning(f"""
                                **⚡ AGGRESSIVE ENTRY (Now)**
                                Entry: Rp {row['EntryAggr']:,.0f}
                                TP1: Rp {row['TP1']:,.0f}
                                TP2: Rp {row['TP2']:,.0f}
                                SL: Rp {row['SL']:,.0f}
                                """)
                    
                    # Technical details
                    st.markdown("### 📋 Analysis Details")
                    for k, v in row['Details'].items():
                        if '❌' in str(k) or '⛔' in str(k):
                            st.error(f"**{k}:** {v}")
                        elif '⚠️' in str(k):
                            st.warning(f"**{k}:** {v}")
                        elif '🟢' in str(v) or '✅' in str(v):
                            st.success(f"**{k}:** {v}")
                        else:
                            st.info(f"**{k}:** {v}")
                    
                    # Action buttons
                    col1, col2 = st.columns(2)
                    if col1.button(f"💾 Track", key=f"t_{row['Ticker']}"):
                        save_recommendation_v4(row['Ticker'].replace('.JK', ''), "Elite", 
                                             row['Score'], row['Confidence'], row['Grade'],
                                             row['Price'], row['Signal'], row['Risk'])
                        st.success("✅ Added to tracking!")
            
            # Download
            csv = df[['Ticker', 'Price', 'Score', 'Confidence', 'Grade', 'Risk', 
                     'Signal', 'WyckoffPhase', 'EntryIdeal', 'TP1', 'TP2', 'SL']].to_csv(index=False)
            st.download_button("📥 Download Elite Picks", csv.encode(), 
                             f"elite_picks_{datetime.now().strftime('%Y%m%d')}.csv")

elif "Full Screener" in menu:
    st.markdown("### 📊 Full Screener - All Quality Grades")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.info(f"📊 Stocks: {limit}")
    col2.info(f"🎯 Score: {min_score}+")
    col3.info(f"📈 Conf: {min_confidence}%+")
    col4.info(f"✅ Grades: {', '.join(grade_filter)}")
    
    if st.button("🚀 Run Full Scan", type="primary"):
        with st.spinner(f"Scanning {limit} stocks..."):
            df = batch_scan_v4(tickers, "Full", period, limit, use_parallel, min_score, min_confidence)
        
        # Apply filters
        if not df.empty:
            df = df[df['Grade'].isin(grade_filter) & df['Risk'].isin(risk_filter)]
        
        if df.empty:
            st.warning("⚠️ No stocks match your criteria")
        else:
            st.success(f"✅ Found {len(df)} opportunities!")
            
            # Summary by grade
            st.markdown("### 📊 Results by Quality Grade")
            grade_summary = df.groupby('Grade').agg({
                'Ticker': 'count',
                'Score': 'mean',
                'Confidence': 'mean'
            }).round(1)
            
            cols = st.columns(len(grade_summary))
            for idx, (grade, row) in enumerate(grade_summary.iterrows()):
                with cols[idx]:
                    st.markdown(display_quality_badge(grade), unsafe_allow_html=True)
                    st.metric("Count", int(row['Ticker']))
                    st.caption(f"Avg Score: {row['Score']:.1f}")
                    st.caption(f"Avg Conf: {row['Confidence']:.1f}%")
            
            # Data table
            show = df[['Ticker', 'Price', 'Score', 'Confidence', 'Grade', 'Risk', 
                      'Signal', 'WyckoffPhase', 'EntryIdeal', 'TP1', 'SL']]
            st.dataframe(show, use_container_width=True, height=400)
            
            # Top picks
            st.markdown(f"### 🏆 Top {min(10, len(df))} Picks")
            for _, row in df.head(10).iterrows():
                wyckoff_icon = "🟢" if "AKUMULASI" in row['WyckoffPhase'] else "🚀" if "MARKUP" in row['WyckoffPhase'] else "⚪"
                
                with st.expander(f"{wyckoff_icon} {row['Ticker']} | Grade {row['Grade']} | Score: {row['Score']}"):
                    st.markdown(display_quality_badge(row['Grade']), unsafe_allow_html=True)
                    st.markdown(f"**{display_risk_badge(row['Risk'])}**")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price", f"Rp {row['Price']:,.0f}")
                    col2.metric("Score", f"{row['Score']}/100")
                    col3.metric("Confidence", f"{row['Confidence']}%")
                    
                    if row['EntryIdeal']:
                        st.info(f"""
                        **Entry:** Rp {row['EntryIdeal']:,.0f}
                        **TP1:** Rp {row['TP1']:,.0f} | **TP2:** Rp {row['TP2']:,.0f}
                        **SL:** Rp {row['SL']:,.0f}
                        """)
                    
                    st.markdown("**Analysis:**")
                    for k, v in row['Details'].items():
                        st.caption(f"• **{k}:** {v}")
                    
                    if st.button(f"💾 Track", key=f"track_{row['Ticker']}"):
                        save_recommendation_v4(row['Ticker'].replace('.JK', ''), "Full", 
                                             row['Score'], row['Confidence'], row['Grade'],
                                             row['Price'], row['Signal'], row['Risk'])
                        st.success("✅ Tracked!")

elif "Single Stock" in menu:
    st.markdown("### 🔍 Deep Analysis - Single Stock")
    
    selected = st.selectbox("Select Stock", tickers)
    
    if st.button("🔍 Analyze", type="primary"):
        with st.spinner(f"Deep analysis on {selected}..."):
            result = process_ticker_v4(selected, "Single", period)
        
        if result is None:
            st.error("❌ Failed to analyze or stock rejected by quality gates")
        else:
            st.markdown(f"## {result['Ticker']}")
            st.markdown(display_quality_badge(result['Grade']), unsafe_allow_html=True)
            st.markdown(f"### {display_risk_badge(result['Risk'])}")
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Price", f"Rp {result['Price']:,.0f}")
            col2.metric("📊 Score", f"{result['Score']}/100")
            col3.metric("🎯 Confidence", f"{result['Confidence']}%")
            col4.metric("🔮 Wyckoff", f"{result['WyckoffScore']}/100")
            
            # Signal
            st.markdown(f'<div class="signal-box {result["signal"]}">{result["Signal"]}</div>', 
                       unsafe_allow_html=True)
            st.markdown(f"### {result['Trend']}")
            st.markdown(f"### {result['WyckoffPhase']}")
            
            # Entry levels
            if result['EntryIdeal']:
                st.markdown("### 🎯 Trading Plan")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"""
                    **📍 IDEAL ENTRY**
                    Wait for pullback to: **Rp {result['EntryIdeal']:,.0f}**
                    
                    **Targets:**
                    🎯 TP1 (Sell 1/3): Rp {result['TP1']:,.0f} (+8%)
                    🎯 TP2 (Sell 1/3): Rp {result['TP2']:,.0f} (+15%)
                    🏃 Trail last 1/3 with 20 EMA
                    
                    **Stop Loss:** Rp {result['SL']:,.0f} (-6%)
                    """)
                
                with col2:
                    if result['EntryAggr']:
                        st.warning(f"""
                        **⚡ AGGRESSIVE ENTRY**
                        Buy now at: **Rp {result['EntryAggr']:,.0f}**
                        
                        **Same targets apply**
                        Higher risk, less room for error
                        
                        💡 Recommended: Wait for ideal entry
                        """)
            
            # Full analysis
            st.markdown("### 📊 Complete Analysis")
            for k, v in result['Details'].items():
                if '❌' in str(k) or '⛔' in str(k):
                    st.error(f"**{k}:** {v}")
                elif '⚠️' in str(k):
                    st.warning(f"**{k}:** {v}")
                elif '🟢' in str(v) or '✅' in str(v):
                    st.success(f"**{k}:** {v}")
                else:
                    st.info(f"**{k}:** {v}")
            
            # Trading decision
            st.markdown("---")
            st.markdown("### ✅ Trading Decision")
            
            if result['Grade'] == 'A' and result['Risk'] == 'LOW' and result['Score'] >= 80:
                st.success("""
                🟢 **HIGH PROBABILITY SETUP**
                - Grade A quality
                - Low risk
                - Strong score & confidence
                
                **Action:** Consider position sizing 100% of normal
                """)
            elif result['Grade'] in ['A', 'B'] and result['Score'] >= 70:
                st.info("""
                🟡 **GOOD SETUP**
                - Acceptable quality
                - Moderate to good score
                
                **Action:** Consider position sizing 50-75% of normal
                """)
            else:
                st.warning("""
                ⚪ **MARGINAL SETUP**
                - Lower quality grade or score
                
                **Action:** Wait for better opportunity or skip
                """)

elif "Performance" in menu:
    st.markdown("### 📊 Performance Analytics")
    
    stats = get_performance_stats()
    
    if stats['total'] == 0:
        st.info("📭 No closed trades yet")
    else:
        # Overall
        col1, col2, col3, col4 = st.columns(4)
        wr = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        col1.metric("Total Trades", stats['total'])
        col2.metric("Win Rate", f"{wr:.1f}%")
        col3.metric("Avg Win", f"+{stats['avg_profit']:.2f}%")
        col4.metric("Avg Loss", f"{stats['avg_loss']:.2f}%")
        
        # By quality grade
        if not stats['by_grade'].empty:
            st.markdown("### 📊 Win Rate by Quality Grade")
            for _, row in stats['by_grade'].iterrows():
                grade_wr = (row['wins'] / row['total'] * 100) if row['total'] > 0 else 0
                
                st.markdown(display_quality_badge(row['quality_grade']), unsafe_allow_html=True)
                st.progress(min(grade_wr / 100, 1.0))
                st.caption(f"WR: {grade_wr:.1f}% | Trades: {int(row['total'])} | Avg: {row['avg_profit']:+.2f}%")
        
        # Insights
        st.markdown("---")
        st.markdown("### 💡 Key Insights")
        
        if wr >= 60:
            st.success("🟢 **Excellent Performance** - Strategy working well!")
        elif wr >= 50:
            st.info("🟡 **On Track** - Keep improving")
        else:
            st.error("🔴 **Needs Improvement** - Focus on Grade A only")

else:  # Test & Validation
    st.markdown("### 🧪 Test & Validation")
    
    test_stocks = st.multiselect("Select test stocks", tickers, default=tickers[:5])
    
    if st.button("🔬 Run Validation Tests"):
        for ticker in test_stocks:
            result = process_ticker_v4(ticker, "Test", period)
            
            if result:
                with st.expander(f"{ticker} - Grade {result['Grade']} | Score: {result['Score']}"):
                    st.markdown(display_quality_badge(result['Grade']), unsafe_allow_html=True)
                    st.markdown(f"**{display_risk_badge(result['Risk'])}**")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Score", result['Score'])
                    col2.metric("Confidence", f"{result['Confidence']}%")
                    col3.metric("Wyckoff", result['WyckoffPhase'])
                    
                    for k, v in result['Details'].items():
                        if '❌' in str(k):
                            st.error(f"{k}: {v}")
                        elif '⚠️' in str(k):
                            st.warning(f"{k}: {v}")
                        else:
                            st.info(f"{k}: {v}")
            else:
                st.error(f"❌ {ticker} - Rejected by quality gates")

elif "Active Positions" in menu:
    st.markdown("### 📈 Active Positions Management")
    
    active = get_active_recommendations()
    
    if active.empty:
        st.info("📭 No active positions. Start tracking from screener results!")
    else:
        st.success(f"📊 Managing {len(active)} active positions")
        
        # Update all button
        if st.button("🔄 Update All Prices"):
            progress = st.progress(0)
            status = st.empty()
            updated = 0
            
            conn = sqlite3.connect('screener_tracking.db')
            for idx, row in active.iterrows():
                status.text(f"Updating {row['ticker']}...")
                progress.progress((idx + 1) / len(active))
                
                ticker = row['ticker'] if row['ticker'].endswith('.JK') else f"{row['ticker']}.JK"
                df = fetch_data_with_retry(ticker, "5d")
                
                if df is not None:
                    c = conn.cursor()
                    c.execute("UPDATE recommendations SET current_price=? WHERE id=?", 
                             (df['Close'].iloc[-1], row['id']))
                    conn.commit()
                    updated += 1
                
                time.sleep(0.5)
            
            conn.close()
            progress.empty()
            status.empty()
            st.success(f"✅ Updated {updated} positions!")
            st.rerun()
        
        # Group by quality grade
        st.markdown("### 📊 Positions by Quality Grade")
        grade_groups = active.groupby('quality_grade').size()
        cols = st.columns(len(grade_groups))
        
        for idx, (grade, count) in enumerate(grade_groups.items()):
            with cols[idx]:
                st.markdown(display_quality_badge(grade), unsafe_allow_html=True)
                st.metric("Positions", int(count))
        
        st.markdown("---")
        
        # Display each position
        for _, row in active.iterrows():
            pnl = ((row['current_price'] - row['entry_price']) / row['entry_price'] * 100)
            color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            
            # Check TP/SL hits
            tp1_hit = pnl >= 8
            tp2_hit = pnl >= 15
            sl_hit = pnl <= -6
            
            status_msg = ""
            if tp2_hit:
                status_msg = " 🎯 TP2 HIT!"
            elif tp1_hit:
                status_msg = " 🎯 TP1 HIT!"
            elif sl_hit:
                status_msg = " 🛑 STOP LOSS!"
            
            with st.expander(f"{color} {row['ticker']} | Grade {row['quality_grade']} | P/L: {pnl:+.2f}%{status_msg}"):
                # Header
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(display_quality_badge(row['quality_grade']), unsafe_allow_html=True)
                    st.markdown(f"**{display_risk_badge(row['risk_level'])}**")
                with col2:
                    st.markdown(f"**{row['signal']}**")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Entry", f"Rp {row['entry_price']:,.0f}")
                col2.metric("Current", f"Rp {row['current_price']:,.0f}")
                col3.metric("P/L", f"{pnl:+.2f}%")
                col4.metric("Score", f"{row['score']}/100")
                
                # Position management guidance
                st.markdown("### 🎯 Position Management")
                
                if tp2_hit:
                    st.success("""
                    **🎉 TP2 (+15%) HIT!**
                    - ✅ Sell 2nd lot (1/3) if not done
                    - 🏃 Trail final 1/3 with 20 EMA
                    - 🛡️ Move SL to +10% (protect profits)
                    - 📊 Let winner run with trailing stop
                    """)
                elif tp1_hit:
                    st.success("""
                    **🎯 TP1 (+8%) HIT!**
                    - ✅ Sell 1st lot (1/3 position)
                    - 🛡️ Move SL to breakeven on rest
                    - 🎯 Target TP2 at +15% for 2nd lot
                    - 📈 Let 2/3 position run
                    """)
                elif sl_hit:
                    st.error("""
                    **🛑 STOP LOSS HIT!**
                    - ❌ Exit entire position NOW
                    - 📝 Document what went wrong
                    - 🔍 Was signal quality good?
                    - 📊 Did you follow entry plan?
                    """)
                elif pnl > 0:
                    st.info(f"""
                    **📈 In Profit (+{pnl:.2f}%)**
                    - 🎯 Next target: TP1 at +8%
                    - 🛡️ Keep SL at -6%
                    - 📊 Current position: Full (3/3)
                    - ⏳ Hold and let it develop
                    """)
                else:
                    st.warning(f"""
                    **📉 In Drawdown ({pnl:.2f}%)**
                    - 🛡️ Stop loss at -6%: Rp {row['entry_price'] * 0.94:,.0f}
                    - ⚠️ Currently {abs(pnl):.1f}% down
                    - 📊 Monitor closely
                    - ❌ Cut if SL hit - no emotions!
                    """)
                
                # Close position section
                st.markdown("---")
                st.markdown("### 📝 Close Position")
                
                col1, col2, col3 = st.columns(3)
                
                if col1.button("✅ WIN", key=f"win_{row['id']}"):
                    notes_options = [
                        "✅ TP hit as planned",
                        "✅ Strong momentum follow-through",
                        "✅ Grade A setup performed well",
                        "✅ Wyckoff accumulation confirmed",
                        "✅ Perfect entry timing"
                    ]
                    note = st.selectbox("Why win?", notes_options, key=f"wn_{row['id']}")
                    
                    conn = sqlite3.connect('screener_tracking.db')
                    c = conn.cursor()
                    c.execute('''UPDATE recommendations 
                                 SET status='CLOSED', result='WIN', profit_pct=?, 
                                     exit_price=?, exit_date=?, notes=?
                                 WHERE id=?''',
                              (pnl, row['current_price'], datetime.now().strftime('%Y-%m-%d'), 
                               note, row['id']))
                    conn.commit()
                    conn.close()
                    st.success("✅ Closed as WIN!")
                    time.sleep(1)
                    st.rerun()
                
                if col2.button("❌ LOSS", key=f"loss_{row['id']}"):
                    notes_options = [
                        "❌ SL hit - respected plan",
                        "❌ Failed support level",
                        "❌ Market reversal/weakness",
                        "❌ Volume dried up",
                        "❌ Grade quality was marginal",
                        "❌ Ignored early warning signs"
                    ]
                    note = st.selectbox("Why loss?", notes_options, key=f"ln_{row['id']}")
                    
                    conn = sqlite3.connect('screener_tracking.db')
                    c = conn.cursor()
                    c.execute('''UPDATE recommendations 
                                 SET status='CLOSED', result='LOSS', profit_pct=?, 
                                     exit_price=?, exit_date=?, notes=?
                                 WHERE id=?''',
                              (pnl, row['current_price'], datetime.now().strftime('%Y-%m-%d'), 
                               note, row['id']))
                    conn.commit()
                    conn.close()
                    st.error("❌ Closed as LOSS")
                    time.sleep(1)
                    st.rerun()
                
                if col3.button("⚪ BE", key=f"be_{row['id']}"):
                    notes_options = [
                        "⚪ Exited at breakeven - protected capital",
                        "⚪ Cut early due to uncertainty",
                        "⚪ Better opportunity elsewhere"
                    ]
                    note = st.selectbox("Why BE?", notes_options, key=f"bn_{row['id']}")
                    
                    conn = sqlite3.connect('screener_tracking.db')
                    c = conn.cursor()
                    c.execute('''UPDATE recommendations 
                                 SET status='CLOSED', result='BE', profit_pct=?, 
                                     exit_price=?, exit_date=?, notes=?
                                 WHERE id=?''',
                              (pnl, row['current_price'], datetime.now().strftime('%Y-%m-%d'), 
                               note, row['id']))
                    conn.commit()
                    conn.close()
                    st.info("⚪ Closed at breakeven")
                    time.sleep(1)
                    st.rerun()

# ============= FOOTER =============
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:2rem;color:#64748b'>
    <h3>🎯 IDX Power Screener v4.0 - Accuracy First Edition</h3>
    <p><strong>Key Features:</strong> Multi-Layer Validation | Wyckoff Phase Detection | Quality Grading A-D | Smart Money Tracking</p>
    <p><strong>Your Recovery Goal:</strong> 60%+ Win Rate by focusing on Grade A setups only</p>
    <p style='margin-top:1rem;font-size:0.9rem'>
        💡 <strong>Trading Wisdom:</strong> "Quality over Quantity - One great setup beats ten mediocre ones"
    </p>
    <p style='margin-top:0.5rem;font-size:0.85rem;color:#94a3b8'>
        ⚠️ <strong>Risk Disclaimer:</strong> This tool is for educational purposes. Always do your own research and never risk more than you can afford to lose.
    </p>
</div>
""", unsafe_allow_html=True)
