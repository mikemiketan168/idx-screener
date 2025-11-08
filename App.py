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

st.set_page_config(page_title="IDX Power Screener v3.2", page_icon="🚀", layout="wide")

st.markdown("""
<style>
.big-title {font-size:2.5rem;font-weight:800;color:#1e40af}
.subtitle {font-size:1.1rem;color:#64748b;margin-bottom:2rem}
.signal-box {padding:1rem;border-radius:0.5rem;margin:1rem 0;font-weight:700;text-align:center}
.strong-buy {background:#10b981;color:white}
.buy {background:#34d399;color:white}
.neutral {background:#fbbf24;color:white}
.sell {background:#ef4444;color:white}
.phase-akum {background:#10b981;color:white;padding:0.5rem;border-radius:0.5rem;font-weight:700}
.phase-markup {background:#3b82f6;color:white;padding:0.5rem;border-radius:0.5rem;font-weight:700}
.phase-dist {background:#ef4444;color:white;padding:0.5rem;border-radius:0.5rem;font-weight:700}
.phase-side {background:#6b7280;color:white;padding:0.5rem;border-radius:0.5rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

# ============= DATABASE SETUP =============
def init_db():
    conn = sqlite3.connect('screener_tracking.db')
    c = conn.cursor()
    
    # Recommendations table
    c.execute('''CREATE TABLE IF NOT EXISTS recommendations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT, ticker TEXT, strategy TEXT, score INTEGER,
                  confidence INTEGER, entry_price REAL, current_price REAL,
                  signal TEXT, status TEXT DEFAULT 'ACTIVE', result TEXT,
                  profit_pct REAL, exit_price REAL, exit_date TEXT, notes TEXT,
                  position_size TEXT DEFAULT '3/3')''')
    
    # Watchlist table
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date_added TEXT, ticker TEXT, strategy TEXT,
                  score INTEGER, confidence INTEGER, target_entry REAL,
                  current_price REAL, notes TEXT, status TEXT DEFAULT 'WATCHING')''')
    
    # Backtest results table (NEW!)
    c.execute('''CREATE TABLE IF NOT EXISTS backtest_results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  strategy TEXT, period TEXT, date_run TEXT,
                  total_signals INTEGER, wins INTEGER, losses INTEGER,
                  win_rate REAL, avg_win REAL, avg_loss REAL,
                  profit_factor REAL, max_drawdown REAL,
                  sharpe_ratio REAL, details TEXT)''')
    
    # Price alerts table (NEW!)
    c.execute('''CREATE TABLE IF NOT EXISTS price_alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ticker TEXT, alert_type TEXT, target_price REAL,
                  current_price REAL, status TEXT DEFAULT 'ACTIVE',
                  created_date TEXT, triggered_date TEXT,
                  message TEXT)''')
    
    conn.commit()
    conn.close()

# ============= TIMEZONE HELPERS =============
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
        return "🟡 LUNCH BREAK (12:00-13:00)", False
    else:
        return "🟢 MARKET OPEN", True

def is_valid_bpjs_time():
    """BPJS best 09:00-09:30 Jakarta time"""
    jkt_time = get_jakarta_time()
    return 9 <= jkt_time.hour < 10

def is_valid_bsjp_time():
    """BSJP best 14:00-15:00 Jakarta time"""
    jkt_time = get_jakarta_time()
    return 14 <= jkt_time.hour < 16

# ============= FETCH DATA =============
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, period="6mo"):
    try:
        end = int(datetime.now().timestamp())
        days = {"5d":5,"1mo":30,"3mo":90,"6mo":180,"1y":365}.get(period,180)
        start = end - (days*86400)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        r = requests.get(url, params={"period1":start,"period2":end,"interval":"1d"}, 
                        headers={'User-Agent':'Mozilla/5.0'}, timeout=15, verify=False)
        if r.status_code != 200:
            return None
        data = r.json()['chart']['result'][0]
        q = data['indicators']['quote'][0]
        df = pd.DataFrame({'Open':q['open'],'High':q['high'],'Low':q['low'],
                          'Close':q['close'],'Volume':q['volume']}, 
                         index=pd.to_datetime(data['timestamp'], unit='s'))
        df = df.dropna()
        if len(df) < 20:
            return None
        
        # Technical Indicators
        df['EMA5'] = df['Close'].ewm(5).mean()
        df['EMA9'] = df['Close'].ewm(9).mean()
        df['EMA21'] = df['Close'].ewm(21).mean()
        df['EMA50'] = df['Close'].ewm(50).mean()
        df['EMA200'] = df['Close'].ewm(200).mean() if len(df) >= 200 else df['Close'].ewm(len(df)).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta>0,0)).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        df['RSI'] = 100 - (100/(1+gain/loss))
        
        exp1 = df['Close'].ewm(12).mean()
        exp2 = df['Close'].ewm(26).mean()
        df['MACD'] = exp1 - exp2
        df['SIGNAL'] = df['MACD'].ewm(9).mean()
        df['MACD_HIST'] = df['MACD'] - df['SIGNAL']
        
        df['BB_MID'] = df['Close'].rolling(20).mean()
        df['BB_STD'] = df['Close'].rolling(20).std()
        df['BB_UPPER'] = df['BB_MID'] + 2*df['BB_STD']
        df['BB_LOWER'] = df['BB_MID'] - 2*df['BB_STD']
        
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['STOCH_K'] = 100*(df['Close']-low14)/(high14-low14)
        df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
        
        df['VOL_SMA20'] = df['Volume'].rolling(20).mean()
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA20']
        
        df['MOM_5D'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        df['MOM_10D'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        
        # ATR for volatility
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        return df
    except:
        return None

def fetch_data_with_retry(ticker, period="6mo", max_retries=3):
    """Fetch with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return fetch_data(ticker, period)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return None

def check_data_freshness(df):
    """Check if data is current"""
    if df is None or len(df) == 0:
        return "❌ No data", "error"
    
    last_date = df.index[-1]
    now = datetime.now()
    
    if last_date.tzinfo is not None:
        last_date = last_date.tz_localize(None)
    
    age_hours = (now - last_date).total_seconds() / 3600
    
    if age_hours < 24:
        return f"✅ Fresh ({age_hours:.0f}h old)", "success"
    elif age_hours < 48:
        return f"⚠️ Stale ({age_hours:.0f}h old)", "warning"
    else:
        return f"❌ Very old ({age_hours/24:.0f}d old)", "error"

# ============= MULTI-TIMEFRAME ANALYSIS (NEW!) =============
def analyze_multi_timeframe(ticker):
    """Analyze ticker across multiple timeframes"""
    results = {}
    
    for tf in ['1mo', '3mo', '6mo']:
        df = fetch_data(ticker, tf)
        if df is not None:
            score, details, conf = score_full_screener_v3(df)
            
            # Determine trend
            if score >= 75:
                trend = "Strong Bull"
            elif score >= 65:
                trend = "Bull"
            elif score >= 50:
                trend = "Neutral"
            elif score > 0:
                trend = "Weak"
            else:
                trend = "Bear"
            
            results[tf] = {
                'score': score,
                'confidence': conf,
                'trend': trend,
                'aligned': score >= 65
            }
    
    # Check alignment
    if len(results) >= 2:
        aligned = all(r['aligned'] for r in results.values())
        avg_score = sum(r['score'] for r in results.values()) / len(results)
        
        if aligned and avg_score >= 70:
            verdict = "🟢 STRONG BUY - All timeframes aligned!"
        elif aligned:
            verdict = "🟡 BUY - Timeframes aligned"
        else:
            verdict = "🔴 CAUTION - Timeframe conflict"
    else:
        verdict = "⚪ Insufficient data"
    
    return results, verdict

# ============= SUPPORT & RESISTANCE FINDER (NEW!) =============
# ============= SUPPORT & RESISTANCE FINDER (NO SCIPY VERSION) =============
def find_support_resistance(df, ticker_price):
    """Find key support and resistance levels - No scipy version"""
    if df is None or len(df) < 50:
        return None
    
    try:
        # Get recent price action
        recent_df = df.tail(100)
        
        # Find resistance levels (local highs)
        resistances = []
        for i in range(5, len(recent_df) - 5):
            current_high = recent_df['High'].iloc[i]
            # Check if it's a local maximum
            is_peak = True
            for j in range(max(0, i-5), min(len(recent_df), i+6)):
                if j != i and recent_df['High'].iloc[j] > current_high:
                    is_peak = False
                    break
            
            if is_peak and current_high > ticker_price:
                resistances.append(current_high)
        
        # Find support levels (local lows)
        supports = []
        for i in range(5, len(recent_df) - 5):
            current_low = recent_df['Low'].iloc[i]
            # Check if it's a local minimum
            is_trough = True
            for j in range(max(0, i-5), min(len(recent_df), i+6)):
                if j != i and recent_df['Low'].iloc[j] < current_low:
                    is_trough = False
                    break
            
            if is_trough and current_low < ticker_price:
                supports.append(current_low)
        
        # Remove duplicates and sort
        resistances = sorted(list(set([round(r, -1) for r in resistances])))[:3]
        supports = sorted(list(set([round(s, -1) for s in supports])), reverse=True)[:3]
        
        # Add EMAs as dynamic support/resistance
        current_ema50 = df['EMA50'].iloc[-1]
        current_ema200 = df['EMA200'].iloc[-1]
        
        sr_data = {
            'resistances': [],
            'supports': [],
            'current_price': ticker_price
        }
        
        # Add resistances
        for i, r in enumerate(resistances):
            strength = "STRONG" if i == 0 else "MEDIUM" if i == 1 else "WEAK"
            sr_data['resistances'].append({
                'level': round(r, 0),
                'strength': strength,
                'distance_pct': ((r - ticker_price) / ticker_price * 100)
            })
        
        # Add supports
        for i, s in enumerate(supports):
            strength = "STRONG" if i == 0 else "MEDIUM" if i == 1 else "WEAK"
            sr_data['supports'].append({
                'level': round(s, 0),
                'strength': strength,
                'distance_pct': ((ticker_price - s) / ticker_price * 100)
            })
        
        # Add EMA50 as dynamic support if below price
        if current_ema50 < ticker_price:
            sr_data['supports'].append({
                'level': round(current_ema50, 0),
                'strength': 'DYNAMIC',
                'distance_pct': ((ticker_price - current_ema50) / ticker_price * 100),
                'type': 'EMA50'
            })
        
        # Add EMA200 as major support/resistance
        if current_ema200 < ticker_price:
            sr_data['supports'].append({
                'level': round(current_ema200, 0),
                'strength': 'MAJOR',
                'distance_pct': ((ticker_price - current_ema200) / ticker_price * 100),
                'type': 'EMA200'
            })
        elif current_ema200 > ticker_price:
            sr_data['resistances'].append({
                'level': round(current_ema200, 0),
                'strength': 'MAJOR',
                'distance_pct': ((current_ema200 - ticker_price) / ticker_price * 100),
                'type': 'EMA200'
            })
        
        return sr_data
        
    except Exception as e:
        # Fallback: Return simple EMA-based S/R
        return {
            'resistances': [
                {'level': round(ticker_price * 1.05, 0), 'strength': 'ESTIMATED', 'distance_pct': 5.0},
                {'level': round(ticker_price * 1.10, 0), 'strength': 'ESTIMATED', 'distance_pct': 10.0}
            ],
            'supports': [
                {'level': round(ticker_price * 0.95, 0), 'strength': 'ESTIMATED', 'distance_pct': 5.0},
                {'level': round(ticker_price * 0.90, 0), 'strength': 'ESTIMATED', 'distance_pct': 10.0}
            ],
            'current_price': ticker_price
        }

# ============= VALIDATION =============
def validate_not_downtrend(df):
    """Validate not in downtrend - CRITICAL CHECK"""
    try:
        r = df.iloc[-1]
        reasons = []
        
        if r['Close'] < r['EMA50'] and r['EMA50'] < r['EMA200']:
            reasons.append("Price < EMA50 < EMA200")
        
        if r['EMA9'] < r['EMA21'] < r['EMA50']:
            reasons.append("EMAs in death cross")
        
        mom_10d = df['MOM_10D'].iloc[-1]
        if mom_10d < -5:
            reasons.append(f"10D Mom: {mom_10d:.1f}%")
        
        if len(reasons) >= 2:
            return False, " | ".join(reasons)
        elif len(reasons) == 1:
            return True, f"⚠️ Warning: {reasons[0]}"
        else:
            return True, "✅ Trend OK"
            
    except Exception as e:
        return True, f"Unable to validate: {str(e)}"

def validate_volume_quality(df):
    try:
        r = df.iloc[-1]
        if r['VOL_RATIO'] > 5:
            if abs(df['MOM_5D'].iloc[-1]) < 2:
                return False, "Suspicious volume spike without price action"
        if df['MOM_5D'].iloc[-1] > 3 and r['VOL_RATIO'] < 0.7:
            return False, "Weak volume during rally"
        return True, "Volume acceptable"
    except:
        return True, "Unable to validate"

def validate_not_overbought(df):
    try:
        r = df.iloc[-1]
        signals = 0
        reasons = []
        if r['RSI'] > 75:
            signals += 1
            reasons.append(f"RSI: {r['RSI']:.1f}")
        if r['STOCH_K'] > 85:
            signals += 1
            reasons.append(f"Stoch: {r['STOCH_K']:.1f}")
        if signals >= 2:
            return False, " | ".join(reasons)
        return True, "Not overbought"
    except:
        return True, "Unable to validate"

# ============= SCORING FUNCTIONS =============
def score_full_screener_v3(df):
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        confidence = 0
        
        is_valid, reason = validate_not_downtrend(df)
        if not is_valid:
            details['⛔ REJECTED'] = reason
            return 0, details, 0
        
        vol_valid, vol_reason = validate_volume_quality(df)
        if not vol_valid:
            details['⚠️ Volume Warning'] = vol_reason
            score -= 20
        
        if r['Close'] > r['EMA9'] > r['EMA21'] > r['EMA50']:
            if r['EMA50'] > r['EMA200']:
                score += 35
                details['Trend'] = '✅ PERFECT BULL (+35)'
                confidence += 35
            else:
                score += 25
                details['Trend'] = '✅ Strong up (+25)'
                confidence += 25
        elif r['Close'] > r['EMA9'] > r['EMA21']:
            score += 18
            details['Trend'] = '✅ Short up (+18)'
            confidence += 18
        
        mom_5d = df['MOM_5D'].iloc[-1]
        mom_10d = df['MOM_10D'].iloc[-1]
        if 2 <= mom_5d <= 8 and mom_10d > 0:
            score += 25
            details['Momentum'] = f'✅ IDEAL {mom_5d:.1f}% (+25)'
            confidence += 25
        elif 0 < mom_5d <= 12:
            score += 15
            details['Momentum'] = f'✅ Good {mom_5d:.1f}% (+15)'
            confidence += 15
        
        if 45 <= r['RSI'] <= 60:
            score += 20
            details['RSI'] = f'✅ SWEET {r["RSI"]:.1f} (+20)'
            confidence += 20
        elif 40 <= r['RSI'] <= 65:
            score += 15
            details['RSI'] = f'✅ Good {r["RSI"]:.1f} (+15)'
            confidence += 15
        
        if r['MACD'] > r['SIGNAL'] and r['MACD_HIST'] > df['MACD_HIST'].iloc[-2]:
            score += 15
            details['MACD'] = '✅ STRONG BULL (+15)'
            confidence += 15
        elif r['MACD'] > r['SIGNAL']:
            score += 10
            details['MACD'] = '✅ Bullish (+10)'
            confidence += 10
        
        if r['VOL_RATIO'] > 1.8 and mom_5d > 1:
            score += 20
            details['Volume'] = f'✅ BREAKOUT {r["VOL_RATIO"]:.2f}x (+20)'
            confidence += 20
        elif r['VOL_RATIO'] > 1.3:
            score += 15
            details['Volume'] = f'✅ Strong {r["VOL_RATIO"]:.2f}x (+15)'
            confidence += 15
        
        confidence = min(int(confidence * 0.9), 100)
        if score < 30:
            return 0, details, 0
        return score, details, confidence
    except:
        return 0, {}, 0

def score_bpjs_v3(df):
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        confidence = 0
        
        is_valid, reason = validate_not_downtrend(df)
        if not is_valid:
            details['⛔ REJECTED'] = reason
            return 0, details, 0
        
        vol_pct = ((df['High']-df['Low'])/df['Low']*100).tail(5).mean()
        if 1.5 < vol_pct < 5:
            score += 30
            details['Volatility'] = f'✅ IDEAL {vol_pct:.2f}% (+30)'
            confidence += 30
        
        if r['VOL_RATIO'] > 2:
            score += 30
            details['Volume'] = f'✅ HUGE {r["VOL_RATIO"]:.2f}x (+30)'
            confidence += 30
        elif r['VOL_RATIO'] > 1.5:
            score += 20
            details['Volume'] = f'✅ Strong {r["VOL_RATIO"]:.2f}x (+20)'
            confidence += 20
        
        if 30 < r['RSI'] < 45:
            score += 25
            details['RSI'] = f"✅ OVERSOLD {r['RSI']:.1f} (+25)"
            confidence += 25
        
        if r['STOCH_K'] < 30 and r['STOCH_K'] > r['STOCH_D']:
            score += 15
            details['Stochastic'] = f"✅ CROSS {r['STOCH_K']:.1f} (+15)"
            confidence += 15
        
        confidence = min(int(confidence), 100)
        return score, details, confidence
    except:
        return 0, {}, 0

def score_bsjp_v3(df):
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        confidence = 0
        
        is_valid, reason = validate_not_downtrend(df)
        if not is_valid:
            details['⛔ REJECTED'] = reason
            return 0, details, 0
        
        bb_pos = (r['Close']-r['BB_LOWER'])/(r['BB_UPPER']-r['BB_LOWER'])*100
        if bb_pos < 15:
            score += 30
            details['BB Position'] = f'✅ EXTREME {bb_pos:.1f}% (+30)'
            confidence += 30
        
        gap = (r['Close']-df['Close'].iloc[-2])/df['Close'].iloc[-2]*100
        if -2 < gap < -0.5:
            score += 25
            details['Gap'] = f'✅ IDEAL {gap:.2f}% (+25)'
            confidence += 25
        
        if 30 < r['RSI'] < 50:
            score += 25
            details['RSI'] = f"✅ OVERSOLD {r['RSI']:.1f} (+25)"
            confidence += 25
        
        confidence = min(int(confidence), 100)
        return score, details, confidence
    except:
        return 0, {}, 0

def score_bandar_v3(df):
    """Enhanced Bandar tracking with better logic"""
    try:
        # Calculate OBV (On-Balance Volume)
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        
        # Calculate OBV MA for trend
        df['OBV_MA20'] = pd.Series(df['OBV']).rolling(20).mean()
        
        # Recent metrics (last 5 vs 20 days)
        recent_vol = df['Volume'].tail(5).mean()
        avg_vol_20 = df['Volume'].tail(20).mean()
        vol_ratio = recent_vol / avg_vol_20 if avg_vol_20 > 0 else 1
        
        # Price change analysis
        current_price = df['Close'].iloc[-1]
        price_20_ago = df['Close'].iloc[-20]
        price_chg_pct = ((current_price - price_20_ago) / price_20_ago * 100)
        
        # OBV trend
        current_obv = df['OBV'].iloc[-1]
        obv_20_ago = df['OBV'].iloc[-20]
        obv_trend = ((current_obv - obv_20_ago) / abs(obv_20_ago) * 100) if obv_20_ago != 0 else 0
        
        # OBV vs Price divergence
        obv_ma_current = df['OBV_MA20'].iloc[-1]
        obv_ma_prev = df['OBV_MA20'].iloc[-10]
        obv_rising = obv_ma_current > obv_ma_prev
        
        price_rising = current_price > df['Close'].iloc[-10]
        
        # ATR for volatility context
        atr_pct = (df['ATR'].iloc[-1] / current_price * 100)
        
        details = {}
        phase = ""
        score = 0
        confidence = 0
        
        # ===== PHASE DETECTION LOGIC =====
        
        # ACCUMULATION: High volume, price sideways/down, OBV rising
        if (vol_ratio > 1.3 and 
            -3 < price_chg_pct < 5 and 
            obv_rising and 
            obv_trend > 5):
            
            phase = '🟢 AKUMULASI'
            score = 90
            confidence = 85
            
            details['Phase'] = 'AKUMULASI (Smart Money Buying)'
            details['Action'] = '🚀 STRONG BUY'
            details['Signal'] = 'Volume ↑ + Price Sideways + OBV ↑'
            details['Strength'] = 'VERY STRONG' if vol_ratio > 1.8 else 'STRONG'
            
        # MARKUP: Price breaking out with volume, OBV confirming
        elif (price_chg_pct > 5 and 
              vol_ratio > 1.1 and 
              obv_rising):
            
            phase = '🚀 MARKUP'
            score = 85
            confidence = 80
            
            details['Phase'] = 'MARKUP (Uptrend Confirmed)'
            details['Action'] = '🎯 HOLD / BUY PULLBACK'
            details['Signal'] = 'Breakout with volume'
            details['Strength'] = 'STRONG' if price_chg_pct > 10 else 'MODERATE'
            
        # DISTRIBUTION: High volume, price not moving up, OBV declining
        elif (vol_ratio > 1.5 and 
              price_chg_pct < 3 and 
              not obv_rising and 
              current_price > df['Close'].rolling(50).mean().iloc[-1]):
            
            phase = '🔴 DISTRIBUSI'
            score = 15
            confidence = 20
            
            details['Phase'] = 'DISTRIBUSI (Smart Money Selling)'
            details['Action'] = '🛑 SELL / AVOID'
            details['Signal'] = 'High volume but no price gain'
            details['Strength'] = 'DANGER!'
            
        # MARKDOWN: Price falling with volume, OBV declining
        elif (price_chg_pct < -5 and 
              not obv_rising):
            
            phase = '⚫ MARKDOWN'
            score = 10
            confidence = 15
            
            details['Phase'] = 'MARKDOWN (Downtrend)'
            details['Action'] = '🚫 STAY AWAY'
            details['Signal'] = 'Falling prices'
            details['Strength'] = 'AVOID'
            
        # SIDEWAYS: No clear pattern
        else:
            phase = '⚪ SIDEWAYS'
            score = 50
            confidence = 50
            
            details['Phase'] = 'SIDEWAYS (No Clear Pattern)'
            details['Action'] = '⏸️ WAIT & WATCH'
            details['Signal'] = 'Mixed signals'
            details['Strength'] = 'NEUTRAL'
        
        # Additional metrics
        details['Volume_Ratio'] = f'{vol_ratio:.2f}x'
        details['Price_Change'] = f'{price_chg_pct:+.2f}%'
        details['OBV_Trend'] = f'{obv_trend:+.1f}%'
        details['Volatility'] = f'{atr_pct:.2f}%'
        
        # Risk assessment
        if phase == '🟢 AKUMULASI':
            details['Risk'] = 'LOW (Best entry point)'
            details['Hold_Period'] = '2-8 weeks'
            details['Target'] = '+15% to +40%'
        elif phase == '🚀 MARKUP':
            details['Risk'] = 'MEDIUM (Trend following)'
            details['Hold_Period'] = '1-4 weeks'
            details['Target'] = '+10% to +25%'
        elif phase == '🔴 DISTRIBUSI':
            details['Risk'] = 'VERY HIGH (Exit zone)'
            details['Hold_Period'] = 'EXIT ASAP'
            details['Target'] = 'Preserve capital'
        else:
            details['Risk'] = 'MEDIUM (Uncertain)'
            details['Hold_Period'] = 'Wait for clear signal'
            details['Target'] = 'TBD'
        
        return score, details, phase, confidence
        
    except Exception as e:
        return 0, {'Error': str(e)}, 'UNKNOWN', 0

def score_value_v3(df):
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        confidence = 0
        
        is_valid, reason = validate_not_downtrend(df)
        if not is_valid:
            details['⛔ REJECTED'] = reason
            return 0, details, 0
        
        high52 = df['High'].tail(252).max() if len(df)>252 else df['High'].max()
        low52 = df['Low'].tail(252).min() if len(df)>252 else df['Low'].min()
        pos52 = (r['Close']-low52)/(high52-low52)*100
        
        if pos52 < 20:
            score += 30
            details['52W'] = f'✅ DEEP VALUE {pos52:.1f}% (+30)'
            confidence += 30
        elif pos52 < 30:
            score += 20
            details['52W'] = f'✅ Undervalued {pos52:.1f}% (+20)'
            confidence += 20
        
        if 20 < r['RSI'] < 35:
            score += 25
            details['RSI'] = f"✅ OVERSOLD {r['RSI']:.1f} (+25)"
            confidence += 25
        
        if r['VOL_RATIO'] > 1.8:
            score += 20
            details['Volume'] = f'✅ BUYING {r["VOL_RATIO"]:.2f}x (+20)'
            confidence += 20
        
        if r['Close'] > r['SMA20']:
            score += 15
            details['Trend'] = '✅ REVERSAL (+15)'
            confidence += 15
        
        confidence = min(int(confidence), 100)
        return score, details, confidence
    except:
        return 0, {}, 0

# ============= POSITION MANAGEMENT =============
def calculate_three_lot_strategy(entry_price):
    """Calculate 3-lot position management"""
    return {
        'lot1_tp': round(entry_price * 1.08, 0),
        'lot2_tp': round(entry_price * 1.15, 0),
        'lot3_trail': 'Trail with 20D EMA',
        'initial_sl': round(entry_price * 0.94, 0)
    }

def calculate_position_size(account_size, risk_per_trade, entry, stop_loss):
    """Calculate position size based on risk"""
    risk_amount = account_size * (risk_per_trade / 100)
    risk_per_share = abs(entry - stop_loss)
    
    if risk_per_share == 0:
        return None
    
    shares = int(risk_amount / risk_per_share)
    position_value = shares * entry
    
    return {
        'shares': shares,
        'position_value': position_value,
        'risk_amount': risk_amount,
        'position_pct': (position_value / account_size * 100)
    }

# ============= TRACKING FUNCTIONS =============
def save_recommendation(ticker, strategy, score, confidence, entry_price, signal):
    conn = sqlite3.connect('screener_tracking.db')
    c = conn.cursor()
    c.execute('''INSERT INTO recommendations 
                 (date, ticker, strategy, score, confidence, entry_price, current_price, signal, status)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker, strategy, score, 
               confidence, entry_price, entry_price, signal, 'ACTIVE'))
    conn.commit()
    conn.close()

def add_to_watchlist(ticker, strategy, score, confidence, target_entry, notes=""):
    """Add stock to watchlist"""
    conn = sqlite3.connect('screener_tracking.db')
    c = conn.cursor()
    c.execute('''INSERT INTO watchlist 
                 (date_added, ticker, strategy, score, confidence, target_entry, current_price, notes)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime('%Y-%m-%d'), ticker, strategy, score, 
               confidence, target_entry, target_entry, notes))
    conn.commit()
    conn.close()

def get_active_recommendations():
    conn = sqlite3.connect('screener_tracking.db')
    df = pd.read_sql("SELECT * FROM recommendations WHERE status='ACTIVE' ORDER BY date DESC", conn)
    conn.close()
    return df

def get_watchlist():
    conn = sqlite3.connect('screener_tracking.db')
    df = pd.read_sql("SELECT * FROM watchlist WHERE status='WATCHING' ORDER BY date_added DESC", conn)
    conn.close()
    return df

def update_recommendation_status(rec_id, status, result, profit_pct, exit_price, notes=""):
    conn = sqlite3.connect('screener_tracking.db')
    c = conn.cursor()
    c.execute('''UPDATE recommendations 
                 SET status=?, result=?, profit_pct=?, exit_price=?, exit_date=?, notes=?
                 WHERE id=?''',
              (status, result, profit_pct, exit_price, datetime.now().strftime('%Y-%m-%d'), notes, rec_id))
    conn.commit()
    conn.close()

def get_performance_stats():
    conn = sqlite3.connect('screener_tracking.db')
    total = pd.read_sql("SELECT COUNT(*) as total FROM recommendations WHERE status='CLOSED'", conn)
    wins = pd.read_sql("SELECT COUNT(*) as wins FROM recommendations WHERE result='WIN'", conn)
    avg_profit = pd.read_sql("SELECT AVG(profit_pct) as avg FROM recommendations WHERE result='WIN'", conn)
    avg_loss = pd.read_sql("SELECT AVG(profit_pct) as avg FROM recommendations WHERE result='LOSS'", conn)
    by_strategy = pd.read_sql("""SELECT strategy, COUNT(*) as total,
                                         SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                                         AVG(profit_pct) as avg_profit
                                  FROM recommendations WHERE status='CLOSED'
                                  GROUP BY strategy""", conn)
    by_confidence = pd.read_sql("""SELECT 
                                          CASE WHEN confidence >= 80 THEN 'High (80+)'
                                               WHEN confidence >= 60 THEN 'Medium (60-79)'
                                               ELSE 'Low (<60)' END as conf_level,
                                          COUNT(*) as total,
                                          SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                                          AVG(profit_pct) as avg_profit
                                   FROM recommendations WHERE status='CLOSED'
                                   GROUP BY conf_level""", conn)
    conn.close()
    return {
        'total': total['total'].iloc[0] if not total.empty else 0,
        'wins': wins['wins'].iloc[0] if not wins.empty else 0,
        'avg_profit': avg_profit['avg'].iloc[0] if not avg_profit.empty else 0,
        'avg_loss': avg_loss['avg'].iloc[0] if not avg_loss.empty else 0,
        'by_strategy': by_strategy,
        'by_confidence': by_confidence
    }

def get_position_summary(active_df):
    """Calculate total P&L across positions"""
    if active_df.empty:
        return None
    
    active_df['pnl_pct'] = ((active_df['current_price'] - active_df['entry_price']) / 
                             active_df['entry_price'] * 100)
    
    return {
        'total_positions': len(active_df),
        'winning': len(active_df[active_df['pnl_pct'] > 0]),
        'losing': len(active_df[active_df['pnl_pct'] < 0]),
        'avg_pnl': active_df['pnl_pct'].mean(),
        'best_trade': active_df.loc[active_df['pnl_pct'].idxmax()] if len(active_df) > 0 else None,
        'worst_trade': active_df.loc[active_df['pnl_pct'].idxmin()] if len(active_df) > 0 else None
    }

def get_strategy_health(stats):
    """Assess strategy health"""
    win_rate = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
    
    if win_rate < 40:
        return "🔴 CRITICAL - Review strategy", "error"
    elif win_rate < 50:
        return "🟡 CAUTION - Needs improvement", "warning"
    elif win_rate < 60:
        return "🟢 ACCEPTABLE - Keep refining", "info"
    else:
        return "🟢 EXCELLENT - Strong edge", "success"

def plot_strategy_comparison(stats):
    """Compare strategy performance"""
    if stats['by_strategy'].empty:
        return
    
    df = stats['by_strategy'].copy()
    df['win_rate'] = (df['wins'] / df['total'] * 100)
    
    st.markdown("### 📊 Strategy Win Rates")
    for _, row in df.iterrows():
        win_rate = row['win_rate']
        color = "🟢" if win_rate >= 60 else "🟡" if win_rate >= 50 else "🔴"
        
        st.markdown(f"**{row['strategy']}** {color}")
        st.progress(min(win_rate / 100, 1.0))
        st.caption(f"WR: {win_rate:.1f}% | Trades: {int(row['total'])} | Avg: {row['avg_profit']:+.2f}%")

def bulk_update_active_positions():
    """Update all active positions with current prices"""
    conn = sqlite3.connect('screener_tracking.db')
    active = pd.read_sql("SELECT * FROM recommendations WHERE status='ACTIVE'", conn)
    
    updated = 0
    failed = 0
    
    progress = st.progress(0)
    status_text = st.empty()
    
    for idx, row in active.iterrows():
        status_text.text(f"Updating {row['ticker']}...")
        progress.progress((idx + 1) / len(active))
        
        ticker = row['ticker'] if row['ticker'].endswith('.JK') else f"{row['ticker']}.JK"
        df = fetch_data_with_retry(ticker, "5d")
        
        if df is not None:
            c = conn.cursor()
            c.execute("UPDATE recommendations SET current_price=? WHERE id=?", 
                     (df['Close'].iloc[-1], row['id']))
            conn.commit()
            updated += 1
        else:
            failed += 1
        
        time.sleep(0.5)
    
    conn.close()
    progress.empty()
    status_text.empty()
    
    return updated, failed

# ============= SIGNAL LEVELS =============
def get_signal_levels(score, price, confidence):
    if score >= 80 and confidence >= 70:
        signal = "STRONG BUY"
        signal_class = "strong-buy"
        trend = "🟢 Strong Uptrend"
        entry_ideal = round(price*0.98,0)
        entry_aggr = round(price,0)
        tp1_ideal = round(entry_ideal*1.08,0)
        tp2_ideal = round(entry_ideal*1.15,0)
        sl_ideal = round(entry_ideal*0.94,0)
        sl_aggr = round(entry_aggr*0.94,0)
    elif score >= 65:
        signal = "BUY"
        signal_class = "buy"
        trend = "🟢 Uptrend"
        entry_ideal = round(price*0.98,0)
        entry_aggr = round(price,0)
        tp1_ideal = round(entry_ideal*1.08,0)
        tp2_ideal = round(entry_ideal*1.15,0)
        sl_ideal = round(entry_ideal*0.94,0)
        sl_aggr = round(entry_aggr*0.94,0)
    elif score >= 50:
        signal = "WATCH"
        signal_class = "neutral"
        trend = "🟡 Monitor"
        entry_ideal = round(price*0.96,0)
        entry_aggr = None
        tp1_ideal = round(entry_ideal*1.08,0)
        tp2_ideal = round(entry_ideal*1.15,0)
        sl_ideal = round(entry_ideal*0.96,0)
        sl_aggr = None
    else:
        signal = "PASS"
        signal_class = "sell"
        trend = "⚪ Wait"
        entry_ideal = None
        entry_aggr = None
        tp1_ideal = None
        tp2_ideal = None
        sl_ideal = None
        sl_aggr = None
    
    return {
        "signal": signal, "signal_class": signal_class, "trend": trend,
        "ideal": {"entry": entry_ideal, "tp1": tp1_ideal, "tp2": tp2_ideal, "sl": sl_ideal},
        "aggr": {"entry": entry_aggr, "tp1": tp1_ideal, "tp2": tp2_ideal, "sl": sl_aggr}
    }

def display_score_breakdown(details, score, confidence):
    """Show how score was calculated"""
    st.markdown("### 📊 Score Breakdown")
    
    components = []
    for key, value in details.items():
        if '(+' in str(value):
            try:
                points = int(str(value).split('(+')[1].split(')')[0])
                components.append((key, value, points))
            except:
                pass
    
    if not components:
        st.info("No detailed scoring breakdown available")
        return
    
    total_points = sum(c[2] for c in components)
    
    for component, description, points in sorted(components, key=lambda x: x[2], reverse=True):
        pct = (points / total_points * 100) if total_points > 0 else 0
        st.markdown(f"**{component}**: {description}")
        st.progress(pct / 100)
        st.caption(f"{points} points ({pct:.0f}% of total)")
    
    st.markdown("---")
    st.markdown(f"**Total Score:** {score}/100")
    st.markdown(f"**Confidence:** {confidence}%")

# ============= PROCESS =============
def process_ticker(ticker, strategy, period):
    try:
        df = fetch_data(ticker, period)
        if df is None or len(df) < 50:
            return None
        
        price = float(df['Close'].iloc[-1])
        
        if strategy == "BPJS":
            score, details, confidence = score_bpjs_v3(df)
        elif strategy == "BSJP":
            score, details, confidence = score_bsjp_v3(df)
        elif strategy == "Bandar":
            score, details, phase, confidence = score_bandar_v3(df)
            details['Phase'] = phase
        elif strategy == "Value":
            score, details, confidence = score_value_v3(df)
        else:
            score, details, confidence = score_full_screener_v3(df)
        
        if score == 0:
            return None
        
        levels = get_signal_levels(score, price, confidence)
        
        return {
            "Ticker": ticker, "Price": price, "Score": score, "Confidence": confidence,
            "Signal": levels["signal"], "Trend": levels["trend"],
            "EntryIdeal": levels["ideal"]["entry"], "EntryAggr": levels["aggr"]["entry"],
            "TP1": levels["ideal"]["tp1"], "TP2": levels["ideal"]["tp2"],
            "SL": levels["ideal"]["sl"], "Details": details
        }
    except:
        return None

def batch_scan(tickers, strategy, period, limit, use_parallel=True):
    results = []
    if limit and limit < len(tickers):
        tickers = tickers[:limit]
    
    progress = st.progress(0)
    status = st.empty()
    total = len(tickers)
    
    if use_parallel and total > 20:
        completed = 0
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_ticker, t, strategy, period): t for t in tickers}
            for future in as_completed(futures):
                completed += 1
                progress.progress(completed / total)
                status.text(f"📊 {completed}/{total}")
                result = future.result()
                if result:
                    results.append(result)
                time.sleep(0.1)
    else:
        for i, ticker in enumerate(tickers):
            progress.progress((i+1)/total)
            status.text(f"📊 {i+1}/{total}: {ticker}")
            result = process_ticker(ticker, strategy, period)
            if result:
                results.append(result)
            time.sleep(0.3)
    
    progress.empty()
    status.empty()
    
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values(["Score", "Confidence"], ascending=False)
    return df[df['Confidence'] >= 40]

def load_tickers():
    try:
        with open("idx_stocks.json","r") as f:
            data = json.load(f)
        tickers = data.get("tickers",[])
        return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except:
        return ["BBCA.JK","BBRI.JK","BMRI.JK","TLKM.JK","ASII.JK",
                "BREN.JK","BRPT.JK","RATU.JK","RAJA.JK","GOTO.JK",
                "ADRO.JK","ANTM.JK","BBNI.JK","INDF.JK","UNVR.JK"]

# ============= MAIN =============
init_db()

st.markdown('<div class="big-title">🚀 IDX Power Screener v3.2</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ultimate Edition: Multi-TF | Bandar Logic | S/R Finder | Advanced Tools</div>', unsafe_allow_html=True)

tickers = load_tickers()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Market status at top
    market_status, is_open = check_idx_market_status()
    if is_open:
        st.success(market_status)
    else:
        st.warning(market_status)
    
    jkt_time = get_jakarta_time()
    st.info(f"🕐 Jakarta: {jkt_time.strftime('%H:%M:%S WIB')}")
    
    st.markdown("---")
    
    menu = st.radio("📋 Menu", [
        "1️⃣ Full Screener", 
        "2️⃣ Single Stock + Multi-TF", 
        "3️⃣ BPJS", 
        "4️⃣ BSJP", 
        "5️⃣ Bandar Tracking 🔥", 
        "6️⃣ Value Hunting",
        "7️⃣ Track Performance", 
        "8️⃣ Active Positions",
        "9️⃣ Watchlist",
        "🧪 Test Cases"
    ])
    st.markdown("---")
    
    if menu not in ["7️⃣ Track Performance", "8️⃣ Active Positions", "9️⃣ Watchlist"]:
        period = st.selectbox("Period", ["3mo","6mo","1y"], index=1)
        if "Single" not in menu and "Test" not in menu:
            limit = st.slider("Max Tickers", 10, len(tickers), min(100, len(tickers)), step=10)
            min_score = st.slider("Min Score", 50, 100, 65, step=5)
            min_confidence = st.slider("Min Confidence", 40, 100, 60, step=5)
            use_parallel = st.checkbox("⚡ Fast Mode", value=True)
    
    st.markdown("---")
    
    # Position calculator
    with st.expander("💰 Position Calculator"):
        account = st.number_input("Account Size (Rp)", value=100_000_000, step=10_000_000, format="%d")
        risk_pct = st.slider("Risk per Trade (%)", 1.0, 5.0, 2.0, 0.5)
        
        st.caption(f"💵 Risk per trade: Rp {account * risk_pct / 100:,.0f}")
        st.caption("📊 Recommended: 2% per trade max")
    
    st.markdown("---")
    st.caption("💡 IDX Traders v3.2 - Ultimate Edition")
    st.caption("🔥 NEW: Enhanced Bandar Logic!")

# ============= MENU HANDLERS =============

if "Test" in menu:
    st.markdown("### 🧪 Test Cases")
    st.info("Testing validation logic and bandar detection on known stocks")
    
    if st.button("🔬 Run Comprehensive Test", type="primary"):
        test_stocks = ["BREN.JK", "BRPT.JK", "RATU.JK", "BBCA.JK", "GOTO.JK"]
        
        for ticker in test_stocks:
            df = fetch_data(ticker, "6mo")
            if df is not None:
                # Full screener score
                score, details, conf = score_full_screener_v3(df)
                
                # Bandar analysis
                band_score, band_details, band_phase, band_conf = score_bandar_v3(df)
                
                with st.expander(f"{ticker} - Score: {score} | Bandar: {band_phase}"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Score", f"{score}/100")
                    col2.metric("Confidence", f"{conf}%")
                    col3.metric("Price", f"Rp {df['Close'].iloc[-1]:,.0f}")
                    
                    st.markdown("### 📊 Full Screener Analysis")
                    if score >= 65:
                        st.success("✅ PASS - Should appear in results")
                    elif score > 0:
                        st.warning("⚠️ LOW SCORE")
                    else:
                        st.error("❌ REJECTED")
                    
                    for k, v in details.items():
                        if '⛔' in k:
                            st.error(f"**{k}:** {v}")
                        elif '⚠️' in k:
                            st.warning(f"**{k}:** {v}")
                        else:
                            st.info(f"**{k}:** {v}")
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Bandar Analysis")
                    
                    # Phase box with color
                    if '🟢' in band_phase:
                        phase_class = "phase-akum"
                    elif '🚀' in band_phase:
                        phase_class = "phase-markup"
                    elif '🔴' in band_phase:
                        phase_class = "phase-dist"
                    else:
                        phase_class = "phase-side"
                    
                    st.markdown(f'<div class="{phase_class}">{band_phase}</div>', unsafe_allow_html=True)
                    
                    for k, v in band_details.items():
                        st.info(f"**{k}:** {v}")
            else:
                st.error(f"❌ {ticker} - Failed to fetch data")
elif "Single" in menu:
    st.markdown("### 📈 Single Stock Analysis + Multi-Timeframe")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("Pilih Saham", tickers)
    with col2:
        analyze_mtf = st.checkbox("Multi-TF", value=True)
    
    if st.button("🔍 Analyze Deep", type="primary"):
        with st.spinner(f"Deep analysis on {selected}..."):
            df = fetch_data(selected, period)
            
            if df is None:
                st.error("❌ Failed to fetch data")
            else:
                freshness, status_type = check_data_freshness(df)
                getattr(st, status_type)(f"📅 Data: {freshness}")
                
                price = df['Close'].iloc[-1]
                
                # Main analysis
                score, details, conf = score_full_screener_v3(df)
                band_score, band_details, band_phase, band_conf = score_bandar_v3(df)
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("💰 Price", f"Rp {price:,.0f}")
                col2.metric("📊 Score", f"{score}/100")
                col3.metric("🎯 Confidence", f"{conf}%")
                col4.metric("🎯 Bandar", band_phase.split()[1] if len(band_phase.split()) > 1 else band_phase)
                
                # Multi-timeframe analysis
                if analyze_mtf:
                    st.markdown("---")
                    st.markdown("### 🔄 Multi-Timeframe Analysis")
                    
                    with st.spinner("Analyzing multiple timeframes..."):
                        mtf_results, mtf_verdict = analyze_multi_timeframe(selected)
                    
                    if "STRONG BUY" in mtf_verdict:
                        st.success(mtf_verdict)
                    elif "BUY" in mtf_verdict:
                        st.info(mtf_verdict)
                    else:
                        st.warning(mtf_verdict)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    for i, (tf, data) in enumerate(mtf_results.items()):
                        col = [col1, col2, col3][i]
                        with col:
                            st.markdown(f"**{tf.upper()} Timeframe**")
                            st.metric("Score", f"{data['score']}/100")
                            st.metric("Trend", data['trend'])
                            st.caption(f"Conf: {data['confidence']}%")
                            
                            if data['aligned']:
                                st.success("✅ Bullish")
                            else:
                                st.error("❌ Bearish")
                
                # Support & Resistance
                # Support & Resistance
                st.markdown("---")
                st.markdown("### 📊 Support & Resistance Levels")
                
                try:
                    sr_data = find_support_resistance(df, price)
                    
                    if sr_data and (sr_data['resistances'] or sr_data['supports']):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🔴 RESISTANCE LEVELS**")
                            if sr_data['resistances']:
                                for idx, r in enumerate(sr_data['resistances']):
                                    strength_color = "🔴" if r['strength'] == "STRONG" else "🟠" if r['strength'] == "MEDIUM" else "🟡"
                                    level_type = f" ({r.get('type', '')})" if 'type' in r else ""
                                    st.info(f"{strength_color} **R{idx+1}:** Rp {r['level']:,.0f} ({r['strength']}){level_type} - {r['distance_pct']:+.1f}%")
                            else:
                                st.caption("No clear resistance detected")
                        
                        with col2:
                            st.markdown("**🟢 SUPPORT LEVELS**")
                            if sr_data['supports']:
                                for idx, s in enumerate(sr_data['supports']):
                                    if 'type' in s:
                                        strength_color = "🔵"
                                        st.success(f"{strength_color} **{s['type']}:** Rp {s['level']:,.0f} ({s['strength']}) - {s['distance_pct']:+.1f}%")
                                    else:
                                        strength_color = "🟢" if s['strength'] == "STRONG" else "🟡" if s['strength'] == "MEDIUM" else "⚪"
                                        st.success(f"{strength_color} **S{idx+1}:** Rp {s['level']:,.0f} ({s['strength']}) - {s['distance_pct']:+.1f}%")
                            else:
                                st.caption("No clear support detected")
                        
                        # Trading plan based on S/R
                        if sr_data['supports'] and sr_data['resistances']:
                            st.markdown("---")
                            st.markdown("### 🎯 S/R Based Trading Plan")
                            
                            nearest_support = sr_data['supports'][0]
                            nearest_resistance = sr_data['resistances'][0]
                            
                            entry_zone = nearest_support['level']
                            target_zone = nearest_resistance['level']
                            sl_zone = entry_zone * 0.96
                            
                            risk_reward = (target_zone - entry_zone) / (entry_zone - sl_zone) if (entry_zone - sl_zone) > 0 else 0
                            
                            st.info(f"""
                            **Optimal Entry:** Near support at Rp {entry_zone:,.0f}
                            **Target:** Resistance at Rp {target_zone:,.0f} ({((target_zone-entry_zone)/entry_zone*100):+.1f}%)
                            **Stop Loss:** Below support at Rp {sl_zone:,.0f}
                            **Risk:Reward:** 1:{risk_reward:.2f} {'✅' if risk_reward >= 2 else '⚠️'}
                            """)
                    else:
                        st.info("📊 S/R levels will be calculated with more data")
                        
                except Exception as e:
                    st.warning("⚠️ Support/Resistance analysis unavailable. Using standard entry levels.")
               
                # Bandar Analysis
                st.markdown("---")
                st.markdown("### 🎯 Bandar / Smart Money Analysis")
                
                # Display phase with color
                if '🟢' in band_phase:
                    st.markdown(f'<div class="phase-akum">{band_phase}</div>', unsafe_allow_html=True)
                elif '🚀' in band_phase:
                    st.markdown(f'<div class="phase-markup">{band_phase}</div>', unsafe_allow_html=True)
                elif '🔴' in band_phase:
                    st.markdown(f'<div class="phase-dist">{band_phase}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="phase-side">{band_phase}</div>', unsafe_allow_html=True)
                
                # Display all bandar details
                for k, v in band_details.items():
                    if 'Action' in k or 'BUY' in str(v):
                        st.success(f"**{k}:** {v}")
                    elif 'SELL' in str(v) or 'AVOID' in str(v) or 'DANGER' in str(v):
                        st.error(f"**{k}:** {v}")
                    elif 'Risk' in k:
                        if 'LOW' in str(v):
                            st.success(f"**{k}:** {v}")
                        elif 'HIGH' in str(v):
                            st.error(f"**{k}:** {v}")
                        else:
                            st.warning(f"**{k}:** {v}")
                    else:
                        st.info(f"**{k}:** {v}")
                
                # Technical Details
                st.markdown("---")
                st.markdown("### 📋 Technical Analysis Details")
                
                for k, v in details.items():
                    if '⛔' in k or '❌' in k:
                        st.error(f"**{k}:** {v}")
                    elif '⚠️' in k:
                        st.warning(f"**{k}:** {v}")
                    else:
                        st.info(f"**{k}:** {v}")
                
                # Entry strategy
                levels = get_signal_levels(score, price, conf)
                
                if levels["ideal"]["entry"]:
                    st.markdown("---")
                    st.markdown("### 🎯 Entry Strategy")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📍 Conservative Entry**")
                        st.info(f"""
                        **Entry:** Rp {levels['ideal']['entry']:,.0f}
                        **TP1 (8%):** Rp {levels['ideal']['tp1']:,.0f}
                        **TP2 (15%):** Rp {levels['ideal']['tp2']:,.0f}
                        **Stop Loss:** Rp {levels['ideal']['sl']:,.0f}
                        """)
                        
                        if levels['ideal']['entry'] and levels['ideal']['tp1'] and levels['ideal']['sl']:
                            rr = (levels['ideal']['tp1'] - levels['ideal']['entry']) / (levels['ideal']['entry'] - levels['ideal']['sl'])
                            st.caption(f"⚖️ Risk:Reward = 1:{rr:.2f}")
                    
                    with col2:
                        if levels['aggr']['entry']:
                            st.markdown("**⚡ Aggressive Entry**")
                            st.warning(f"""
                            **Entry:** Rp {levels['aggr']['entry']:,.0f}
                            **TP1:** Rp {levels['ideal']['tp1']:,.0f}
                            **TP2:** Rp {levels['ideal']['tp2']:,.0f}
                            **SL:** Rp {levels['aggr']['sl']:,.0f}
                            """)
                        else:
                            st.info("⏳ Wait for pullback")
                    
                    # 3-lot strategy
                    st.markdown("### 📊 3-Lot Position Management")
                    three_lot = calculate_three_lot_strategy(levels['ideal']['entry'])
                    
                    st.success(f"""
                    **Split into 3 equal lots:**
                    
                    🎯 **Lot 1/3:** Sell at Rp {three_lot['lot1_tp']:,.0f} (+8%)
                    🎯 **Lot 2/3:** Sell at Rp {three_lot['lot2_tp']:,.0f} (+15%)
                    🏃 **Lot 3/3:** {three_lot['lot3_trail']}
                    
                    🛑 **Initial SL:** Rp {three_lot['initial_sl']:,.0f}
                    """)
                    
                    # Position size
                    if account > 0:
                        pos_calc = calculate_position_size(account, risk_pct, 
                                                          levels['ideal']['entry'], 
                                                          levels['ideal']['sl'])
                        if pos_calc:
                            st.markdown("### 💰 Position Size")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Shares", f"{pos_calc['shares']:,}")
                            col2.metric("Value", f"Rp {pos_calc['position_value']:,.0f}")
                            col3.metric("% Portfolio", f"{pos_calc['position_pct']:.1f}%")
                            st.caption(f"💵 Risk: Rp {pos_calc['risk_amount']:,.0f} ({risk_pct}%)")
                
                # Score breakdown
                display_score_breakdown(details, score, conf)
                
                # Action buttons
                st.markdown("---")
                col1, col2 = st.columns(2)
                if col1.button("💾 Track Position", use_container_width=True):
                    save_recommendation(selected.replace('.JK',''), "Single Stock", 
                                      score, conf, price, levels['signal'])
                    st.success("✅ Tracked!")
                    
                if col2.button("🔖 Add to Watchlist", use_container_width=True):
                    add_to_watchlist(selected.replace('.JK',''), "Single Stock",
                                    score, conf, levels['ideal']['entry'] if levels['ideal']['entry'] else price)
                    st.success("✅ Watchlisted!")

elif "Watchlist" in menu:
    st.markdown("### 🔖 Watchlist")
    
    watchlist = get_watchlist()
    
    if watchlist.empty:
        st.info("📭 Your watchlist is empty. Add stocks from screener results!")
    else:
        if st.button("🔄 Update All Prices"):
            progress = st.progress(0)
            status = st.empty()
            
            conn = sqlite3.connect('screener_tracking.db')
            for idx, row in watchlist.iterrows():
                status.text(f"Updating {row['ticker']}...")
                progress.progress((idx + 1) / len(watchlist))
                
                ticker = row['ticker'] if row['ticker'].endswith('.JK') else f"{row['ticker']}.JK"
                df = fetch_data_with_retry(ticker, "5d")
                
                if df is not None:
                    c = conn.cursor()
                    c.execute("UPDATE watchlist SET current_price=? WHERE id=?", 
                             (df['Close'].iloc[-1], row['id']))
                    conn.commit()
                
                time.sleep(0.5)
            
            conn.close()
            progress.empty()
            status.empty()
            st.success("✅ Prices updated!")
            st.rerun()
        
        st.markdown(f"**Total: {len(watchlist)} stocks**")
        
        for _, row in watchlist.iterrows():
            if row['current_price'] <= row['target_entry']:
                alert = "🎯 TARGET HIT!"
                box_color = "success"
            else:
                pct_away = ((row['current_price'] - row['target_entry']) / row['target_entry'] * 100)
                alert = f"⏳ {pct_away:+.2f}% from target"
                box_color = "info"
            
            with st.expander(f"{row['ticker']} - {alert}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current", f"Rp {row['current_price']:,.0f}")
                col2.metric("Target Entry", f"Rp {row['target_entry']:,.0f}")
                col3.metric("Score", f"{row['score']}/100")
                col4.metric("Confidence", f"{row['confidence']}%")
                
                if row['notes']:
                    st.caption(f"📝 {row['notes']}")
                
                col1, col2, col3 = st.columns(3)
                
                if col1.button("💾 Move to Active", key=f"ma{row['id']}"):
                    save_recommendation(row['ticker'], row['strategy'], row['score'],
                                      row['confidence'], row['current_price'], "BUY")
                    
                    conn = sqlite3.connect('screener_tracking.db')
                    c = conn.cursor()
                    c.execute("UPDATE watchlist SET status='MOVED' WHERE id=?", (row['id'],))
                    conn.commit()
                    conn.close()
                    
                    st.success("✅ Moved to active positions!")
                    st.rerun()
                
                if col2.button("🔄 Refresh", key=f"rp{row['id']}"):
                    ticker = row['ticker'] if row['ticker'].endswith('.JK') else f"{row['ticker']}.JK"
                    df = fetch_data_with_retry(ticker, "5d")
                    if df is not None:
                        conn = sqlite3.connect('screener_tracking.db')
                        c = conn.cursor()
                        c.execute("UPDATE watchlist SET current_price=? WHERE id=?", 
                                 (df['Close'].iloc[-1], row['id']))
                        conn.commit()
                        conn.close()
                        st.success("✅ Updated!")
                        st.rerun()
                
                if col3.button("❌ Remove", key=f"rm{row['id']}"):
                    conn = sqlite3.connect('screener_tracking.db')
                    c = conn.cursor()
                    c.execute("DELETE FROM watchlist WHERE id=?", (row['id'],))
                    conn.commit()
                    conn.close()
                    st.success("✅ Removed!")
                    st.rerun()

elif "Track" in menu and "Performance" in menu:
    st.markdown("### 📊 Performance Tracking")
    
    stats = get_performance_stats()
    
    if stats['total'] == 0:
        st.info("👋 No closed trades yet. Start tracking your trades!")
        st.markdown("""
        **How to use:**
        1. Run screener and save recommendations
        2. Enter positions from Active Positions tab
        3. Close them as WIN/LOSS/BE
        4. Track your progress here!
        """)
    else:
        col1, col2, col3, col4 = st.columns(4)
        win_rate = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        col1.metric("Total Trades", stats['total'])
        col2.metric("Win Rate", f"{win_rate:.1f}%", 
                   delta="Target: 60%" if win_rate < 60 else "Above target!")
        col3.metric("Avg Win", f"+{stats['avg_profit']:.2f}%")
        col4.metric("Avg Loss", f"{stats['avg_loss']:.2f}%")
        
        health_msg, health_type = get_strategy_health(stats)
        getattr(st, health_type)(health_msg)
        
        if not stats['by_strategy'].empty:
            st.markdown("---")
            plot_strategy_comparison(stats)
        
        if not stats['by_confidence'].empty:
            st.markdown("---")
            st.markdown("### 🎯 Performance by Confidence Level")
            
            for _, row in stats['by_confidence'].iterrows():
                wr = (row['wins'] / row['total'] * 100) if row['total'] > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"**{row['conf_level']}**")
                col2.metric("Win Rate", f"{wr:.1f}%")
                col3.metric("Avg Profit", f"{row['avg_profit']:+.2f}%")
        
        st.markdown("---")
        st.markdown("### 💡 Performance Insights")
        
        if win_rate < 40:
            st.error("""
            **🔴 CRITICAL - Immediate Action Needed:**
            - Stop trading and review ALL past trades
            - Identify common mistakes
            - Focus ONLY on 80+ confidence signals
            - Consider paper trading until win rate improves
            """)
        elif win_rate < 50:
            st.warning("""
            **🟡 CAUTION - Strategy Adjustment Required:**
            - Review losing trades - find patterns
            - Increase minimum confidence to 70%
            - Stick to your best-performing strategy only
            - Reduce position sizes by 50%
            """)
        elif win_rate < 60:
            st.info("""
            **🟢 ON TRACK - Keep Improving:**
            - You're approaching the target zone
            - Keep detailed trade journals
            - Focus on risk management
            - Gradually increase position sizes
            """)
        else:
            st.success("""
            **🟢 EXCELLENT - Strong Edge:**
            - Your strategy is working!
            - Maintain discipline and consistency
            - Consider scaling up carefully
            - Keep tracking to maintain edge
            """)

elif "Active" in menu and "Positions" in menu:
    st.markdown("### 📋 Active Positions")
    
    active = get_active_recommendations()
    
    if active.empty:
        st.info("📭 No active positions. Start by saving recommendations from screener results!")
    else:
        summary = get_position_summary(active)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Positions", summary['total_positions'])
        col2.metric("Winning", summary['winning'], 
                   delta=f"{summary['winning']/summary['total_positions']*100:.0f}%")
        col3.metric("Losing", summary['losing'])
        
        avg_pnl = summary['avg_pnl']
        col4.metric("Avg P&L", f"{avg_pnl:+.2f}%",
                   delta_color="normal" if avg_pnl > 0 else "inverse")
        
        if st.button("🔄 Update All Prices"):
            with st.spinner("Updating prices..."):
                updated, failed = bulk_update_active_positions()
                st.success(f"✅ Updated {updated} positions")
                if failed > 0:
                    st.warning(f"⚠️ Failed to update {failed} positions")
                st.rerun()
        
        st.markdown("---")
        
        for _, row in active.iterrows():
            pnl = ((row['current_price'] - row['entry_price']) / row['entry_price'] * 100)
            color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            
            tp1_pct = 8
            tp2_pct = 15
            sl_pct = -6
            
            tp1_hit = pnl >= tp1_pct
            tp2_hit = pnl >= tp2_pct
            sl_hit = pnl <= sl_pct
            
            status_msg = ""
            if tp2_hit:
                status_msg = " 🎯 TP2 HIT!"
            elif tp1_hit:
                status_msg = " 🎯 TP1 HIT!"
            elif sl_hit:
                status_msg = " 🛑 STOP LOSS!"
            
            with st.expander(f"{color} {row['ticker']} | {row['signal']} | P/L: {pnl:+.2f}%{status_msg}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Entry", f"Rp {row['entry_price']:,.0f}")
                col2.metric("Current", f"Rp {row['current_price']:,.0f}")
                col3.metric("P/L", f"{pnl:+.2f}%")
                
                col1, col2, col3 = st.columns(3)
                col1.info(f"**Strategy:** {row['strategy']}")
                col2.info(f"**Score:** {row['score']}/100")
                col3.info(f"**Confidence:** {row['confidence']}%")
                
                st.markdown("### 🎯 Position Management")
                
                if tp2_hit:
                    st.success("""
                    **TP2 (+15%) HIT!**
                    - Sell 2nd lot (if not done)
                    - Trail final 1/3 with 20 EMA
                    - Move SL to breakeven+
                    """)
                elif tp1_hit:
                    st.success("""
                    **TP1 (+8%) HIT!**
                    - Sell 1st lot (1/3 position)
                    - Move SL to breakeven
                    - Let 2/3 run to TP2 (+15%)
                    """)
                elif sl_hit:
                    st.error("""
                    **STOP LOSS HIT!**
                    - Consider exiting entire position
                    - Review what went wrong
                    - Update trade notes before closing
                    """)
                else:
                    st.info(f"""
                    **Holding Position**
                    - Current P/L: {pnl:+.2f}%
                    - Next target: TP1 at +{tp1_pct}%
                    - Stop loss at {sl_pct}%
                    """)
                
                st.markdown("### 📝 Close Position")
                
                close_notes = {
                    'WIN': [
                        "✅ Target hit as planned",
                        "✅ Strong momentum follow-through",
                        "✅ Market catalyst helped",
                        "✅ Early entry - optimal zone"
                    ],
                    'LOSS': [
                        "❌ Failed support/resistance",
                        "❌ Market reversal/weakness",
                        "❌ Volume dried up",
                        "❌ Entered too late/early",
                        "❌ News/fundamental change",
                        "❌ Ignored stop loss initially"
                    ],
                    'BE': [
                        "⚪ Exited at breakeven",
                        "⚪ Cut losses early",
                        "⚪ Market uncertainty"
                    ]
                }
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("✅ WIN", key=f"w{row['id']}", use_container_width=True):
                        note = st.selectbox(f"Why won?", close_notes['WIN'], key=f"wn{row['id']}")
                        update_recommendation_status(row['id'], 'CLOSED', 'WIN', pnl, 
                                                    row['current_price'], notes=note)
                        st.success("✅ Closed as WIN!")
                        time.sleep(1)
                        st.rerun()
                
                with col2:
                    if st.button("❌ LOSS", key=f"l{row['id']}", use_container_width=True):
                        note = st.selectbox(f"What went wrong?", close_notes['LOSS'], key=f"ln{row['id']}")
                        update_recommendation_status(row['id'], 'CLOSED', 'LOSS', pnl, 
                                                    row['current_price'], notes=note)
                        st.error("❌ Closed as LOSS")
                        time.sleep(1)
                        st.rerun()
                
                with col3:
                    if st.button("⚪ BE", key=f"b{row['id']}", use_container_width=True):
                        note = st.selectbox(f"BE reason?", close_notes['BE'], key=f"bn{row['id']}")
                        update_recommendation_status(row['id'], 'CLOSED', 'BE', pnl, 
                                                    row['current_price'], notes=note)
                        st.info("⚪ Closed at breakeven")
                        time.sleep(1)
                        st.rerun()
                
                st.markdown("---")
                if st.button("🔄 Update Price Only", key=f"up{row['id']}"):
                    ticker = row['ticker'] if row['ticker'].endswith('.JK') else f"{row['ticker']}.JK"
                    df = fetch_data_with_retry(ticker, "5d")
                    if df is not None:
                        conn = sqlite3.connect('screener_tracking.db')
                        c = conn.cursor()
                        c.execute("UPDATE recommendations SET current_price=? WHERE id=?", 
                                 (df['Close'].iloc[-1], row['id']))
                        conn.commit()
                        conn.close()
                        st.success("✅ Price updated!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to update")

elif "Bandar" in menu and "🔥" in menu:
    st.markdown("### 🎯 Bandar Tracking - Smart Money Scanner 🔥")
    st.caption("Wyckoff Accumulation/Distribution Detector - Find where institutions are buying!")
    
    jkt_time = get_jakarta_time()
    st.info(f"🕐 Analysis Time: {jkt_time.strftime('%H:%M WIB')}")
    
    with st.expander("📚 How Bandar Tracking Works - READ THIS!"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🟢 AKUMULASI (BUY ZONE) ⭐⭐⭐⭐⭐
            **Best time to enter!**
            
            **Characteristics:**
            - High volume but price sideways/slight down
            - OBV (On-Balance Volume) trending UP
            - Smart money quietly accumulating
            - Price compressed in tight range
            
            **What's Happening:**
            - Institutions buying from weak hands
            - Retail sellers, smart money buyers
            - Building position before markup
            
            **Action:** 
            - START BUYING in tranches
            - Best risk:reward ratio
            - Hold 2-8 weeks
            - Target: +15% to +40%
            
            ---
            
            ### 🚀 MARKUP (HOLD/FOLLOW ZONE) ⭐⭐⭐⭐
            **Trend is your friend**
            
            **Characteristics:**
            - Price breaking out with volume
            - Clear uptrend established
            - OBV confirming move
            
            **What's Happening:**
            - Institutions driving price up
            - Retail FOMO buying helps
            - Momentum phase
            
            **Action:**
            - HOLD if already in
            - Can add on pullbacks
            - Trail stop loss
            - Target: +10% to +25%
            """)
        
        with col2:
            st.markdown("""
            ### 🔴 DISTRIBUSI (SELL ZONE) ❌❌❌
            **DANGER! Exit immediately!**
            
            **Characteristics:**
            - High volume but price NOT moving up
            - OBV trending DOWN
            - Price at resistance
            
            **What's Happening:**
            - Institutions SELLING to retail
            - Retail buyers, smart money sellers
            - Top formation
            
            **Action:**
            - EXIT ALL POSITIONS
            - DO NOT BUY
            - Markdown coming (-15% to -40%)
            
            ---
            
            ### ⚪ SIDEWAYS (WAIT ZONE) ⭐⭐
            **No clear edge**
            
            **Characteristics:**
            - Mixed signals
            - Low volume
            - No clear OBV trend
            
            **What's Happening:**
            - Market indecision
            - Could go either way
            
            **Action:**
            - WAIT for clear signal
            - Watch for volume increase
            - Patience is key
            
            ---
            
            ### 💡 KEY PRINCIPLE:
            **Follow the Smart Money!**
            - They accumulate BEFORE price rises
            - They distribute BEFORE price falls
            - Volume + OBV = footprints
            - Be early, not late!
            """)
    
    if st.button("🎯 Scan for Smart Money Activity", type="primary"):
        with st.spinner(f"Analyzing {limit} stocks for institutional activity..."):
            results = []
            
            progress = st.progress(0)
            status = st.empty()
            
            scan_tickers = tickers[:limit] if limit < len(tickers) else tickers
            
            for i, ticker in enumerate(scan_tickers):
                progress.progress((i+1)/len(scan_tickers))
                status.text(f"🔍 Scanning {ticker}... ({i+1}/{len(scan_tickers)})")
                
                df = fetch_data(ticker, period)
                if df is None or len(df) < 50:
                    continue
                
                price = float(df['Close'].iloc[-1])
                score, details, phase, confidence = score_bandar_v3(df)
                
                if score >= 30:  # Show all meaningful signals
                    results.append({
                        "Ticker": ticker,
                        "Price": price,
                        "Phase": phase,
                        "Score": score,
                        "Confidence": confidence,
                        "Action": details.get('Action', ''),
                        "Signal": details.get('Signal', ''),
                        "Risk": details.get('Risk', ''),
                        "Volume_Ratio": details.get('Volume_Ratio', ''),
                        "Price_Change": details.get('Price_Change', ''),
                        "OBV_Trend": details.get('OBV_Trend', ''),
                        "Details": details
                    })
                
                time.sleep(0.3)
            
            progress.empty()
            status.empty()
            
            if not results:
                st.warning("⚠️ No significant smart money activity detected")
                st.info("""
                **Possible reasons:**
                - Market in low-activity period
                - No clear accumulation/distribution patterns
                - Try scanning more stocks
                - Check different time period
                """)
            else:
                df_results = pd.DataFrame(results).sort_values("Score", ascending=False)
                
                # Summary by phase
                st.success(f"✅ Found {len(df_results)} stocks with bandar activity!")
                
                col1, col2, col3, col4 = st.columns(4)
                akum_count = len(df_results[df_results['Phase'] == '🟢 AKUMULASI'])
                markup_count = len(df_results[df_results['Phase'] == '🚀 MARKUP'])
                dist_count = len(df_results[df_results['Phase'] == '🔴 DISTRIBUSI'])
                side_count = len(df_results[df_results['Phase'] == '⚪ SIDEWAYS'])
                
                col1.metric("🟢 Akumulasi", akum_count, delta="BUY ZONE!" if akum_count > 0 else None)
                col2.metric("🚀 Markup", markup_count, delta="HOLD ZONE" if markup_count > 0 else None)
                col3.metric("🔴 Distribusi", dist_count, delta="DANGER!" if dist_count > 0 else None)
                col4.metric("⚪ Sideways", side_count)
                
                # Filter by phase
                st.markdown("### 🎯 Filter by Phase")
                col1, col2, col3, col4 = st.columns(4)
                show_akum = col1.checkbox("🟢 Akumulasi", value=True)
                show_markup = col2.checkbox("🚀 Markup", value=True)
                show_dist = col3.checkbox("🔴 Distribusi", value=False)
                show_side = col4.checkbox("⚪ Sideways", value=False)
                
                # Apply filters
                filtered = df_results.copy()
                phase_filters = []
                if show_akum:
                    phase_filters.append('🟢 AKUMULASI')
                if show_markup:
                    phase_filters.append('🚀 MARKUP')
                if show_dist:
                    phase_filters.append('🔴 DISTRIBUSI')
                if show_side:
                    phase_filters.append('⚪ SIDEWAYS')
                
                if phase_filters:
                    filtered = filtered[filtered['Phase'].isin(phase_filters)]
                
                if filtered.empty:
                    st.info("No stocks match selected filters")
                else:
                    # Data table
                    show_cols = ["Ticker", "Price", "Phase", "Action", "Score", "Confidence", 
                                "Volume_Ratio", "Price_Change", "OBV_Trend"]
                    st.dataframe(filtered[show_cols], use_container_width=True, height=400)
                    
                    # Detailed cards
                    st.markdown("### 📊 Detailed Bandar Analysis")
                    
                    for _, row in filtered.head(20).iterrows():
                        # Phase color
                        if '🟢' in row['Phase']:
                            phase_html = f'<div class="phase-akum">{row['Phase']} - {row['Action']}</div>'
                        elif '🚀' in row['Phase']:
                            phase_html = f'<div class="phase-markup">{row['Phase']} - {row['Action']}</div>'
                        elif '🔴' in row['Phase']:
                            phase_html = f'<div class="phase-dist">{row['Phase']} - {row['Action']}</div>'
                        else:
                            phase_html = f'<div class="phase-side">{row['Phase']} - {row['Action']}</div>'
                        
                        with st.expander(f"{row['Ticker']} - {row['Phase']} (Score: {row['Score']})"):
                            st.markdown(phase_html, unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("💰 Price", f"Rp {row['Price']:,.0f}")
                            col2.metric("📊 Score", f"{row['Score']}/100")
                            col3.metric("🎯 Confidence", f"{row['Confidence']}%")
                            
                            # Bandar metrics
                            st.markdown("### 📊 Smart Money Indicators")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.info(f"**Volume:** {row['Volume_Ratio']}")
                            col2.info(f"**Price Δ:** {row['Price_Change']}")
                            col3.info(f"**OBV Trend:** {row['OBV_Trend']}")
                            col4.info(f"**Signal:** {row['Signal']}")
                            
                            # Trading plan
                            st.markdown("### 🎯 Trading Plan")
                            
                            if '🟢' in row['Phase']:  # Akumulasi
                                st.success(f"""
                                **🟢 ACCUMULATION PHASE - PRIME BUY ZONE!**
                                
                                ✅ **Why This is Perfect:**
                                - {row['Signal']}
                                - {row['Risk']}
                                - Best risk:reward entry point
                                
                                🎯 **Entry Strategy:**
                                - Start
                                - Start buying NOW (25% position)
                                - Add on dips (3 tranches total)
                                - Average cost = key to success
                                - Entry range: Rp {row['Price']*0.97:.0f} - {row['Price']*1.02:.0f}
                                
                                📈 **Targets:**
                                - Conservative: +15% (Rp {row['Price']*1.15:.0f})
                                - Aggressive: +25-40% (Rp {row['Price']*1.25:.0f} - {row['Price']*1.40:.0f})
                                - Timeline: 2-8 weeks
                                
                                🛑 **Stop Loss:**
                                - Below support: Rp {row['Price']*0.92:.0f} (-8%)
                                - Exit if distribution signals appear
                                
                                💡 **Pro Tip:**
                                This is THE BEST time to buy! Smart money is accumulating.
                                Patience will be rewarded. HOLD through markup phase.
                                """)
                                
                                col1, col2 = st.columns(2)
                                if col1.button(f"💾 Track", key=f"t{row['Ticker']}", use_container_width=True):
                                    save_recommendation(row['Ticker'].replace('.JK',''), 
                                                      "Bandar - Akumulasi", 
                                                      row['Score'], row['Confidence'], 
                                                      row['Price'], "STRONG BUY")
                                    st.success("✅ Added to tracking!")
                                
                                if col2.button(f"🔖 Watch", key=f"w{row['Ticker']}", use_container_width=True):
                                    add_to_watchlist(row['Ticker'].replace('.JK',''), 
                                                    "Bandar - Akumulasi",
                                                    row['Score'], row['Confidence'], 
                                                    row['Price'] * 0.97,
                                                    notes="Accumulation phase - prime buy zone")
                                    st.success("✅ Added to watchlist!")
                            
                            elif '🚀' in row['Phase']:  # Markup
                                st.info(f"""
                                **🚀 MARKUP PHASE - TREND FOLLOWING ZONE**
                                
                                ✅ **What's Happening:**
                                - {row['Signal']}
                                - Smart money driving price up
                                - Momentum established
                                
                                📊 **If Already In Position:**
                                - ✅ HOLD your position
                                - ✅ Trail stop loss below swing lows
                                - ✅ Take partial profits at resistance
                                - ✅ Let winners run!
                                
                                🆕 **If Entering Now:**
                                - Wait for pullback to support (Rp {row['Price']*0.96:.0f})
                                - Smaller position size (risk higher than accumulation)
                                - Quick profits, don't overstay
                                - Entry: Only on 3-5% dip
                                
                                🎯 **Targets:**
                                - Short-term: +8-12% (Rp {row['Price']*1.08:.0f} - {row['Price']*1.12:.0f})
                                - Medium-term: +15-25% (if momentum continues)
                                
                                🛑 **Exit Signals:**
                                - Volume spike without price gain
                                - Break below major support
                                - Distribution phase begins
                                
                                💡 **Pro Tip:**
                                The trend is your friend, BUT be ready to exit quickly.
                                Late entries = higher risk. Watch for distribution!
                                """)
                                
                                col1, col2 = st.columns(2)
                                if col1.button(f"💾 Track", key=f"t{row['Ticker']}", use_container_width=True):
                                    save_recommendation(row['Ticker'].replace('.JK',''), 
                                                      "Bandar - Markup", 
                                                      row['Score'], row['Confidence'], 
                                                      row['Price'], "BUY")
                                    st.success("✅ Added to tracking!")
                                
                                if col2.button(f"🔖 Wait for Pullback", key=f"w{row['Ticker']}", use_container_width=True):
                                    add_to_watchlist(row['Ticker'].replace('.JK',''), 
                                                    "Bandar - Markup",
                                                    row['Score'], row['Confidence'], 
                                                    row['Price'] * 0.95,
                                                    notes="Wait for 5% pullback before entry")
                                    st.success("✅ Added to watchlist - will alert on pullback!")
                            
                            elif '🔴' in row['Phase']:  # Distribusi
                                st.error(f"""
                                **🔴 DISTRIBUTION PHASE - EXTREME DANGER ZONE!**
                                
                                🚨 **What's Happening:**
                                - {row['Signal']}
                                - Smart money is SELLING
                                - Retail buying from institutions
                                - Top formation in progress
                                
                                ⛔ **If You're In This Stock:**
                                - 🚨 EXIT IMMEDIATELY - NO QUESTIONS!
                                - Don't wait for "confirmation"
                                - Don't hope for recovery
                                - Cut losses NOW while you can
                                - Smart money is dumping on you
                                
                                ❌ **If You're Not In:**
                                - DO NOT BUY - NO MATTER WHAT!
                                - Ignore the "dip buying" urge
                                - This is NOT a buying opportunity
                                - Stay away completely
                                
                                📉 **What's Coming Next:**
                                - Markdown phase (sharp decline)
                                - Expected drop: 15-40%
                                - Timeline: 2-8 weeks
                                - Pain for late buyers
                                
                                ⏰ **When to Consider Again:**
                                - After markdown completes
                                - New accumulation phase starts
                                - Might take 2-6 months
                                
                                💡 **Critical Lesson:**
                                Smart money sells at tops, buys at bottoms.
                                Don't be the exit liquidity! Preserve capital!
                                """)
                            
                            else:  # Sideways
                                st.warning(f"""
                                **⚪ SIDEWAYS PHASE - NO CLEAR EDGE**
                                
                                📊 **Current Situation:**
                                - {row['Signal']}
                                - Mixed signals
                                - Market indecision
                                - No clear smart money activity
                                
                                ⏸️ **Recommended Action:**
                                - WAIT for clearer signal
                                - Don't force a trade
                                - Patience is profitable
                                - Watch for pattern change
                                
                                👀 **Watch For These Signals:**
                                
                                **Bullish Signs:**
                                - Volume increase + Price steady = Possible accumulation
                                - OBV rising while price flat = Hidden buying
                                - Breakout above resistance = Markup starting
                                
                                **Bearish Signs:**
                                - High volume + No breakout = Distribution
                                - OBV falling + Price flat = Hidden selling
                                - Break below support = Markdown starting
                                
                                💡 **Pro Tip:**
                                Not every stock is tradeable all the time.
                                The best trade is sometimes NO trade.
                                Wait for AKUMULASI phase!
                                """)
                                
                                if st.button(f"🔖 Add to Watchlist", key=f"w{row['Ticker']}", use_container_width=True):
                                    add_to_watchlist(row['Ticker'].replace('.JK',''), 
                                                    "Bandar - Sideways",
                                                    row['Score'], row['Confidence'], 
                                                    row['Price'],
                                                    notes="Waiting for accumulation signal")
                                    st.success("✅ Will alert when phase changes!")
                            
                            # All bandar details
                            st.markdown("---")
                            st.markdown("### 📋 Complete Bandar Metrics")
                            details_col1, details_col2 = st.columns(2)
                            
                            with details_col1:
                                for k, v in list(row['Details'].items())[:len(row['Details'])//2]:
                                    if 'Action' in k or 'BUY' in str(v):
                                        st.success(f"**{k}:** {v}")
                                    elif 'SELL' in str(v) or 'AVOID' in str(v):
                                        st.error(f"**{k}:** {v}")
                                    else:
                                        st.info(f"**{k}:** {v}")
                            
                            with details_col2:
                                for k, v in list(row['Details'].items())[len(row['Details'])//2:]:
                                    if 'Target' in k:
                                        st.success(f"**{k}:** {v}")
                                    elif 'Risk' in k:
                                        if 'LOW' in str(v):
                                            st.success(f"**{k}:** {v}")
                                        elif 'HIGH' in str(v) or 'VERY HIGH' in str(v):
                                            st.error(f"**{k}:** {v}")
                                        else:
                                            st.warning(f"**{k}:** {v}")
                                    else:
                                        st.info(f"**{k}:** {v}")
                    
                    # Download results
                    csv = filtered[show_cols].to_csv(index=False).encode()
                    st.download_button("📥 Download Bandar Scan Results", csv,
                                     f"bandar_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                     mime="text/csv")
                    
                    # Educational footer
                    st.markdown("---")
                    st.markdown("### 📚 Understanding Smart Money / Bandar")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info("""
                        **What is "Bandar" / Smart Money?**
                        
                        Bandar (Indonesian) = Big Players = Institutions = Smart Money
                        
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
                        **Your Edge: Follow Their Footprints!**
                        
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
                        
                        **💡 Golden Rule:**
                        Buy in AKUMULASI, Hold through MARKUP,
                        Sell BEFORE distribution!
                        """)
                    
                    # Quick reference
                    st.markdown("---")
                    st.markdown("### 🎯 Quick Reference Card")
                    
                    st.warning("""
                    **CRITICAL DECISION TREE:**
                    
                    1. Is it 🟢 AKUMULASI? → **BUY NOW** (best opportunity)
                    2. Is it 🚀 MARKUP? → **HOLD** if in, **WAIT FOR DIP** if out
                    3. Is it 🔴 DISTRIBUSI? → **SELL EVERYTHING** / **STAY AWAY**
                    4. Is it ⚪ SIDEWAYS? → **WAIT PATIENTLY** for accumulation
                    
                    **Remember:**
                    - Patience in accumulation = Profits in markup
                    - Greed in distribution = Losses in markdown
                    - When in doubt, WAIT for 🟢 AKUMULASI!
                    """)

elif "BPJS" in menu or "BSJP" in menu or "Value" in menu:
    strategy_map = {
        "3️⃣ BPJS": ("BPJS", "⚡ BPJS Scanner", "Day Trading - High Volatility Plays"),
        "4️⃣ BSJP": ("BSJP", "🌙 BSJP Scanner", "Overnight Trading - Gap Recovery"),
        "6️⃣ Value Hunting": ("Value", "💎 Value Hunter", "Undervalued Reversal Plays")
    }
    strategy, title, description = strategy_map[menu]
    
    st.markdown(f"### {title}")
    st.caption(description)
    
    # Timing alerts
    if strategy == "BPJS":
        if is_valid_bpjs_time():
            st.success("✅ OPTIMAL BPJS ENTRY TIME (09:00-09:30 WIB)")
        else:
            st.warning("⏰ Best BPJS time: 09:00-09:30 WIB")
            st.caption("💡 BPJS works best at market open with high volatility")
    
    elif strategy == "BSJP":
        if is_valid_bsjp_time():
            st.success("✅ OPTIMAL BSJP ENTRY TIME (14:00-15:00 WIB)")
        else:
            st.warning("⏰ Best BSJP time: 14:00-15:00 WIB")
            st.caption("💡 BSJP targets gap-down stocks for overnight recovery")
    
    if st.button(f"🚀 Run {strategy} Scanner", type="primary"):
        with st.spinner(f"Scanning {limit} stocks for {strategy} opportunities..."):
            df = batch_scan(tickers, strategy, period, limit, use_parallel)
        
        if df.empty:
            st.warning(f"⚠️ No {strategy} signals found")
        else:
            df = df[(df["Score"] >= min_score) & (df["Confidence"] >= min_confidence)]
            
            if df.empty:
                st.warning(f"No stocks meeting criteria")
            else:
                st.success(f"✅ Found {len(df)} {strategy} opportunities!")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Score", f"{df['Score'].mean():.1f}")
                col2.metric("Avg Conf", f"{df['Confidence'].mean():.1f}%")
                col3.metric("Strong Buy", len(df[df['Signal'] == 'STRONG BUY']))
                col4.metric("Buy", len(df[df['Signal'] == 'BUY']))
                
                show = df[["Ticker","Price","Score","Confidence","Signal","EntryIdeal","TP1","TP2","SL"]]
                st.dataframe(show, use_container_width=True, height=400)
                
                st.markdown(f"### 🏆 Top {min(15, len(df))} Recommendations")
                
                for _, row in df.head(15).iterrows():
                    conf_color = "🟢" if row['Confidence'] >= 80 else "🟡" if row['Confidence'] >= 60 else "🟠"
                    
                    with st.expander(f"{conf_color} {row['Ticker']} - {row['Score']} | {row['Confidence']}% | {row['Signal']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**💰 Price:** Rp {row['Price']:,.0f}")
                            st.markdown(f"**📊 Score:** {row['Score']}/100")
                            st.markdown(f"**🎯 Confidence:** {row['Confidence']}%")
                            st.markdown(f"**📈 Signal:** {row['Signal']}")
                        
                        with col2:
                            if row['EntryIdeal']:
                                st.markdown(f"**🎯 Entry:** Rp {row['EntryIdeal']:,.0f}")
                                st.markdown(f"**🟢 TP1 (8%):** Rp {row['TP1']:,.0f}")
                                st.markdown(f"**🟢 TP2 (15%):** Rp {row['TP2']:,.0f}")
                                st.markdown(f"**🔴 SL:** Rp {row['SL']:,.0f}")
                                
                                if row['EntryIdeal'] and row['TP1'] and row['SL']:
                                    rr = (row['TP1'] - row['EntryIdeal']) / (row['EntryIdeal'] - row['SL'])
                                    st.markdown(f"**⚖️ R:R:** 1:{rr:.2f}")
                        
                        st.markdown("---")
                        for k, v in row['Details'].items():
                            if '⛔' in str(k):
                                st.error(f"- **{k}:** {v}")
                            elif '⚠️' in str(k):
                                st.warning(f"- **{k}:** {v}")
                            else:
                                st.info(f"- **{k}:** {v}")
                        
                        col1, col2 = st.columns(2)
                        if col1.button(f"💾 Track", key=f"t{row['Ticker']}", use_container_width=True):
                            save_recommendation(row['Ticker'].replace('.JK',''), strategy, 
                                              row['Score'], row['Confidence'], 
                                              row['Price'], row['Signal'])
                            st.success("✅ Tracked!")
                        
                        if col2.button(f"🔖 Watch", key=f"w{row['Ticker']}", use_container_width=True):
                            add_to_watchlist(row['Ticker'].replace('.JK',''), strategy,
                                           row['Score'], row['Confidence'], 
                                           row['EntryIdeal'] if row['EntryIdeal'] else row['Price'])
                            st.success("✅ Watchlisted!")
                
                csv = show.to_csv(index=False).encode()
                st.download_button("📥 Download CSV", csv, 
                                 f"{strategy}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

else:  # Full Screener
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
                st.warning(f"No stocks meeting criteria")
            else:
                st.success(f"✅ Found {len(df)} quality opportunities!")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Score", f"{df['Score'].mean():.1f}/100")
                col2.metric("Avg Confidence", f"{df['Confidence'].mean():.1f}%")
                col3.metric("Strong Buy", len(df[df['Signal'] == 'STRONG BUY']))
                col4.metric("Buy", len(df[df['Signal'] == 'BUY']))
                
                show = df[["Ticker","Price","Score","Confidence","Signal","Trend","EntryIdeal","TP1","TP2","SL"]]
                st.dataframe(show, use_container_width=True, height=400)
                
                st.markdown("### 🏆 Top 15 Recommendations")
                
                for _, row in df.head(15).iterrows():
                    conf_color = "🟢" if row['Confidence'] >= 80 else "🟡" if row['Confidence'] >= 60 else "🟠"
                    
                    with st.expander(f"{conf_color} {row['Ticker']} - {row['Score']} | {row['Confidence']}% | {row['Signal']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**💰 Price:** Rp {row['Price']:,.0f}")
                            st.markdown(f"**📊 Score:** {row['Score']}/100")
                            st.markdown(f"**🎯 Confidence:** {row['Confidence']}%")
                            st.markdown(f"**📈 Signal:** {row['Signal']}")
                            st.markdown(f"**🔄 Trend:** {row['Trend']}")
                        
                        with col2:
                            if row['EntryIdeal']:
                                st.markdown("**🎯 Entry Levels:**")
                                st.markdown(f"**Entry:** Rp {row['EntryIdeal']:,.0f}")
                                st.markdown(f"**🟢 TP1 (8%):** Rp {row['TP1']:,.0f}")
                                st.markdown(f"**🟢 TP2 (15%):** Rp {row['TP2']:,.0f}")
                                st.markdown(f"**🔴 SL (6%):** Rp {row['SL']:,.0f}")
                                
                                if row['EntryIdeal'] and row['TP1'] and row['SL']:
                                    rr = (row['TP1'] - row['EntryIdeal']) / (row['EntryIdeal'] - row['SL'])
                                    st.markdown(f"**⚖️ Risk:Reward:** 1:{rr:.2f}")
                        
                        if row['EntryIdeal']:
                            st.markdown("---")
                            st.markdown("**📊 3-Lot Strategy:**")
                            three_lot = calculate_three_lot_strategy(row['EntryIdeal'])
                            st.info(f"""
                            🎯 **Lot 1/3:** Exit at Rp {three_lot['lot1_tp']:,.0f} (+8%)
                            🎯 **Lot 2/3:** Exit at Rp {three_lot['lot2_tp']:,.0f} (+15%)
                            🏃 **Lot 3/3:** {three_lot['lot3_trail']}
                            🛑 **Stop Loss:** Rp {three_lot['initial_sl']:,.0f}
                            """)
                        
                        st.markdown("---")
                        for k, v in row['Details'].items():
                            if '⛔' in str(k):
                                st.error(f"- **{k}:** {v}")
                            elif '⚠️' in str(k):
                                st.warning(f"- **{k}:** {v}")
                            else:
                                st.info(f"- **{k}:** {v}")
                        
                        col1, col2 = st.columns(2)
                        if col1.button(f"💾 Track", key=f"track_{row['Ticker']}", use_container_width=True):
                            save_recommendation(row['Ticker'].replace('.JK',''), "Full Screener", 
                                              row['Score'], row['Confidence'], 
                                              row['Price'], row['Signal'])
                            st.success("✅ Tracked!")
                        
                        if col2.button(f"🔖 Watchlist", key=f"watch_{row['Ticker']}", use_container_width=True):
                            add_to_watchlist(row['Ticker'].replace('.JK',''), "Full Screener",
                                           row['Score'], row['Confidence'], 
                                           row['EntryIdeal'] if row['EntryIdeal'] else row['Price'])
                            st.success("✅ Watchlisted!")
                
                csv = show.to_csv(index=False).encode()
                st.download_button("📥 Download CSV", csv, 
                                 f"screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
                
                st.markdown("---")
                st.markdown("### 💡 Trading Guidelines")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success("""
                    **✅ High Probability Setups:**
                    - Score > 75 + Confidence > 70%
                    - Strong volume confirmation
                    - Clear uptrend alignment
                    - Risk:Reward > 1:2
                    
                    **🎯 Best Practices:**
                    - Use 3-lot position management
                    - Always set stop loss BEFORE entry
                    - Take partial profits at TP1
                    - Let winners run with trailing stop
                    """)
                
                with col2:
                    st.warning("""
                    **⚠️ Use Caution:**
                    - Confidence < 60%
                    - Overbought warnings
                    - Volume quality concerns
                    - Market in downtrend
                    
                    **🛑 Avoid:**
                    - Trading during lunch
                    - Entering without SL
                    - Averaging down on losers
                    - Ignoring position size limits
                    """)
                
                st.info("""
                **📊 Your Recovery Plan (Target: 60%+ Win Rate)**
                
                1. **Selection:** Only 70+ confidence signals
                2. **Entry:** Wait for ideal entry levels
                3. **Position Size:** Risk only 2% max
                4. **Management:** Use 3-lot strategy
                5. **Tracking:** Record EVERY trade
                6. **Review:** Weekly analysis
                
                💪 **Focus:** Consistency before scaling!
                """)
