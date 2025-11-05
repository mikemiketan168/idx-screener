#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3

st.set_page_config(page_title="IDX Power Screener v3", page_icon="🚀", layout="wide")

st.markdown("""
<style>
.big-title {font-size:2.5rem;font-weight:800;color:#1e40af}
.subtitle {font-size:1.1rem;color:#64748b;margin-bottom:2rem}
.signal-box {padding:1rem;border-radius:0.5rem;margin:1rem 0;font-weight:700;text-align:center}
.strong-buy {background:#10b981;color:white}
.buy {background:#34d399;color:white}
.neutral {background:#fbbf24;color:white}
.sell {background:#ef4444;color:white}
</style>
""", unsafe_allow_html=True)

# ============= DATABASE SETUP =============
def init_db():
    conn = sqlite3.connect('screener_tracking.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS recommendations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT, ticker TEXT, strategy TEXT, score INTEGER,
                  confidence INTEGER, entry_price REAL, current_price REAL,
                  signal TEXT, status TEXT DEFAULT 'ACTIVE', result TEXT,
                  profit_pct REAL, exit_price REAL, exit_date TEXT, notes TEXT)''')
    conn.commit()
    conn.close()

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
        
        return df
    except:
        return None

# ============= VALIDATION =============
def validate_not_downtrend(df):
    try:
        r = df.iloc[-1]
        if r['Close'] < r['EMA50'] and r['EMA50'] < r['EMA200']:
            return False, "Price below EMA50 & EMA200 - DOWNTREND"
        if r['EMA9'] < r['EMA21'] < r['EMA50']:
            return False, "EMAs in death cross - DOWNTREND"
        if df['MOM_10D'].iloc[-1] < -5:
            return False, f"10D Momentum: {df['MOM_10D'].iloc[-1]:.1f}% - DOWNTREND"
        return True, "Valid uptrend/consolidation"
    except:
        return True, "Unable to validate"

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
    try:
        obv = [0]
        for i in range(1,len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1]+df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1]-df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        
        vol_ratio = df['Volume'].tail(5).mean()/df['Volume'].rolling(20).mean().iloc[-1]
        price_chg = (df['Close'].iloc[-1]-df['Close'].iloc[-20])/df['Close'].iloc[-20]*100
        obv_trend = (df['OBV'].iloc[-1]-df['OBV'].iloc[-20])/abs(df['OBV'].iloc[-20]) if df['OBV'].iloc[-20] != 0 else 0
        
        details = {}
        if vol_ratio > 1.5 and -2 < price_chg < 5 and obv_trend > 0.1:
            phase = '🟢 AKUMULASI'
            score = 90
            confidence = 85
            details['Phase'] = 'AKUMULASI'
            details['Action'] = '🚀 BUY'
        elif price_chg > 5 and obv_trend > 0.05:
            phase = '🚀 MARKUP'
            score = 85
            confidence = 80
            details['Phase'] = 'MARKUP'
            details['Action'] = '🎯 HOLD'
        elif vol_ratio > 1.5 and price_chg < -3:
            phase = '🔴 DISTRIBUSI'
            score = 10
            confidence = 15
            details['Phase'] = 'DISTRIBUSI'
            details['Action'] = '🛑 AVOID'
        else:
            phase = '⚪ SIDEWAYS'
            score = 50
            confidence = 50
            details['Phase'] = 'SIDEWAYS'
            details['Action'] = '⏸️ WAIT'
        
        details['Volume'] = f'{vol_ratio:.2f}x'
        details['Price'] = f'{price_chg:+.2f}%'
        return score, details, phase, confidence
    except:
        return 0, {}, 'UNKNOWN', 0

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

# ============= TRACKING =============
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

def get_active_recommendations():
    conn = sqlite3.connect('screener_tracking.db')
    df = pd.read_sql("SELECT * FROM recommendations WHERE status='ACTIVE' ORDER BY date DESC", conn)
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

# ============= SIGNAL LEVELS =============
def get_signal_levels(score, price, confidence):
    if score >= 80 and confidence >= 70:
        signal = "STRONG BUY"
        signal_class = "strong-buy"
        trend = "🟢 Strong Uptrend"
        entry_ideal = round(price*0.98,0)
        entry_aggr = round(price,0)
        tp1_ideal = round(entry_ideal*1.08,0)
        tp2_ideal = round(entry_ideal*1.12,0)
        sl_ideal = round(entry_ideal*0.94,0)
        sl_aggr = round(entry_aggr*0.94,0)
    elif score >= 65:
        signal = "BUY"
        signal_class = "buy"
        trend = "🟢 Uptrend"
        entry_ideal = round(price*0.98,0)
        entry_aggr = round(price,0)
        tp1_ideal = round(entry_ideal*1.06,0)
        tp2_ideal = round(entry_ideal*1.10,0)
        sl_ideal = round(entry_ideal*0.95,0)
        sl_aggr = round(entry_aggr*0.95,0)
    elif score >= 50:
        signal = "WATCH"
        signal_class = "neutral"
        trend = "🟡 Monitor"
        entry_ideal = round(price*0.96,0)
        entry_aggr = None
        tp1_ideal = round(entry_ideal*1.05,0)
        tp2_ideal = round(entry_ideal*1.08,0)
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
                "BREN.JK","BRPT.JK","RATU.JK","RAJA.JK"]

# ============= MAIN =============
init_db()

st.markdown('<div class="big-title">🚀 IDX Power Screener v3</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fixed Logic | Validation | Tracking</div>', unsafe_allow_html=True)

tickers = load_tickers()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    menu = st.radio("📋 Menu", [
        "1️⃣ Full Screener", "2️⃣ Single Stock", "3️⃣ BPJS", "4️⃣ BSJP", 
        "5️⃣ Bandar Tracking", "6️⃣ Value Hunting",
        "7️⃣ Track Performance", "8️⃣ Active Positions", "9️⃣ Test Cases"
    ])
    st.markdown("---")
    
    if menu not in ["7️⃣ Track Performance", "8️⃣ Active Positions"]:
        period = st.selectbox("Period", ["3mo","6mo","1y"], index=1)
        if "Single" not in menu and "Test" not in menu:
            limit = st.slider("Max Tickers", 10, len(tickers), min(100, len(tickers)), step=10)
            min_score = st.slider("Min Score", 50, 100, 65, step=5)
            min_confidence = st.slider("Min Confidence", 40, 100, 60, step=5)
            use_parallel = st.checkbox("⚡ Fast Mode", value=True)
    
    st.markdown("---")
    st.caption("💡 IDX Traders v3")

# ============= MENUS =============
if "Test" in menu:
    st.markdown("### 🧪 Test Cases")
    if st.button("🔬 Run Test", type="primary"):
        for ticker in ["BREN.JK", "BRPT.JK", "RATU.JK", "RAJA.JK"]:
            df = fetch_data(ticker, period)
            if df is not None:
                score, details, conf = score_full_screener_v3(df)
                with st.expander(f"{ticker} - Score: {score} | Conf: {conf}%"):
                    st.metric("Score", f"{score}/100")
                    if score >= 65:
                        st.success("✅ SHOULD APPEAR")
                    else:
                        st.error("❌ REJECTED")
                    for k, v in details.items():
                        st.markdown(f"- **{k}:** {v}")

elif "Single" in menu:
    st.markdown("### 📈 Single Stock")
    selected = st.selectbox("Pilih Saham", tickers)
    if st.button("🔍 Analyze", type="primary"):
        df = fetch_data(selected, period)
        if df is not None:
            score, details, conf = score_full_screener_v3(df)
            price = df['Close'].iloc[-1]
            levels = get_signal_levels(score, price, conf)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Price", f"Rp {price:,.0f}")
            col2.metric("Score", f"{score}/100")
            col3.metric("Confidence", f"{conf}%")
            
            st.markdown st.markdown(f"### {levels['trend']}")
            
            for k, v in details.items():
                if '⛔' in k:
                    st.error(f"**{k}:** {v}")
                elif '⚠️' in k:
                    st.warning(f"**{k}:** {v}")
                else:
                    st.info(f"**{k}:** {v}")
            
            if levels["ideal"]["entry"]:
                st.markdown("### 🎯 Entry Strategy")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **Entry:** Rp {levels['ideal']['entry']:,.0f}
                    **TP1:** Rp {levels['ideal']['tp1']:,.0f}
                    **TP2:** Rp {levels['ideal']['tp2']:,.0f}
                    **SL:** Rp {levels['ideal']['sl']:,.0f}
                    """)

elif "Track" in menu:
    st.markdown("### 📊 Performance Tracking")
    stats = get_performance_stats()
    
    if stats['total'] == 0:
        st.info("👋 Belum ada data. Mulai dengan save recommendations!")
    else:
        col1, col2, col3, col4 = st.columns(4)
        win_rate = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
        col1.metric("Total", stats['total'])
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Avg Win", f"+{stats['avg_profit']:.2f}%")
        col4.metric("Avg Loss", f"{stats['avg_loss']:.2f}%")
        
        if not stats['by_strategy'].empty:
            st.markdown("### By Strategy")
            for _, row in stats['by_strategy'].iterrows():
                wr = (row['wins'] / row['total'] * 100) if row['total'] > 0 else 0
                st.write(f"**{row['strategy']}** - WR: {wr:.1f}% | Trades: {int(row['total'])}")

elif "Active" in menu:
    st.markdown("### 📋 Active Positions")
    active = get_active_recommendations()
    
    if active.empty:
        st.info("📭 No active positions")
    else:
        if st.button("🔄 Update Prices"):
            for _, row in active.iterrows():
                df = fetch_data(row['ticker'] + ".JK", "5d")
                if df is not None:
                    conn = sqlite3.connect('screener_tracking.db')
                    c = conn.cursor()
                    c.execute("UPDATE recommendations SET current_price=? WHERE id=?", 
                             (df['Close'].iloc[-1], row['id']))
                    conn.commit()
                    conn.close()
            st.success("✅ Updated!")
            st.rerun()
        
        for _, row in active.iterrows():
            pnl = ((row['current_price'] - row['entry_price']) / row['entry_price'] * 100)
            color = "🟢" if pnl > 0 else "🔴"
            
            with st.expander(f"{color} {row['ticker']} | {row['signal']} | P/L: {pnl:+.2f}%"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Entry", f"Rp {row['entry_price']:,.0f}")
                col2.metric("Current", f"Rp {row['current_price']:,.0f}")
                col3.metric("P/L", f"{pnl:+.2f}%")
                
                col1, col2, col3 = st.columns(3)
                if col1.button("✅ WIN", key=f"w{row['id']}"):
                    update_recommendation_status(row['id'], 'CLOSED', 'WIN', pnl, row['current_price'])
                    st.rerun()
                if col2.button("❌ LOSS", key=f"l{row['id']}"):
                    update_recommendation_status(row['id'], 'CLOSED', 'LOSS', pnl, row['current_price'])
                    st.rerun()
                if col3.button("⏸️ BE", key=f"b{row['id']}"):
                    update_recommendation_status(row['id'], 'CLOSED', 'BE', pnl, row['current_price'])
                    st.rerun()

elif "BPJS" in menu or "BSJP" in menu or "Bandar" in menu or "Value" in menu:
    strategy_map = {
        "3️⃣ BPJS": ("BPJS", "⚡ BPJS Scanner"),
        "4️⃣ BSJP": ("BSJP", "🌙 BSJP Scanner"),
        "5️⃣ Bandar Tracking": ("Bandar", "🎯 Bandar Tracker"),
        "6️⃣ Value Hunting": ("Value", "💎 Value Hunter")
    }
    strategy, title = strategy_map[menu]
    
    st.markdown(f"### {title}")
    
    if st.button(f"🚀 Run {strategy}", type="primary"):
        df = batch_scan(tickers, strategy, period, limit, use_parallel)
        
        if df.empty:
            st.warning(f"⚠️ No {strategy} signals found")
        else:
            df = df[(df["Score"] >= min_score) & (df["Confidence"] >= min_confidence)]
            
            if df.empty:
                st.warning("No stocks meeting criteria")
            else:
                st.success(f"✅ Found {len(df)} opportunities!")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Score", f"{df['Score'].mean():.1f}")
                col2.metric("Avg Conf", f"{df['Confidence'].mean():.1f}%")
                col3.metric("Strong Buy", len(df[df['Signal'] == 'STRONG BUY']))
                col4.metric("Buy", len(df[df['Signal'] == 'BUY']))
                
                show = df[["Ticker","Price","Score","Confidence","Signal","EntryIdeal","TP1","SL"]]
                st.dataframe(show, use_container_width=True, height=400)
                
                st.markdown("### 🏆 Top 15")
                for _, row in df.head(15).iterrows():
                    conf_color = "🟢" if row['Confidence'] >= 80 else "🟡" if row['Confidence'] >= 60 else "🟠"
                    
                    with st.expander(f"{conf_color} {row['Ticker']} - {row['Score']} | {row['Confidence']}% | {row['Signal']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Price:** Rp {row['Price']:,.0f}")
                            st.markdown(f"**Score:** {row['Score']}/100")
                            st.markdown(f"**Confidence:** {row['Confidence']}%")
                        with col2:
                            if row['EntryIdeal']:
                                st.markdown(f"**Entry:** Rp {row['EntryIdeal']:,.0f}")
                                st.markdown(f"**TP1:** Rp {row['TP1']:,.0f}")
                                st.markdown(f"**SL:** Rp {row['SL']:,.0f}")
                        
                        st.markdown("**Details:**")
                        for k, v in row['Details'].items():
                            st.markdown(f"- {k}: {v}")
                        
                        if st.button(f"💾 Track", key=f"t{row['Ticker']}"):
                            save_recommendation(row['Ticker'], strategy, row['Score'], 
                                              row['Confidence'], row['Price'], row['Signal'])
                            st.success("✅ Saved!")
                
                csv = show.to_csv(index=False).encode()
                st.download_button("📥 CSV", csv, f"{strategy}_{datetime.now().strftime('%Y%m%d')}.csv")
                
                if strategy == "BPJS":
                    st.info("💡 Entry 09:00-09:30 | Exit before 14:00 | Target 2-5%")
                elif strategy == "BSJP":
                    st.info("💡 Entry 14:00-15:00 | Exit next day 09:30-10:30 | Target 2-4%")

else:  # Full Screener
    st.markdown("### 🚀 Full Screener")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.info(f"📊 **Stocks**\n{limit}")
    col2.info(f"🎯 **Min Score**\n{min_score}")
    col3.info(f"📈 **Min Conf**\n{min_confidence}%")
    col4.info(f"⚡ **Mode**\n{'Parallel' if use_parallel else 'Sequential'}")
    
    if st.button("🚀 Run Screener", type="primary"):
        df = batch_scan(tickers, "Full Screener", period, limit, use_parallel)
        
        if df.empty:
            st.warning("⚠️ No stocks found")
        else:
            df = df[(df["Score"] >= min_score) & (df["Confidence"] >= min_confidence)]
            
            if df.empty:
                st.warning(f"No stocks with Score>={min_score} AND Conf>={min_confidence}")
            else:
                st.success(f"✅ Found {len(df)} quality stocks!")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Score", f"{df['Score'].mean():.1f}")
                col2.metric("Avg Confidence", f"{df['Confidence'].mean():.1f}%")
                col3.metric("Strong Buy", len(df[df['Signal'] == 'STRONG BUY']))
                col4.metric("Buy", len(df[df['Signal'] == 'BUY']))
                
                show = df[["Ticker","Price","Score","Confidence","Signal","Trend","EntryIdeal","TP1","TP2","SL"]]
                st.dataframe(show, use_container_width=True, height=400)
                
                st.markdown("### 🏆 Top 15 Recommendations")
                
                for _, row in df.head(15).iterrows():
                    conf_color = "🟢" if row['Confidence'] >= 80 else "🟡" if row['Confidence'] >= 60 else "🟠"
                    
                    with st.expander(f"{conf_color} {row['Ticker']} - Score: {row['Score']} | Conf: {row['Confidence']}% | {row['Signal']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**💰 Price:** Rp {row['Price']:,.0f}")
                            st.markdown(f"**📊 Score:** {row['Score']}/100")
                            st.markdown(f"**🎯 Confidence:** {row['Confidence']}%")
                            st.markdown(f"**📈 Signal:** {row['Signal']}")
                            st.markdown(f"**🔄 Trend:** {row['Trend']}")
                        
                        with col2:
                            if row['EntryIdeal']:
                                st.markdown(f"**🎯 Entry:** Rp {row['EntryIdeal']:,.0f}")
                                st.markdown(f"**🟢 TP1:** Rp {row['TP1']:,.0f}")
                                st.markdown(f"**🟢 TP2:** Rp {row['TP2']:,.0f}")
                                st.markdown(f"**🔴 SL:** Rp {row['SL']:,.0f}")
                                
                                if row['EntryIdeal'] and row['TP1'] and row['SL']:
                                    rr = (row['TP1'] - row['EntryIdeal']) / (row['EntryIdeal'] - row['SL'])
                                    st.markdown(f"**⚖️ R:R:** 1:{rr:.2f}")
                        
                        st.markdown("---")
                        st.markdown("**📋 Technical Details:**")
                        for k, v in row['Details'].items():
                            if '⛔' in k or '❌' in k:
                                st.markdown(f"- {k}: {v}")
                            else:
                                st.markdown(f"- **{k}:** {v}")
                        
                        if st.button(f"💾 Track This", key=f"track_{row['Ticker']}"):
                            save_recommendation(row['Ticker'], "Full Screener", 
                                              row['Score'], row['Confidence'], 
                                              row['Price'], row['Signal'])
                            st.success("✅ Saved to tracking!")
                
                csv = show.to_csv(index=False).encode()
                st.download_button("📥 Download Results", csv, 
                                 f"screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
                
                st.markdown("---")
                st.markdown("### 💡 Quick Tips")
                st.info("""
                ✅ **High Probability:**
                - Score > 75 + Confidence > 70%
                - Strong volume + Clear uptrend
                
                ⚠️ **Use Caution:**
                - Confidence < 60%
                - Overbought warnings
                
                🎯 **Best Practice:**
                - Verify with Single Stock analysis
                - Always use Stop Loss
                - Track results in Performance tab
                """)
