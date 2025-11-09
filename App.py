#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

st.set_page_config(page_title="IDX Power Screener v4.0", page_icon="🎯", layout="wide")

# ============= LOAD TICKERS =============
def load_tickers():
    try:
        if os.path.exists("idx_stocks.json"):
            with open("idx_stocks.json", "r") as f:
                data = json.load(f)
            tickers = data.get("tickers", [])
            return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except:
        pass
    
    # Default 100+ tickers
    return ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK",
            "TLKM.JK", "EXCL.JK", "ISAT.JK", "ASII.JK", "AUTO.JK",
            "UNTR.JK", "PTBA.JK", "ADRO.JK", "ITMG.JK", "BREN.JK",
            "BRPT.JK", "PGAS.JK", "INDF.JK", "ICBP.JK", "MYOR.JK",
            "KLBF.JK", "KAEF.JK", "SIDO.JK", "CPIN.JK", "JPFA.JK",
            "SMGR.JK", "WSBP.JK", "INTP.JK", "UNVR.JK", "HMSP.JK",
            "GOTO.JK", "BUKA.JK", "EMTK.JK", "AMMN.JK", "ANTM.JK",
            "TOWR.JK", "TBIG.JK", "PLIN.JK", "MAPI.JK", "LPPF.JK"]

# ============= FETCH DATA =============
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
        if 'chart' not in data:
            return None
            
        result = data['chart']['result'][0]
        q = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Close':q['close'],'Volume':q['volume'],'High':q['high'],'Low':q['low']
        }, index=pd.to_datetime(result['timestamp'], unit='s'))
        
        df = df.dropna()
        if len(df) < 50:
            return None
        
        # Simple indicators
        df['EMA9'] = df['Close'].ewm(span=9).mean()
        df['EMA21'] = df['Close'].ewm(span=21).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        
        delta = df['Close'].diff()
        gain = delta.where(delta>0,0).rolling(14).mean()
        loss = -delta.where(delta<0,0).rolling(14).mean()
        df['RSI'] = 100 - (100/(1+gain/loss))
        
        df['VOL_SMA'] = df['Volume'].rolling(20).mean()
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA']
        
        df['MOM_5D'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        
        return df
    except:
        return None

# ============= SCORING =============
def score_stock(df):
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        
        # Reject downtrends
        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "Downtrend"}, 0, "D"
        
        # Trend
        if r['Close'] > r['EMA9'] > r['EMA21'] > r['EMA50']:
            score += 40
            details['Trend'] = '🟢 Strong uptrend'
        elif r['Close'] > r['EMA9']:
            score += 20
            details['Trend'] = '🟡 Short uptrend'
        
        # RSI
        if 45 <= r['RSI'] <= 60:
            score += 30
            details['RSI'] = f'🟢 Sweet spot {r["RSI"]:.0f}'
        elif 40 <= r['RSI'] <= 70:
            score += 15
            details['RSI'] = f'🟡 OK {r["RSI"]:.0f}'
        
        # Volume
        vol = df['VOL_RATIO'].tail(5).mean()
        if vol > 1.5:
            score += 30
            details['Volume'] = f'🟢 Strong {vol:.1f}x'
        elif vol > 1.0:
            score += 15
            details['Volume'] = f'🟡 Normal {vol:.1f}x'
        
        # Grade
        if score >= 80:
            grade = "A"
        elif score >= 60:
            grade = "B"
        else:
            grade = "C"
        
        confidence = min(score, 100)
        
        return score, details, confidence, grade
    except:
        return 0, {}, 0, "D"

# ============= PROCESS =============
def process_ticker(ticker, period):
    try:
        df = fetch_data(ticker, period)
        if df is None:
            return None
        
        price = float(df['Close'].iloc[-1])
        score, details, confidence, grade = score_stock(df)
        
        if score < 50:
            return None
        
        entry = round(price * 0.98, 0)
        tp1 = round(entry * 1.08, 0)
        sl = round(entry * 0.94, 0)
        
        return {
            "Ticker": ticker.replace('.JK',''),
            "Price": price,
            "Score": score,
            "Confidence": confidence,
            "Grade": grade,
            "Entry": entry,
            "TP1": tp1,
            "SL": sl,
            "Details": details
        }
    except:
        return None

def scan_stocks(tickers, period, limit_stage1, limit_stage2):
    st.info(f"🔍 **STAGE 1**: Quick scan {len(tickers)} stocks...")
    
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ticker, t, period): t for t in tickers}
        completed = 0
        
        for future in as_completed(futures):
            completed += 1
            progress.progress(completed / len(tickers))
            status.text(f"📊 {completed}/{len(tickers)} | Found: {len(results)}")
            
            result = future.result()
            if result:
                results.append(result)
            
            time.sleep(0.05)
    
    progress.empty()
    status.empty()
    
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    
    df1 = pd.DataFrame(results).sort_values("Score", ascending=False).head(limit_stage1)
    st.success(f"✅ Stage 1: Found {len(df1)} candidates")
    
    df2 = df1[df1['Grade'].isin(['A','B'])].head(limit_stage2)
    st.success(f"🏆 Stage 2: {len(df2)} elite picks!")
    
    return df1, df2

# ============= UI =============
st.title("🎯 IDX Power Screener v4.0")
st.caption("2-Stage Filter: Scan All → Top 50 → Top 10 Elite")

tickers = load_tickers()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.info(f"📊 Total stocks: {len(tickers)}")
    
    period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
    
    st.markdown("### 🎯 Filtering")
    limit1 = st.slider("Stage 1: Top N", 20, 100, 50, 10)
    limit2 = st.slider("Stage 2: Elite", 5, 30, 10, 5)
    
    st.caption(f"Scan {len(tickers)} → Top {limit1} → Elite {limit2}")

if st.button("🚀 START SCAN", type="primary"):
    df1, df2 = scan_stocks(tickers, period, limit1, limit2)
    
    if df2.empty:
        st.warning("⚠️ No elite stocks found")
        if not df1.empty:
            st.info(f"But found {len(df1)} candidates in Stage 1")
            st.dataframe(df1)
    else:
        st.markdown(f"### 🏆 TOP {len(df2)} ELITE PICKS")
        
        for _, row in df2.iterrows():
            emoji = "💎" if row['Grade']=='A' else "🔹"
            
            with st.expander(f"{emoji} {row['Ticker']} | Grade {row['Grade']} | Score: {row['Score']}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"Rp {row['Price']:,.0f}")
                col2.metric("Score", f"{row['Score']}/100")
                col3.metric("Confidence", f"{row['Confidence']}%")
                col4.metric("Grade", row['Grade'])
                
                st.success(f"""
                **Entry:** Rp {row['Entry']:,.0f}
                **TP1 (+8%):** Rp {row['TP1']:,.0f}
                **SL (-6%):** Rp {row['SL']:,.0f}
                """)
                
                for k, v in row['Details'].items():
                    st.caption(f"• {k}: {v}")
        
        with st.expander(f"📊 Stage 1: All {len(df1)} Candidates"):
            st.dataframe(df1)

st.markdown("---")
st.caption("🎯 IDX Power Screener v4.0 | Educational purposes only")
