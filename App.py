#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime

st.set_page_config(page_title="IDX Power Screener", page_icon="🚀", layout="wide")

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

@st.cache_data(ttl=600, show_spinner=False)
def fetch_data(ticker, period="6mo"):
    try:
        end = int(datetime.now().timestamp())
        days = {"5d":5,"1mo":30,"3mo":90,"6mo":180,"1y":365}.get(period,180)
        start = end - (days*86400)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        r = requests.get(url, params={"period1":start,"period2":end,"interval":"1d"}, headers={'User-Agent':'Mozilla/5.0'}, timeout=15, verify=False)
        if r.status_code != 200:
            return None
        data = r.json()['chart']['result'][0]
        q = data['indicators']['quote'][0]
        df = pd.DataFrame({'Open':q['open'],'High':q['high'],'Low':q['low'],'Close':q['close'],'Volume':q['volume']}, index=pd.to_datetime(data['timestamp'], unit='s'))
        df = df.dropna()
        if len(df) < 20:
            return None
        df['EMA9'] = df['Close'].ewm(9).mean()
        df['EMA21'] = df['Close'].ewm(21).mean()
        df['EMA50'] = df['Close'].ewm(50).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta>0,0)).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        df['RSI'] = 100 - (100/(1+gain/loss))
        exp1 = df['Close'].ewm(12).mean()
        exp2 = df['Close'].ewm(26).mean()
        df['MACD'] = exp1 - exp2
        df['SIGNAL'] = df['MACD'].ewm(9).mean()
        df['BB_MID'] = df['Close'].rolling(20).mean()
        df['BB_STD'] = df['Close'].rolling(20).std()
        df['BB_UPPER'] = df['BB_MID'] + 2*df['BB_STD']
        df['BB_LOWER'] = df['BB_MID'] - 2*df['BB_STD']
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['STOCH'] = 100*(df['Close']-low14)/(high14-low14)
        return df
    except:
        return None

def get_signal_levels(score, price):
    if score >= 80:
        signal = "STRONG BUY"
        signal_class = "strong-buy"
        trend = "🟢 Strong Uptrend"
        entry_ideal = round(price*0.97,0)
        entry_aggr = round(price,0)
        tp1_ideal = round(entry_ideal*1.10,0)
        tp2_ideal = round(entry_ideal*1.15,0)
        tp1_aggr = round(entry_aggr*1.08,0)
        tp2_aggr = round(entry_aggr*1.12,0)
        sl_ideal = round(entry_ideal*0.93,0)
        sl_aggr = round(entry_aggr*0.93,0)
    elif score >= 60:
        signal = "BUY"
        signal_class = "buy"
        trend = "🟢 Uptrend"
        entry_ideal = round(price*0.98,0)
        entry_aggr = round(price,0)
        tp1_ideal = round(entry_ideal*1.08,0)
        tp2_ideal = round(entry_ideal*1.12,0)
        tp1_aggr = round(entry_aggr*1.06,0)
        tp2_aggr = round(entry_aggr*1.10,0)
        sl_ideal = round(entry_ideal*0.95,0)
        sl_aggr = round(entry_aggr*0.95,0)
    elif score >= 40:
        signal = "NEUTRAL"
        signal_class = "neutral"
        trend = "🟡 Sideways"
        entry_ideal = round(price*0.95,0)
        entry_aggr = None
        tp1_ideal = round(entry_ideal*1.05,0)
        tp2_ideal = round(entry_ideal*1.08,0)
        tp1_aggr = None
        tp2_aggr = None
        sl_ideal = round(entry_ideal*0.97,0)
        sl_aggr = None
    else:
        signal = "SELL/HOLD"
        signal_class = "sell"
        trend = "🔴 Downtrend"
        entry_ideal = None
        entry_aggr = None
        tp1_ideal = None
        tp2_ideal = None
        tp1_aggr = None
        tp2_aggr = None
        sl_ideal = None
        sl_aggr = None
    return {"signal":signal,"signal_class":signal_class,"trend":trend,"ideal":{"entry":entry_ideal,"tp1":tp1_ideal,"tp2":tp2_ideal,"sl":sl_ideal},"aggr":{"entry":entry_aggr,"tp1":tp1_aggr,"tp2":tp2_aggr,"sl":sl_aggr}}

def score_full_screener(df):
    try:
        r = df.iloc[-1]
        score, det = 0, {}
        if r['Close'] > r['EMA9'] > r['EMA21'] > r['EMA50']:
            score += 40
            det['Trend'] = '✅ Strong Uptrend (+40)'
        elif r['Close'] > r['EMA9']:
            score += 20
            det['Trend'] = '⚠️ Short Up (+20)'
        else:
            det['Trend'] = '❌ Down (+0)'
        if 40 <= r['RSI'] <= 70:
            score += 20
            det['RSI'] = f"✅ {r['RSI']:.1f} (+20)"
        elif 30 < r['RSI'] < 80:
            score += 10
            det['RSI'] = f"⚠️ {r['RSI']:.1f} (+10)"
        else:
            det['RSI'] = f"❌ {r['RSI']:.1f} (+0)"
        if r['MACD'] > r['SIGNAL']:
            score += 20
            det['MACD'] = '✅ Bullish (+20)'
        else:
            det['MACD'] = '❌ Bearish (+0)'
        vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        if r['Volume'] > vol_avg*1.3:
            score += 10
            det['Volume'] = '✅ High (+10)'
        else:
            det['Volume'] = '❌ Low (+0)'
        mom = (r['Close']-df['Close'].iloc[-5])/df['Close'].iloc[-5]*100
        if 2 <= mom <= 15:
            score += 10
            det['Momentum'] = f'✅ {mom:.1f}% (+10)'
        else:
            det['Momentum'] = f'❌ {mom:.1f}% (+0)'
        return score, det
    except:
        return 0, {}

def score_bpjs(df):
    try:
        r = df.iloc[-1]
        score, det = 0, {}
        vol_pct = ((df['High']-df['Low'])/df['Low']*100).tail(5).mean()
        if 2 < vol_pct < 5:
            score += 30
            det['Volatility'] = f'✅ {vol_pct:.2f}% (+30)'
        else:
            det['Volatility'] = f'❌ {vol_pct:.2f}% (+0)'
        vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        if r['Volume'] > vol_avg*1.5:
            score += 25
            det['Volume'] = '✅ Surge (+25)'
        else:
            det['Volume'] = '❌ Low (+0)'
        if 30 < r['RSI'] < 50:
            score += 25
            det['RSI'] = f"✅ {r['RSI']:.1f} (+25)"
        else:
            det['RSI'] = f"❌ {r['RSI']:.1f} (+0)"
        if r['STOCH'] < 30:
            score += 20
            det['Stochastic'] = f"✅ {r['STOCH']:.1f} (+20)"
        else:
            det['Stochastic'] = f"❌ {r['STOCH']:.1f} (+0)"
        return score, det
    except:
        return 0, {}

def score_bsjp(df):
    try:
        r = df.iloc[-1]
        score, det = 0, {}
        bb_pos = (r['Close']-r['BB_LOWER'])/(r['BB_UPPER']-r['BB_LOWER'])*100
        if bb_pos < 20:
            score += 30
            det['BB Position'] = f'✅ {bb_pos:.1f}% (+30)'
        else:
            det['BB Position'] = f'❌ {bb_pos:.1f}% (+0)'
        gap = (r['Close']-df['Close'].iloc[-2])/df['Close'].iloc[-2]*100
        if -1.5 < gap < 0:
            score += 25
            det['Gap'] = f'✅ {gap:.2f}% (+25)'
        else:
            det['Gap'] = f'❌ {gap:.2f}% (+0)'
        if 35 < r['RSI'] < 55:
            score += 25
            det['RSI'] = f"✅ {r['RSI']:.1f} (+25)"
        else:
            det['RSI'] = f"❌ {r['RSI']:.1f} (+0)"
        gains = []
        for i in range(-5,-1):
            try:
                gain = (df['Open'].iloc[i+1]-df['Close'].iloc[i])/df['Close'].iloc[i]*100
                gains.append(gain)
            except:
                pass
        if gains and np.mean(gains) > 0.3:
            score += 20
            det['History'] = f'✅ +{np.mean(gains):.2f}% (+20)'
        else:
            det['History'] = '❌ Negative (+0)'
        return score, det
    except:
        return 0, {}

def score_bandar(df):
    try:
        score, det = 0, {}
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
        obv_trend = (df['OBV'].iloc[-1]-df['OBV'].iloc[-20])/abs(df['OBV'].iloc[-20])
        if vol_ratio > 1.3 and price_chg > -2 and obv_trend > 0.1:
            phase = '🟢 AKUMULASI'
            score = 85
            det['Phase'] = 'AKUMULASI'
            det['Action'] = '🚀 BUY'
        elif vol_ratio > 1.3 and price_chg < -3:
            phase = '🔴 DISTRIBUSI'
            score = 15
            det['Phase'] = 'DISTRIBUSI'
            det['Action'] = '🛑 AVOID'
        elif price_chg > 5:
            phase = '🚀 MARKUP'
            score = 90
            det['Phase'] = 'MARKUP'
            det['Action'] = '🎯 HOLD'
        else:
            phase = '⚪ SIDEWAYS'
            score = 50
            det['Phase'] = 'SIDEWAYS'
            det['Action'] = '⏸️ WAIT'
        det['Volume'] = f'{vol_ratio:.2f}x avg'
        det['Price'] = f'{price_chg:+.2f}%'
        det['OBV'] = f"{'📈 Up' if obv_trend>0 else '📉 Down'}"
        return score, det, phase
    except:
        return 0, {}, 'UNKNOWN'

def score_value(df):
    try:
        r = df.iloc[-1]
        score, det = 0, {}
        high52 = df['High'].tail(252).max() if len(df)>252 else df['High'].max()
        low52 = df['Low'].tail(252).min() if len(df)>252 else df['Low'].min()
        pos52 = (r['Close']-low52)/(high52-low52)*100
        if pos52 < 25:
            score += 30
            det['52W Pos'] = f'✅ {pos52:.1f}% MURAH (+30)'
        elif pos52 < 40:
            score += 20
            det['52W Pos'] = f'⚠️ {pos52:.1f}% (+20)'
        else:
            det['52W Pos'] = f'❌ {pos52:.1f}% (+0)'
        if 25 < r['RSI'] < 40:
            score += 25
            det['RSI'] = f"✅ {r['RSI']:.1f} (+25)"
        else:
            det['RSI'] = f"❌ {r['RSI']:.1f} (+0)"
        vol_ratio = df['Volume'].tail(5).mean()/df['Volume'].rolling(20).mean().iloc[-1]
        if vol_ratio > 1.5:
            score += 20
            det['Volume'] = f'✅ {vol_ratio:.2f}x (+20)'
        else:
            det['Volume'] = f'❌ {vol_ratio:.2f}x (+0)'
        if r['Close'] > r['SMA20']:
            score += 15
            det['Trend'] = '✅ Above SMA20 (+15)'
        else:
            det['Trend'] = '❌ Below SMA20 (+0)'
        vol_pct = df['Close'].pct_change().std()*100
        if vol_pct < 2.5:
            score += 10
            det['Stability'] = f'✅ {vol_pct:.2f}% (+10)'
        else:
            det['Stability'] = f'❌ {vol_pct:.2f}% (+0)'
        target = low52+(high52-low52)*0.6
        potential = (target-r['Close'])/r['Close']*100
        det['Potential'] = f'🎯 +{potential:.1f}%'
        return score, det
    except:
        return 0, {}

def batch_scan(tickers, strategy, period, limit):
    results = []
    if limit:
        tickers = tickers[:limit]
    progress = st.progress(0)
    status = st.empty()
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        progress.progress((i+1)/total)
        status.text(f"📊 {i+1}/{total}: {ticker}")
        try:
            df = fetch_data(ticker, period)
            if df is None or len(df) < 50:
                continue
            price = float(df['Close'].iloc[-1])
            if strategy == "Full Screener":
                score, details = score_full_screener(df)
            elif strategy == "BPJS":
                score, details = score_bpjs(df)
            elif strategy == "BSJP":
                score, details = score_bsjp(df)
            elif strategy == "Bandar":
                score, details, phase = score_bandar(df)
            elif strategy == "Value":
                score, details = score_value(df)
            levels = get_signal_levels(score, price)
            results.append({"Ticker":ticker,"Price":price,"Score":score,"Signal":levels["signal"],"Trend":levels["trend"],"EntryIdeal":levels["ideal"]["entry"],"EntryAggr":levels["aggr"]["entry"],"TP1":levels["ideal"]["tp1"],"TP2":levels["ideal"]["tp2"],"SL":levels["ideal"]["sl"],"Details":details})
        except:
            pass
        time.sleep(0.4)
    progress.empty()
    status.empty()
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("Score", ascending=False)

def load_tickers():
    try:
        with open("idx_stocks.json","r") as f:
            data = json.load(f)
        tickers = data.get("tickers",[])
        return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except:
        return ["BBCA.JK","BBRI.JK","BMRI.JK","TLKM.JK","ASII.JK"]

st.markdown('<div class="big-title">🚀 IDX Power Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Professional Stock Screening - 900+ Saham IDX</div>', unsafe_allow_html=True)
tickers = load_tickers()
st.success(f"✅ **{len(tickers)} saham** loaded")

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    menu = st.radio("📋 Menu", ["1️⃣ Full Screener","2️⃣ Single Stock","3️⃣ BPJS","4️⃣ BSJP","5️⃣ Bandar Tracking","6️⃣ Value Hunting"])
    st.markdown("---")
    if "Single" not in menu:
        period = st.selectbox("Period", ["3mo","6mo","1y"], index=1)
        limit = st.slider("Max Tickers", 10, 300, 100, step=10)
        min_score = st.slider("Min Score", 0, 100, 60, step=5)
    else:
        period = st.selectbox("Period", ["3mo","6mo","1y"], index=1)
    st.markdown("---")
    st.caption("💡 IDX Traders")

if "Single" in menu:
    st.markdown("### 📈 Single Stock Analysis")
    col1, col2 = st.columns([3,1])
    with col1:
        selected = st.selectbox("Pilih Saham", tickers, index=tickers.index("BBCA.JK") if "BBCA.JK" in tickers else 0)
    with col2:
        analyze_btn = st.button("🔍 Analyze", type="primary")
    if analyze_btn:
        with st.spinner(f"Analyzing {selected}..."):
            df = fetch_data(selected, period)
        if df is None:
            st.error("❌ Data tidak tersedia")
        else:
            price = float(df['Close'].iloc[-1])
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview","⚡ BPJS","🌙 BSJP","🎯 Bandar","💎 Value"])
            with tab1:
                score, det = score_full_screener(df)
                levels = get_signal_levels(score, price)
                col1, col2, col3 = st.columns(3)
                col1.metric("Price", f"Rp {price:,.0f}")
                col2.metric("Score", f"{score}/100")
                col3.markdown(f'<div class="signal-box {levels["signal_class"]}">{levels["signal"]}</div>', unsafe_allow_html=True)
                st.markdown(f"### {levels['trend']}")
                st.line_chart(df[['Close','EMA9','EMA21','EMA50']])
                st.markdown("---")
                st.markdown("## 🎯 Entry & Exit Strategy")
                if levels["ideal"]["entry"] is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### 💎 Ideal Entry")
                        entry_i = levels["ideal"]["entry"]
                        tp1_i = levels["ideal"]["tp1"]
                        tp2_i = levels["ideal"]["tp2"]
                        sl_i = levels["ideal"]["sl"]
                        gain_tp1 = ((tp1_i/entry_i)-1)*100
                        gain_tp2 = ((tp2_i/entry_i)-1)*100
                        loss_sl = ((sl_i/entry_i)-1)*100
                        st.info(f"**Entry:** Rp {entry_i:,.0f}\n\n**TP1:** Rp {tp1_i:,.0f} (+{gain_tp1:.1f}%)\n\n**TP2:** Rp {tp2_i:,.0f} (+{gain_tp2:.1f}%)\n\n**SL:** Rp {sl_i:,.0f} ({loss_sl:.1f}%)")
                        st.caption("⏳ Tunggu dip 2-3%")
                    with col2:
                        if levels["aggr"]["entry"] is not None:
                            st.markdown("### ⚡ Aggressive")
                            entry_a = levels["aggr"]["entry"]
                            tp1_a = levels["aggr"]["tp1"]
                            tp2_a = levels["aggr"]["tp2"]
                            sl_a = levels["aggr"]["sl"]
                            gain_tp1_a = ((tp1_a/entry_a)-1)*100
                            gain_tp2_a = ((tp2_a/entry_a)-1)*100
                            loss_sl_a = ((sl_a/entry_a)-1)*100
                            st.success(f"**Entry:** Rp {entry_a:,.0f} (NOW)\n\n**TP1:** Rp {tp1_a:,.0f} (+{gain_tp1_a:.1f}%)\n\n**TP2:** Rp {tp2_a:,.0f} (+{gain_tp2_a:.1f}%)\n\n**SL:** Rp {sl_a:,.0f} ({loss_sl_a:.1f}%)")
                            st.caption("🚀 Entry sekarang!")
                        else:
                            st.warning("⚠️ Not recommended")
                else:
                    st.error("🛑 **NO ENTRY**")
                    st.warning("Tunggu setup lebih baik")
                st.markdown("---")
                st.markdown("## 📊 Analysis")
                for k, v in det.items():
                    st.markdown(f"**{k}:** {v}")
            with tab2:
                st.markdown("#### ⚡ BPJS")
                score, det = score_bpjs(df)
                if score >= 70:
                    st.success(f"🟢 {score}/100 - **BUY PAGI**")
                else:
                    st.warning(f"🟡 {score}/100 - **WAIT**")
                for k, v in det.items():
                    st.markdown(f"- {k}: {v}")
            with tab3:
                st.markdown("#### 🌙 BSJP")
                score, det = score_bsjp(df)
                if score >= 70:
                    st.success(f"🟢 {score}/100 - **BUY SORE**")
                else:
                    st.warning(f"🟡 {score}/100 - **WAIT**")
                for k, v in det.items():
                    st.markdown(f"- {k}: {v}")
            with tab4:
                st.markdown("#### 🎯 Bandar")
                score, det, phase = score_bandar(df)
                if "AKUMULASI" in phase or "MARKUP" in phase:
                    st.success(f"🟢 {phase} ({score}/100)")
                elif "DISTRIBUSI" in phase:
                    st.error(f"🔴 {phase} ({score}/100)")
                else:
                    st.warning(f"🟡 {phase} ({score}/100)")
                for k, v in det.items():
                    st.markdown(f"- {k}: {v}")
            with tab5:
                st.markdown("#### 💎 Value")
                score, det = score_value(df)
                if score >= 70:
                    st.success(f"🟢 {score}/100 - **VALUE BUY**")
                else:
                    st.warning(f"🟡 {score}/100 - **MONITOR**")
                for k, v in det.items():
                    st.markdown(f"- {k}: {v}")
else:
    strategy_map = {"1️⃣ Full Screener":"Full Screener","3️⃣ BPJS":"BPJS","4️⃣ BSJP":"BSJP","5️⃣ Bandar Tracking":"Bandar","6️⃣ Value Hunting":"Value"}
    strategy = strategy_map[menu]
    st.markdown(f"### {menu}")
    if st.button("🚀 Run Screener", type="primary"):
        df = batch_scan(tickers, strategy, period, limit)
        if df.empty:
            st.warning("⚠️ No data")
        else:
            df = df[df["Score"] >= min_score].head(50)
            if df.empty:
                st.info(f"No stocks >= {min_score}")
            else:
                st.success(f"✅ {len(df)} stocks!")
                show = df[["Ticker","Price","Score","Signal","Trend","EntryIdeal","TP1","TP2","SL"]]
                st.dataframe(show, use_container_width=True, height=400)
                st.markdown("### 🏆 Top 5")
                for idx, row in df.head(5).iterrows():
                    with st.expander(f"{row['Ticker']} - {row['Score']} ({row['Signal']})"):
                        st.markdown(f"**Price:** Rp {row['Price']:,.0f}")
                        st.markdown(f"**Signal:** {row['Signal']}")
                        st.markdown(f"**Trend:** {row['Trend']}")
                        if row['EntryIdeal']:
                            st.markdown(f"**Entry:** Rp {row['EntryIdeal']:,.0f}")
                            st.markdown(f"**TP1:** Rp {row['TP1']:,.0f} | **TP2:** Rp {row['TP2']:,.0f}")
                            st.markdown(f"**SL:** Rp {row['SL']:,.0f}")
                        for k, v in row['Details'].items():
                            st.markdown(f"- {k}: {v}")
                csv = show.to_csv(index=False).encode()
                st.download_button("📥 CSV", csv, f"{strategy}_{datetime.now().strftime('%Y%m%d')}.csv")
