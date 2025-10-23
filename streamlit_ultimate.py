#!/usr/bin/env python3
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

st.set_page_config(page_title="IDX ULTIMATE", page_icon="🚀", layout="wide")

st.markdown("""<style>
.main-header {font-size: 3rem; font-weight: bold; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 2rem;}
.buy-signal {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 15px; color: white; font-size: 1.5rem; font-weight: bold; text-align: center;}
.sell-signal {background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); padding: 1.5rem; border-radius: 15px; color: white; font-size: 1.5rem; font-weight: bold; text-align: center;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_stocks():
    try:
        with open('idx_stocks.json', 'r') as f:
            data = json.load(f)
            return sorted(data) if isinstance(data, list) else sorted(data.get('stocks', []))
    except:
        return []

@st.cache_data(ttl=1800)
def get_data(ticker, period='6mo'):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty or len(df) < 50:
            return None
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        return df
    except:
        return None

def calc_score(df):
    if df is None or len(df) < 50:
        return 0, {}
    latest = df.iloc[-1]
    score = 0
    details = {}
    if latest['Close'] > latest['EMA9']:
        score += 10
        details['Price > EMA9'] = '✅ +10'
    else:
        details['Price > EMA9'] = '❌ 0'
    if latest['Close'] > latest['EMA21']:
        score += 10
        details['Price > EMA21'] = '✅ +10'
    else:
        details['Price > EMA21'] = '❌ 0'
    if latest['Close'] > latest['EMA50']:
        score += 10
        details['Price > EMA50'] = '✅ +10'
    else:
        details['Price > EMA50'] = '❌ 0'
    if latest['EMA9'] > latest['EMA21'] > latest['EMA50']:
        score += 10
        details['EMA Alignment'] = '✅ +10'
    else:
        details['EMA Alignment'] = '❌ 0'
    rsi = latest['RSI']
    if 40 <= rsi <= 70:
        score += 20
        details[f'RSI ({rsi:.1f})'] = '✅ +20'
    elif 30 < rsi < 40 or 70 < rsi < 80:
        score += 10
        details[f'RSI ({rsi:.1f})'] = '⚠️ +10'
    else:
        details[f'RSI ({rsi:.1f})'] = '❌ 0'
    if latest['MACD'] > latest['Signal_Line']:
        score += 20
        details['MACD'] = '✅ +20'
    else:
        details['MACD'] = '❌ 0'
    avg_vol = df['Volume'].tail(20).mean()
    if latest['Volume'] > avg_vol * 1.2:
        score += 10
        details['Volume'] = '✅ +10'
    else:
        details['Volume'] = '❌ 0'
    week_ago = df['Close'].iloc[-5] if len(df) >= 5 else df['Close'].iloc[0]
    wc = ((latest['Close'] - week_ago) / week_ago) * 100
    if 2 <= wc <= 20:
        score += 10
        details[f'Weekly Change ({wc:.1f}%)'] = '✅ +10'
    else:
        details[f'Weekly Change ({wc:.1f}%)'] = '❌ 0'
    return score, details

def get_signal(score):
    if score >= 70:
        return "STRONG BUY", "🟢", min(95, 50 + (score - 70))
    elif score >= 60:
        return "BUY", "🟢", 50 + (score - 60) * 2
    elif score >= 40:
        return "NEUTRAL", "🟡", 50
    else:
        return "SELL", "🔴", max(10, 50 - (40 - score))

def calc_strategy(price, signal):
    if "BUY" not in signal:
        return {'ideal': {'entry': None, 'tp1': None, 'tp2': None, 'sl': None, 'rr': 0, 'msg': '⚠️ NO BUY'}, 'aggr': {'entry': None, 'tp1': None, 'tp2': None, 'sl': None, 'rr': 0, 'msg': '⚠️ NO BUY'}}
    ie = price * 0.97
    it1 = ie * 1.10
    it2 = ie * 1.15
    isl = ie * 0.93
    irr = (it1 - ie) / (ie - isl) if ie > isl else 0
    ae = price
    at1 = ae * 1.08
    at2 = ae * 1.12
    asl = ae * 0.93
    arr = (at1 - ae) / (ae - asl) if ae > asl else 0
    return {'ideal': {'entry': ie, 'tp1': it1, 'tp2': it2, 'sl': isl, 'rr': irr, 'msg': '🎯 Wait -3%'}, 'aggr': {'entry': ae, 'tp1': at1, 'tp2': at2, 'sl': asl, 'rr': arr, 'msg': '⚡ Buy NOW'}}

def make_chart(df, ticker, strat):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50', line=dict(color='red', width=1)))
    if strat['ideal']['entry']:
        fig.add_hline(y=strat['ideal']['entry'], line_dash="dash", line_color="green", annotation_text="Ideal")
        fig.add_hline(y=strat['ideal']['tp1'], line_dash="dot", line_color="lime", annotation_text="TP1")
        fig.add_hline(y=strat['ideal']['sl'], line_dash="dash", line_color="red", annotation_text="SL")
    fig.update_layout(title=f'{ticker}', yaxis_title='Price', height=600, template='plotly_white', xaxis_rangeslider_visible=False)
    return fig

def analyze_stock(ticker):
    try:
        df = get_data(ticker, '6mo')
        if df is None:
            return None
        score, _ = calc_score(df)
        signal, _, prob = get_signal(score)
        price = df['Close'].iloc[-1]
        strat = calc_strategy(price, signal)
        return {'Ticker': ticker, 'Price': f'Rp {price:,.0f}', 'Score': score, 'Signal': signal, 'Prob': f'{prob:.1f}%', 'Entry': f'Rp {strat["ideal"]["entry"]:,.0f}' if strat['ideal']['entry'] else 'N/A'}
    except:
        return None

def main():
    st.markdown('<div class="main-header">🚀 IDX SCREENER ULTIMATE</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('## ⚙️ Settings')
        mode = st.radio('Mode:', ['📈 Single Stock', '🔍 Multi-Stock Screener'])
        st.markdown('---')
        if mode == '📈 Single Stock':
            stocks = load_stocks()
            if not stocks:
                st.error('No stocks!')
                return
            sel = st.selectbox('Stock:', stocks, index=stocks.index('TLKM.JK') if 'TLKM.JK' in stocks else 0)
            period = st.selectbox('Period:', ['3mo', '6mo', '1y'], index=1)
            btn = st.button('📊 ANALYZE', use_container_width=True, type='primary')
        else:
            min_score = st.slider('Min Score:', 0, 100, 60)
            max_stocks = st.slider('Max Stocks:', 10, 100, 50)
            scan_btn = st.button('🚀 START SCAN', use_container_width=True, type='primary')
    
    if mode == '📈 Single Stock' and btn:
        with st.spinner(f'Analyzing {sel}...'):
            df = get_data(sel, period)
            if df is None:
                st.error('❌ No data')
                return
            score, details = calc_score(df)
            signal, color, prob = get_signal(score)
            price = df['Close'].iloc[-1]
            strat = calc_strategy(price, signal)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric('Price', f'Rp {price:,.0f}')
            with c2:
                st.metric('Score', f'{score}/100')
            with c3:
                if 'BUY' in signal:
                    st.markdown(f'<div class="buy-signal">{color} {signal}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="sell-signal">{color} {signal}</div>', unsafe_allow_html=True)
            with c4:
                st.metric('Probability', f'{prob:.1f}%')
            st.markdown('---')
            st.plotly_chart(make_chart(df, sel, strat), use_container_width=True)
            st.markdown('---')
            if 'BUY' in signal:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('### 🎯 IDEAL')
                    st.markdown(f'**Entry:** Rp {strat["ideal"]["entry"]:,.0f}')
                    st.markdown(f'**TP1:** Rp {strat["ideal"]["tp1"]:,.0f} (+10%)')
                    st.markdown(f'**TP2:** Rp {strat["ideal"]["tp2"]:,.0f} (+15%)')
                    st.markdown(f'**SL:** Rp {strat["ideal"]["sl"]:,.0f} (-7%)')
                    st.markdown(f'**R:R:** 1:{strat["ideal"]["rr"]:.2f}')
                    st.info(strat['ideal']['msg'])
                with c2:
                    st.markdown('### ⚡ AGGRESSIVE')
                    st.markdown(f'**Entry:** Rp {strat["aggr"]["entry"]:,.0f}')
                    st.markdown(f'**TP1:** Rp {strat["aggr"]["tp1"]:,.0f} (+8%)')
                    st.markdown(f'**TP2:** Rp {strat["aggr"]["tp2"]:,.0f} (+12%)')
                    st.markdown(f'**SL:** Rp {strat["aggr"]["sl"]:,.0f} (-7%)')
                    st.markdown(f'**R:R:** 1:{strat["aggr"]["rr"]:.2f}')
                    st.warning(strat['aggr']['msg'])
            else:
                st.warning('⚠️ NO BUY signal!')
            st.markdown('---')
            st.markdown('### 📊 Score Breakdown')
            st.table(pd.DataFrame.from_dict(details, orient='index', columns=['Status']))
    
    elif mode == '🔍 Multi-Stock Screener' and scan_btn:
        stocks = load_stocks()[:max_stocks]
        st.info(f'🔍 Scanning {len(stocks)} stocks (~3-5 min)...')
        progress = st.progress(0)
        status = st.empty()
        results = []
        start = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(analyze_stock, t): t for t in stocks}
            for i, future in enumerate(as_completed(futures)):
                ticker = futures[future]
                elapsed = time.time() - start
                remain = (elapsed / (i + 1)) * (len(stocks) - i - 1) if i > 0 else 0
                status.text(f'Analyzing {ticker}... ({i+1}/{len(stocks)}) | {elapsed:.0f}s | Remain: {remain:.0f}s')
                try:
                    result = future.result()
                    if result and result['Score'] >= min_score:
                        results.append(result)
                except:
                    pass
                progress.progress((i + 1) / len(stocks))
        status.text('✅ Complete!')
        if results:
            df_res = pd.DataFrame(results).sort_values('Score', ascending=False)
            st.markdown(f'## 🎯 Found {len(df_res)} stocks')
            st.dataframe(df_res, use_container_width=True, height=600)
            csv = df_res.to_csv(index=False)
            st.download_button('📥 Download CSV', csv, f'scan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', 'text/csv')
        else:
            st.warning(f'No stocks found with score >= {min_score}')

if __name__ == '__main__':
    main()
