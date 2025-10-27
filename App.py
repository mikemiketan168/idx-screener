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

st.markdown("""
<style>
.main-header {font-size: 3rem; font-weight: bold; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 2rem;}
.buy-signal {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 15px; color: white; font-size: 1.5rem; font-weight: bold; text-align: center;}
.sell-signal {background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); padding: 1.5rem; border-radius: 15px; color: white; font-size: 1.5rem; font-weight: bold; text-align: center;}
</style>
""", unsafe_allow_html=True)

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
        return {'ideal': {'entry': None}}

    ie = price * 0.97
    isl = ie * 0.93
    it1 = ie * 1.10
    it2 = ie * 1.15
    irr = (it1 - ie) / (ie - isl)

    ae = price
    asl = ae * 0.93
    at1 = ae * 1.08
    at2 = ae * 1.12
    arr = (at1 - ae) / (ae - asl)

    return {
        'ideal': {'entry': ie, 'tp1': it1, 'tp2': it2, 'sl': isl, 'rr': irr},
        'aggr': {'entry': ae, 'tp1': at1, 'tp2': at2, 'sl': asl, 'rr': arr}
    }

def make_chart(df, ticker, strat):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50'))
    fig.update_layout(title=f'{ticker}', height=600, xaxis_rangeslider_visible=False)
    return fig

def analyze_stock(ticker):
    df = get_data(ticker, '6mo')
    if df is None:
        return None
    score, _ = calc_score(df)
    signal, _, prob = get_signal(score)
    price = df['Close'].iloc[-1]
    strat = calc_strategy(price, signal)
    return {
        'Ticker': ticker,
        'Price': f'Rp {price:,.0f}',
        'Score': score,
        'Signal': signal,
        'Prob': f'{prob:.1f}%'
    }

def main():
    st.markdown('<div class="main-header">🚀 IDX SCREENER ULTIMATE</div>', unsafe_allow_html=True)

    with st.sidebar:
        mode = st.radio('Mode:', ['📈 Single Stock', '🔍 Multi-Stock Screener'])

    if mode == '📈 Single Stock':
        stocks = load_stocks()
        if not stocks:
            st.error("Stock list failed to load!")
            return
        sel = st.selectbox('Stock:', stocks)
        if st.button('📊 Analyze'):
            df = get_data(sel)
            if df is None:
                st.error('No data!')
                return
            score, details = calc_score(df)
            signal, col, prob = get_signal(score)
            price = df['Close'].iloc[-1]
            strat = calc_strategy(price, signal)
            st.metric('Price', f'Rp {price:,.0f}')
            st.metric('Score', f'{score}/100')
            st.metric('Signal', signal)
            st.metric('Probability', f'{prob:.1f}%')
            st.plotly_chart(make_chart(df, sel, strat))

    elif mode == '🔍 Multi-Stock Screener':
        stocks = load_stocks()
        results = []
        with st.spinner("Scanning stocks..."):
            for t in stocks[:50]:
                r = analyze_stock(t)
                if r and r['Signal'] in ['BUY', 'STRONG BUY']:
                    results.append(r)
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
        else:
            st.warning("No opportunities found.")

if __name__ == "__main__":
    main()
