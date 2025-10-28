import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

st.set_page_config(page_title="IDX Screener Ultimate",
                   layout="wide",
                   page_icon="🚀")

@st.cache_data
def load_stocks():
    with open('idx_stocks.json', 'r') as f:
        return json.load(f)

stocks = load_stocks()

def get_data(ticker):
    data = yf.download(ticker, period="6mo", interval="1d")
    if data.empty:
        return None
    data["EMA9"] = data["Close"].ewm(span=9).mean()
    data["EMA21"] = data["Close"].ewm(span=21).mean()
    data["EMA50"] = data["Close"].ewm(span=50).mean()
    return data

def analyze_stock(df):
    price = df["Close"].iloc[-1]
    ema9 = df["EMA9"].iloc[-1]
    ema21 = df["EMA21"].iloc[-1]
    ema50 = df["EMA50"].iloc[-1]

    score = 0
    signal = "HOLD"
    probability = 50

    if price > ema9 > ema21 > ema50:
        score = 100
        signal = "STRONG BUY"
        probability = 85
    elif price > ema21 > ema50:
        score = 90
        signal = "BUY"
        probability = 70
    elif price < ema50:
        score = 20
        signal = "SELL"
        probability = 30

    entry_ideal = round(ema9, 2)
    entry_aggr = round(ema21, 2)
    tp1 = round(price * 1.03, 2)
    tp2 = round(price * 1.05, 2)
    cutloss = round(price * 0.97, 2)

    return price, score, signal, probability, entry_ideal, entry_aggr, tp1, tp2, cutloss

st.title("🚀 IDX SCREENER ULTIMATE")

mode = st.sidebar.radio("Mode:",
                        ["📊 Single Stock", "🔍 Multi-Stock Screener"])

if mode == "📊 Single Stock":
    ticker = st.selectbox("Stock:", stocks)
    if st.button("📈 Analyze"):
        df = get_data(ticker)
        if df is not None:
            price, score, signal, prob, entry, entry_aggr, tp1, tp2, cl = analyze_stock(df)

            st.subheader(f"📌 {ticker}")
            st.metric("Price", f"Rp {price:,.0f}")
            st.metric("Score", f"{score}/100")
            st.metric("Signal", signal)
            st.metric("Probability", f"{prob}%")

            st.info(f"""
*🎯 Entry Ideal:* Rp {entry:,.0f}  
*⚡ Entry Agresif:* Rp {entry_aggr:,.0f}  
*🥇 Target Profit 1:* Rp {tp1:,.0f}  
*🏆 Target Profit 2:* Rp {tp2:,.0f}  
*🛑 Cut Loss:* Rp {cl:,.0f}
""")

            st.line_chart(df[["Close", "EMA9", "EMA21", "EMA50"]])

        else:
            st.error("❌ Data not available!")

else:
    if st.button("🚀 Screener Now!"):
        result = []

        for t in stocks:
            df = get_data(t)
            if df is not None:
                price, score, signal, prob, entry, entry_aggr, tp1, tp2, cl = analyze_stock(df)
                result.append([t, price, score, signal, prob])

        df_res = pd.DataFrame(result,
                              columns=["Ticker", "Price", "Score", "Signal", "Prob"])
        df_res = df_res.sort_values(by=["Score", "Prob"], ascending=False)
        st.dataframe(df_res)
