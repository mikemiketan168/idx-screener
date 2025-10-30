import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf

st.set_page_config(page_title="🚀 IDX Screener — Auto", layout="wide")

st.title("🚀 IDX Screener — Auto Mode (No API Key)")
st.caption("Hybrid FastIDX + Yahoo auto fallback | by Chef 🧠")

# ========== Sidebar ==========
st.sidebar.header("⚙️ Pengaturan")
max_tickers = st.sidebar.slider("Jumlah saham (Top N)", 10, 200, 50)
period = st.sidebar.selectbox("Periode data", ["1mo", "3mo", "6mo", "1y"], index=1)

# ========== List Saham ==========
tickers = [
    "TLKM.JK","BREN.JK","CUAN.JK","BRPT.JK","PGUN.JK","PTRO.JK",
    "BBCA.JK","BBRI.JK","ADRO.JK","TPIA.JK","AMMN.JK",
    "RAJA.JK","CDIA.JK","GTSI.JK","DADA.JK"
]

# ========== Fungsi ambil data ==========
def fetch_fastidx_public(ticker):
    """Ambil data dari public FastIDX endpoint"""
    try:
        url = f"https://fastidx.pro/api/public/{ticker}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "historical" in data:
            df = pd.DataFrame(data["historical"])
            df["Date"] = pd.to_datetime(df["date"])
            df.set_index("Date", inplace=True)
            df["Close"] = df["close"]
            return df[["Close"]]
        else:
            return pd.DataFrame()
    except Exception as e:
        print("FastIDX public error:", e)
        return pd.DataFrame()

def fetch_yahoo(ticker):
    """Fallback ke yfinance"""
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, threads=False)
        if not df.empty:
            df = df[["Close"]]
        return df
    except Exception as e:
        print("Yahoo error:", e)
        return pd.DataFrame()

def fetch_one(ticker):
    df = fetch_fastidx_public(ticker)
    if df is None or df.empty:
        df = fetch_yahoo(ticker)
    return df

# ========== Analisa Tunggal ==========
st.subheader("📈 Analisa Tunggal Saham")
symbol = st.selectbox("Pilih saham", tickers)
if st.button("Analyze"):
    df = fetch_one(symbol)
    if df is None or df.empty:
        st.warning("Data tidak tersedia / gagal fetch.")
    else:
        st.line_chart(df["Close"])
        st.success(f"✅ Data {symbol} berhasil ditampilkan ({len(df)} hari)")

# ========== Screener Cepat ==========
st.subheader("🚀 Screener Cepat")
if st.button("🚀 Screener Now!"):
    results = []
    for t in tickers[:max_tickers]:
        df = fetch_one(t)
        if df is None or df.empty:
            continue
        score = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
        results.append([t, score])
    if results:
        dfres = pd.DataFrame(results, columns=["Ticker", "Return"])
        dfres = dfres.sort_values("Return", ascending=False).reset_index(drop=True)
        st.dataframe(dfres.style.format({"Return": "{:.2%}"}))
        st.success("✅ Screener selesai — data valid.")
    else:
        st.warning("Tidak ada data valid (cek koneksi / limit).")

st.caption("© IDX Screener — Auto Mode | FastIDX + Yahoo Backup | by Chef 🧠")
