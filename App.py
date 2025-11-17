#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import requests

st.set_page_config(page_title="IDX Power Screener v5.7", layout="wide")

# ==========================================================
# CONFIG
# ==========================================================
MAX_RESULTS = 100  # limit top 100
PER_PAGE_OPTIONS = [20, 40, 60, 80, 100]

# ==========================================================
# DUMMY FUNCTION — GANTI DENGAN FETCH DATA API MU
# ==========================================================
def load_data():
    df = pd.read_csv("data.csv") if False else pd.DataFrame({
        "Ticker": ["CUAN", "BUMI", "INET", "CDIA", "RATA", "BREN", "BBCA", "WINS", "BACA"],
        "Price": [2230, 230, 535, 1830, 10150, 9800, 8525, 458, 148],
        "Score": [92, 88, 90, 87, 80, 77, 95, 78, 79],
        "Grade": ["A+", "A", "A", "B+", "B", "B", "A+", "B", "B"],
        "Trend": ["Strong Uptrend"] * 9,
        "Signal": ["Strong Buy", "Buy", "Strong Buy", "Hold", "Hold", "Hold", "Buy", "Hold", "Hold"],
        "Volume": [5e8, 4e8, 3e8, 2e8, 1.2e8, 5e7, 1e9, 6e7, 4e7],
        "Value": [1e11, 8e10, 5e10, 4e10, 2e10, 9e9, 1.5e11, 6e9, 5e9]
    })
    return df

df = load_data()

# ==========================================================
# FILTER ANTI-TUYUL
# ==========================================================
df = df[df["Value"] > 5e9]      # >5B value
df = df[df["Volume"] > 5e7]     # >50M volume
df = df.sort_values("Score", ascending=False).head(MAX_RESULTS)

# ==========================================================
# HEADER
# ==========================================================
st.title("🔥 IDX Power Screener v5.7")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.info("⚡ Stage 1 = Top 100 candidates (filter + sort)\n⚡ Stage 2 = Elite Picks (A+/A + Strong Buy detected)")

# ==========================================================
# PAGINATION
# ==========================================================
st.subheader("Stage 1 – Top Candidates")
items_per_page = st.selectbox("Rows per page", PER_PAGE_OPTIONS, index=1)
num_pages = int(np.ceil(len(df) / items_per_page))
page = st.slider("Page", 1, num_pages, 1)

start = (page - 1) * items_per_page
end = start + items_per_page
st.dataframe(df[start:end], height=380)

# ==========================================================
# ELITE PICKS
# ==========================================================
elite = df[(df["Grade"].isin(["A+", "A"])) & (df["Signal"] == "Strong Buy")]

with st.expander("🚀 Stage 2 – Elite Picks (Strong Buy Only!)", expanded=True):
    if elite.empty:
        st.warning("Belum ada saham elite hari ini 📛")
    else:
        st.success(f"🔥 {len(elite)} ticker memenuhi syarat A+/A Strong Buy")
        st.dataframe(elite, height=300)

# ==========================================================
# OPTIONAL FOOTER
# ==========================================================
st.markdown("---")
st.caption("IDX Power Screener v5.7 • Developed with ❤️ by mentor & Mike")
