#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np

# =========================
# BASIC CONFIG
# =========================
st.set_page_config(
    page_title="IDX Power Screener v6.9 EXTREME",
    page_icon="🚀",
    layout="wide"
)

# Simple dark + blue theme (bukan hijau)
st.markdown(
    """
    <style>
    .main {
        background-color: #060815;
    }
    .stSidebar {
        background-color: #050712;
    }
    .tag-strong {
        background: linear-gradient(90deg, #2563eb, #22d3ee);
        padding: 2px 8px;
        border-radius: 999px;
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .tag-buy {
        background: rgba(59,130,246,0.15);
        padding: 2px 8px;
        border-radius: 999px;
        color: #bfdbfe;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .tag-hold {
        background: rgba(148,163,184,0.2);
        padding: 2px 8px;
        border-radius: 999px;
        color: #e5e7eb;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .tag-sell {
        background: rgba(248,113,113,0.2);
        padding: 2px 8px;
        border-radius: 999px;
        color: #fecaca;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 IDX Power Screener – EXTREME BUILD v6.9")
st.caption("⚡ 2-Stage Smart Filter • ⚡ Strong Buy Only • ⚡ Anti-Downtrend • ⚡ Scalping 1–3 Hari")

# =========================
# DATA LOADER
# =========================
@st.cache_data(ttl=600)
def load_universe() -> pd.DataFrame:
    """
    Default: coba baca universe.csv (kalau Chef nanti mau pakai file sendiri).
    Kalau gagal → bikin data dummy 799 ticker supaya app TIDAK error.
    Kolom yang dipakai:
    Ticker, Price, Score, Grade, Trend, Signal,
    Entry_Ideal, Entry_Aggressive, TP1, TP2, TP3, CL
    """
    try:
        df = pd.read_csv("universe.csv")
    except Exception:
        # ================ DUMMY UNIVERSE (fallback) ================
        n = 799
        rng = np.random.default_rng(42)
        tickers = [f"T{str(i).zfill(3)}" for i in range(1, n + 1)]
        price = rng.integers(50, 700, size=n)
        score = rng.integers(60, 100, size=n)

        grade = pd.cut(
            score,
            bins=[0, 79, 84, 89, 1000],
            labels=["B", "B+", "A-", "A"],
            include_lowest=True,
        )

        trend_choices = [
            "Strong Uptrend",
            "Uptrend",
            "Sideways",
            "Weak Uptrend",
            "Downtrend",
        ]
        trend = rng.choice(trend_choices, size=n, p=[0.25, 0.35, 0.2, 0.1, 0.1])

        signal = []
        for s, t in zip(score, trend):
            if t == "Downtrend":
                signal.append("Sell")
            elif s >= 90 and "Uptrend" in t:
                signal.append("Strong Buy")
            elif s >= 80:
                signal.append("Buy")
            elif s >= 70:
                signal.append("Hold")
            else:
                signal.append("Sell")

        entry_ideal = price
        entry_aggr = np.maximum(price - 5, 50)
        tp1 = price + (price * 0.03).astype(int)
        tp2 = price + (price * 0.06).astype(int)
        tp3 = price + (price * 0.10).astype(int)
        cl = np.maximum(price - (price * 0.05).astype(int), 50)

        df = pd.DataFrame(
            {
                "Ticker": tickers,
                "Price": price,
                "Score": score,
                "Grade": grade.astype(str),
                "Trend": trend,
                "Signal": signal,
                "Entry_Ideal": entry_ideal,
                "Entry_Aggressive": entry_aggr,
                "TP1": tp1,
                "TP2": tp2,
                "TP3": tp3,
                "CL": cl,
            }
        )

    # safety: pastikan semua kolom ada
    required = [
        "Ticker",
        "Price",
        "Score",
        "Grade",
        "Trend",
        "Signal",
        "Entry_Ideal",
        "Entry_Aggressive",
        "TP1",
        "TP2",
        "TP3",
        "CL",
    ]
    df = df.copy()
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0).astype(int)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).astype(int)

    return df


df_universe = load_universe()

# =========================
# SIDEBAR – MODE & LIMIT
# =========================
st.sidebar.header("⚙️ Screener Settings")

mode = st.sidebar.radio(
    "Mode Trading",
    [
        "⚡ SPEED Trader (1–2 Hari)",
        "🔄 SWING Trader (3–10 Hari)",
        "🌙 BSJP (Beli Sore Jual Pagi)",
        "🌞 BPJS (Beli Pagi Jual Sore)",
        "🟩 VALUE Plays (Murah & Prospek)",
        "🎯 Bandarmology Style",
        "🔍 Single Stock View",
    ],
)

period = st.sidebar.selectbox(
    "Period Data",
    ["3 Bulan", "6 Bulan", "12 Bulan"],
    index=1,
)

stage1_limit = st.sidebar.select_slider(
    "Stage 1 – Top N (Strong Buy / Buy)",
    options=[50, 100, 150, 200],
    value=100,
)

stage2_limit = st.sidebar.select_slider(
    "Stage 2 – Elite Picks dari Stage 1",
    options=[10, 20, 30, 40, 50],
    value=20,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "📌 Style: hold cepat 1–3 hari.\n"
    "Stage 1: buang downtrend & saham tuyul.\n"
    "Stage 2: hanya Strong Buy grade A+/A/A-."
)

# =========================
# MODE BIAS (untuk Score_Adj)
# =========================
def apply_mode_bias(df: pd.DataFrame, mode_name: str) -> pd.DataFrame:
    df = df.copy()
    bias = np.zeros(len(df))

    if "SPEED" in mode_name or "BPJS" in mode_name:
        bias += np.where(df["Trend"].isin(["Strong Uptrend", "Uptrend"]), 5, 0)

    if "SWING" in mode_name or "BSJP" in mode_name:
        bias += np.where(df["Grade"].isin(["A", "A-"]), 3, 0)
        bias += np.where(df["Signal"] == "Strong Buy", 4, 0)

    if "VALUE" in mode_name:
        cheap = df["Price"] <= df["Price"].quantile(0.4)
        bias += np.where(cheap, 4, 0)

    if "Bandarmology" in mode_name:
        bias += np.where(df["Score"] >= 90, 3, 0)

    df["Score_Adj"] = df["Score"] + bias
    return df


# =========================
# STAGE FILTERS
# =========================
def stage1_filter(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = df.copy()
    df = df[df["Signal"].isin(["Strong Buy", "Buy"])]
    df = df[~df["Trend"].str.contains("Downtrend", case=False, na=False)]
    df = df.sort_values(["Score_Adj", "Score"], ascending=False)
    return df.head(top_n)


def stage2_filter(df_stage1: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = df_stage1.copy()
    df = df[
        (df["Signal"] == "Strong Buy")
        & (df["Grade"].isin(["A+", "A", "A-"]) | df["Grade"].isin(["A", "A-"]))
    ]
    if df.empty:
        df = df_stage1.copy()
    df = df.sort_values(["Score_Adj", "Score"], ascending=False)
    return df.head(top_n)


# =========================
# UTIL: PAGINATION & TABLE VIEW
# =========================
def safe_pagination(label: str, total_rows: int, per_page: int = 20) -> int:
    if per_page <= 0:
        per_page = 20
    if total_rows <= 0:
        st.caption(f"{label}: 1 / 1")
        return 1
    num_pages = int(np.ceil(total_rows / per_page))
    if num_pages <= 1:
        st.caption(f"{label}: 1 / 1")
        return 1
    return st.slider(label, 1, num_pages, 1)


def show_trade_table(df: pd.DataFrame, *, title: str, key: str):
    st.markdown(f"### {title}")
    if df.empty:
        st.info("❕ Tidak ada kandidat yang memenuhi kriteria untuk mode & filter saat ini.")
        return

    per_page = st.selectbox(
        "Rows per page",
        options=[20, 40, 60, 80, 100],
        index=1,
        key=f"per_page_{key}",
    )
    page = safe_pagination("Page", len(df), per_page)
    start = (page - 1) * per_page
    end = start + per_page

    cols_show = [
        "Ticker",
        "Price",
        "Score",
        "Grade",
        "Trend",
        "Signal",
        "Entry_Ideal",
        "Entry_Aggressive",
        "TP1",
        "TP2",
        "TP3",
        "CL",
    ]
    cols_show = [c for c in cols_show if c in df.columns]

    st.dataframe(
        df.iloc[start:end][cols_show].style.format(
            {
                "Price": "{:,.0f}",
                "Entry_Ideal": "{:,.0f}",
                "Entry_Aggressive": "{:,.0f}",
                "TP1": "{:,.0f}",
                "TP2": "{:,.0f}",
                "TP3": "{:,.0f}",
                "CL": "{:,.0f}",
            }
        ),
        use_container_width=True,
        height=420,
    )


# =========================
# MAIN – APPLY MODE & RENDER
# =========================
df_mode = apply_mode_bias(df_universe, mode)

# ----- SINGLE STOCK VIEW -----
if "Single Stock" in mode:
    st.subheader("🔍 Single Stock Analyzer")
    col_a, col_b = st.columns([2, 3])
    with col_a:
        ticker_query = st.text_input(
            "Ticker (contoh: CUAN, BUMI, INET)",
            "",
        ).strip().upper()

    if ticker_query:
        d = df_mode[df_mode["Ticker"].str.upper() == ticker_query]
        if d.empty:
            st.warning("Ticker tidak ditemukan di universe (mungkin beda penulisan).")
        else:
            row = d.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Harga", f"{row['Price']:,.0f}")
            c2.metric("Score", int(row["Score"]))
            c3.metric("Grade", str(row["Grade"]))
            c4.metric("Trend", str(row["Trend"]))

            st.markdown(
                f"""
                **Signal:** <span class="tag-strong">{row['Signal']}</span>  
                **Entry Ideal:** `{int(row['Entry_Ideal']):,}`  
                **Entry Agresif:** `{int(row['Entry_Aggressive']):,}`  
                **TP1 / TP2 / TP3:** `{int(row['TP1']):,}` / `{int(row['TP2']):,}` / `{int(row['TP3']):,}`  
                **Cut Loss (CL):** `{int(row['CL']):,}`
                """,
                unsafe_allow_html=True,
            )

            st.markdown("---")

# ----- STAGE 1 -----
st.subheader("🥇 Stage 1 – Top Kandidat (Strong Buy / Buy • NO Downtrend)")
df_stage1 = stage1_filter(df_mode, stage1_limit)
show_trade_table(df_stage1, title="Stage 1 – Hasil Filter", key="stage1")

# ----- STAGE 2 -----
st.subheader("🏆 Stage 2 – Elite Picks (A+/A Strong Buy Only)")
df_stage2 = stage2_filter(df_stage1, stage2_limit)
show_trade_table(df_stage2, title="Stage 2 – Elite Strong Buy", key="stage2")

# ----- DOWNLOAD BUTTONS -----
c_dl1, c_dl2 = st.columns(2)
with c_dl1:
    st.download_button(
        "📥 Download Stage 1 (CSV)",
        data=df_stage1.to_csv(index=False).encode("utf-8"),
        file_name="stage1_candidates.csv",
        mime="text/csv",
        key="dl_stage1",
    )
with c_dl2:
    st.download_button(
        "📥 Download Stage 2 Elite Picks (CSV)",
        data=df_stage2.to_csv(index=False).encode("utf-8"),
        file_name="stage2_elite_picks.csv",
        mime="text/csv",
        key="dl_stage2",
    )

# ----- FOOTNOTE / LOGIC EXPLAIN -----
st.markdown(
    """
    ---
    #### 🧠 Ringkasan Logika Trading Plan
    - **Stage 1**
      - Filter hanya `Signal = Strong Buy / Buy`
      - Buang semua yang `Trend` mengandung kata **Downtrend**
      - Urutkan pakai `Score_Adj` (Score + bias sesuai mode)
    - **Stage 2**
      - Dari Stage 1, ambil yang:
        - `Signal = Strong Buy`
        - `Grade ∈ {A+, A, A-}`
      - Kalau hasil kosong → fallback pakai Stage 1 lagi, urut Score_Adj
    - Mode (Speed / Swing / BSJP / BPJS / Value / Bandarmology)
      - Hanya mempengaruhi **bias** ke `Score_Adj`, jadi ranking menyesuaikan gaya Chef
    - Style rekomendasi: **hold 1–3 hari**, selalu cek lagi:
      - Orderbook real-time
      - Bandarmology / Broker Summary
      - News & corporate action
     """
)