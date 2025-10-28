#!/usr/bin/env python3
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

st.set_page_config(page_title="IDX Screener LIVE", page_icon="📈", layout="wide")

# ====== UI THEME ======
st.markdown("""
<style>
.main-title{font-size:2rem;font-weight:800;margin-bottom:.25rem}
.badge{display:inline-block;padding:.25rem .5rem;border-radius:.5rem;color:#fff;font-weight:700}
.buy{background:#16a34a}.sell{background:#dc2626}.neutral{background:#f59e0b}
.card{padding:.75rem;border:1px solid #e5e7eb;border-radius:.75rem;background:#fff}
.small{font-size:.85rem;color:#6b7280}
</style>
""", unsafe_allow_html=True)

# ====== OPTIONS ======
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    live = st.toggle("📡 Live (1m)", value=True, help="Gunakan candle 1 menit saat market buka")
    auto = st.toggle("🔁 Auto Refresh", value=True)
    interval_sec = st.slider("Interval refresh (detik)", 10, 120, 30)
    min_score = st.slider("Min Score", 0, 100, 60)
    max_stocks = st.slider("Max Stocks di-scan", 20, 500, 150)
    mode = st.radio("Mode", ["🔍 Screener", "📈 Single Stock"], index=0)
    st.divider()
    st.caption("Tips: saat market buka, pakai Live + Auto Refresh 30–60 detik.")

# ====== AUT0 REFRESH ======
if auto:
    # st_autorefresh tanpa library eksternal
    st.session_state._tick = st.session_state.get("_tick", 0) + 1
    st.experimental_set_query_params(t=st.session_state._tick)
    st.caption(f"⏱️ Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# ====== LOAD TICKER LIST ======
# Pakai idx_stocks.json jika ada; kalau tidak, fallback ke list likuid inti
def load_tickers():
    import json, os
    p = "idx_stocks.json"
    if os.path.exists(p):
        try:
            data = json.load(open(p))
            if isinstance(data, list):
                return sorted(list(dict.fromkeys(data)))
            return sorted(list(dict.fromkeys(data.get("stocks", []))))
        except Exception:
            pass
    # fallback likuid inti (isi cepat—bisa kamu lengkapi nanti)
    return sorted(list(dict.fromkeys("""
BBCA.JK, BBRI.JK, BBNI.JK, BMRI.JK, BRIS.JK, TLKM.JK, TOWR.JK, ICBP.JK, INDF.JK,
ASII.JK, AUTO.JK, ERAA.JK, ACES.JK, MAPI.JK, AMRT.JK, MDKA.JK, INCO.JK, ANTM.JK,
ADRO.JK, ITMG.JK, MEDC.JK, PGAS.JK, PTBA.JK, ARTO.JK, BUKA.JK, GOTO.JK, TPIA.JK,
CPIN.JK, JPFA.JK, SMGR.JK, INTP.JK, KLBF.JK, EXCL.JK, ISAT.JK, MTEL.JK, UNVR.JK,
UNTR.JK, TKIM.JK, INKP.JK, MDLN.JK, CTRA.JK, PWON.JK, BSDE.JK, SCMA.JK, MORA.JK,
PGEO.JK, ADMR.JK, CUAN.JK, PTRO.JK, CDIA.JK, KBLV.JK
""".replace("\n","").replace(" ","").split(","))))

TICKERS = load_tickers()

# ====== DATA FETCH ======
def fetch_df(ticker: str, live_mode=True):
    try:
        if live_mode:
            # 2 hari 1m biar indikator jalan
            df = yf.download(ticker, period="2d", interval="1m", progress=False)
        else:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        df = df.dropna().copy()
        # standar kolom
        for c in ["Open","High","Low","Close","Volume"]:
            if c not in df.columns: return None

        # indikator minimum
        df["EMA9"]  = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

        delta = df["Close"].diff()
        gain = np.where(delta>0, delta, 0.0)
        loss = np.where(delta<0, -delta, 0.0)
        roll = 14
        avg_gain = pd.Series(gain, index=df.index).rolling(roll).mean()
        avg_loss = pd.Series(loss, index=df.index).rolling(roll).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        df["RSI"] = 100 - (100/(1+rs))
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        return df
    except Exception:
        return None

# ====== SCORING ======
def score_row(df: pd.DataFrame):
    if df is None or len(df) < 30: return 0, "SELL", 20
    last = df.iloc[-1]
    s = 0

    # trend / MA structure
    s += 10 if last.Close > last.EMA9 else 0
    s += 10 if last.Close > last.EMA21 else 0
    s += 10 if last.Close > last.EMA50 else 0
    s += 10 if (last.EMA9 > last.EMA21 > last.EMA50) else 0

    # momentum
    if not np.isnan(last.RSI):
        s += 20 if 40 <= last.RSI <= 70 else (10 if (30 < last.RSI < 40 or 70 < last.RSI < 80) else 0)

    # macd
    s += 20 if last.MACD > last.Signal else 0

    # volume boost (gunakan 1d jika 1m volume tidak tersedia stabil)
    try:
        v20 = df.Volume.tail(20).mean()
        s += 10 if last.Volume > v20 * 1.2 else 0
    except Exception:
        pass

    if s >= 70:  sig, prob = "STRONG BUY", min(95, 50 + (s-70))
    elif s >= 60: sig, prob = "BUY", 50 + (s-60)*2
    elif s >= 40: sig, prob = "NEUTRAL", 50
    else:         sig, prob = "SELL", max(10, 50-(40-s))
    return int(s), sig, int(prob)

# ====== ENTRY/EXIT PLAN ======
def plan_from_price(price: float, signal: str):
    if price is None or price <= 0:
        return dict(ideal=None, aggr=None)
    if "BUY" not in signal:
        return dict(ideal=None, aggr=None, note="⚠️ NO BUY")

    # IDEAL: tunggu diskon -3%, TP +10/15, SL -7
    ie = round(price*0.97, 2)
    it1 = round(ie*1.10, 2)
    it2 = round(ie*1.15, 2)
    isl = round(ie*0.93, 2)
    irr = round((it1 - ie) / max(ie - isl, 1e-9), 2)

    # AGGRESSIVE: harga sekarang, TP +8/12, SL -7
    ae = round(price, 2)
    at1 = round(ae*1.08, 2)
    at2 = round(ae*1.12, 2)
    asl = round(ae*0.93, 2)
    arr = round((at1 - ae) / max(ae - asl, 1e-9), 2)

    return dict(
        ideal=dict(entry=ie, tp1=it1, tp2=it2, sl=isl, rr=irr),
        aggr =dict(entry=ae, tp1=at1, tp2=at2, sl=asl, rr=arr),
        note ="🎯 Wait -3% (ideal)  /  ⚡ Buy NOW (aggr)"
    )

# ====== SINGLE STOCK ======
def single_view():
    tick = st.selectbox("Pilih saham", TICKERS, index=min(0, len(TICKERS)-1))
    df = fetch_df(tick, live_mode=live)
    if df is None:
        st.error("Data kosong.")
        return
    last = df.iloc[-1]
    score, sig, prob = score_row(df)
    plan = plan_from_price(float(last.Close), sig)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Price", f"{last.Close:,.2f}")
    with c2: st.metric("Score", f"{score}/100")
    with c3:
        color = "buy" if "BUY" in sig else ("neutral" if sig=="NEUTRAL" else "sell")
        st.markdown(f'<span class="badge {color}">{sig}</span>', unsafe_allow_html=True)
    with c4: st.metric("Prob", f"{prob}%")

    st.divider()
    st.markdown("### 🎯 Trading Plan")
    if "BUY" in sig:
        t = pd.DataFrame([
            ["🎯 IDEAL", plan["ideal"]["entry"], plan["ideal"]["tp1"], plan["ideal"]["tp2"], plan["ideal"]["sl"], plan["ideal"]["rr"]],
            ["⚡ AGGRESSIVE", plan["aggr"]["entry"], plan["aggr"]["tp1"], plan["aggr"]["tp2"], plan["aggr"]["sl"], plan["aggr"]["rr"]],
        ], columns=["Strategy","Entry","TP1","TP2","SL","R:R"])
        st.dataframe(t, use_container_width=True)
        st.caption(plan["note"])
    else:
        st.warning("⚠️ NO BUY — tunggu setup rapi.")

# ====== SCREENER ======
def analyze_one(tick):
    df = fetch_df(tick, live_mode=live)
    if df is None or len(df)<30: return None
    s, sig, p = score_row(df)
    last = df.iloc[-1].Close
    plan = plan_from_price(float(last), sig)
    row = dict(Ticker=tick, Price=round(float(last),2), Score=s, Signal=sig, Prob=p)
    if "BUY" in sig:
        row.update({
            "Entry_Ideal": plan["ideal"]["entry"],
            "TP1_Ideal":   plan["ideal"]["tp1"],
            "TP2_Ideal":   plan["ideal"]["tp2"],
            "SL_Ideal":    plan["ideal"]["sl"],
            "RR_Ideal":    plan["ideal"]["rr"],
            "Entry_Aggr":  plan["aggr"]["entry"],
            "TP1_Aggr":    plan["aggr"]["tp1"],
            "TP2_Aggr":    plan["aggr"]["tp2"],
            "SL_Aggr":     plan["aggr"]["sl"],
            "RR_Aggr":     plan["aggr"]["rr"],
        })
    return row

def screener_view():
    st.markdown('<div class="main-title">🚀 IDX Screener LIVE</div>', unsafe_allow_html=True)
    st.caption(f"Total tickers: {len(TICKERS)}")
    scan_list = TICKERS[:max_stocks]
    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(analyze_one, t): t for t in scan_list}
        pb = st.progress(0.0)
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r: out.append(r)
            pb.progress((i+1)/len(scan_list))
    if not out:
        st.warning("Tidak ada data (coba matikan Live, atau kurangi Max Stocks).")
        return
    df = pd.DataFrame(out)
    df = df.sort_values(["Score","Prob"], ascending=[False, False])
    df = df[df["Score"]>=min_score]
    st.dataframe(df, use_container_width=True, height=600)
    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name=f"screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

# ====== MAIN ======
if mode == "🔍 Screener":
    screener_view()
else:
    single_view()
