#!/usr/bin/env python3
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="IDX Screener PRO", page_icon="🚀", layout="wide")

# ============ THEME ============
st.markdown("""<style>
.main-header{font-size:2.1rem;font-weight:800;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin:.5rem 0 1rem}
.badge{padding:.45rem .7rem;border-radius:10px;color:#fff;font-weight:700;display:inline-block}
.buy{background:linear-gradient(135deg,#11998e 0%,#38ef7d 100%)}
.sell{background:linear-gradient(135deg,#eb3349 0%,#f45c43 100%)}
.small{font-size:.9rem;opacity:.85}
.tbl thead tr th{position:sticky; top:0; background:#111}
</style>""", unsafe_allow_html=True)

# ============ DATA SOURCES ============
@st.cache_data
def load_stocks() -> list[str]:
    """
    Load daftar ticker dari idx_stocks.json (array ['TLKM.JK', ...] atau {'stocks':[...]}).
    NOTE: kalau file kosong, fallback ke subset populer agar app tetap jalan.
    """
    paths = ["idx_stocks.json", os.path.join(os.path.dirname(__file__), "idx_stocks.json")]
    tickers = []
    for p in paths:
        if os.path.exists(p):
            try:
                data = json.load(open(p,"r"))
                tickers = data if isinstance(data, list) else data.get("stocks", [])
                break
            except Exception:
                pass
    # fallback minimal (biar tetap jalan kalau JSON belum lengkap)
    if not tickers:
        tickers = ["BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK","TLKM.JK","AMMN.JK","ADRO.JK","ASII.JK","BREN.JK","ICBP.JK","INDF.JK","ANTM.JK","MDKA.JK","UNTR.JK","UNVR.JK","BRIS.JK","ARTO.JK","ESSA.JK","FILM.JK","NCKL.JK"]
    # unique + sorted + format .JK
    tickers = list(dict.fromkeys([t for t in tickers if t.upper().endswith(".JK")]))
    return sorted(tickers)

@st.cache_data(ttl=60*60)
def get_daily(ticker: str, period="6mo") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
        if df is None or df.empty or len(df) < 50: return None
        df = df.dropna().copy()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        delta = df["Close"].diff()
        gain = (delta.where(delta>0,0)).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        rs = (gain / loss).replace([pd.NA, pd.NaT], 0).fillna(0)
        df["RSI"] = 100 - (100 / (1 + rs.replace([float("inf"),-float("inf")],0)))
        return df
    except Exception:
        return None

@st.cache_data(ttl=60*30)
def get_intraday(ticker: str, period="30d", interval="15m") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty or len(df) < 30: return None
        df = df.dropna().copy()
        df["EMA9"]  = df["Close"].ewm(span=9,  adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        delta = df["Close"].diff()
        gain = (delta.where(delta>0,0)).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        rs = (gain / loss).replace([pd.NA, pd.NaT], 0).fillna(0)
        df["RSI"] = 100 - (100 / (1 + rs.replace([float("inf"),-float("inf")],0)))
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
        return df
    except Exception:
        return None

# ============ FILTER & SIGNAL ============
def daily_bullish(d1: pd.DataFrame) -> bool:
    if d1 is None or len(d1) < 50: return False
    last = d1.iloc[-1]
    return (last["Close"] > last["EMA21"]) and (last["EMA21"] > last["EMA50"]) and (float(last["RSI"]) > 50)

def score_intraday(d: pd.DataFrame) -> tuple[int, dict]:
    if d is None or len(d) < 30: return 0, {}
    last = d.iloc[-1]
    s, det = 0, {}
    for span, pts in [(9,10),(21,10),(50,10)]:
        cond = last["Close"] > last[f"EMA{span}"]
        s += pts if cond else 0
        det[f"Price>EMA{span}"] = "✅ +%d" % pts if cond else "❌ 0"
    align = last["EMA9"] > last["EMA21"] > last["EMA50"]
    s += 10 if align else 0
    det["EMA Align"] = "✅ +10" if align else "❌ 0"
    rsi = float(last["RSI"])
    if 50 <= rsi <= 70: s+=20; det[f"RSI({rsi:.1f})"]="✅ +20"
    elif 40 <= rsi < 50: s+=10; det[f"RSI({rsi:.1f})"]="⚠️ +10"
    else: det[f"RSI({rsi:.1f})"]="❌ 0"
    macdok = last["MACD"] > last["Signal_Line"]
    s += 20 if macdok else 0
    det["MACD"] = "✅ +20" if macdok else "❌ 0"
    avg20 = d["Volume"].tail(20).mean()
    volok = last["Volume"] > (avg20*1.2 if pd.notna(avg20) else 0)
    s += 10 if volok else 0
    det["Vol"] = "✅ +10" if volok else "❌ 0"
    return int(s), det

def signal_from(score:int) -> tuple[str,str,float]:
    if score>=85: return "🔥 STRONG BUY","BUY",92.0
    if score>=70: return "BUY","BUY",80.0
    if score>=55: return "NEUTRAL","NEUTRAL",50.0
    return "SELL","SELL",30.0

def build_strategy(price: float, buy_like: bool) -> dict|None:
    if not buy_like: return None
    ie = price*0.97; it1=ie*1.10; it2=ie*1.15; isl=ie*0.93
    irr = (it1-ie)/max(ie-isl,1e-6)
    ae = price; at1=ae*1.08; at2=ae*1.12; asl=ae*0.93
    arr = (at1-ae)/max(ae-asl,1e-6)
    return {"ideal":{"entry":ie,"tp1":it1,"tp2":it2,"sl":isl,"rr":irr},
            "aggr":{"entry":ae,"tp1":at1,"tp2":at2,"sl":asl,"rr":arr}}

# ============ ANALYZERS ============
def analyze_single(ticker: str):
    d1 = get_daily(ticker, "6mo")
    d1_ok = daily_bullish(d1)
    d15 = get_intraday(ticker, "30d", "15m")
    if d15 is None: return {"error":"No intraday data"}
    score, det = score_intraday(d15)
    sig_txt, sig_class, prob = signal_from(score)
    buy_like = (sig_class=="BUY") and d1_ok
    price = float(d15["Close"].iloc[-1])
    strat = build_strategy(price, buy_like)
    return {"ticker":ticker,"price":price,"score":score,"sig_txt":sig_txt,"sig_class":sig_class,
            "prob":prob,"buy_like":buy_like,"strat":strat,"det":det,"d15":d15}

def analyze_for_mode(ticker:str, mode:str):
    # Daily filter dulu (Entry Mode A = aman)
    if not daily_bullish(get_daily(ticker,"6mo")): return None
    d15 = get_intraday(ticker,"30d","15m")
    if d15 is None: return None
    score, _ = score_intraday(d15)
    sig_txt, sig_class, prob = signal_from(score)
    if sig_class != "BUY": return None  # BUY-only: buang Neutral/Sell
    price = float(d15["Close"].iloc[-1])
    strat = build_strategy(price, True)
    return {"Ticker":ticker,"Price":price,"Score":score,"Signal":sig_txt,"Prob%":prob,
            "Entry Ideal":strat["ideal"]["entry"],"Entry Aggressive":strat["aggr"]["entry"],
            "TP1":strat["ideal"]["tp1"],"TP2":strat["ideal"]["tp2"],"SL":strat["ideal"]["sl"],
            "RR":strat["ideal"]["rr"]}

# ============ CHART ============
def plot_chart(df: pd.DataFrame, ticker: str, strat: dict|None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    for k,c in [("EMA9","blue"),("EMA21","orange"),("EMA50","red")]:
        if k in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df[k], name=k, line=dict(color=c,width=1)))
    if strat:
        ie=strat["ideal"]["entry"]; it1=strat["ideal"]["tp1"]; it2=strat["ideal"]["tp2"]; isl=strat["ideal"]["sl"]
        fig.add_hline(y=ie, line_dash="dash", line_color="green", annotation_text="Ideal")
        fig.add_hline(y=it1, line_dash="dot",  line_color="lime",  annotation_text="TP1")
        fig.add_hline(y=it2, line_dash="dot",  line_color="lime",  annotation_text="TP2")
        fig.add_hline(y=isl, line_dash="dash", line_color="red",   annotation_text="SL")
    fig.update_layout(title=ticker, height=560, template="plotly_white", xaxis_rangeslider_visible=False)
    return fig

# ============ APP ============
def main():
    st.markdown('<div class="main-header">🚀 IDX SCREENER PRO — Mode B1 (Chart → Detail)</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        mode = st.radio("Mode", ["📈 Manual (Single Stock)", "🔍 Screener — BPJS (Pagi→Sore)", "🔍 Screener — BSJP (Sore→Pagi)"], index=1)
        topN = st.select_slider("Top Results", options=[20,30,50], value=30)
        st.caption("Daily filter: Close>EMA21>EMA50 & RSI>50 • BUY-only")

    stocks = load_stocks()
    if not stocks:
        st.error("❌ idx_stocks.json kosong. Tambahkan daftar ticker IDX (format TICKER.JK).")
        return

    # === MANUAL ===
    if mode.startswith("📈"):
        c1, c2 = st.columns([2,1])
        with c1:
            sel = st.selectbox("Pilih Saham", stocks, index=(stocks.index("TLKM.JK") if "TLKM.JK" in stocks else 0))
        with c2:
            st.write("")
            go_btn = st.button("📊 ANALYZE", type="primary", use_container_width=True)
        if go_btn:
            with st.spinner(f"Analysing {sel} (15m + Daily) …"):
                res = analyze_single(sel)
                if res.get("error"):
                    st.error("❌ Data tidak tersedia.")
                    return
                # B1: CHART di atas
                st.plotly_chart(plot_chart(res["d15"], sel, res["strat"] if res["buy_like"] else None), use_container_width=True)
                # DETAIL di bawah
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Price", f"Rp {res['price']:,.0f}")
                m2.metric("Score", f"{res['score']}/100")
                m3.markdown(f'<span class="badge {"buy" if res["buy_like"] else "sell"}'>{ "BUY" if res["buy_like"] else "SELL" } — {res["sig_txt"]}</span>', unsafe_allow_html=True)
                m4.metric("Prob", f"{res['prob']:.1f}%")
                st.markdown("### 📊 Score Breakdown")
                st.table(pd.DataFrame.from_dict(res["det"], orient="index", columns=["Status"]))

                if res["buy_like"] and res["strat"]:
                    st.markdown("---")
                    cL, cR = st.columns(2)
                    with cL:
                        st.markdown("### 🎯 IDEAL")
                        st.write(f"Entry: **Rp {res['strat']['ideal']['entry']:,.0f}**")
                        st.write(f"TP1 / TP2: **Rp {res['strat']['ideal']['tp1']:,.0f} / Rp {res['strat']['ideal']['tp2']:,.0f}**")
                        st.write(f"SL: **Rp {res['strat']['ideal']['sl']:,.0f}**")
                        st.write(f"R:R **1:{res['strat']['ideal']['rr']:.2f}**")
                    with cR:
                        st.markdown("### ⚡ AGGRESSIVE")
                        st.write(f"Entry: **Rp {res['strat']['aggr']['entry']:,.0f}**")
                        st.write(f"TP1 / TP2: **Rp {res['strat']['aggr']['tp1']:,.0f} / Rp {res['strat']['aggr']['tp2']:,.0f}**")
                        st.write(f"SL: **Rp {res['strat']['aggr']['sl']:,.0f}**")
                        st.write(f"R:R **1:{res['strat']['aggr']['rr']:.2f}**")
                else:
                    st.warning("⚠️ NO BUY — Daily filter atau intraday belum valid.")
        return

    # === SCREENER ===
    st.info(("🌅 **BPJS**: Beli pagi → jual sore" if "BPJS" in mode else "🌙 **BSJP**: Beli sore → jual pagi")
            + " • Daily bullish filter ✅ • Hanya BUY/STRONG BUY ✅")
    btn = st.button("🚀 RUN SCREENER", type="primary", use_container_width=True)
    if not btn: return

    with st.spinner("Scanning IDX (15m + Daily) …"):
        results, progress, status = [], st.progress(0), st.empty()
        start = time.time()
        def task(tk): return analyze_for_mode(tk, "BPJS" if "BPJS" in mode else "BSJP")
        tickers = stocks
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(task, tk): tk for tk in tickers}
            for i, fut in enumerate(as_completed(futures)):
                tk = futures[fut]
                try:
                    r = fut.result()
                    if r: results.append(r)
                except Exception:
                    pass
                progress.progress((i+1)/len(tickers))
                status.write(f"Analysing {tk} — {i+1}/{len(tickers)} • {time.time()-start:.0f}s")

        if not results:
            st.warning("Tidak ada kandidat BUY sesuai filter.")
            return

        df = pd.DataFrame(results).sort_values("Score", ascending=False)
        topN_val = min(len(df), st.session_state.get("topN_val", st.session_state.setdefault("topN_val", 30)))
        df = df.head(topN_val) if topN_val else df.head(30)

        view = df.copy()
        for c in ["Price","Entry Ideal","Entry Aggressive","TP1","TP2","SL"]:
            view[c] = view[c].apply(lambda x: f"Rp {x:,.0f}")
        view["RR"] = view["RR"].apply(lambda x: f"1:{x:.2f}")
        view["Prob%"] = view["Prob%"].apply(lambda x: f"{x:.1f}%")

        st.markdown(f"### 🎯 Top {len(view)} Candidates")
        st.dataframe(view[["Ticker","Price","Score","Signal","Prob%","Entry Ideal","Entry Aggressive","TP1","TP2","SL","RR"]],
                     use_container_width=True, height=520)

        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, file_name=f"idx_top_{len(view)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

if __name__ == "__main__":
    main()
