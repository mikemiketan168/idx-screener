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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

# ================== BASIC CONFIG ==================
st.set_page_config(
    page_title="IDX Power Screener – EXTREME BUILD",
    page_icon="⚡",
    layout="wide"
)

# ============= SESSION STATE =============
if "last_scan_df" not in st.session_state:
    st.session_state.last_scan_df = None
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "last_scan_strategy" not in st.session_state:
    st.session_state.last_scan_strategy = None
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

# ============= COMPLETE IDX TICKERS =============
def load_tickers():
    """Load complete IDX tickers list"""
    complete_tickers = [
        "AALI", "ABBA", "ABDA", "ABMM", "ACES", "ACST", "ADHI", "ADMF", "ADRO", "AGAR",
        "AGII", "AGRO", "AGRS", "AHAP", "AIMS", "AISA", "AKKU", "AKPI", "AKRA", "AKSI",
        "ALII", "ALKA", "ALMI", "ALTO", "AMMN", "AMMS", "AMRT", "ANDI", "ANJT", "ANTM",
        "APIC", "APII", "APLI", "APLN", "ARCI", "ARGO", "ARII", "ARKA", "ARMY", "ARTA",
        "ARTI", "ARTO", "ASBI", "ASDM", "ASGR", "ASHA", "ASII", "ASJT", "ASLC", "ASMI",
        "ASPI", "ASRI", "ASRM", "ASSA", "ATAP", "ATIC", "AUTO", "AVIP", "AWAN", "AYLS",
        "BABP", "BACA", "BAJA", "BAIK", "BALI", "BAPA", "BAPI", "BATA", "BATR", "BAYU",
        "BBCA", "BBHI", "BBKP", "BBLD", "BBMD", "BBNI", "BBRI", "BBRM", "BBSI", "BBTN",
        "BCAP", "BCIC", "BCIP", "BEKS", "BELL", "BELT", "BESS", "BEST", "BFIN", "BGTG",
        "BHIT", "BIKA", "BIKE", "BIMA", "BINA", "BIPI", "BIPP", "BIRD", "BISI", "BJBR",
        "BJTM", "BKDP", "BKSL", "BLTA", "BLTZ", "BLUE", "BMAS", "BMHS", "BMRI", "BMSR",
        "BMTR", "BNBA", "BNBR", "BNGA", "BNII", "BNLI", "BOAT", "BOBA", "BOGA", "BOLT",
        "BOSS", "BPTR", "BRAM", "BREN", "BRMS", "BRNA", "BRPT", "BSDE", "BSIM", "BSSR",
        "BSWD", "BTEK", "BTON", "BTPN", "BTPS", "BUDI", "BUKK", "BUMI", "BUVA", "BVIC",
        "BWPT", "BYAN", "CAKK", "CAMP", "CANI", "CARS", "CASA", "CASH", "CASS", "CBDK",
        "CBMF", "CBPE", "CBUT", "CCSI", "CEKA", "CEKO", "CENT", "CFIN", "CGAS", "CHEM",
        "CHME", "CINT", "CITA", "CITY", "CLAY", "CLEO", "CLPI", "CMNP", "CMNT", "CMPP",
        "CMRY", "CNKO", "CNTB", "CNTX", "COAL", "COCO", "CODE", "COWL", "CPGT", "CPIN",
        "CPRI", "CPRO", "CSAP", "CSIS", "CSMI", "CSRA", "CTBN", "CTRA", "CTRS", "CTTH",
        "DADA", "DAJK", "DAMA", "DANA", "DAYA", "DCII", "DECI", "DEFI", "DEWA", "DFAM",
        "DGIK", "DIGI", "DILD", "DIVA", "DKFT", "DLTA", "DMAS", "DMMX", "DMND", "DNAR",
        "DNET", "DOID", "DPUM", "DRMA", "DSFI", "DSNG", "DSSA", "DUCK", "DVLA", "DWGL",
        "DYAN", "EAST", "ECII", "EDGE", "EKAD", "ELPI", "ELSA", "ELTY", "EMDE", "EMTK",
        "ENAK", "ENRG", "ENVY", "EPAC", "EPMT", "ERAA", "ERAL", "ERIC", "ERTX", "ESIP",
        "ESSA", "ESTA", "EURO", "EXCL", "FAPA", "FAST", "FASW", "FILM", "FIRE", "FISH",
        "FITT", "FMII", "FOLK", "FOOD", "FORZ", "FPNI", "FREN", "FUJI", "GAMA", "GDST",
        "GDYR", "GEMA", "GEMS", "GGRM", "GGRP", "GHON", "GIAA", "GJTL", "GLOB", "GLVA",
        "GMFI", "GMTD", "GOLD", "GOLL", "GOOD", "GOTO", "GPRA", "GRDP", "GRIA", "GRPM",
        "GTBO", "GTRA", "GTSI", "GUNA", "GWSA", "HADE", "HAIS", "HAKA", "HALO", "HAPB",
        "HAPS", "HATM", "HBAT", "HDIT", "HDTX", "HEAL", "HEAR", "HELI", "HERO", "HEXA",
        "HILL", "HKMU", "HOKI", "HOME", "HOPE", "HOTL", "HRME", "HRTA", "IATA", "IBFN",
        "IBOS", "IBST", "ICBP", "ICON", "IDEA", "IDPR", "IGAR", "IIKP", "IKAI", "IKBI",
        "IKAN", "IKLH", "IMAS", "IMJS", "IMPC", "IMPG", "INAF", "INAL", "INCO", "INCR",
        "INDF", "INDR", "INDS", "INDX", "INDY", "INET", "INOV", "INPC", "INPP", "INPS",
        "INRU", "INTA", "INTD", "INTP", "IPAC", "IPCC", "IPCM", "IPOL", "IQCO", "IPTV",
        "IRRA", "ISAT", "ISSP", "ITIC", "ITMA", "ITMG", "JAKA", "JALI", "JAMF", "JAST",
        "JAWA", "JAYA", "JECC", "JGLE", "JIHD", "JKON", "JKSW", "JMAS", "JMTO", "JNPL",
        "JOEL", "JOGE", "JOJO", "JPFA", "JRPT", "JSKY", "JSMR", "JSPT", "JTPE", "KAEF",
        "KARW", "KAYU", "KBAG", "KBLI", "KBLM", "KBLV", "KBRI", "KDSI", "KEEN", "KIAS",
        "KICI", "KIJA", "KINO", "KIOS", "KJEN", "KLIN", "KLBF", "KMDS", "KMTR", "KOBX",
        "KOIN", "KOKA", "KONI", "KOPI", "KOTA", "KPAL", "KPAS", "KPIG", "KRAH", "KRAS",
        "KREN", "KUAS", "LAMB", "LAND", "LAPD", "LAUT", "LCKM", "LEAD", "LIFE", "LINK",
        "LION", "LMAS", "LMPI", "LMSH", "LPCK", "LPGI", "LPIN", "LPKR", "LPLI", "LPPF",
        "LPPS", "LRNA", "LUCY", "LUCK", "MAIN", "MAMI", "MANG", "MAPA", "MAPB", "MAPI",
        "MARK", "MASA", "MAYA", "MBAP", "MBMA", "MBSS", "MBTO", "MCAS", "MCOR", "MDIA",
        "MDKA", "MDKI", "MDLN", "MDRN", "MEDC", "MEGA", "MEJA", "MEKO", "MERC", "MERK",
        "META", "MFIN", "MFMI", "MGNA", "MGRO", "MICE", "MIDI", "MIKA", "MINA", "MITI",
        "MKNT", "MKPI", "MLBI", "MLIA", "MLPL", "MLPT", "MMLP", "MNCN", "MNGA", "MOLI",
        "MPMX", "MPOW", "MPPA", "MRAT", "MREI", "MSKY", "MTDL", "MTFN", "MTLA", "MTOR",
        "MTRA", "MTSM", "MTWI", "MYOH", "MYOR", "MYRX", "MYTX", "NAGA", "NASA", "NASI",
        "NATO", "NELY", "NETV", "NFCX", "NICK", "NICL", "NIKL", "NIKP", "NIPS", "NIRO",
        "NISP", "NOBU", "NRCA", "NUSA", "OASA", "OBMD", "OCAP", "OKAS", "OMRE", "OPMS",
        "ORIN", "PACK", "PADI", "PALM", "PANI", "PANR", "PANS", "PAPA", "PARS", "PAXE",
        "PBID", "PBRX", "PBSA", "PCAR", "PDES", "PEGE", "PGAS", "PGEO", "PGLI", "PGUN",
        "PICO", "PJAA", "PKPK", "PLAS", "PLIN", "PNBN", "PNBS", "PNGO", "PNIN", "PNLF",
        "PNSE", "POLA", "POLI", "POLL", "POLU", "POLY", "POML", "POND", "POOL", "PORT",
        "POSA", "POWR", "PPGL", "PPRE", "PPRO", "PRAS", "PRDA", "PRIM", "PSAB", "PSDN",
        "PSGO", "PSKT", "PSSI", "PTBA", "PTIS", "PTPN", "PTPP", "PTPS", "PTRO", "PTSN",
        "PUDP", "PURA", "PURE", "PURI", "PWON", "PYFA", "PZZA", "RAJA", "RALS", "RANC",
        "RAND", "RBMS", "RDTX", "REAL", "RELI", "RICY", "RIDE", "RIGS", "RISE", "RIMO",
        "RION", "RLIS", "ROCK", "RODA", "ROTI", "RUIS", "RUNS", "SAFE", "SAGA", "SAMP",
        "SAMR", "SATU", "SBAT", "SBMA", "SCBD", "SCCO", "SCMA", "SCNP", "SCPI", "SDMU",
        "SDPC", "SDRA", "SGER", "SGRO", "SHID", "SICO", "SIDO", "SIER", "SILO", "SIMA",
        "SIMP", "SIMZ", "SINI", "SIPD", "SKBM", "SKLT", "SKRN", "SKYB", "SLIS", "SMAR",
        "SMBR", "SMCB", "SMDM", "SMDR", "SMGR", "SMKL", "SMKM", "SMMA", "SMMT", "SMRA",
        "SMRU", "SMSM", "SNLK", "SOCI", "SODA", "SOFA", "SOHO", "SONA", "SOUL", "SOSS",
        "SPMA", "SPTO", "SQMI", "SRAJ", "SRIL", "SRSN", "SRTG", "SSIA", "SSMS", "SSTM",
        "STAR", "STTP", "SUGI", "SULI", "SUPE", "SUPX", "SURE", "SWAT", "TAFA", "TAKA",
        "TALF", "TAMA", "TANK", "TAPG", "TARA", "TAXI", "TAYS", "TBIG", "TBLA", "TBMS",
        "TCID", "TEBE", "TECH", "TELE", "TFAS", "TFCO", "TGKA", "TGRA", "TGUK", "TIFA",
        "TINS", "TIRA", "TIRT", "TISI", "TKIM", "TLKM", "TMAS", "TMPO", "TNCA", "TOBA",
        "TOSK", "TOTL", "TOUR", "TPIA", "TPMA", "TRAM", "TRIL", "TRIM", "TRIN", "TRIS",
        "TRJA", "TRST", "TRUE", "TRUK", "TRUS", "TSPC", "TSTD", "TTMA", "TURI", "UANG",
        "UCID", "UFOE", "UFLY", "ULTJ", "UMIC", "UNIC", "UNIQ", "UNIT", "UNTR", "UNVR",
        "URBN", "UVCR", "VICO", "VINS", "VIVA", "VKTR", "VNTR", "VOKS", "VRNA", "WAPO",
        "WAPU", "WEGE", "WEHA", "WIFI", "WIIM", "WIKA", "WINE", "WINS", "WIRG", "WOOD",
        "WSKT", "WTON", "YELO", "YPAS", "YULE", "ZBRA", "ZINC", "ZONE", "RATU", "CDIA",
        "CUAN", "COIN", "JARR", "GZCO", "NCKL", "CBRE", "AADI", "FUTR", "ZPAY"
    ]
    
    return [f"{ticker}.JK" for ticker in complete_tickers]

# ============= MARKET STATUS =============
def get_market_status():
    """Cek status pasar IDX (Buka/Tutup)"""
    jkt_time = get_jakarta_time()
    current_hour = jkt_time.hour
    current_minute = jkt_time.minute
    
    is_morning_session = (9 <= current_hour < 12) or (current_hour == 12 and current_minute == 0)
    is_afternoon_session = (13 <= current_hour < 16) or (current_hour == 16 and current_minute == 0)
    
    if is_morning_session or is_afternoon_session:
        next_close = "12:00" if current_hour < 12 else "16:00"
        return "🟢 OPEN", f"Pasar buka • Tutup {next_close} WIB"
    else:
        return "🔴 CLOSED", "Pasar tutup • Buka besok 09:00 WIB"

# ============= IHSG WIDGET =============
@st.cache_data(ttl=180)
def fetch_ihsg_data():
    """Fetch IHSG data"""
    try:
        import yfinance as yf
        ihsg = yf.Ticker("^JKSE")
        hist = ihsg.history(period="1d")
        if hist.empty:
            return None

        current = float(hist["Close"].iloc[-1])
        open_price = float(hist["Open"].iloc[-1])
        high = float(hist["High"].iloc[-1])
        low = float(hist["Low"].iloc[-1])
        change = current - open_price
        change_pct = (change / open_price) * 100 if open_price != 0 else 0

        return {
            "price": current,
            "change": change,
            "change_pct": change_pct,
            "high": high,
            "low": low,
            "status": "up" if change >= 0 else "down",
        }
    except Exception:
        return None

def display_ihsg_widget():
    ihsg = fetch_ihsg_data()
    market_status, status_desc = get_market_status()
    
    if not ihsg:
        st.info("📊 IHSG data temporarily unavailable")
        return

    status_emoji = "🟢" if ihsg["status"] == "up" else "🔴"
    status_text = "BULLISH" if ihsg["status"] == "up" else "BEARISH"

    if ihsg["change_pct"] > 1.5:
        condition = "🔥 Strong uptrend - Good for momentum!"
        guidance = "✅ Excellent for SPEED / SWING trades"
    elif ihsg["change_pct"] > 0.5:
        condition = "📈 Moderate uptrend - Good conditions"
        guidance = "✅ Good for all strategies"
    elif ihsg["change_pct"] > -0.5:
        condition = "➡️ Sideways - Mixed conditions"
        guidance = "⚠️ Be selective, gunakan stop ketat"
    elif ihsg["change_pct"] > -1.5:
        condition = "📉 Moderate downtrend - Caution"
        guidance = "⚠️ Fokus VALUE, kurangi SPEED"
    else:
        condition = "🔻 Strong downtrend - High risk"
        guidance = "❌ Lebih baik tunggu atau super selektif"

    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
                    padding: 16px; border-radius: 12px; margin-bottom: 18px;
                    border-left: 5px solid {"#22c55e" if ihsg['status']=="up" else "#ef4444"}'>
          <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div>
              <h3 style='margin:0;color:white;'>📊 MARKET OVERVIEW – IHSG</h3>
              <p style='margin:4px 0;color:#e0e7ff;font-size:0.9em;'>
                Jakarta Composite Index • {market_status}
              </p>
            </div>
            <div style='text-align:right;'>
              <h2 style='margin:0;color:white;'>
                {status_emoji} {ihsg['price']:,.2f}
              </h2>
              <p style='margin:4px 0;color:{"#22c55e" if ihsg['status']=="up" else "#ef4444"};
                        font-size:1.05em;font-weight:bold;'>
                {ihsg['change']:+,.2f} ({ihsg['change_pct']:+.2f}%)
              </p>
            </div>
          </div>
          <div style='margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.2);'>
            <p style='margin:3px 0;color:#e0e7ff;font-size:0.85em;'>
              📊 High: {ihsg['high']:,.2f} | Low: {ihsg['low']:,.2f}
              | Status: <strong>{status_text}</strong>
            </p>
            <p style='margin:3px 0;color:#fbbf24;font-size:0.9em;'> {condition} </p>
            <p style='margin:3px 0;color:#a5b4fc;font-size:0.85em;'> {guidance} </p>
            <p style='margin:5px 0 0 0;color:#94a3b8;font-size:0.75em;'>
              ⏰ Last update: {datetime.now().strftime('%H:%M:%S')} WIB | {status_desc}
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def get_jakarta_time():
    return datetime.now(timezone(timedelta(hours=7)))

# ============= SIMPLE SCORING SYSTEM =============
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, period="3mo"):
    """Fetch stock data"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception:
        return None

def simple_score(df, ticker):
    """Simple scoring function"""
    if df is None or len(df) < 20:
        return None
    
    try:
        last_close = df['Close'].iloc[-1]
        volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].tail(20).mean()
        
        # Basic filters
        if volume < 100000 or last_close < 50:
            return None
            
        # Simple momentum
        if len(df) > 5:
            price_5d_ago = df['Close'].iloc[-6]
            momentum_5d = (last_close / price_5d_ago - 1) * 100
        else:
            momentum_5d = 0
            
        if len(df) > 20:
            price_20d_ago = df['Close'].iloc[-21]
            momentum_20d = (last_close / price_20d_ago - 1) * 100
        else:
            momentum_20d = 0
            
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        # Simple score calculation
        score = 0
        details = {}
        
        # Price momentum
        if momentum_5d > 5:
            score += 30
            details["Momentum"] = f"🟢 Strong +{momentum_5d:.1f}%"
        elif momentum_5d > 2:
            score += 20
            details["Momentum"] = f"🟡 Positive +{momentum_5d:.1f}%"
        elif momentum_5d > 0:
            score += 10
            details["Momentum"] = f"🟠 Neutral +{momentum_5d:.1f}%"
        else:
            details["Momentum"] = f"🔴 Negative {momentum_5d:.1f}%"
            
        # Volume
        if volume_ratio > 2:
            score += 25
            details["Volume"] = f"🟢 High {volume_ratio:.1f}x"
        elif volume_ratio > 1.2:
            score += 20
            details["Volume"] = f"🟡 Good {volume_ratio:.1f}x"
        elif volume_ratio > 0.8:
            score += 15
            details["Volume"] = f"🟠 Normal {volume_ratio:.1f}x"
        else:
            details["Volume"] = f"🔴 Low {volume_ratio:.1f}x"
            
        # Trend (simple)
        if momentum_20d > 10:
            score += 25
            details["Trend"] = f"🟢 Uptrend +{momentum_20d:.1f}%"
        elif momentum_20d > 0:
            score += 15
            details["Trend"] = f"🟡 Sideways +{momentum_20d:.1f}%"
        else:
            details["Trend"] = f"🔴 Downtrend {momentum_20d:.1f}%"
            
        # Price level
        if last_close > 1000:
            score += 20
            details["Price"] = f"🟢 Bluechip Rp {last_close:,.0f}"
        elif last_close > 500:
            score += 15
            details["Price"] = f"🟡 Midcap Rp {last_close:,.0f}"
        else:
            score += 10
            details["Price"] = f"🟠 Smallcap Rp {last_close:,.0f}"
            
        # Grade
        if score >= 80:
            grade, signal = "A+", "Strong Buy"
        elif score >= 70:
            grade, signal = "A", "Buy"
        elif score >= 60:
            grade, signal = "B+", "Hold"
        elif score >= 50:
            grade, signal = "B", "Hold"
        else:
            grade, signal = "C", "Watch"
            
        return {
            "Ticker": ticker.replace(".JK", ""),
            "Price": last_close,
            "Score": score,
            "Grade": grade,
            "Signal": signal,
            "Momentum_5D": momentum_5d,
            "Momentum_20D": momentum_20d,
            "Volume_Ratio": volume_ratio,
            "Details": details
        }
    except Exception:
        return None

def enhanced_scan_universe(tickers, strategy, period):
    """Enhanced scan dengan performance tracking"""
    start_time = time.time()
    results = []
    
    with st.status(f"🔍 Scanning {len(tickers)} stocks...", expanded=True) as status:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_ticker, t, strategy, period): t for t in tickers}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, future in enumerate(as_completed(futures)):
                progress = (i + 1) / len(futures)
                progress_bar.progress(progress)
                
                res = future.result()
                if res:
                    results.append(res)
                
                status_text.text(f"📊 Progress: {i+1}/{len(tickers)} • Found: {len(results)} candidates")
        
        status.update(label=f"✅ Scan complete! Found {len(results)} candidates", state="complete")
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    
    # Update session state
    st.session_state.last_scan_df = df
    st.session_state.last_scan_time = datetime.now()
    st.session_state.last_scan_strategy = strategy
    st.session_state.scan_count += 1
    
    return df

def process_ticker(ticker, strategy, period):
    """Process single ticker"""
    try:
        df = fetch_data(ticker, period)
        result = simple_score(df, ticker)
        return result
    except Exception:
        return None

def display_paginated_dataframe(df, rows_per_page, key_suffix=""):
    """Fixed pagination system"""
    if df.empty:
        st.info("No data to display")
        return
    
    total_rows = len(df)
    num_pages = max((total_rows - 1) // rows_per_page + 1, 1)
    
    # Gunakan selectbox sebagai ganti slider
    if num_pages > 1:
        page_options = list(range(1, num_pages + 1))
        page = st.selectbox(f"Page {key_suffix}", page_options, key=f"page_{key_suffix}")
    else:
        page = 1
    
    start_idx = (page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    
    st.dataframe(
        df.iloc[start_idx:end_idx][
            ["Ticker", "Price", "Score", "Grade", "Signal", "Momentum_5D", "Volume_Ratio"]
        ],
        use_container_width=True,
        height=400,
    )
    
    st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_rows} results")

def manage_stage_filtering(df_all, stage1_limit, stage2_limit):
    """Fixed stage filtering system"""
    if df_all.empty:
        return None, None
    
    # Stage 1: Basic filtering
    stage1_df = df_all[df_all["Score"] >= 60].head(stage1_limit)
    
    # Stage 2: Elite picks dari Stage 1
    stage2_df = stage1_df[stage1_df["Score"] >= 70].head(stage2_limit)
    
    # Display info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Scanned", len(df_all))
    with col2:
        st.metric("Stage 1 Candidates", len(stage1_df))
    with col3:
        st.metric("Stage 2 Elite", len(stage2_df))
    
    return stage1_df, stage2_df

def main():
    st.title("⚡ IDX Power Screener – COMPLETE WITH BUTTONS")
    st.caption("799 Saham Lengkap • Tombol Jelas • Hasil Akurat")

    display_ihsg_widget()
    tickers = load_tickers()

    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.info(f"📊 Total stocks loaded: **{len(tickers)}**")

        jkt = get_jakarta_time()
        st.caption(f"🕒 Jakarta time: {jkt.strftime('%d %b %Y • %H:%M WIB')}")

        st.markdown("---")

        menu = st.radio("📋 Menu Utama", [
            "🔎 Screen ALL IDX", "🔍 Single Stock", "🌙 BSJP Strategy",
            "⚡ BPJS Strategy", "💎 Value Strategy", "🔮 Bandarmology"
        ])

        st.markdown("---")

        if menu != "🔍 Single Stock":
            period = st.selectbox("Period data", ["1mo", "3mo", "6mo"], index=1)

            st.markdown("### 🎯 Stage Filter")
            stage1_limit = st.selectbox("Stage 1 – Top Kandidat", options=[50, 100, 150, 200], index=1)
            stage2_limit = st.selectbox("Stage 2 – Elite Picks", options=[10, 20, 30, 40, 50], index=2)

            st.markdown("### 📄 Tabel Tampilan")
            rows_per_page = st.selectbox("Rows per page", options=[20, 40, 60, 80, 100], index=1)

        st.markdown("---")
        st.caption("v8.0 – Complete with Clear Buttons")

    # ============= 🔎 SCREEN ALL IDX =============
    if menu == "🔎 Screen ALL IDX":
        st.markdown("## 🔎 Screen ALL IDX – Full Universe")
        
        st.markdown("### 🚀 **TEKAN TOMBOL INI UNTUK JALANKAN SCREENER:**")
        
        # BIG RED BUTTON - VERY CLEAR!
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            scan_button = st.button(
                "🔥 JALANKAN SCAN SEMUA SAHAM! 🔥", 
                type="primary", 
                use_container_width=True,
                key="main_scan_button"
            )
        
        if scan_button:
            with st.spinner(f"Scanning {len(tickers)} saham..."):
                df_all = enhanced_scan_universe(tickers, "General", period)

            if df_all.empty:
                st.error("❌ Tidak ada saham yang lolos filter.")
            else:
                stage1_df, stage2_df = manage_stage_filtering(df_all, stage1_limit, stage2_limit)
                
                if stage1_df is not None:
                    st.markdown("### 🥇 Stage 1 – Top Kandidat")
                    display_paginated_dataframe(stage1_df, rows_per_page, "stage1")
                
                if stage2_df is not None and not stage2_df.empty:
                    st.markdown("### 🏆 Stage 2 – Elite Picks")
                    st.dataframe(stage2_df, use_container_width=True, height=300)
                    
                    # Download button
                    csv = stage2_df.to_csv(index=False)
                    st.download_button(
                        "💾 Download Elite Picks (CSV)",
                        data=csv,
                        file_name=f"IDX_Elite_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

    # ============= 🔍 SINGLE STOCK =============
    elif menu == "🔍 Single Stock":
        st.markdown("## 🔍 Single Stock Analysis")
        
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected = st.selectbox("Pilih saham", [t.replace(".JK", "") for t in tickers])
        with col_sel2:
            period = st.selectbox("Period data", ["1mo", "3mo", "6mo"], index=1)
        
        st.markdown("### 🚀 **TEKAN TOMBOL INI UNTUK ANALISA:**")
        
        # ANALYZE BUTTON
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            analyze_button = st.button(
                "🔍 ANALYZE SAHAM INI!", 
                type="primary", 
                use_container_width=True,
                key="analyze_button"
            )
        
        if analyze_button:
            ticker_full = f"{selected}.JK"
            with st.spinner(f"Menganalisa {selected}..."):
                df = fetch_data(ticker_full, period)
                result = simple_score(df, ticker_full)

            if result is None:
                st.error("❌ Gagal menganalisa saham ini.")
            else:
                st.markdown(f"## 💎 {result['Ticker']} – {result['Signal']} ({result['Grade']})")
                
                colm1, colm2, colm3, colm4 = st.columns(4)
                colm1.metric("Price", f"Rp {result['Price']:,.0f}")
                colm2.metric("Score", f"{result['Score']}/100")
                colm3.metric("Grade", result["Grade"])
                colm4.metric("Signal", result["Signal"])
                
                st.markdown("### 📊 Technical Details")
                for k, v in result["Details"].items():
                    st.write(f"**{k}**: {v}")

    # ============= STRATEGY MENUS =============
    elif menu == "🌙 BSJP Strategy":
        st.markdown("## 🌙 BSJP Strategy")
        st.info("Beli Sore Jual Pagi - Overnight trading")
        
        # BSJP BUTTON
        if st.button("🚀 SCAN BSJP NOW!", type="primary", use_container_width=True):
            with st.spinner("Scanning untuk BSJP..."):
                df_all = enhanced_scan_universe(tickers, "BSJP", period)
                if not df_all.empty:
                    stage1_df = df_all.head(stage1_limit)
                    display_paginated_dataframe(stage1_df, rows_per_page, "bsjp")

    elif menu == "⚡ BPJS Strategy":
        st.markdown("## ⚡ BPJS Strategy") 
        st.info("Beli Pagi Jual Sore - Day trading")
        
        # BPJS BUTTON
        if st.button("🚀 SCAN BPJS NOW!", type="primary", use_container_width=True):
            with st.spinner("Scanning untuk BPJS..."):
                df_all = enhanced_scan_universe(tickers, "BPJS", period)
                if not df_all.empty:
                    stage1_df = df_all.head(stage1_limit)
                    display_paginated_dataframe(stage1_df, rows_per_page, "bpjs")

    elif menu == "💎 Value Strategy":
        st.markdown("## 💎 Value Strategy")
        st.info("Saham fundamental kuat & murah")
        
        # VALUE BUTTON
        if st.button("🚀 SCAN VALUE NOW!", type="primary", use_container_width=True):
            with st.spinner("Scanning untuk Value..."):
                df_all = enhanced_scan_universe(tickers, "Value", period)
                if not df_all.empty:
                    stage1_df = df_all.head(stage1_limit)
                    display_paginated_dataframe(stage1_df, rows_per_page, "value")

    elif menu == "🔮 Bandarmology":
        st.markdown("## 🔮 Bandarmology")
        st.info("Deteksi pergerakan uang besar")
        
        # BANDAR BUTTON
        if st.button("🚀 SCAN BANDARMOLOGY NOW!", type="primary", use_container_width=True):
            with st.spinner("Scanning untuk Bandarmology..."):
                df_all = enhanced_scan_universe(tickers, "Bandar", period)
                if not df_all.empty:
                    stage1_df = df_all.head(stage1_limit)
                    display_paginated_dataframe(stage1_df, rows_per_page, "bandar")

    st.markdown("---")
    st.caption("⚡ IDX Power Screener v8.0 • Tombol jelas di setiap menu!")

if __name__ == "__main__":
    main()
