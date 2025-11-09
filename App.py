#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="IDX Power Screener v4.0", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom
st.markdown("""
<style>
    .big-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1e40af;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .signal-box {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        font-weight: 700;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .strong-buy {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    .buy {
        background: linear-gradient(135deg, #34d399, #10b981);
        color: white;
    }
    .hold {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        color: white;
    }
    .cl {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
    }
    .stage-box {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============= KONFIGURASI =============
class Config:
    MAX_RETRIES = 3
    TIMEOUT = 20
    WORKERS = 8
    CACHE_TTL = 300

def init_db():
    conn = sqlite3.connect('screener.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scan_results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, ticker TEXT, 
                  strategy TEXT, score INTEGER, signal TEXT, price REAL,
                  entry_ideal REAL, entry_agresif REAL, tp1 REAL, tp2 REAL, 
                  cut_loss REAL, trend TEXT, volume_ratio REAL, rsi REAL)''')
    conn.commit()
    conn.close()

# ============= UTILITIES =============
def get_jakarta_time():
    jkt_tz = timezone(timedelta(hours=7))
    return datetime.now(jkt_tz)

def check_market_hours():
    jkt_time = get_jakarta_time()
    hour = jkt_time.hour
    return {
        'bpjs_time': 9 <= hour < 10,
        'bsjp_time': 14 <= hour < 16,
        'market_open': 9 <= hour < 16 and jkt_time.weekday() < 5
    }

# ============= DATA FETCHING =============
@st.cache_data(ttl=Config.CACHE_TTL)
def fetch_yahoo_data(ticker, period="3mo"):
    try:
        end = int(datetime.now().timestamp())
        period_sec = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}.get(period, 90) * 86400
        start = end - period_sec
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        response = requests.get(
            url,
            params={"period1": start, "period2": end, "interval": "1d"},
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=Config.TIMEOUT
        )
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        result = data['chart']['result'][0]
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': quote['open'], 'High': quote['high'], 'Low': quote['low'],
            'Close': quote['close'], 'Volume': quote['volume']
        }, index=pd.to_datetime(result['timestamp'], unit='s'))
        
        df = df.dropna()
        if len(df) < 20:
            return None
            
        # Calculate indicators
        df['EMA9'] = df['Close'].ewm(span=9).mean()
        df['EMA21'] = df['Close'].ewm(span=21).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # Volume
        df['Volume_SMA20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']
        
        # Momentum
        df['Momentum_5D'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        
        return df
        
    except Exception as e:
        return None

def fetch_with_retry(ticker, period="3mo"):
    for attempt in range(Config.MAX_RETRIES):
        df = fetch_yahoo_data(ticker, period)
        if df is not None:
            return df
        time.sleep(2 ** attempt)
    return None

# ============= SCORING SYSTEMS =============
def calculate_trend_strength(df):
    current = df.iloc[-1]
    trend_score = 0
    trend_info = []
    
    # EMA Alignment
    if current['Close'] > current['EMA9'] > current['EMA21'] > current['EMA50']:
        trend_score += 40
        trend_info.append("EMA Bullish Alignment")
    elif current['Close'] > current['EMA9'] > current['EMA21']:
        trend_score += 25
        trend_info.append("Short-term Bullish")
    elif current['Close'] > current['EMA50']:
        trend_score += 15
        trend_info.append("Above EMA50")
    else:
        trend_score -= 10
        trend_info.append("Downtrend")
    
    # Price vs SMA20
    if current['Close'] > current['SMA20']:
        trend_score += 10
        trend_info.append("Above SMA20")
    
    # Momentum
    if current['Momentum_5D'] > 2:
        trend_score += 15
        trend_info.append("Positive Momentum")
    
    return trend_score, " | ".join(trend_info)

def score_full_screener(df):
    try:
        trend_score, trend_info = calculate_trend_strength(df)
        current = df.iloc[-1]
        
        total_score = trend_score
        
        # Volume scoring
        if current['Volume_Ratio'] > 2.0:
            total_score += 25
        elif current['Volume_Ratio'] > 1.5:
            total_score += 15
        elif current['Volume_Ratio'] > 1.0:
            total_score += 5
            
        # RSI scoring
        if 40 <= current['RSI'] <= 60:
            total_score += 20
        elif 30 <= current['RSI'] <= 70:
            total_score += 10
            
        # Price position
        if current['Close'] > current['EMA9']:
            total_score += 10
            
        return max(0, min(100, total_score)), trend_info
        
    except:
        return 0, "Error in scoring"

# ============= SIGNAL & LEVELS CALCULATION =============
def calculate_trading_levels(price, score, trend_info):
    if score >= 80:
        signal = "STRONG BUY"
        signal_class = "strong-buy"
        entry_ideal = price * 0.97
        entry_agresif = price
        tp1 = price * 1.08
        tp2 = price * 1.15
        cut_loss = price * 0.92
        
    elif score >= 65:
        signal = "BUY"
        signal_class = "buy"
        entry_ideal = price * 0.98
        entry_agresif = price
        tp1 = price * 1.06
        tp2 = price * 1.12
        cut_loss = price * 0.94
        
    elif score >= 50:
        signal = "HOLD"
        signal_class = "hold"
        entry_ideal = price * 0.95
        entry_agresif = None
        tp1 = price * 1.05
        tp2 = price * 1.10
        cut_loss = price * 0.90
        
    else:
        signal = "CL"
        signal_class = "cl"
        entry_ideal = None
        entry_agresif = None
        tp1 = None
        tp2 = None
        cut_loss = None
    
    # Determine trend
    if "Bullish" in trend_info:
        trend = "🟢 UPTREND"
    elif "Downtrend" in trend_info:
        trend = "🔴 DOWNTREND"
    else:
        trend = "🟡 SIDEWAYS"
    
    return {
        'signal': signal,
        'signal_class': signal_class,
        'entry_ideal': round(entry_ideal, 2) if entry_ideal else None,
        'entry_agresif': round(entry_agresif, 2) if entry_agresif else None,
        'tp1': round(tp1, 2) if tp1 else None,
        'tp2': round(tp2, 2) if tp2 else None,
        'cut_loss': round(cut_loss, 2) if cut_loss else None,
        'trend': trend
    }

# ============= LOAD TICKERS =============
def load_all_tickers():
    """Load 800+ Indonesian stocks"""
    try:
        with open("idx_stocks.json", "r") as f:
            data = json.load(f)
        tickers = data.get("tickers", [])
        return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except:
        # Fallback - 800+ Indonesian stocks
        base_stocks = [
            "BBCA", "BBRI", "BMRI", "BBNI", "BBTN", "BRIS", "BJBR", "BJTM", "BACA", "BAJA",
            "TLKM", "EXCL", "FREN", "ISAT", "TELK", "TKIM", "BTEL", "CITA", "DNET", "EDGE",
            "ASII", "AUTO", "BRPT", "GJTL", "HMSP", "ICBP", "INDF", "JPFA", "KAEF", "KLBF",
            "MBSS", "MLBI", "MYOR", "ROTI", "SCMA", "STTP", "ULTJ", "ADRO", "AKRA", "ANTM",
            "BUMI", "BYAN", "DOID", "ELSA", "EMTK", "ENRG", "HRUM", "ITMG", "MDKA", "PGAS",
            "PTBA", "PTPP", "SMBR", "SSIA", "TINS", "TOWR", "AKSI", "ARTO", "ASRM", "BOLT",
            "BRAM", "CASS", "CLEO", "DMMX", "FAST", "GDST", "HOKI", "ICON", "IGAR", "IKAI",
            "IMAS", "INKP", "IPCC", "JAST", "JAYA", "JSMR", "KBLI", "KBLM", "KIJA", "LION",
            "LPCK", "MAPI", "MCAS", "MIKA", "MTDL", "PANI", "PBSA", "PCAR", "POLY", "POWR",
            "PRIM", "PSDN", "PSSI", "RALS", "RICY", "SAME", "SAPX", "SDMU", "SEMA", "SIDO",
            "SILO", "SIMP", "SMMA", "SMSM", "SOCL", "SONA", "SOSS", "SULI", "TARA", "TCID",
            "TCPI", "TFCO", "TGRA", "TOTO", "TOYS", "TRST", "TSPC", "UNIC", "UNTR", "WEGE",
            "WICO", "WIIM", "WSBP", "WTON", "YELO", "ZBRA", "ZYRX", "ACES", "ADES", "ADMG",
            "AGAR", "AGII", "AGRO", "AIMS", "AISA", "AKPI", "ALDO", "ALKA", "ALMI", "AMAG",
            "AMFG", "AMIN", "AMOR", "ANAF", "ANJT", "APEX", "APIC", "APII", "APLI", "APLN",
            "ARNA", "ARTA", "ASBI", "ASDM", "ASGR", "ASII", "ASJT", "ASMI", "ASPI", "ASRI",
            "ASRM", "ASSA", "ATAP", "ATIC", "AUTO", "AVIA", "AYLS", "BABP", "BACA", "BAJA",
            "BALI", "BANK", "BAPA", "BAPI", "BATA", "BAYU", "BBCA", "BBHI", "BBKP", "BBLD",
            "BBMD", "BBNI", "BBRI", "BBRM", "BBSI", "BBSS", "BBTN", "BBYB", "BCAP", "BCIC",
            "BCIP", "BDMN", "BEEF", "BEKS", "BELL", "BEST", "BFIN", "BGTG", "BHAT", "BHIT",
            "BIKA", "BIMA", "BINA", "BIPI", "BIPP", "BIRD", "BISI", "BJBR", "BJTM", "BKDP",
            "BKSL", "BKSW", "BLTA", "BLTZ", "BLUE", "BMAS", "BMRI", "BMSR", "BMTR", "BNBA",
            "BNBR", "BNGA", "BNII", "BNLI", "BOBA", "BOLT", "BORN", "BOSS", "BPFI", "BPII",
            "BPTR", "BRAM", "BRIS", "BRMS", "BRNA", "BRPT", "BSDE", "BSIM", "BSML", "BSSR",
            "BTEK", "BTEL", "BTON", "BUDI", "BUKA", "BUKK", "BULL", "BUMI", "BUVA", "BVIC",
            "BWPT", "BYAN", "CAKK", "CAMP", "CANI", "CARE", "CARS", "CASA", "CASH", "CASS",
            "CBMF", "CCSI", "CEKA", "CENT", "CFIN", "CINT", "CITA", "CITY", "CLAY", "CLEO",
            "CLPI", "CMNP", "CMPP", "CMRY", "CNKO", "CNTX", "COCO", "COWL", "CPRI", "CPRO",
            "CSAP", "CSIS", "CSMI", "CSRA", "CTBN", "CTRA", "CTTH", "DART", "DAYA", "DCII",
            "DEAL", "DEFI", "DEWA", "DFAM", "DGIK", "DIGI", "DILD", "DIVA", "DKFT", "DLTA",
            "DMAS", "DMMX", "DMND", "DNET", "DOID", "DPNS", "DPUM", "DSFI", "DSNG", "DSSA",
            "DUCK", "DUTI", "DVLA", "DWGL", "DYAN", "EAST", "ECII", "EDGE", "EKAD", "ELSA",
            "ELTY", "EMDE", "EMTK", "ENRG", "ENVY", "ENZO", "EPAC", "ERAA", "ERTX", "ESIP",
            "ESSA", "ESTI", "ETWA", "EXCL", "FAST", "FASW", "FILM", "FIRE", "FISH", "FITT",
            "FLMC", "FMII", "FOOD", "FORU", "FORZ", "FPNI", "FREN", "FUJI", "GAMA", "GDST",
            "GDYR", "GEMA", "GEMS", "GGRM", "GGRP", "GIAA", "GJTL", "GLOB", "GLVA", "GMFI",
            "GOLD", "GOLL", "GOOD", "GPRA", "GSMF", "GTBO", "GTSI", "GWSA", "GZCO", "HADE",
            "HDFA", "HDIT", "HEAL", "HELI", "HERO", "HEXA", "HITS", "HKMU", "HMSP", "HOKI",
            "HOME", "HOMI", "HOPE", "HOTL", "HRTA", "HRUM", "IATA", "IBFN", "IBST", "ICBP",
            "ICON", "IDPR", "IGAR", "IIKP", "IKAI", "IKAN", "IKBI", "IMAS", "IMPC", "INAI",
            "INCF", "INCI", "INCO", "INDF", "INDO", "INDR", "INDS", "INDX", "INDY", "INKP",
            "INOV", "INPC", "INPP", "INPS", "INRU", "INTA", "INTD", "INTP", "IPCC", "IPCM",
            "IPOL", "IPCC", "ISAT", "ISSP", "ITIC", "ITMA", "ITMG", "JAST", "JAYA", "JECC",
            "JGLE", "JIHD", "JKON", "JKSW", "JMAS", "JPFA", "JRPT", "JSKY", "JSMR", "JSPT",
            "JTPE", "KAEF", "KARW", "KAYU", "KBAG", "KBLI", "KBLM", "KBLV", "KBRI", "KDSI",
            "KEEN", "KEJU", "KINO", "KIJA", "KKGI", "KLBF", "KMDS", "KMTR", "KOBX", "KOIN",
            "KONI", "KOPI", "KOTA", "KPAL", "KPAS", "KPIG", "KRAH", "KRAS", "KREN", "KRYA",
            "LAMI", "LAND", "LAPD", "LCGP", "LCKM", "LEAD", "LIFE", "LINK", "LION", "LMAS",
            "LMPI", "LMSH", "LPCK", "LPGI", "LPIN", "LPKR", "LPLI", "LPPF", "LPPS", "LRNA",
            "LSIP", "LTLS", "LUCK", "MABA", "MAGP", "MAIN", "MAMI", "MAPA", "MAPI", "MASA",
            "MAYA", "MBAP", "MBSS", "MCAS", "MCOL", "MDIA", "MDKA", "MDLN", "MDRN", "MEDC",
            "MEGA", "MERK", "META", "MFMI", "MGNA", "MICE", "MIDI", "MIKA", "MINA", "MIRA",
            "MITI", "MKNT", "MLBI", "MLIA", "MLPL", "MLPT", "MMLP", "MNCN", "MOLI", "MPMX",
            "MPOW", "MPPA", "MRAT", "MREI", "MSIN", "MSKY", "MTDL", "MTFN", "MTLA", "MTPS",
            "MTSM", "MYOH", "MYOR", "MYRX", "MYTX", "NASA", "NATO", "NELY", "NFCX", "NICK",
            "NICL", "NIKL", "NIPS", "NIRO", "NISP", "NOBU", "NPGF", "NRCA", "NUSA", "NZIA",
            "OASA", "OCAP", "OKAS", "OMRE", "OPMS", "PADI", "PALM", "PAMI", "PAMG", "PANI",
            "PANR", "PANS", "PBRX", "PBSA", "PCAR", "PDPP", "PEGE", "PEHA", "PGAS", "PGJO",
            "PGLI", "PGUN", "PICO", "PJAA", "PKPK", "PLAN", "PLAS", "PLIN", "PMLI", "PNBN",
            "PNBS", "PNGO", "PNIN", "PNLF", "PNSE", "POLA", "POLY", "POLL", "POLU", "POWR",
            "PPRE", "PPRO", "PRAS", "PRDA", "PRIM", "PSAB", "PSDN", "PSGO", "PSKT", "PSSI",
            "PTBA", "PTDU", "PTIS", "PTPP", "PTPW", "PTRO", "PTSN", "PTSP", "PUDP", "PURA",
            "PURE", "PURI", "PWON", "PYFA", "RAJA", "RALS", "RANC", "RBMS", "RDTX", "REAL",
            "RELI", "RICY", "RIGS", "RIMO", "RISE", "RMBA", "ROCK", "RODA", "ROTI", "RSGK",
            "RUIS", "SAFE", "SAME", "SAPX", "SATU", "SBAT", "SCCO", "SCMA", "SCNP", "SCPI",
            "SDMU", "SDPC", "SDRA", "SEMA", "SFAN", "SGER", "SGRO", "SHID", "SHIP", "SIDO",
            "SILO", "SIMA", "SIMP", "SIPD", "SKBM", "SKLT", "SKRN", "SKYB", "SLIS", "SMAR",
            "SMBR", "SMCB", "SMDR", "SMGR", "SMKL", "SMMA", "SMMT", "SMSM", "SOCI", "SOCL",
            "SOFA", "SOHO", "SONA", "SOSS", "SOTS", "SPMA", "SPTO", "SQMI", "SRAJ", "SRIL",
            "SRSN", "SRTG", "SSIA", "SSMS", "SSTM", "STAR", "STTP", "SUGI", "SULI", "SUPR",
            "SURE", "SURY", "SWID", "TALF", "TAMA", "TAMU", "TAPG", "TARA", "TAXI", "TBLA",
            "TCID", "TCPI", "TDPM", "TEBE", "TECH", "TECH", "TELE", "TELK", "TFCO", "TFLO",
            "TGKA", "TGRA", "TIFA", "TINS", "TIRA", "TIRT", "TITIS", "TKIM", "TLKM", "TMPO",
            "TNBA", "TNCA", "TOPS", "TOTL", "TOTO", "TOWR", "TOYS", "TPIA", "TPMA", "TRAM",
            "TRIO", "TRIS", "TRST", "TRUK", "TSPC", "TUGU", "TURI", "UANG", "UCID", "UFOE",
            "ULTJ", "UNIC", "UNIQ", "UNIT", "UNSP", "UNTR", "UNVR", "URBN", "VICI", "VINS",
            "VOKS", "VRNA", "WAPO", "WEGE", "WEHA", "WICO", "WIIM", "WIKA", "WINE", "WINR",
            "WINS", "WOMF", "WOOD", "WOWS", "WSBP", "WSKT", "WTON", "YELO", "ZBRA", "ZONE",
            "ZYRX"
        ]
        return [f"{ticker}.JK" for ticker in base_stocks]

# ============= PROCESS TICKERS =============
def process_ticker(ticker, strategy, period):
    try:
        df = fetch_with_retry(ticker, period)
        if df is None or len(df) < 20:
            return None
            
        current_price = df['Close'].iloc[-1]
        current_volume_ratio = df['Volume_Ratio'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        
        # Select scoring function
        score, trend_info = score_full_screener(df)
        
        if score < 40:  # Minimum threshold
            return None
            
        levels = calculate_trading_levels(current_price, score, trend_info)
        
        return {
            'Ticker': ticker,
            'Price': current_price,
            'Score': score,
            'Signal': levels['signal'],
            'SignalClass': levels['signal_class'],
            'Entry Ideal': levels['entry_ideal'],
            'Entry Agresif': levels['entry_agresif'],
            'TP1': levels['tp1'],
            'TP2': levels['tp2'],
            'Cut Loss': levels['cut_loss'],
            'Trend': levels['trend'],
            'Volume Ratio': current_volume_ratio,
            'RSI': current_rsi,
            'Trend Info': trend_info
        }
        
    except Exception as e:
        return None

def batch_process(tickers, strategy, period, max_workers=8):
    results = []
    total = len(tickers)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_ticker, ticker, strategy, period): ticker 
            for ticker in tickers
        }
        
        completed = 0
        for future in as_completed(future_to_ticker):
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"📊 Processing {completed}/{total} stocks...")
            
            result = future.result()
            if result:
                results.append(result)
            
            time.sleep(0.05)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    return results

# ============= STREAMLIT UI =============
def main():
    init_db()
    
    st.markdown('<div class="big-title">🚀 IDX Power Screener v4.0</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #64748b; margin-bottom: 2rem;">Advanced Technical Analysis | Real-time Alerts | Portfolio Management</div>', unsafe_allow_html=True)
    
    all_tickers = load_all_tickers()
    market_hours = check_market_hours()
    
    with st.sidebar:
        st.markdown("## ⚙️ Menu")
        
        # Market status
        if market_hours['market_open']:
            st.success("🟢 MARKET OPEN")
        else:
            st.warning("🔴 MARKET CLOSED")
        
        st.markdown("---")
        
        menu = st.radio("Pilih Menu:", [
            "1. Full Screener",
            "2. Single Analysis", 
            "3. Bandar",
            "4. BPJS",
            "5. BSJP"
        ])
        
        st.markdown("---")
        
        if menu == "1. Full Screener":
            stage = st.radio("Screening Stage:", [
                "Stage 1: 800 → 60 Stocks",
                "Stage 2: 60 → 15 Stocks"
            ])
            
            min_score = st.slider("Minimum Score:", 40, 90, 65)
            use_fast_mode = st.checkbox("⚡ Fast Mode", value=True)
        
        period = st.selectbox("Period:", ["1mo", "3mo", "6mo"], index=1)
        
        st.markdown("---")
        st.caption(f"Total Tickers: {len(all_tickers)}")
    
    # Main content
    if menu == "1. Full Screener":
        show_full_screener(all_tickers, stage, min_score, period, use_fast_mode)
    elif menu == "2. Single Analysis":
        show_single_analysis(all_tickers, period)
    elif menu == "3. Bandar":
        show_strategy_screener(all_tickers, "Bandar", stage, min_score, period, use_fast_mode)
    elif menu == "4. BPJS":
        show_strategy_screener(all_tickers, "BPJS", stage, min_score, period, use_fast_mode)
    elif menu == "5. BSJP":
        show_strategy_screener(all_tickers, "BSJP", stage, min_score, period, use_fast_mode)

def show_full_screener(tickers, stage, min_score, period, use_fast_mode):
    st.markdown("## 📊 Full Market Screener")
    
    # Display stage info
    if "Stage 1" in stage:
        st.markdown('<div class="stage-box">🎯 STAGE 1: Screening 800+ stocks → 60 Best Stocks</div>', unsafe_allow_html=True)
        target_count = 60
    else:
        st.markdown('<div class="stage-box">🎯 STAGE 2: Screening 60 stocks → 15 Best Stocks</div>', unsafe_allow_html=True)
        target_count = 15
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Stocks", len(tickers))
    col2.metric("Target", f"{target_count} stocks")
    col3.metric("Min Score", min_score)
    
    if st.button("🚀 Run Screening", type="primary", use_container_width=True):
        if "Stage 1" in stage:
            # Stage 1: Screen 800 stocks to 60
            with st.spinner("Stage 1: Screening 800+ stocks to find 60 best..."):
                workers = 10 if use_fast_mode else 5
                stage1_tickers = tickers[:800]  # Take first 800 stocks
                
                results = batch_process(stage1_tickers, "Full Screener", period, workers)
                results_df = pd.DataFrame(results)
                
                if results_df.empty:
                    st.error("No stocks found meeting criteria")
                    return
                
                # Filter by minimum score and take top 60
                filtered_results = results_df[results_df['Score'] >= min_score]
                if len(filtered_results) > 60:
                    stage1_final = filtered_results.nlargest(60, 'Score')
                else:
                    stage1_final = filtered_results
                
                display_results(stage1_final, f"Stage 1 Results - Top {len(stage1_final)} Stocks")
                
        else:
            # Stage 2: Screen 60 stocks to 15
            with st.spinner("Stage 1: Initial screening to get 60 stocks..."):
                # First get 60 stocks from stage 1
                workers = 8 if use_fast_mode else 4
                stage1_tickers = tickers[:800]
                stage1_results = batch_process(stage1_tickers, "Full Screener", period, workers)
                stage1_df = pd.DataFrame(stage1_results)
                
                if stage1_df.empty:
                    st.error("No stocks found in Stage 1")
                    return
                
                stage1_filtered = stage1_df[stage1_df['Score'] >= min_score]
                if len(stage1_filtered) > 60:
                    top_60 = stage1_filtered.nlargest(60, 'Score')
                else:
                    top_60 = stage1_filtered
            
            st.success(f"Stage 1 completed: Found {len(top_60)} stocks")
            
            # Stage 2: Detailed analysis of top 60
            with st.spinner("Stage 2: Detailed analysis of 60 stocks to find 15 best..."):
                detailed_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (_, row) in enumerate(top_60.iterrows()):
                    progress_bar.progress((i + 1) / len(top_60))
                    status_text.text(f"🔍 Analyzing {row['Ticker']}...")
                    
                    # Use longer period for more accurate analysis
                    detailed_df = fetch_with_retry(row['Ticker'], "6mo")
                    
                    if detailed_df is not None:
                        # Recalculate with more data
                        current_price = detailed_df['Close'].iloc[-1]
                        current_volume = detailed_df['Volume_Ratio'].iloc[-1]
                        current_rsi = detailed_df['RSI'].iloc[-1]
                        
                        score, trend_info = score_full_screener(detailed_df)
                        
                        if score >= min_score:
                            levels = calculate_trading_levels(current_price, score, trend_info)
                            detailed_results.append({
                                'Ticker': row['Ticker'],
                                'Price': current_price,
                                'Score': score,
                                'Signal': levels['signal'],
                                'SignalClass': levels['signal_class'],
                                'Entry Ideal': levels['entry_ideal'],
                                'Entry Agresif': levels['entry_agresif'],
                                'TP1': levels['tp1'],
                                'TP2': levels['tp2'],
                                'Cut Loss': levels['cut_loss'],
                                'Trend': levels['trend'],
                                'Volume Ratio': current_volume,
                                'RSI': current_rsi,
                                'Trend Info': trend_info
                            })
                    
                    time.sleep(0.1)  # Rate limiting
                
                progress_bar.empty()
                status_text.empty()
                
                # Take top 15
                if detailed_results:
                    final_df = pd.DataFrame(detailed_results)
                    if len(final_df) > 15:
                        final_results = final_df.nlargest(15, 'Score')
                    else:
                        final_results = final_df
                    
                    display_results(final_results, f"Stage 2 Results - Top {len(final_results)} Stocks")
                else:
                    st.error("No stocks found in Stage 2")

def display_results(results_df, title):
    st.markdown(f"### {title}")
    st.success(f"🎯 Found {len(results_df)} qualifying stocks!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Score", f"{results_df['Score'].mean():.1f}")
    col2.metric("Strong Buy", len(results_df[results_df['Signal'] == 'STRONG BUY']))
    col3.metric("Buy", len(results_df[results_df['Signal'] == 'BUY']))
    col4.metric("Avg RSI", f"{results_df['RSI'].mean():.1f}")
    
    # Display table
    display_cols = ['Ticker', 'Price', 'Score', 'Signal', 'Trend', 'Entry Ideal', 'TP1', 'TP2', 'Cut Loss']
    display_df = results_df[display_cols].copy()
    
    st.dataframe(
        display_df.style.format({
            'Price': '{:,.0f}',
            'Entry Ideal': '{:,.0f}',
            'TP1': '{:,.0f}',
            'TP2': '{:,.0f}',
            'Cut Loss': '{:,.0f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Detailed view
    st.markdown("### 📋 Detailed Analysis")
    for _, row in results_df.iterrows():
        with st.expander(f"{row['Ticker']} - {row['Signal']} (Score: {row['Score']})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Price", f"Rp {row['Price']:,.0f}")
                st.metric("RSI", f"{row['RSI']:.1f}")
                st.metric("Volume Ratio", f"{row['Volume Ratio']:.2f}x")
                
            with col2:
                st.metric("Trend", row['Trend'])
                st.metric("Entry Ideal", f"Rp {row['Entry Ideal']:,.0f}" if row['Entry Ideal'] else "N/A")
                st.metric("Cut Loss", f"Rp {row['Cut Loss']:,.0f}" if row['Cut Loss'] else "N/A")
            
            # Signal box
            st.markdown(f'<div class="signal-box {row["SignalClass"]}">{row["Signal"]} - Score: {row["Score"]}/100</div>', unsafe_allow_html=True)
            
            st.write(f"**Trend Analysis:** {row['Trend Info']}")

def show_single_analysis(tickers, period):
    st.markdown("## 🔍 Single Stock Analysis")
    
    selected_ticker = st.selectbox("Pilih Saham:", tickers)
    
    if st.button("Analyze Stock", type="primary"):
        with st.spinner(f"Analyzing {selected_ticker}..."):
            df = fetch_with_retry(selected_ticker, period)
            
            if df is None:
                st.error("Failed to fetch data")
                return
            
            # Calculate score
            current_price = df['Close'].iloc[-1]
            current_volume = df['Volume_Ratio'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            
            score, trend_info = score_full_screener(df)
            levels = calculate_trading_levels(current_price, score, trend_info)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💰 Current Price", f"Rp {current_price:,.0f}")
                st.metric("📊 Score", f"{score}/100")
            
            with col2:
                st.metric("🎯 Signal", levels['signal'])
                st.metric("📈 Trend", levels['trend'])
            
            with col3:
                st.metric("📊 RSI", f"{current_rsi:.1f}")
                st.metric("📈 Volume Ratio", f"{current_volume:.2f}x")
            
            # Signal box
            st.markdown(f'<div class="signal-box {levels["signal_class"]}">{levels["signal"]} - Score: {score}/100</div>', unsafe_allow_html=True)
            
            # Trading levels
            st.markdown("### 🎯 Trading Levels")
            if levels['entry_ideal']:
                col1, col2, col3, col4, col5 = st.columns(5)
                
                col1.success(f"**Entry Ideal**\nRp {levels['entry_ideal']:,.0f}")
                col2.warning(f"**Entry Agresif**\nRp {levels['entry_agresif']:,.0f}")
                col3.info(f"**TP1**\nRp {levels['tp1']:,.0f}")
                col4.info(f"**TP2**\nRp {levels['tp2']:,.0f}")
                col5.error(f"**Cut Loss**\nRp {levels['cut_loss']:,.0f}")

def show_strategy_screener(tickers, strategy, stage, min_score, period, use_fast_mode):
    st.markdown(f"## 📊 {strategy} Screener")
    st.info(f"**{strategy} Strategy**: Specialized screening for this strategy")
    
    if st.button(f"🚀 Run {strategy} Screening", type="primary", use_container_width=True):
        with st.spinner(f"Running {strategy} screening..."):
            workers = 8 if use_fast_mode else 4
            results = batch_process(tickers[:800], "Full Screener", period, workers)
            results_df = pd.DataFrame(results)
            
            if results_df.empty:
                st.error("No stocks found meeting criteria")
                return
            
            if "Stage 1" in stage:
                filtered_results = results_df[results_df['Score'] >= min_score]
                if len(filtered_results) > 60:
                    final_results = filtered_results.nlargest(60, 'Score')
                else:
                    final_results = filtered_results
                display_results(final_results, f"{strategy} - Top {len(final_results)} Stocks")
            else:
                filtered_results = results_df[results_df['Score'] >= min_score]
                if len(filtered_results) > 15:
                    final_results = filtered_results.nlargest(15, 'Score')
                else:
                    final_results = filtered_results
                display_results(final_results, f"{strategy} - Top {len(final_results)} Stocks")

if __name__ == "__main__":
    main()
