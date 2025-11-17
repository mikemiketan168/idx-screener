#!/usr/bin/env python3
"""
IDX Power Screener v6.0 STOCKBOT ULTIMATE
Complete production-ready version
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime, timedelta, timezone
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
from typing import Dict, List, Optional, Tuple
import yfinance as yf

# ================== LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('screener.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="IDX Power Screener v6.0 ULTIMATE",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== SESSION STATE ==================
def init_session_state():
    defaults = {
        'last_scan_results': None,
        'last_scan_time': None,
        'last_scan_strategy': None,
        'scan_count': 0,
        'portfolio_size': 100000000,
        'risk_per_trade': 2.0,
        'strict_mode': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ================== IHSG WIDGET ==================
@st.cache_data(ttl=180, show_spinner=False)
def fetch_ihsg_data():
    try:
        ihsg = yf.Ticker("^JKSE")
        hist = ihsg.history(period="5d")
        if hist.empty:
            return None
        
        current = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current
        high = hist['High'].iloc[-1]
        low = hist['Low'].iloc[-1]
        volume = hist['Volume'].iloc[-1]
        
        change = current - prev_close
        change_pct = (change / prev_close) * 100
        trend_5d = ((current - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100 if len(hist) >= 5 else change_pct
        
        return {
            'price': current, 'change': change, 'change_pct': change_pct,
            'high': high, 'low': low, 'volume': volume, 'trend_5d': trend_5d,
            'status': 'up' if change >= 0 else 'down', 'timestamp': datetime.now()
        }
    except Exception as e:
        logger.error(f"IHSG error: {e}")
        return None

def display_ihsg_widget():
    ihsg = fetch_ihsg_data()
    if not ihsg:
        st.info("📊 IHSG data unavailable")
        return None

    status_emoji = "🟢" if ihsg['status'] == 'up' else "🔴"
    
    if ihsg['change_pct'] > 1.5:
        condition, color = "🔥 Strong Uptrend", "#059669"
        guidance = "✅ EXCELLENT for all strategies"
    elif ihsg['change_pct'] > 0.5:
        condition, color = "📈 Moderate Uptrend", "#10b981"
        guidance = "✅ GOOD conditions"
    elif ihsg['change_pct'] > -0.5:
        condition, color = "➡️ Sideways", "#fbbf24"
        guidance = "⚠️ Be selective"
    elif ihsg['change_pct'] > -1.5:
        condition, color = "📉 Downtrend", "#f59e0b"
        guidance = "⚠️ CAUTION - Tight stops"
    else:
        condition, color = "🔻 Strong Downtrend", "#ef4444"
        guidance = "❌ HIGH RISK - Consider cash"

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                padding: 18px; border-radius: 12px; margin-bottom: 20px;
                border-left: 6px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
        <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div>
                <h2 style='margin:0;color:white;'>📊 JAKARTA COMPOSITE INDEX</h2>
                <p style='margin:5px 0;color:#e0e7ff;font-size:0.9em;'>Real-time Market Analysis</p>
            </div>
            <div style='text-align:right;'>
                <h1 style='margin:0;color:white;font-size:2em;'>{status_emoji} {ihsg['price']:,.2f}</h1>
                <p style='margin:5px 0;color:{color};font-size:1.3em;font-weight:bold;'>
                    {ihsg['change']:+,.2f} ({ihsg['change_pct']:+.2f}%)
                </p>
            </div>
        </div>
        <div style='margin-top:12px;padding:12px;background:rgba(0,0,0,0.2);border-radius:8px;'>
            <p style='margin:0 0 5px 0;color:#fbbf24;font-size:1em;font-weight:600;'>{condition}</p>
            <p style='margin:0;color:#a5b4fc;font-size:0.9em;'>📈 {guidance}</p>
        </div>
        <p style='margin:10px 0 0 0;color:#94a3b8;font-size:0.75em;text-align:right;'>
            ⏰ {ihsg['timestamp'].strftime('%H:%M:%S')} WIB | 🔄 Refresh: 3 min
        </p>
    </div>
    """, unsafe_allow_html=True)
    return ihsg

# ================== TICKER LOADING ==================
@st.cache_data(ttl=3600)
def load_tickers() -> List[str]:
    try:
        if os.path.exists("idx_stocks.json"):
            with open("idx_stocks.json", "r") as f:
                data = json.load(f)
            tickers = data.get("tickers", [])
            return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except Exception as e:
        logger.error(f"JSON load error: {e}")
    
    # Comprehensive fallback
    return [
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

# ================== TIME UTILS ==================
def get_jakarta_time():
    try:
        import pytz
        return datetime.now(pytz.timezone('Asia/Jakarta'))
    except:
        return datetime.now(timezone(timedelta(hours=7)))

def is_bpjs_time():
    return 9 <= get_jakarta_time().hour < 10

def is_bsjp_time():
    return 14 <= get_jakarta_time().hour < 16

# ================== VALIDATION ==================
def validate_dataframe(df: pd.DataFrame) -> bool:
    try:
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            return False
        if len(df) < 50 or (df['Close'] <= 0).any():
            return False
        return True
    except:
        return False

# ================== DATA FETCHING ==================
@st.cache_data(ttl=300, show_spinner=False, max_entries=200)
def fetch_data(ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    try:
        end = int(datetime.now().timestamp())
        days = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}.get(period, 180)
        start = end - (days * 86400)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        r = requests.get(url, params={"period1": start, "period2": end, "interval": "1d"},
                        headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        
        if r.status_code != 200:
            return None
        
        data = r.json()
        result = data['chart']['result'][0]
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': quote['open'], 'High': quote['high'],
            'Low': quote['low'], 'Close': quote['close'], 'Volume': quote['volume']
        }, index=pd.to_datetime(result['timestamp'], unit='s'))
        
        df = df.dropna()
        if not validate_dataframe(df):
            return None
        
        return calculate_indicators(df)
    except Exception as e:
        logger.error(f"{ticker}: {e}")
        return None

# ================== INDICATORS ==================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # EMAs
        for p in [9, 21, 50, 200]:
            df[f'EMA{p}'] = df['Close'].ewm(span=min(p, len(df)), adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume
        df['VOL_SMA20'] = df['Volume'].rolling(20).mean()
        df['VOL_SMA50'] = df['Volume'].rolling(50).mean()
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA20'].replace(0, np.nan)
        
        # Momentum
        for p in [5, 10, 20]:
            df[f'MOM_{p}D'] = ((df['Close'] - df['Close'].shift(p)) / df['Close'].shift(p)) * 100
        
        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        df['OBV_EMA'] = pd.Series(df['OBV']).ewm(span=10, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_MID'] = df['Close'].rolling(20).mean()
        df['BB_STD'] = df['Close'].rolling(20).std()
        df['BB_UPPER'] = df['BB_MID'] + 2 * df['BB_STD']
        df['BB_LOWER'] = df['BB_MID'] - 2 * df['BB_STD']
        df['BB_POSITION'] = ((df['Close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])) * 100
        
        # Stochastic
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['STOCH_K'] = 100 * (df['Close'] - low14) / (high14 - low14).replace(0, np.nan)
        df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
        
        # ATR
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        df['ATR_PCT'] = (df['ATR'] / df['Close']) * 100
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
        
        return df
    except Exception as e:
        logger.error(f"Indicator error: {e}")
        return df

# ================== LIQUIDITY FILTER ==================
def apply_liquidity_filter(df: pd.DataFrame) -> Tuple[bool, str]:
    try:
        r = df.iloc[-1]
        price = r['Close']
        vol_avg = df['Volume'].tail(20).mean()
        
        if price < 50:
            return False, f"Price too low: Rp {price:.0f}"
        if vol_avg < 500000:
            return False, f"Volume too low"
        if price * vol_avg < 100_000_000:
            return False, f"Turnover too low"
        
        zero_days = (df['Volume'].tail(5) == 0).sum()
        if zero_days >= 2:
            return False, f"Inactive stock"
        
        return True, "Passed"
    except:
        return False, "Filter error"

# ================== SCORING: GENERAL ==================
def score_general(df: pd.DataFrame) -> Tuple[int, Dict, int, str]:
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        
        passed, reason = apply_liquidity_filter(df)
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"
        
        if r['Close'] < r['EMA21'] < r['EMA50'] < r['EMA200']:
            return 0, {"Rejected": "Strong downtrend"}, 0, "F"
        
        if r['Close'] < r['EMA50'] and st.session_state.strict_mode:
            return 0, {"Rejected": "Below EMA50"}, 0, "F"
        
        mom_20d = r['MOM_20D']
        if mom_20d < -8:
            return 0, {"Rejected": f"Weak momentum {mom_20d:.1f}%"}, 0, "F"
        
        vol_ratio = r['VOL_RATIO']
        if vol_ratio < 0.8:
            return 0, {"Rejected": f"Weak volume"}, 0, "F"
        
        # Momentum factor
        momentum_factor = 1.0
        if mom_20d < -5:
            momentum_factor = 0.7
        elif mom_20d < 0:
            momentum_factor = 0.9
        elif mom_20d > 10:
            momentum_factor = 1.15
        
        # EMA alignment (40 pts)
        ema_score = sum([
            r['Close'] > r['EMA9'], r['EMA9'] > r['EMA21'],
            r['EMA21'] > r['EMA50'], r['EMA50'] > r['EMA200']
        ])
        
        if ema_score == 4:
            score += 40
            details['Trend'] = '🟢 Perfect (4/4)'
        elif ema_score == 3:
            score += 28
            details['Trend'] = '🟡 Strong (3/4)'
        elif ema_score == 2:
            score += 15
            details['Trend'] = '🟠 Moderate (2/4)'
        else:
            score += 5
            details['Trend'] = '🔴 Weak'
        
        # RSI (25 pts)
        rsi = r['RSI']
        if 50 <= rsi <= 65:
            score += 25
            details['RSI'] = f'🟢 Ideal {rsi:.0f}'
        elif 45 <= rsi < 50:
            score += 20
            details['RSI'] = f'🟡 Good {rsi:.0f}'
        elif 40 <= rsi < 45:
            score += 15
            details['RSI'] = f'🟠 OK {rsi:.0f}'
        else:
            score += 10
            details['RSI'] = f'⚪ {rsi:.0f}'
        
        # Volume (20 pts)
        if vol_ratio > 3.0:
            score += 20
            details['Volume'] = f'🔥 Huge {vol_ratio:.1f}x'
        elif vol_ratio > 2.0:
            score += 17
            details['Volume'] = f'🟢 Strong {vol_ratio:.1f}x'
        elif vol_ratio > 1.5:
            score += 13
            details['Volume'] = f'🟡 Good {vol_ratio:.1f}x'
        elif vol_ratio > 1.0:
            score += 8
            details['Volume'] = f'🟠 Normal {vol_ratio:.1f}x'
        else:
            score += 4
            details['Volume'] = f'🔴 Weak {vol_ratio:.1f}x'
        
        # Momentum (15 pts)
        mom_5d = r['MOM_5D']
        if mom_5d > 5:
            score += 15
            details['Momentum'] = f'🟢 Strong {mom_5d:.1f}%'
        elif mom_5d > 3:
            score += 12
            details['Momentum'] = f'🟡 Good {mom_5d:.1f}%'
        elif mom_5d > 1:
            score += 9
            details['Momentum'] = f'🟠 Positive {mom_5d:.1f}%'
        else:
            score += 4
            details['Momentum'] = '⚪ Weak'
        
        score = int(score * momentum_factor)
        score = min(100, score)
        
        if score >= 90:
            grade, conf = "A+", 92
        elif score >= 80:
            grade, conf = "A", 85
        elif score >= 70:
            grade, conf = "B+", 75
        elif score >= 60:
            grade, conf = "B", 65
        elif score >= 50:
            grade, conf = "C", 52
        else:
            grade, conf = "D", max(score, 0)
        
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"

# ================== SCORING: BPJS ==================
def score_bpjs(df: pd.DataFrame) -> Tuple[int, Dict, int, str]:
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        
        passed, reason = apply_liquidity_filter(df)
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"
        
        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "No uptrend"}, 0, "F"
        
        atr_pct = r['ATR_PCT']
        if 2.5 <= atr_pct <= 4.5:
            score += 35
            details['Volatility'] = f'🟢 PERFECT {atr_pct:.1f}%'
        elif 2.0 <= atr_pct <= 5.5:
            score += 25
            details['Volatility'] = f'🟡 Good {atr_pct:.1f}%'
        else:
            return 0, {"Rejected": f"Volatility {atr_pct:.1f}%"}, 0, "F"
        
        vol_ratio = r['VOL_RATIO']
        if vol_ratio > 4.0:
            score += 30
            details['Volume'] = f'🔥 MASSIVE {vol_ratio:.1f}x'
        elif vol_ratio > 2.0:
            score += 23
            details['Volume'] = f'🟢 Strong {vol_ratio:.1f}x'
        elif vol_ratio > 1.5:
            score += 15
            details['Volume'] = f'🟡 Moderate {vol_ratio:.1f}x'
        else:
            return 0, {"Rejected": "Weak volume"}, 0, "F"
        
        rsi = r['RSI']
        if 25 <= rsi <= 35:
            score += 20
            details['RSI'] = f'🟢 Oversold {rsi:.0f}'
        elif 35 < rsi <= 42:
            score += 15
            details['RSI'] = f'🟡 Good {rsi:.0f}'
        else:
            score += 8
            details['RSI'] = f'⚪ {rsi:.0f}'
        
        if r['STOCH_K'] < 25:
            score += 15
            details['Stoch'] = f"🟢 Oversold {r['STOCH_K']:.0f}"
        
        if score >= 85:
            grade, conf = "A+", 88
        elif score >= 75:
            grade, conf = "A", 78
        elif score >= 65:
            grade, conf = "B+", 68
        else:
            grade, conf = "B", 58
        
        return score, details, conf, grade
    except:
        return 0, {"Error": "Scoring failed"}, 0, "F"

# ================== SCORING: BSJP ==================
def score_bsjp(df: pd.DataFrame) -> Tuple[int, Dict, int, str]:
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        
        passed, reason = apply_liquidity_filter(df)
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"
        
        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "No uptrend"}, 0, "F"
        
        prev_close = df['Close'].iloc[-2]
        gap = ((r['Close'] - prev_close) / prev_close) * 100
        
        if -5 < gap < -2:
            score += 35
            details['Gap'] = f'🟢 PERFECT {gap:.1f}%'
        elif -2 <= gap < -0.3:
            score += 20
            details['Gap'] = f'🟡 Good {gap:.1f}%'
        else:
            return 0, {"Rejected": f"No gap ({gap:.1f}%)"}, 0, "F"
        
        bb_pos = r['BB_POSITION']
        if bb_pos < 20:
            score += 30
            details['BB'] = f'🟢 Lower band {bb_pos:.0f}%'
        elif bb_pos < 35:
            score += 22
            details['BB'] = f'🟡 Below mid {bb_pos:.0f}%'
        
        rsi = r['RSI']
        if 20 <= rsi <= 35:
            score += 20
            details['RSI'] = f'🟢 Oversold {rsi:.0f}'
        elif 35 < rsi <= 45:
            score += 14
            details['RSI'] = f'🟡 Good {rsi:.0f}'
        
        if r['VOL_RATIO'] > 1.5:
            score += 15
            details['Volume'] = f"🟢 Strong {r['VOL_RATIO']:.1f}x"
        
        if score >= 80:
            grade, conf = "A", 82
        elif score >= 70:
            grade, conf = "B+", 72
        else:
            grade, conf = "B", 62
        
        return score, details, conf, grade
    except:
        return 0, {"Error": "Scoring failed"}, 0, "F"

# ================== SCORING: BANDAR ==================
def score_bandar(df: pd.DataFrame) -> Tuple[int, Dict, int, str]:
    try:
        r = df.iloc[-1]
        details = {}
        
        passed, reason = apply_liquidity_filter(df)
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"
        
        vol_10d = df['Volume'].tail(10).mean() / df['VOL_SMA50'].iloc[-1]
        price_20d = ((r['Close'] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
        
        obv_cur = df['OBV'].iloc[-1]
        obv_20 = df['OBV'].iloc[-20]
        obv_chg = (obv_cur - obv_20) / abs(obv_20) if abs(obv_20) > 0 else 0
        obv_rising = obv_cur > df['OBV_EMA'].iloc[-1]
        
        # ACCUMULATION
        if (vol_10d > 1.2 and -3 < price_20d < 8 and obv_chg > 0.03 and 
            obv_rising and r['Close'] > r['EMA50']):
            score, conf = 90, 85
            details['Phase'] = '🟢 ACCUMULATION'
            details['Signal'] = 'STRONG BUY'
            details['Action'] = 'Enter now'
        # MARKUP
        elif price_20d > 8 and obv_chg > 0.08:
            score, conf = 75, 70
            details['Phase'] = '🚀 MARKUP'
            details['Signal'] = 'HOLD/TRAIL'
        # DISTRIBUTION
        elif r['VOL_RATIO'] > 1.8 and r['RSI'] > 70:
            score, conf = 20, 25
            details['Phase'] = '🔴 DISTRIBUTION'
            details['Signal'] = 'SELL'
        else:
            score, conf = 45, 40
            details['Phase'] = '⚪ RANGING'
            details['Signal'] = 'WAIT'
        
        details['Vol_10D'] = f'{vol_10d:.1f}x'
        details['Price_20D'] = f'{price_20d:+.1f}%'
        details['OBV'] = f'{obv_chg*100:+.1f}%'
        
        grade = "A" if score >= 80 else "B" if score >= 60 else "C"
        return score, details, conf, grade
    except:
        return 0, {"Error": "Scoring failed"}, 0, "F"

# ================== SCORING: SWING ==================
def score_swing(df: pd.DataFrame) -> Tuple[int, Dict, int, str]:
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        
        passed, reason = apply_liquidity_filter(df)
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"
        
        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "Below EMA50"}, 0, "F"
        
        if r['MOM_20D'] < -5:
            return 0, {"Rejected": f"Negative momentum"}, 0, "F"
        
        ema_cnt = sum([
            r['EMA9'] > r['EMA21'], r['EMA21'] > r['EMA50'],
            r['EMA50'] > r['EMA200'], r['Close'] > r['EMA9']
        ])
        
        if ema_cnt == 4:
            score += 35
            details['Trend'] = '🟢 Perfect (4/4)'
        elif ema_cnt == 3:
            score += 28
            details['Trend'] = '🟢 Strong (3/4)'
        else:
            score += 15
            details['Trend'] = '🟡 Moderate'
        
        rsi = r['RSI']
        if 45 <= rsi <= 65:
            score += 25
            details['RSI'] = f'🟢 Ideal {rsi:.0f}'
        elif 40 <= rsi <= 70:
            score += 20
            details['RSI'] = f'🟡 Good {rsi:.0f}'
        else:
            score += 10
            details['RSI'] = f'🟠 {rsi:.0f}'
        
        vol = r['VOL_RATIO']
        if vol > 2.5:
            score += 20
            details['Volume'] = f'🟢 Strong {vol:.1f}x'
        elif vol > 1.5:
            score += 15
            details['Volume'] = f'🟡 Good {vol:.1f}x'
        else:
            score += 8
            details['Volume'] = f'🟠 {vol:.1f}x'
        
        mom10 = r['MOM_10D']
        if mom10 > 8:
            score += 20
            details['Momentum'] = f'🟢 Strong {mom10:.1f}%'
        elif mom10 > 5:
            score += 15
            details['Momentum'] = f'🟡 Good {mom10:.1f}%'
        else:
            score += 8
            details['Momentum'] = f'🟠 {mom10:.1f}%'
        
        if score >= 85:
            grade, conf = "A+", 88
        elif score >= 75:
            grade, conf = "A", 78
        elif score >= 65:
            grade, conf = "B+", 68
        else:
            grade, conf = "B", 58
        
        return score, details, conf, grade
    except:
        return 0, {"Error": "Scoring failed"}, 0, "F"

# ================== SCORING: VALUE ==================
def score_value(df: pd.DataFrame) -> Tuple[int, Dict, int, str]:
    try:
        r = df.iloc[-1]
        price = r['Close']
        score = 0
        details = {}
        
        if price >= 1000:
            return 0, {"Rejected": f"Too expensive"}, 0, "F"
        
        if r['Close'] < r['EMA200']:
            return 0, {"Rejected": "Below 200 EMA"}, 0, "F"
        
        if price < 300:
            score += 25
            details['Price'] = f'🟢 Very cheap Rp {price:.0f}'
        elif price < 500:
            score += 20
            details['Price'] = f'🟡 Cheap Rp {price:.0f}'
        else:
            score += 12
            details['Price'] = f'🟠 Rp {price:.0f}'
        
        bb_pos = r['BB_POSITION']
        if bb_pos < 25:
            score += 20
            details['Support'] = f'🟢 BB lower'
        elif bb_pos < 40:
            score += 12
            details['Support'] = f'🟡 Below mid'
        
        rsi = r['RSI']
        if rsi < 30:
            score += 20
            details['RSI'] = f'🟢 Oversold {rsi:.0f}'
        elif rsi < 40:
            score += 15
            details['RSI'] = f'🟡 Low {rsi:.0f}'
        
        vol_5d = df['Volume'].tail(5).mean()
        vol_20d = df['VOL_SMA20'].iloc[-1]
        vol_trend = vol_5d / vol_20d if vol_20d > 0 else 1
        
        if vol_trend > 1.5:
            score += 20
            details['Volume'] = f'🟢 Increasing {vol_trend:.1f}x'
        elif vol_trend > 1.2:
            score += 15
            details['Volume'] = f'🟡 Rising {vol_trend:.1f}x'
        
        green = sum(1 for i in range(-5, 0) if df['Close'].iloc[i] > df['Open'].iloc[i])
        if green >= 3:
            score += 15
            details['Pattern'] = f'🟢 Reversal ({green}/5)'
        
        if score >= 75:
            grade, conf = "A", 78
        elif score >= 65:
            grade, conf = "B+", 68
        else:
            grade, conf = "B", 58
        
        return score, details, conf, grade
    except:
        return 0, {"Error": "Scoring failed"}, 0, "F"

# ================== TREND & SIGNAL ==================
def detect_trend(r: pd.Series) -> str:
    try:
        p, e9, e21, e50, e200 = r['Close'], r['EMA9'], r['EMA21'], r['EMA50'], r['EMA200']
        if p > e9 > e21 > e50 > e200:
            return "Strong Uptrend"
        elif p > e50 and e9 > e21:
            return "Uptrend"
        elif abs(p - e50) / p < 0.03:
            return "Sideways"
        else:
            return "Downtrend"
    except:
        return "Unknown"

def classify_signal(r: pd.Series, score: int, grade: str, trend: str) -> str:
    try:
        if trend in ["Strong Uptrend", "Uptrend"] and grade in ["A+", "A"] and score >= 80:
            return "Strong Buy"
        elif trend in ["Strong Uptrend", "Uptrend"] and grade in ["A+", "A", "B+", "B"]:
            return "Buy"
        elif grade in ["B", "C"]:
            return "Hold"
        else:
            return "Watch"
    except:
        return "Unknown"

# ================== TRADE PLAN ==================
def compute_trade_plan(df: pd.DataFrame, strategy: str, trend: str) -> Dict:
    try:
        r = df.iloc[-1]
        price = r['Close']
        
        if strategy == "BPJS":
            entry_ideal = round(price * 0.998, 0)
            tp1 = round(entry_ideal * 1.03, 0)
            tp2 = round(entry_ideal * 1.05, 0)
            tp3 = None
            sl = round(entry_ideal * 0.975, 0)
        elif strategy == "BSJP":
            entry_ideal = round(price * 0.995, 0)
            tp1 = round(entry_ideal * 1.04, 0)
            tp2 = round(entry_ideal * 1.06, 0)
            tp3 = None
            sl = round(entry_ideal * 0.97, 0)
        elif strategy == "Swing":
            entry_ideal = round(price * 0.99, 0)
            tp1 = round(entry_ideal * 1.06, 0)
            tp2 = round(entry_ideal * 1.10, 0)
            tp3 = round(entry_ideal * 1.15, 0)
            sl = round(entry_ideal * 0.95, 0)
        elif strategy == "Value":
            entry_ideal = round(price * 0.98, 0)
            tp1 = round(entry_ideal * 1.15, 0)
            tp2 = round(entry_ideal * 1.25, 0)
            tp3 = round(entry_ideal * 1.40, 0)
            sl = round(entry_ideal * 0.92, 0)
        else:
            entry_ideal = round(price * 0.995, 0)
            tp1 = round(entry_ideal * 1.04, 0)
            tp2 = round(entry_ideal * 1.07, 0)
            tp3 = None
            sl = round(entry_ideal * 0.97, 0)
        
        entry_aggressive = round(price, 0)
        risk = entry_ideal - sl
        reward = tp1 - entry_ideal
        rr = reward / risk if risk > 0 else 0
        
        return {
            "entry_ideal": entry_ideal,
            "entry_aggressive": entry_aggressive,
            "tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": sl,
            "rr_ratio": rr,
            "volatility": r['ATR_PCT']
        }
    except:
        return {
            "entry_ideal": price, "entry_aggressive": price,
            "tp1": price * 1.05, "tp2": price * 1.10, "tp3": None,
            "sl": price * 0.95, "rr_ratio": 1.0, "volatility": 0
        }

# ================== CHART ==================
def create_chart(df: pd.DataFrame, ticker: str, period_days: int = 90):
    try:
        df_chart = df.tail(period_days).copy()
        
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{ticker} - Price & EMAs', 'Volume', 'RSI')
        )
        
        fig.add_trace(go.Candlestick(
            x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
            low=df_chart['Low'], close=df_chart['Close'], name='Price',
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ), row=1, col=1)
        
        colors = {'EMA9': '#2196F3', 'EMA21': '#FF9800', 'EMA50': '#F44336', 'EMA200': '#9E9E9E'}
        for ema in ['EMA9', 'EMA21', 'EMA50', 'EMA200']:
            if ema in df_chart.columns:
                fig.add_trace(go.Scatter(
                    x=df_chart.index, y=df_chart[ema], name=ema,
                    line=dict(color=colors[ema], width=1.5)
                ), row=1, col=1)
        
        vol_colors = ['#ef5350' if df_chart['Close'].iloc[i] < df_chart['Open'].iloc[i]
                      else '#26a69a' for i in range(len(df_chart))]
        fig.add_trace(go.Bar(
            x=df_chart.index, y=df_chart['Volume'], name='Volume',
            marker_color=vol_colors, showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart['RSI'], name='RSI',
            line=dict(color='#9C27B0', width=2)
        ), row=3, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        
        fig.update_layout(
            height=700, showlegend=True, xaxis_rangeslider_visible=False,
            hovermode='x unified', template='plotly_dark',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333')
        
        return fig
    except:
        return None

# ================== PROCESS TICKER ==================
def process_ticker(ticker: str, strategy: str, period: str) -> Optional[Dict]:
    try:
        df = fetch_data(ticker, period)
        if df is None:
            return None
        
        price = float(df['Close'].iloc[-1])
        
        if strategy == "BPJS":
            score, details, conf, grade = score_bpjs(df)
        elif strategy == "BSJP":
            score, details, conf, grade = score_bsjp(df)
        elif strategy == "Bandar":
            score, details, conf, grade = score_bandar(df)
        elif strategy == "Swing":
            score, details, conf, grade = score_swing(df)
        elif strategy == "Value":
            score, details, conf, grade = score_value(df)
        else:
            score, details, conf, grade = score_general(df)
        
        if grade not in ['A+', 'A', 'B+', 'B', 'C']:
            return None
        
        r = df.iloc[-1]
        trend = detect_trend(r)
        signal = classify_signal(r, score, grade, trend)
        plan = compute_trade_plan(df, strategy, trend)
        
        return {
            "Ticker": ticker.replace('.JK', ''),
            "Price": price,
            "Score": score,
            "Confidence": conf,
            "Grade": grade,
            "Trend": trend,
            "Signal": signal,
            "Entry": plan["entry_ideal"],
            "EntryAggressive": plan["entry_aggressive"],
            "TP1": plan["tp1"],
            "TP2": plan["tp2"],
            "TP3": plan.get("tp3"),
            "SL": plan["sl"],
            "R/R": plan["rr_ratio"],
            "Vol": plan["volatility"],
            "Details": details
        }
    except Exception as e:
        logger.error(f"{ticker}: {e}")
        return None

# ================== SCAN ENGINE ==================
def scan_stocks(tickers: List[str], strategy: str, period: str, limit1: int, limit2: int):
    st.info(f"🔍 Scanning {len(tickers)} stocks for {strategy}...")
    
    results = []
    progress = st.progress(0)
    status = st.empty()
    results_lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ticker, t, strategy, period): t for t in tickers}
        completed = 0
        
        for future in as_completed(futures):
            completed += 1
            progress.progress(completed / len(tickers))
            status.text(f"📊 {completed}/{len(tickers)} | Found: {len(results)}")
            
            result = future.result()
            if result:
                with results_lock:
                    results.append(result)
            time.sleep(0.05)
    
    progress.empty()
    status.empty()
    
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    
    df1 = pd.DataFrame(results).sort_values("Score", ascending=False).head(limit1)
    st.success(f"✅ Stage 1: Found {len(df1)} candidates (Avg: {df1['Score'].mean():.0f})")
    
    df2 = df1[df1['Grade'].isin(['A+', 'A', 'B+', 'B'])].head(limit2)
    st.success(f"🏆 Stage 2: {len(df2)} elite picks")
    
    st.session_state.last_scan_results = (df2, df1)
    st.session_state.last_scan_time = datetime.now()
    st.session_state.last_scan_strategy = strategy
    
    return df1, df2

# ================== UI ==================
st.title("🚀 IDX Power Screener v6.0 ULTIMATE")
st.caption("Professional Multi-Strategy Scanner")

display_ihsg_widget()
tickers = load_tickers()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ SETTINGS")
    st.info(f"📊 Total: {len(tickers)} stocks")
    st.caption(f"🕐 Jakarta: {get_jakarta_time().strftime('%H:%M WIB')}")
    
    st.markdown("---")
    menu = st.radio("📋 STRATEGY", [
        "⚡ SPEED (1-2d)", "🎯 SWING (3-5d)", "💎 VALUE (Undervalued)",
        "⚡ BPJS (Intraday)", "🌙 BSJP (Overnight)", "🔮 BANDAR (Wyckoff)",
        "🔍 SINGLE STOCK"
    ])
    
    st.markdown("---")
    if "SINGLE" not in menu:
        period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
        st.markdown("### 🎯 FILTERS")
        limit1 = st.slider("Stage 1: Top N", 20, 100, 50, 10)
        limit2 = st.slider("Stage 2: Elite", 5, 30, 10, 5)
        st.session_state.strict_mode = st.checkbox("Strict Mode", value=False)
    
    st.markdown("---")
    st.caption("v6.0 ULTIMATE")

# Main content
if "SINGLE" in menu:
    st.markdown("### 🔍 SINGLE STOCK ANALYSIS")
    
    selected = st.selectbox("Select Stock", [t.replace('.JK', '') for t in tickers])
    strategy_single = st.selectbox("Strategy", ["General", "BPJS", "BSJP", "Bandar", "Swing", "Value"])
    period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
    
    if st.button("🔍 ANALYZE", type="primary"):
        ticker_full = selected if selected.endswith('.JK') else f"{selected}.JK"
        
        with st.spinner(f"Analyzing {selected}..."):
            df = fetch_data(ticker_full, period)
            
            if df is None:
                st.error("❌ Failed to fetch data")
            else:
                st.markdown("### 📊 CHART")
                chart = create_chart(df, selected)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                result = process_ticker(ticker_full, strategy_single, period)
                
                if result is None:
                    st.warning("⚠️ Stock rejected by filters")
                else:
                    st.markdown(f"## 💎 {result['Ticker']}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Price", f"Rp {result['Price']:,.0f}")
                    c2.metric("Score", f"{result['Score']}/100")
                    c3.metric("Confidence", f"{result['Confidence']}%")
                    c4.metric("Grade", result['Grade'])
                    
                    c5, c6 = st.columns(2)
                    c5.metric("Trend", result['Trend'])
                    c6.metric("Signal", result['Signal'])
                    
                    tp3_text = f"• **TP3:** Rp {result['TP3']:,.0f}\n" if result.get('TP3') else ""
                    
                    st.success(f"""
                    **🎯 TRADE PLAN ({strategy_single}):**
                    
                    • **Entry Ideal:** Rp {result['Entry']:,.0f}
                    • **Entry Aggressive:** Rp {result['EntryAggressive']:,.0f}
                    
                    • **TP1:** Rp {result['TP1']:,.0f}
                    • **TP2:** Rp {result['TP2']:,.0f}
                    {tp3_text}• **Stop Loss:** Rp {result['SL']:,.0f}
                    
                    💰 **R/R:** {result['R/R']:.2f}:1 | 📊 **Vol:** {result['Vol']:.1f}%
                    
                    **Signal:** {result['Signal']} | **Trend:** {result['Trend']}
                    """)
                    
                    with st.expander("📋 Details"):
                        for k, v in result['Details'].items():
                            st.caption(f"• **{k}**: {v}")

else:
    # Extract strategy name
    if "BPJS" in menu:
        strategy = "BPJS"
        st.markdown("### ⚡ BPJS - Beli Pagi Jual Sore")
        st.info("Entry: 09:00-09:30 | Exit: Same day 14:30-15:15 | Target: 3-5%")
    elif "BSJP" in menu:
        strategy = "BSJP"
        st.markdown("### 🌙 BSJP - Beli Sore Jual Pagi")
        st.info("Entry: 14:00-15:20 | Exit: Next morning 09:30-10:30 | Target: 3-5%")
    elif "BANDAR" in menu:
        strategy = "Bandar"
        st.markdown("### 🔮 BANDAR - Wyckoff Smart Money")
        st.info("🟢 ACCUMULATION = BUY | 🚀 MARKUP = HOLD | 🔴 DISTRIBUTION = SELL")
    elif "SWING" in menu:
        strategy = "Swing"
        st.markdown("### 🎯 SWING - Hold 3-5 Days")
        st.info("Holding: 3-5 hari | Target: 6-15% | Focus: Trend + Momentum")
    elif "VALUE" in menu:
        strategy = "Value"
        st.markdown("### 💎 VALUE - Undervalued Gems")
        st.info("Focus: Price < Rp 1000 | Oversold | Near support | Hold 5-10 hari")
    else:
        strategy = "General"
        st.markdown("### ⚡ SPEED - Quick 1-2 Days")
        st.info("Holding: 1-2 hari max | Target: 4-7% | Exit cepat")
    
    if st.button("🚀 START SCAN", type="primary"):
        df1, df2 = scan_stocks(tickers, strategy, period, limit1, limit2)
        
        if df2.empty:
            st.warning("⚠️ No A/B grade setups found")
            if not df1.empty:
                with st.expander(f"📊 Stage 1: {len(df1)} candidates"):
                    st.dataframe(df1.drop('Details', axis=1), use_container_width=True)
        else:
            st.markdown(f"### 🏆 TOP {len(df2)} PICKS")
            
            # Download button
            if not df2.empty:
                csv = df2.drop('Details', axis=1).to_csv(index=False)
                st.download_button(
                    "💾 Download CSV",
                    csv,
                    f"IDX_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )
            
            for _, row in df2.iterrows():
                emoji = "⭐" if row['Grade'] in ['A+', 'A'] else "💎"
                with st.expander(
                    f"{emoji} **{row['Ticker']}** | Grade **{row['Grade']}** | "
                    f"Score: {row['Score']}/100 | {row['Signal']}",
                    expanded=True
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Price", f"Rp {row['Price']:,.0f}")
                    c2.metric("Score", row['Score'])
                    c3.metric("Confidence", f"{row['Confidence']}%")
                    c4.metric("Grade", row['Grade'])
                    
                    c5, c6 = st.columns(2)
                    c5.metric("Trend", row['Trend'])
                    c6.metric("Signal", row['Signal'])
                    
                    tp3_text = f"• **TP3:** Rp {row['TP3']:,.0f}\n" if row.get('TP3') else ""
                    
                    st.success(f"""
                    **🎯 TRADE PLAN:**
                    
                    • **Entry Ideal:** Rp {row['Entry']:,.0f}
                    • **Entry Aggressive:** Rp {row['EntryAggressive']:,.0f}
                    
                    • **TP1:** Rp {row['TP1']:,.0f}
                    • **TP2:** Rp {row['TP2']:,.0f}
                    {tp3_text}• **Stop Loss:** Rp {row['SL']:,.0f}
                    
                    💰 R/R: {row['R/R']:.2f}:1
                    """)
                    
                    st.markdown("**Analysis:**")
                    for k, v in row['Details'].items():
                        st.caption(f"• **{k}**: {v}")

st.markdown("---")
st.caption("🚀 IDX Power Screener v6.0 ULTIMATE | Educational purposes only")
