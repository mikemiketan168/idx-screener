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
from io import StringIO

st.set_page_config(page_title="IDX Power Screener v4.5 ENHANCED", page_icon="🚀", layout="wide")

# ============= SESSION STATE INITIALIZATION =============
# Initialize session state for persistence (lock screen safe!)
if 'last_scan_results' not in st.session_state:
    st.session_state.last_scan_results = None
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'last_scan_strategy' not in st.session_state:
    st.session_state.last_scan_strategy = None
if 'scan_count' not in st.session_state:
    st.session_state.scan_count = 0

# ============= IHSG MARKET WIDGET =============
@st.cache_data(ttl=180)  # Cache for 3 minutes
def fetch_ihsg_data():
    """Fetch IHSG (Jakarta Composite Index) data"""
    try:
        import yfinance as yf
        ihsg = yf.Ticker("^JKSE")
        hist = ihsg.history(period="1d")
        
        if not hist.empty:
            current = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[-1]
            high = hist['High'].iloc[-1]
            low = hist['Low'].iloc[-1]
            change = current - open_price
            change_pct = (change / open_price) * 100
            
            return {
                'price': current,
                'change': change,
                'change_pct': change_pct,
                'high': high,
                'low': low,
                'status': 'up' if change >= 0 else 'down'
            }
    except:
        pass
    return None

def display_ihsg_widget():
    """Display IHSG market overview widget"""
    ihsg = fetch_ihsg_data()
    
    if ihsg:
        status_emoji = "🟢" if ihsg['status'] == 'up' else "🔴"
        status_text = "BULLISH" if ihsg['status'] == 'up' else "BEARISH"
        
        # Determine market condition
        if ihsg['change_pct'] > 1.5:
            condition = "🔥 Strong uptrend - Good for momentum!"
            guidance = "✅ Excellent for SPEED/SWING trades"
        elif ihsg['change_pct'] > 0.5:
            condition = "📈 Moderate uptrend - Good conditions"
            guidance = "✅ Good for all strategies"
        elif ihsg['change_pct'] > -0.5:
            condition = "➡️ Sideways - Mixed conditions"
            guidance = "⚠️ Be selective, use tight stops"
        elif ihsg['change_pct'] > -1.5:
            condition = "📉 Moderate downtrend - Caution"
            guidance = "⚠️ Focus on VALUE plays, avoid SPEED"
        else:
            condition = "🔻 Strong downtrend - High risk"
            guidance = "❌ Consider sitting out or very selective"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); 
                    padding: 15px; border-radius: 10px; margin-bottom: 20px;
                    border-left: 5px solid {"#22c55e" if ihsg['status'] == 'up' else "#ef4444"}'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h3 style='margin: 0; color: white;'>📊 MARKET OVERVIEW</h3>
                    <p style='margin: 5px 0; color: #e0e7ff; font-size: 0.9em;'>
                        Jakarta Composite Index
                    </p>
                </div>
                <div style='text-align: right;'>
                    <h2 style='margin: 0; color: white;'>
                        {status_emoji} {ihsg['price']:,.2f}
                    </h2>
                    <p style='margin: 5px 0; color: {"#22c55e" if ihsg['status'] == 'up' else "#ef4444"}; 
                              font-size: 1.1em; font-weight: bold;'>
                        {ihsg['change']:+,.2f} ({ihsg['change_pct']:+.2f}%)
                    </p>
                </div>
            </div>
            <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.2);'>
                <p style='margin: 3px 0; color: #e0e7ff; font-size: 0.85em;'>
                    📊 High: {ihsg['high']:,.2f} | Low: {ihsg['low']:,.2f} | Status: <strong>{status_text}</strong>
                </p>
                <p style='margin: 3px 0; color: #fbbf24; font-size: 0.9em;'>
                    {condition}
                </p>
                <p style='margin: 3px 0; color: #a5b4fc; font-size: 0.85em;'>
                    {guidance}
                </p>
                <p style='margin: 5px 0 0 0; color: #94a3b8; font-size: 0.75em;'>
                    ⏰ Last update: {datetime.now().strftime('%H:%M:%S')} WIB | 🔄 Auto-refresh: 3 min
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📊 IHSG data temporarily unavailable")


# ============= LOAD TICKERS =============
def load_tickers():
    try:
        if os.path.exists("idx_stocks.json"):
            with open("idx_stocks.json", "r") as f:
                data = json.load(f)
            tickers = data.get("tickers", [])
            return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except:
        pass
    
    return ["AALI.JK","ABBA.JK","ABMM.JK","ACES.JK","ADRO.JK","AGII.JK","AGRO.JK","AIMS.JK",
            "AKKU.JK","AKPI.JK","AKRA.JK","AKSI.JK","ALDO.JK","ALKA.JK","ALMI.JK","ALTO.JK",
            "AMMN.JK","AMRT.JK","ANDI.JK","ANTM.JK","APLI.JK","APLN.JK","ARGO.JK","ARII.JK",
            "ARNA.JK","ARTA.JK","ARTI.JK","ASII.JK","ASMI.JK","ASPI.JK","ASRI.JK","ASSA.JK",
            "AUTO.JK","AVIA.JK","AYLS.JK","BABP.JK","BAJA.JK","BBCA.JK","BBHI.JK","BBKP.JK",
            "BBNI.JK","BBRI.JK","BBTN.JK","BCAP.JK","BCIP.JK","BDMN.JK","BEEF.JK","BELL.JK",
            "BEST.JK","BFIN.JK","BGTG.JK","BHAT.JK","BHIT.JK","BIKA.JK","BIMA.JK","BIPP.JK",
            "BIRD.JK","BISI.JK","BJBR.JK","BJTM.JK","BLTA.JK","BLTZ.JK","BLUE.JK","BMAS.JK",
            "BMRI.JK","BMSR.JK","BMTR.JK","BNBA.JK","BNGA.JK","BNII.JK","BNLI.JK","BOGA.JK",
            "BOLT.JK","BOSS.JK","BPFI.JK","BPTR.JK","BRAM.JK","BREN.JK","BRIS.JK","BRMS.JK",
            "BRNA.JK","BRPT.JK","BSDE.JK","BSIM.JK","BSSR.JK","BTEK.JK","BTEL.JK","BTPN.JK",
            "BTPS.JK","BUDI.JK","BUKA.JK","BULL.JK","BUMI.JK","BUVA.JK","BVIC.JK","BWPT.JK",
            "BYAN.JK","CAKK.JK","CAMP.JK","CANI.JK","CARE.JK","CARS.JK","CASA.JK","CASS.JK",
            "CEKA.JK","CENT.JK","CFIN.JK","CINT.JK","CITA.JK","CITY.JK","CKRA.JK","CLAY.JK",
            "CLPI.JK","CMNP.JK","CMNT.JK","COAL.JK","COCO.JK","COWL.JK","CPIN.JK","CPRI.JK",
            "CPRO.JK","CSAP.JK","CTBN.JK","CTRA.JK","CTTH.JK","DADA.JK","DAYA.JK","DCII.JK",
            "DEWA.JK","DGIK.JK","DGNS.JK","DIGI.JK","DILD.JK","DIVA.JK","DLTA.JK","DMAS.JK",
            "DMND.JK","DNAR.JK","DNET.JK","DOID.JK","DPNS.JK","DPUM.JK","DRMA.JK","DSSA.JK",
            "DUCK.JK","DVLA.JK","DYAN.JK","EAST.JK","ECII.JK","EDGE.JK","EKAD.JK","ELSA.JK",
            "ELTY.JK","EMDE.JK","EMTK.JK","ENRG.JK","ENVY.JK","EPMT.JK","ERAA.JK","ESSA.JK",
            "ESTE.JK","EURO.JK","EXCL.JK","FAST.JK","FILM.JK","FINN.JK","FIRE.JK","FISH.JK",
            "FITT.JK","FOOD.JK","FORU.JK","FPNI.JK","FREN.JK","FUJI.JK","GADO.JK","GEMA.JK",
            "GEMS.JK","GGRM.JK","GGRP.JK","GHON.JK","GIAA.JK","GJTL.JK","GMFI.JK","GMTD.JK",
            "GOLD.JK","GOLL.JK","GOOD.JK","GOTO.JK","GPRA.JK","GRIA.JK","GSMF.JK","GWSA.JK",
            "HADE.JK","HAIS.JK","HEAL.JK","HERO.JK","HEXA.JK","HILL.JK","HMSP.JK","HOME.JK",
            "HOMI.JK","HOPE.JK","HOTL.JK","HRME.JK","HRTA.JK","HRUM.JK","IATA.JK","IBST.JK",
            "ICBP.JK","ICON.JK","IDEA.JK","IFII.JK","IGAR.JK","IIKP.JK","IKAI.JK","IKAN.JK",
            "IKBI.JK","IMAS.JK","IMJS.JK","IMPC.JK","INAF.JK","INAR.JK","INCO.JK","INDF.JK",
            "INDO.JK","INDR.JK","INDS.JK","INET.JK","INOV.JK","INPC.JK","INPP.JK","INPS.JK",
            "INRU.JK","INTA.JK","INTD.JK","INTP.JK","IPAC.JK","IPCC.JK","IPCM.JK","IPOL.JK",
            "IPPE.JK","IPTV.JK","IRRA.JK","ISAT.JK","ISSP.JK","ITMA.JK","ITMG.JK","JAST.JK",
            "JAWA.JK","JAYA.JK","JECC.JK","JGLE.JK","JKON.JK","JKSW.JK","JMAS.JK","JMTO.JK",
            "JPFA.JK","JPRT.JK","JRPT.JK","JSKY.JK","JSMR.JK","JSPT.JK","JTPE.JK","KAEF.JK",
            "KARW.JK","KAYU.JK","KBAG.JK","KBLI.JK","KBLM.JK","KBLV.JK","KBRI.JK","KDSI.JK",
            "KEEN.JK","KEJU.JK","KIAS.JK","KICI.JK","KIJA.JK","KINO.JK","KIOS.JK","KJEN.JK",
            "KKGI.JK","KLBF.JK","KOBX.JK","KOIN.JK","KOKA.JK","KONI.JK","KOPI.JK","KOTA.JK",
            "KPAS.JK","KPIG.JK","KRAH.JK","KRAS.JK","KREN.JK","LAND.JK","LAPD.JK","LAUT.JK",
            "LEAD.JK","LIFE.JK","LINK.JK","LION.JK","LMAS.JK","LMPI.JK","LMSH.JK","LPCK.JK",
            "LPGI.JK","LPIN.JK","LPKR.JK","LPLI.JK","LPPF.JK","LPPS.JK","LRNA.JK","LSIP.JK",
            "LTLS.JK","LUCK.JK","MAIN.JK","MAMI.JK","MANM.JK","MAPB.JK","MAPI.JK","MARK.JK",
            "MASB.JK","MAYA.JK","MBAP.JK","MBMA.JK","MBSS.JK","MBTO.JK","MCAS.JK","MCOL.JK",
            "MCOR.JK","MDIA.JK","MDKA.JK","MDKI.JK","MDLN.JK","MDRN.JK","MEDC.JK","MEDP.JK",
            "MEGA.JK","MERK.JK","META.JK","MFIN.JK","MFMI.JK","MGNA.JK","MGLV.JK","MGRO.JK",
            "MICE.JK","MIDI.JK","MIKA.JK","MINA.JK","MITI.JK","MKNT.JK","MKPI.JK","MLBI.JK",
            "MLIA.JK","MLPL.JK","MLPT.JK","MMLP.JK","MNCN.JK","MOLI.JK","MPMX.JK","MPPA.JK",
            "MPRO.JK","MRAT.JK","MREI.JK","MRSV.JK","MSJA.JK","MSKY.JK","MTDL.JK","MTFN.JK",
            "MTLA.JK","MTRA.JK","MTSM.JK","MYOH.JK","MYOR.JK","MYRX.JK","NASA.JK","NASI.JK",
            "NATO.JK","NCKL.JK","NETV.JK","NFCX.JK","NICK.JK","NIPS.JK","NISP.JK","NOBU.JK",
            "NRCA.JK","NUSA.JK","OASA.JK","OBMD.JK","OCAP.JK","OILS.JK","OKUR.JK","OMRE.JK",
            "OOMA.JK","OPLK.JK","OPMS.JK","PACK.JK","PALM.JK","PANI.JK","PANR.JK","PANS.JK",
            "PCAR.JK","PBRX.JK","PDES.JK","PEHA.JK","PEJA.JK","PENS.JK","PGAS.JK","PGEO.JK",
            "PGLI.JK","PGUN.JK","PICO.JK","PJAA.JK","PKPK.JK","PLAN.JK","PLIN.JK","PNBN.JK",
            "PNBS.JK","PNGO.JK","PNIN.JK","PNLF.JK","PNSE.JK","POWR.JK","PPGL.JK","PPRE.JK",
            "PPRO.JK","PRAS.JK","PRDA.JK","PRIM.JK","PSAB.JK","PSDN.JK","PSGO.JK","PSKT.JK",
            "PSSI.JK","PTBA.JK","PTDU.JK","PTMP.JK","PTPP.JK","PTPW.JK","PTRO.JK","PTSN.JK",
            "PUDP.JK","PURA.JK","PURE.JK","PURW.JK","PWON.JK","PWSI.JK","PYFA.JK","PZZA.JK",
            "RAJA.JK","RALS.JK","RANC.JK","RBMS.JK","RDTX.JK","REAL.JK","RELI.JK","RICY.JK",
            "RIGS.JK","RISE.JK","RMBA.JK","RIMO.JK","ROCK.JK","RODA.JK","RONY.JK","ROTI.JK",
            "SAFE.JK","SAME.JK","SAMF.JK","SAPX.JK","SATU.JK","SBAT.JK","SBMA.JK","SCCO.JK",
            "SCMA.JK","SCNP.JK","SCPI.JK","SDMU.JK","SDPC.JK","SDRA.JK","SGRO.JK","SHID.JK",
            "SHIP.JK","SILO.JK","SIMA.JK","SIMP.JK","SINI.JK","SIPD.JK","SKBM.JK","SKLT.JK",
            "SKRN.JK","SKYB.JK","SLIS.JK","SMAR.JK","SMBR.JK","SMCB.JK","SMDM.JK","SMDR.JK",
            "SMGR.JK","SMKL.JK","SMMA.JK","SMMT.JK","SMRA.JK","SMRU.JK","SMSM.JK","SNOW.JK",
            "SOBI.JK","SOCI.JK","SOHO.JK","SONA.JK","SOSS.JK","SOTS.JK","SOVA.JK","SPMA.JK",
            "SPTO.JK","SQMI.JK","SRAJ.JK","SRNA.JK","SRSN.JK","SRTG.JK","SSIA.JK","SSMS.JK",
            "SSTM.JK","STAR.JK","STTP.JK","SUGI.JK","SULI.JK","SUNI.JK","SURE.JK","SWAT.JK",
            "SWIN.JK","TALF.JK","TALL.JK","TAMA.JK","TANK.JK","TAPG.JK","TARA.JK","TAXI.JK",
            "TBIG.JK","TBLA.JK","TBMS.JK","TCID.JK","TEBE.JK","TECH.JK","TENT.JK","TFAS.JK",
            "TFCO.JK","TGKA.JK","TGRA.JK","TINS.JK","TIRA.JK","TIRT.JK","TKIM.JK","TKRN.JK",
            "TLKM.JK","TMAS.JK","TMPO.JK","TNCA.JK","TOBA.JK","TOSK.JK","TOTL.JK","TOTO.JK",
            "TOUR.JK","TOYS.JK","TPIA.JK","TPMA.JK","TRJA.JK","TRIL.JK","TRIM.JK","TRIN.JK",
            "TRIO.JK","TRIS.JK","TRST.JK","TRUE.JK","TRUK.JK","TSPC.JK","TUGU.JK","TURI.JK",
            "UANG.JK","UBIX.JK","UCID.JK","UGRO.JK","ULTJ.JK","UNIC.JK","UNIQ.JK","UNIT.JK",
            "UNSP.JK","UNTR.JK","UNVR.JK","URBN.JK","UVCR.JK","VICO.JK","VINS.JK","VIVA.JK",
            "VKTR.JK","VRNA.JK","WAPO.JK","WARA.JK","WEGE.JK","WEHA.JK","WICO.JK","WIFI.JK",
            "WIIM.JK","WINS.JK","WIRA.JK","WMPP.JK","WMUU.JK","WOOD.JK","WOWS.JK","WPAK.JK",
            "WRNA.JK","WSBP.JK","WSKT.JK","WTON.JK","YELO.JK","YPAS.JK","YULE.JK","ZBRA.JK",
            "ZINC.JK","ZONE.JK"]

def get_jakarta_time():
    return datetime.now(timezone(timedelta(hours=7)))

def is_bpjs_time():
    jkt_hour = get_jakarta_time().hour
    return 9 <= jkt_hour < 10

def is_bsjp_time():
    jkt_hour = get_jakarta_time().hour
    return 14 <= jkt_hour < 16

# ============= CHART VISUALIZATION =============
def create_chart(df, ticker, period_days=30):
    """Create interactive chart with technical indicators"""
    try:
        # Get last N days
        df_chart = df.tail(period_days).copy()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{ticker} - Price & EMAs', 'Volume', 'RSI')
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df_chart.index,
                open=df_chart['Open'],
                high=df_chart['High'],
                low=df_chart['Low'],
                close=df_chart['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # EMAs
        colors = {'EMA9': '#2196F3', 'EMA21': '#FF9800', 'EMA50': '#F44336', 'EMA200': '#9E9E9E'}
        for ema in ['EMA9', 'EMA21', 'EMA50', 'EMA200']:
            if ema in df_chart.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_chart.index,
                        y=df_chart[ema],
                        name=ema,
                        line=dict(color=colors[ema], width=1.5)
                    ),
                    row=1, col=1
                )
        
        # Volume bars
        colors_vol = ['#ef5350' if df_chart['Close'].iloc[i] < df_chart['Open'].iloc[i] 
                      else '#26a69a' for i in range(len(df_chart))]
        fig.add_trace(
            go.Bar(
                x=df_chart.index,
                y=df_chart['Volume'],
                name='Volume',
                marker_color=colors_vol,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart['RSI'],
                name='RSI',
                line=dict(color='#9C27B0', width=2)
            ),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
        
        # Update layout
        fig.update_layout(
            height=700,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_dark',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333')
        
        return fig
    except Exception as e:
        return None

# ============= FETCH DATA =============
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, period="6mo"):
    try:
        end = int(datetime.now().timestamp())
        days = {"5d":5,"1mo":30,"3mo":90,"6mo":180,"1y":365}.get(period,180)
        start = end - (days*86400)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        r = requests.get(url, params={"period1":start,"period2":end,"interval":"1d"}, 
                        headers={'User-Agent':'Mozilla/5.0'}, timeout=10)
        
        if r.status_code != 200:
            return None
            
        data = r.json()
        result = data['chart']['result'][0]
        q = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': q['open'],
            'High': q['high'],
            'Low': q['low'],
            'Close': q['close'],
            'Volume': q['volume']
        }, index=pd.to_datetime(result['timestamp'], unit='s'))
        
        df = df.dropna()
        if len(df) < 50:
            return None
        
        # EMAs
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean() if len(df)>=200 else df['Close'].ewm(span=len(df), adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta>0,0).rolling(14).mean()
        loss = -delta.where(delta<0,0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100/(1+rs))
        
        # Volume analysis
        df['VOL_SMA20'] = df['Volume'].rolling(20).mean()
        df['VOL_SMA50'] = df['Volume'].rolling(50).mean()
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA20']
        
        # Momentum
        df['MOM_5D'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['MOM_10D'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        df['MOM_20D'] = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100
        
        # OBV for Wyckoff
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
        df['BB_UPPER'] = df['BB_MID'] + 2*df['BB_STD']
        df['BB_LOWER'] = df['BB_MID'] - 2*df['BB_STD']
        df['BB_WIDTH'] = ((df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MID']) * 100
        
        # Stochastic
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['STOCH_K'] = 100*(df['Close']-low14)/(high14-low14)
        df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
        
        # ATR for volatility
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
        return None

# ============= PRE-FILTERS =============
def apply_liquidity_filter(df, ticker):
    """Strict liquidity requirements"""
    try:
        r = df.iloc[-1]
        price = r['Close']
        vol_avg = df['Volume'].tail(20).mean()
        
        # Minimum price (avoid penny stocks)
        if price < 50:
            return False, "Price too low"
        
        # Minimum average volume
        if vol_avg < 500000:
            return False, "Volume too low"
        
        # Minimum turnover (price * volume)
        turnover = price * vol_avg
        if turnover < 100_000_000:  # 100M IDR daily turnover
            return False, "Turnover too low"
        
        return True, "Passed"
    except:
        return False, "Error"

# ============= ENHANCED SCORING =============
def score_general(df):
    """ULTRA STRICT General Swing Trading Scoring - v4.2"""
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        rejection_reasons = []
        
        # === CRITICAL FILTERS (AUTO-REJECT) ===
        
        # 1. Liquidity check
        passed, reason = apply_liquidity_filter(df, "")
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"
        
        # 2. Strong downtrend rejection (STRICTER)
        if r['Close'] < r['EMA21'] < r['EMA50'] < r['EMA200']:
            return 0, {"Rejected": "Strong downtrend"}, 0, "F"
        
        # 3. Weak positioning - below key EMAs
        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "Below EMA50"}, 0, "F"
        
        # 4. MANDATORY: Negative momentum rejection (NEW!)
        mom_20d = r['MOM_20D']
        if mom_20d < -8:
            return 0, {"Rejected": f"Strong negative momentum ({mom_20d:.1f}%)"}, 0, "F"
        
        # 5. MANDATORY: Weak volume rejection (NEW!)
        vol_ratio = r['VOL_RATIO']
        if vol_ratio < 1.0:
            return 0, {"Rejected": f"Insufficient volume ({vol_ratio:.1f}x)"}, 0, "F"
        
        # === MOMENTUM PENALTY (for -5% to -8%) ===
        if -8 <= mom_20d < -5:
            momentum_penalty = 0.6  # 40% penalty
            details['⚠️ Warning'] = f'Weak momentum {mom_20d:.1f}%'
        elif -5 <= mom_20d < 0:
            momentum_penalty = 0.8  # 20% penalty
            details['⚠️ Warning'] = f'Slight negative momentum {mom_20d:.1f}%'
        else:
            momentum_penalty = 1.0  # No penalty
        
        # === TREND STRENGTH (40 points max) ===
        # Perfect bullish alignment: 9 > 21 > 50 > 200
        ema_alignment = 0
        if r['EMA9'] > r['EMA21']:
            ema_alignment += 1
        if r['EMA21'] > r['EMA50']:
            ema_alignment += 1
        if r['EMA50'] > r['EMA200']:
            ema_alignment += 1
        if r['Close'] > r['EMA9']:
            ema_alignment += 1
        
        if ema_alignment == 4:
            score += 40
            details['Trend'] = '🟢 Perfect alignment'
        elif ema_alignment == 3:
            score += 25
            details['Trend'] = '🟡 Strong'
        elif ema_alignment == 2:
            score += 10
            details['Trend'] = '🟠 Moderate'
        else:
            score += 0
            details['Trend'] = '🔴 Weak'
        
        # === RSI (25 points max) ===
        rsi = r['RSI']
        if 50 <= rsi <= 65:
            score += 25
            details['RSI'] = f'🟢 Ideal {rsi:.0f}'
        elif 45 <= rsi < 50:
            score += 20
            details['RSI'] = f'🟡 Good {rsi:.0f}'
        elif 40 <= rsi < 45:
            score += 10
            details['RSI'] = f'🟠 OK {rsi:.0f}'
        elif rsi > 70:
            score += 0
            details['RSI'] = f'🔴 Overbought {rsi:.0f}'
        elif rsi < 35:
            score += 0
            details['RSI'] = f'🔴 Oversold {rsi:.0f}'
        else:
            score += 5
            details['RSI'] = f'⚪ Neutral {rsi:.0f}'
        
        # === VOLUME (20 points max) ===
        vol_ratio = r['VOL_RATIO']
        if vol_ratio > 2.0:
            score += 20
            details['Volume'] = f'🟢 Surge {vol_ratio:.1f}x'
        elif vol_ratio > 1.5:
            score += 15
            details['Volume'] = f'🟡 Strong {vol_ratio:.1f}x'
        elif vol_ratio > 1.0:
            score += 5
            details['Volume'] = f'🟠 Normal {vol_ratio:.1f}x'
        else:
            score += 0
            details['Volume'] = f'🔴 Weak {vol_ratio:.1f}x'
        
        # === MOMENTUM (15 points max) ===
        # For 1-2 day trading, focus on SHORT-TERM momentum
        mom_5d = r['MOM_5D']
        mom_10d = r['MOM_10D']
        mom_20d = r['MOM_20D']
        
        # Short-term momentum is MORE important for speed trading
        if mom_5d > 3 and mom_10d > 5:
            score += 15
            details['Momentum'] = f'🟢 Strong short-term +{mom_5d:.1f}% (5D)'
        elif mom_5d > 1 and mom_10d > 2:
            score += 10
            details['Momentum'] = f'🟡 Good +{mom_5d:.1f}% (5D)'
        elif mom_5d > 0:
            score += 5
            details['Momentum'] = f'🟠 Positive +{mom_5d:.1f}% (5D)'
        elif mom_20d > 5:
            # Fallback to 20D if 5D weak but 20D strong
            score += 8
            details['Momentum'] = f'🟡 20D momentum +{mom_20d:.1f}%'
        # Note: Negative momentum already handled above
        
        # === APPLY MOMENTUM PENALTY ===
        score = int(score * momentum_penalty)
        
        # === GRADE CALCULATION ===
        if score >= 85:
            grade = "A+"
            conf = 85
        elif score >= 75:
            grade = "A"
            conf = 75
        elif score >= 65:
            grade = "B+"
            conf = 65
        elif score >= 55:
            grade = "B"
            conf = 55
        elif score >= 45:
            grade = "C"
            conf = 45
        else:
            grade = "D"
            conf = max(score, 0)
        
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"

def score_bpjs(df):
    """Enhanced BPJS (Day Trading) Scoring"""
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        
        # Liquidity check
        passed, reason = apply_liquidity_filter(df, "")
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"
        
        # Must be above EMA50 (trend confirmation)
        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "Below EMA50 - No uptrend"}, 0, "F"
        
        # === VOLATILITY (35 points) ===
        atr_pct = r['ATR_PCT']
        if 2.5 < atr_pct < 5:
            score += 35
            details['Volatility'] = f'🟢 PERFECT {atr_pct:.1f}%'
        elif 2 < atr_pct <= 2.5 or 5 <= atr_pct < 6:
            score += 20
            details['Volatility'] = f'🟡 Good {atr_pct:.1f}%'
        elif atr_pct >= 6:
            score += 0
            details['Volatility'] = f'🔴 Too high {atr_pct:.1f}%'
        else:
            score += 0
            details['Volatility'] = f'🔴 Too low {atr_pct:.1f}%'
        
        # === VOLUME SURGE (30 points) ===
        if r['VOL_RATIO'] > 3.0:
            score += 30
            details['Volume'] = f'🟢 HUGE {r["VOL_RATIO"]:.1f}x'
        elif r['VOL_RATIO'] > 2.0:
            score += 20
            details['Volume'] = f'🟡 Strong {r["VOL_RATIO"]:.1f}x'
        elif r['VOL_RATIO'] > 1.5:
            score += 10
            details['Volume'] = f'🟠 Moderate {r["VOL_RATIO"]:.1f}x'
        else:
            return 0, {"Rejected": "Insufficient volume"}, 0, "F"
        
        # === RSI OVERSOLD (20 points) ===
        if 30 < r['RSI'] < 40:
            score += 20
            details['RSI'] = f"🟢 Oversold {r['RSI']:.0f}"
        elif 40 <= r['RSI'] < 50:
            score += 10
            details['RSI'] = f"🟡 OK {r['RSI']:.0f}"
        else:
            score += 0
            details['RSI'] = f"🔴 Not oversold {r['RSI']:.0f}"
        
        # === STOCHASTIC (15 points) ===
        if r['STOCH_K'] < 30:
            score += 15
            details['Stoch'] = f"🟢 Oversold {r['STOCH_K']:.0f}"
        elif r['STOCH_K'] < 50:
            score += 5
            details['Stoch'] = f"🟡 OK {r['STOCH_K']:.0f}"
        
        # === GRADING ===
        if score >= 80:
            grade = "A"
            conf = 80
        elif score >= 65:
            grade = "B"
            conf = 65
        elif score >= 50:
            grade = "C"
            conf = 50
        else:
            grade = "D"
            conf = max(score, 0)
        
        return score, details, conf, grade
    except:
        return 0, {}, 0, "F"

def score_bsjp(df):
    """Enhanced BSJP (Overnight) Scoring"""
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        
        # Liquidity check
        passed, reason = apply_liquidity_filter(df, "")
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"
        
        # Must have uptrend context
        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "No uptrend"}, 0, "F"
        
        # === BB POSITION (35 points) ===
        bb_pos = ((r['Close'] - r['BB_LOWER']) / (r['BB_UPPER'] - r['BB_LOWER'])) * 100
        if bb_pos < 20:
            score += 35
            details['BB'] = f'🟢 Near lower {bb_pos:.0f}%'
        elif bb_pos < 35:
            score += 20
            details['BB'] = f'🟡 Below mid {bb_pos:.0f}%'
        else:
            score += 0
            details['BB'] = f'🔴 Too high {bb_pos:.0f}%'
        
        # === GAP DOWN (30 points) ===
        gap = ((r['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        if -4 < gap < -1:
            score += 30
            details['Gap'] = f'🟢 Perfect {gap:.1f}%'
        elif -1 <= gap < -0.3:
            score += 15
            details['Gap'] = f'🟡 Small {gap:.1f}%'
        elif gap < -4:
            score += 0
            details['Gap'] = f'🔴 Too large {gap:.1f}%'
        else:
            score += 0
            details['Gap'] = f'⚪ No gap {gap:.1f}%'
        
        # === RSI (20 points) ===
        if 25 < r['RSI'] < 40:
            score += 20
            details['RSI'] = f"🟢 Oversold {r['RSI']:.0f}"
        elif 40 <= r['RSI'] < 50:
            score += 10
            details['RSI'] = f"🟡 OK {r['RSI']:.0f}"
        
        # === VOLUME (15 points) ===
        if r['VOL_RATIO'] > 1.5:
            score += 15
            details['Volume'] = f"🟢 Strong {r['VOL_RATIO']:.1f}x"
        elif r['VOL_RATIO'] > 1.0:
            score += 5
            details['Volume'] = f"🟡 Normal {r['VOL_RATIO']:.1f}x"
        
        # === GRADING ===
        if score >= 80:
            grade = "A"
            conf = 80
        elif score >= 65:
            grade = "B"
            conf = 65
        elif score >= 50:
            grade = "C"
            conf = 50
        else:
            grade = "D"
            conf = max(score, 0)
        
        return score, details, conf, grade
    except:
        return 0, {}, 0, "F"

def score_bandar(df):
    """Enhanced Wyckoff Bandar Detection"""
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        
        # Liquidity check
        passed, reason = apply_liquidity_filter(df, "")
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"
        
        # === VOLUME ANALYSIS ===
        vol_ratio_10d = df['Volume'].tail(10).mean() / df['VOL_SMA50'].iloc[-1]
        vol_ratio_recent = df['Volume'].tail(3).mean() / df['VOL_SMA20'].iloc[-1]
        
        # === PRICE ACTION ===
        price_chg_20d = ((r['Close'] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
        price_chg_5d = ((r['Close'] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
        
        # === OBV TREND ===
        obv_chg = ((df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / abs(df['OBV'].iloc[-20])) if df['OBV'].iloc[-20] != 0 else 0
        obv_vs_ema = df['OBV'].iloc[-1] > df['OBV_EMA'].iloc[-1]
        
        # === WYCKOFF PHASE DETECTION ===
        
        # PHASE 1: ACCUMULATION (Best for entry)
        if (vol_ratio_10d > 1.3 and  # Elevated volume
            -2 < price_chg_20d < 5 and  # Sideways/slight up
            obv_chg > 0.05 and  # OBV rising
            obv_vs_ema and  # OBV above its EMA
            r['Close'] > r['EMA50']):  # Above trend
            
            phase = "🟢 ACCUMULATION"
            score = 90
            conf = 85
            details['Phase'] = 'Accumulation Zone'
            details['Signal'] = 'STRONG BUY'
            details['Action'] = 'Enter position'
        
        # PHASE 2: MARKUP (Hold/Trail)
        elif (price_chg_20d > 8 and  # Strong uptrend
              obv_chg > 0.1 and  # OBV strongly rising
              r['Close'] > r['EMA9'] > r['EMA21']):  # Uptrend confirmed
            
            phase = "🚀 MARKUP"
            score = 75
            conf = 70
            details['Phase'] = 'Markup Phase'
            details['Signal'] = 'HOLD/TRAIL'
            details['Action'] = 'Trail stop'
        
        # PHASE 3: DISTRIBUTION (Exit)
        elif (vol_ratio_recent > 2.0 and  # Volume spike
              price_chg_5d < -2 and  # Recent decline
              (r['RSI'] > 70 or price_chg_20d > 15)):  # Overbought or extended
            
            phase = "🔴 DISTRIBUTION"
            score = 20
            conf = 25
            details['Phase'] = 'Distribution Zone'
            details['Signal'] = 'SELL/AVOID'
            details['Action'] = 'Exit now'
        
        # PHASE 4: MARKDOWN (Avoid)
        elif (r['Close'] < r['EMA50'] and
              obv_chg < -0.1):
            
            phase = "⚫ MARKDOWN"
            score = 10
            conf = 15
            details['Phase'] = 'Markdown Phase'
            details['Signal'] = 'AVOID'
            details['Action'] = 'Stay away'
        
        # PHASE 5: RANGING/UNCERTAIN
        else:
            phase = "⚪ RANGING"
            score = 45
            conf = 40
            details['Phase'] = 'No clear phase'
            details['Signal'] = 'WAIT'
            details['Action'] = 'Monitor'
        
        # Add metrics
        details['Vol_10D'] = f'{vol_ratio_10d:.1f}x'
        details['Price_20D'] = f'{price_chg_20d:+.1f}%'
        details['OBV_Trend'] = f'{obv_chg*100:+.1f}%'
        
        # === GRADING ===
        if score >= 80:
            grade = "A"
        elif score >= 60:
            grade = "B"
        elif score >= 40:
            grade = "C"
        else:
            grade = "D"
        
        return score, details, conf, grade
    except Exception as e:
        return 0, {"Error": str(e)}, 0, "F"

def score_swing(df):
    """Swing Trader Scoring (3-5 days holding)"""
    try:
        r = df.iloc[-1]
        score = 0
        details = {}
        
        # Liquidity check
        passed, reason = apply_liquidity_filter(df, "")
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"
        
        # Must be in uptrend
        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "Below EMA50"}, 0, "F"
        
        # MANDATORY: Momentum must be positive
        mom_10d = r['MOM_10D']
        mom_20d = r['MOM_20D']
        if mom_20d < -5:
            return 0, {"Rejected": f"Negative momentum ({mom_20d:.1f}%)"}, 0, "F"
        
        # Volume check
        if r['VOL_RATIO'] < 1.0:
            return 0, {"Rejected": f"Weak volume ({r['VOL_RATIO']:.1f}x)"}, 0, "F"
        
        # === TREND (35 points) ===
        ema_alignment = 0
        if r['EMA9'] > r['EMA21']: ema_alignment += 1
        if r['EMA21'] > r['EMA50']: ema_alignment += 1
        if r['EMA50'] > r['EMA200']: ema_alignment += 1
        if r['Close'] > r['EMA9']: ema_alignment += 1
        
        if ema_alignment == 4:
            score += 35
            details['Trend'] = '🟢 Perfect alignment'
        elif ema_alignment == 3:
            score += 25
            details['Trend'] = '🟡 Strong'
        else:
            score += 10
            details['Trend'] = '🟠 Moderate'
        
        # === RSI (25 points) ===
        rsi = r['RSI']
        if 45 <= rsi <= 65:
            score += 25
            details['RSI'] = f'🟢 Ideal {rsi:.0f}'
        elif 40 <= rsi < 45 or 65 < rsi <= 70:
            score += 15
            details['RSI'] = f'🟡 OK {rsi:.0f}'
        else:
            score += 5
            details['RSI'] = f'🟠 Acceptable {rsi:.0f}'
        
        # === VOLUME (20 points) ===
        vol = r['VOL_RATIO']
        if vol > 1.8:
            score += 20
            details['Volume'] = f'🟢 Strong {vol:.1f}x'
        elif vol > 1.3:
            score += 15
            details['Volume'] = f'🟡 Good {vol:.1f}x'
        else:
            score += 5
            details['Volume'] = f'🟠 Normal {vol:.1f}x'
        
        # === MOMENTUM (20 points) - Focus on 10D + 20D ===
        if mom_10d > 5 and mom_20d > 8:
            score += 20
            details['Momentum'] = f'🟢 Strong 10D:{mom_10d:.1f}% 20D:{mom_20d:.1f}%'
        elif mom_10d > 2 and mom_20d > 4:
            score += 15
            details['Momentum'] = f'🟡 Good 10D:{mom_10d:.1f}% 20D:{mom_20d:.1f}%'
        elif mom_10d > 0 and mom_20d > 0:
            score += 8
            details['Momentum'] = f'🟠 Positive 10D:{mom_10d:.1f}% 20D:{mom_20d:.1f}%'
        
        # === GRADING ===
        if score >= 85:
            grade, conf = "A+", 85
        elif score >= 75:
            grade, conf = "A", 75
        elif score >= 65:
            grade, conf = "B+", 65
        elif score >= 55:
            grade, conf = "B", 55
        else:
            grade, conf = "C", max(score, 0)
        
        return score, details, conf, grade
    except:
        return 0, {}, 0, "F"

def score_value(df):
    """Value Plays Scoring (Undervalued stocks with potential)"""
    try:
        r = df.iloc[-1]
        price = r['Close']
        score = 0
        details = {}
        
        # === VALUE CRITERIA ===
        
        # 1. MUST be affordable (under 1000)
        if price >= 1000:
            return 0, {"Rejected": "Price too high (not undervalued)"}, 0, "F"
        
        # 2. Liquidity check (but more lenient)
        vol_avg = df['Volume'].tail(20).mean()
        if vol_avg < 300000:  # Lower than general filter
            return 0, {"Rejected": "Volume too low"}, 0, "F"
        
        # 3. Must have SOME uptrend context (not in strong downtrend)
        if r['Close'] < r['EMA200']:
            return 0, {"Rejected": "Below 200 EMA"}, 0, "F"
        
        # === UNDERVALUED SIGNALS (40 points) ===
        undervalued_score = 0
        
        # Near support (EMA50 or BB lower)
        dist_to_ema50 = ((r['Close'] - r['EMA50']) / r['EMA50']) * 100
        bb_position = ((r['Close'] - r['BB_LOWER']) / (r['BB_UPPER'] - r['BB_LOWER'])) * 100
        
        if -5 < dist_to_ema50 < 3:
            undervalued_score += 20
            details['Support'] = f'🟢 Near EMA50 ({dist_to_ema50:+.1f}%)'
        elif bb_position < 25:
            undervalued_score += 20
            details['Support'] = f'🟢 Near BB lower ({bb_position:.0f}%)'
        elif -10 < dist_to_ema50 < 5:
            undervalued_score += 10
            details['Support'] = f'🟡 Approaching EMA50'
        
        # Oversold RSI
        rsi = r['RSI']
        if rsi < 35:
            undervalued_score += 20
            details['RSI'] = f'🟢 Oversold {rsi:.0f}'
        elif rsi < 45:
            undervalued_score += 10
            details['RSI'] = f'🟡 Low {rsi:.0f}'
        
        score += undervalued_score
        
        # === REVERSAL SIGNS (35 points) ===
        reversal_score = 0
        
        # Volume increasing (accumulation)
        vol_trend = df['Volume'].tail(5).mean() / df['VOL_SMA20'].iloc[-1]
        if vol_trend > 1.3:
            reversal_score += 20
            details['Volume'] = f'🟢 Increasing {vol_trend:.1f}x'
        elif vol_trend > 1.0:
            reversal_score += 10
            details['Volume'] = f'🟡 Normal {vol_trend:.1f}x'
        
        # Recent green candles (bottoming)
        green_candles = sum(1 for i in range(-5, 0) if df['Close'].iloc[i] > df['Open'].iloc[i])
        if green_candles >= 3:
            reversal_score += 15
            details['Pattern'] = f'🟢 {green_candles}/5 green candles'
        elif green_candles >= 2:
            reversal_score += 8
            details['Pattern'] = f'🟡 {green_candles}/5 green candles'
        
        score += reversal_score
        
        # === AFFORDABILITY BONUS (25 points) ===
        if price < 500:
            score += 25
            details['Price'] = f'🟢 Very affordable Rp {price:.0f}'
        elif price < 750:
            score += 15
            details['Price'] = f'🟡 Affordable Rp {price:.0f}'
        else:
            score += 5
            details['Price'] = f'🟠 Rp {price:.0f}'
        
        # === GRADING ===
        if score >= 75:
            grade, conf = "A", 75
        elif score >= 65:
            grade, conf = "B+", 70
        elif score >= 50:
            grade, conf = "B", 60
        else:
            grade, conf = "C", max(score, 0)
        
        return score, details, conf, grade
    except:
        return 0, {}, 0, "F"

# ============= PROCESS =============
def process_ticker(ticker, strategy, period):
    try:
        df = fetch_data(ticker, period)
        if df is None:
            return None
        
        price = float(df['Close'].iloc[-1])
        
        # Apply strategy scoring
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
        
        # Stricter filter: only keep grade A, B, C+
        if grade not in ['A+', 'A', 'B+', 'B', 'C']:
            return None
        
        # Calculate levels based on strategy
        if strategy == "Swing":
            # SWING TRADING (3-5 days)
            entry = round(price * 0.99, 0)   # -1%
            tp1 = round(entry * 1.06, 0)     # +6% (day 2-3)
            tp2 = round(entry * 1.10, 0)     # +10% (day 4-5)
            tp3 = round(entry * 1.15, 0)     # +15% (extended)
            sl = round(entry * 0.95, 0)      # -5%
        elif strategy == "Value":
            # VALUE PLAYS (5-10 days)
            entry = round(price * 0.98, 0)   # -2% (buy the dip)
            tp1 = round(entry * 1.15, 0)     # +15%
            tp2 = round(entry * 1.25, 0)     # +25%
            tp3 = round(entry * 1.35, 0)     # +35% (moon!)
            sl = round(entry * 0.93, 0)      # -7%
        else:
            # SPEED TRADING (1-2 days) - default
            entry = round(price * 0.995, 0)  # -0.5%
            tp1 = round(entry * 1.04, 0)     # +4% (day 1)
            tp2 = round(entry * 1.07, 0)     # +7% (day 2)
            tp3 = None
            sl = round(entry * 0.97, 0)      # -3%
        
        result = {
            "Ticker": ticker.replace('.JK',''),
            "Price": price,
            "Score": score,
            "Confidence": conf,
            "Grade": grade,
            "Entry": entry,
            "TP1": tp1,
            "TP2": tp2,
            "SL": sl,
            "Details": details
        }
        
        if tp3:
            result["TP3"] = tp3
            
        return result
    except:
        return None

def save_scan_to_session(df2, df1, strategy):
    """Save scan results to session state"""
    st.session_state.last_scan_results = (df2, df1)
    st.session_state.last_scan_time = datetime.now()
    st.session_state.last_scan_strategy = strategy
    st.session_state.scan_count += 1

def display_last_scan_info():
    """Display last scan summary if available"""
    if st.session_state.last_scan_results:
        df2, df1 = st.session_state.last_scan_results
        time_ago = datetime.now() - st.session_state.last_scan_time
        mins_ago = int(time_ago.total_seconds() / 60)
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #064e3b 0%, #065f46 100%); 
                    padding: 12px; border-radius: 8px; margin-bottom: 15px;
                    border-left: 4px solid #10b981;'>
            <p style='margin: 0; color: white; font-weight: bold;'>
                📂 LAST SCAN RESULTS
            </p>
            <p style='margin: 5px 0 0 0; color: #d1fae5; font-size: 0.9em;'>
                Strategy: {st.session_state.last_scan_strategy} | 
                Time: {st.session_state.last_scan_time.strftime('%H:%M:%S')} ({mins_ago} min ago) | 
                Found: {len(df2)} Grade A+/A stocks
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return True
    return False

def create_csv_download(df, strategy):
    """Create CSV download button for results"""
    if not df.empty:
        # Prepare dataframe for export
        export_df = df.copy()
        if 'Details' in export_df.columns:
            export_df = export_df.drop('Details', axis=1)
        
        csv = export_df.to_csv(index=False)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"IDX_{strategy}_scan_{timestamp}.csv"
        
        st.download_button(
            label="💾 Download Results (CSV)",
            data=csv,
            file_name=filename,
            mime="text/csv",
            help="Save scan results to your device"
        )

def scan_stocks(tickers, strategy, period, limit1, limit2):
    st.info(f"🔍 **STAGE 1**: Scanning {len(tickers)} stocks for {strategy}...")
    
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ticker, t, strategy, period): t for t in tickers}
        completed = 0
        
        for future in as_completed(futures):
            completed += 1
            progress.progress(completed / len(tickers))
            status.text(f"📊 {completed}/{len(tickers)} | Found: {len(results)}")
            
            result = future.result()
            if result:
                results.append(result)
            
            time.sleep(0.05)
    
    progress.empty()
    status.empty()
    
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    
    # Stage 1: Top N by score
    df1 = pd.DataFrame(results).sort_values("Score", ascending=False).head(limit1)
    st.success(f"✅ Stage 1: Found {len(df1)} candidates (Avg score: {df1['Score'].mean():.0f})")
    
    # Stage 2: Only A and B grades
    df2 = df1[df1['Grade'].isin(['A+','A','B+','B'])].head(limit2)
    st.success(f"🏆 Stage 2: {len(df2)} elite picks (Avg conf: {df2['Confidence'].mean():.0f}%)")
    
    # Save to session state for persistence
    save_scan_to_session(df2, df1, strategy)
    
    return df1, df2


# ============= UI =============
st.title("🚀 IDX Power Screener v4.5 ENHANCED")
st.caption("3 Trading Styles + IHSG Dashboard + Session Persistence | Lock Screen Safe!")

# Display IHSG Market Overview
display_ihsg_widget()

tickers = load_tickers()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.info(f"📊 Total stocks: {len(tickers)}")
    
    jkt_time = get_jakarta_time()
    st.caption(f"🕐 Jakarta: {jkt_time.strftime('%H:%M WIB')}")
    
    st.markdown("---")
    
    menu = st.radio("📋 Strategy", [
        "⚡ SPEED Trader (1-2d)",
        "🎯 SWING Trader (3-5d)",
        "💎 VALUE Plays (Undervalued)",
        "⚡ BPJS (Day Trading)",
        "🌙 BSJP (Overnight)",
        "🔮 Bandar Tracking",
        "🔍 Single Stock"
    ])
    
    st.markdown("---")
    
    if "Single" not in menu:
        period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
        
        st.markdown("### 🎯 Filtering")
        limit1 = st.slider("Stage 1: Top N", 20, 100, 50, 10)
        limit2 = st.slider("Stage 2: Elite", 5, 30, 10, 5)
        
        st.caption(f"Scan {len(tickers)} → Top {limit1} → Elite {limit2}")
    
    st.markdown("---")
    st.caption("v4.5 ENHANCED - Session Persistence + IHSG")

# ============= MENU HANDLERS =============

if "Single Stock" in menu:
    st.markdown("### 🔍 Single Stock Analysis")
    
    selected = st.selectbox("Select Stock", [t.replace('.JK','') for t in tickers])
    strategy_single = st.selectbox("Strategy", ["General", "BPJS", "BSJP", "Bandar"])
    period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
    
    if st.button("🔍 ANALYZE", type="primary"):
        ticker_full = selected if selected.endswith('.JK') else f"{selected}.JK"
        
        with st.spinner(f"Analyzing {selected}..."):
            df = fetch_data(ticker_full, period)
            
            if df is None:
                st.error("❌ Failed to fetch data")
            else:
                result = process_ticker(ticker_full, strategy_single, period)
                
                if result is None:
                    st.error("❌ Analysis failed or stock rejected by filters")
                    
                    # Show chart anyway for rejected stocks
                    st.markdown("### 📊 Chart (For Reference)")
                    chart = create_chart(df, selected)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    # Show chart FIRST
                    st.markdown("### 📊 Interactive Chart")
                    chart = create_chart(df, selected)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    
                    # Then show analysis
                    st.markdown(f"## 💎 {result['Ticker']}")
                    st.markdown(f"### Grade: **{result['Grade']}**")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price", f"Rp {result['Price']:,.0f}")
                    col2.metric("Score", f"{result['Score']}/100")
                    col3.metric("Confidence", f"{result['Confidence']}%")
                    
                    st.success(f"""
                    **🎯 SPEED TRADING PLAN (1-2 Days):**
                    
                    **Entry:** Rp {result['Entry']:,.0f} (-0.5%)
                    **TP1 (Day 1):** Rp {result['TP1']:,.0f} (+4%)
                    **TP2 (Day 2):** Rp {result['TP2']:,.0f} (+7%)
                    **Stop Loss:** Rp {result['SL']:,.0f} (-3%)
                    
                    💡 **Exit dalam 1-2 hari maksimal!**
                    """)
                    
                    st.markdown("**Technical Analysis:**")
                    for k, v in result['Details'].items():
                        st.caption(f"• **{k}**: {v}")

elif "BPJS" in menu:
    st.markdown("### ⚡ BPJS - Beli Pagi Jual Sore")
    
    if is_bpjs_time():
        st.success("✅ OPTIMAL TIME! (09:00-10:00 WIB)")
    else:
        st.warning("⏰ Best time: 09:00-10:00 WIB")
    
    st.info("""
    **Strategy:**
    - Entry: 09:00-09:30 WIB (first 30 min)
    - Exit: Same day 14:30-15:15
    - Target: 3-5% intraday
    - Focus: Oversold stocks with volume surge
    - Risk: High volatility, fast execution needed
    """)
    
    if st.button("🚀 SCAN BPJS", type="primary"):
        df1, df2 = scan_stocks(tickers, "BPJS", period, limit1, limit2)
        
        if df2.empty:
            st.warning("⚠️ No A/B grade BPJS setups found today")
            if not df1.empty:
                with st.expander(f"📊 Stage 1: {len(df1)} Lower-Grade Candidates"):
                    st.dataframe(df1, use_container_width=True)
        else:
            st.markdown(f"### 🏆 TOP {len(df2)} BPJS PICKS")
            
            for idx, row in df2.iterrows():
                emoji = "⚡" if row['Grade'] in ['A+','A'] else "🔸"
                
                with st.expander(f"{emoji} **{row['Ticker']}** | Grade **{row['Grade']}** | Score: {row['Score']}/100", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"Rp {row['Price']:,.0f}")
                    col2.metric("Score", f"{row['Score']}")
                    col3.metric("Confidence", f"{row['Confidence']}%")
                    col4.metric("Grade", row['Grade'])
                    
                    st.success(f"""
                    **🎯 BUY NOW:** Rp {row['Price']:,.0f} - {row['Entry']:,.0f}
                    **💰 Target:** Rp {row['TP1']:,.0f} (+8%)
                    **🛑 Stop Loss:** Rp {row['SL']:,.0f} (-6%)
                    **⏰ EXIT by 15:00 WIB LATEST!**
                    """)
                    
                    st.markdown("**Analysis:**")
                    for k, v in row['Details'].items():
                        st.caption(f"• **{k}**: {v}")
            
            if not df1.empty:
                with st.expander(f"📊 Stage 1: All {len(df1)} Candidates"):
                    st.dataframe(df1, use_container_width=True)

elif "BSJP" in menu:
    st.markdown("### 🌙 BSJP - Beli Sore Jual Pagi")
    
    if is_bsjp_time():
        st.success("✅ OPTIMAL TIME! (14:00-15:30 WIB)")
    else:
        st.warning("⏰ Best time: 14:00-15:30 WIB")
    
    st.info("""
    **Strategy:**
    - Entry: 14:00-15:20 WIB (gap down stocks)
    - Exit: Next morning 09:30-10:30
    - Target: 3-5% gap recovery
    - Focus: Oversold at BB lower, gap down
    - Risk: Overnight holding risk
    """)
    
    if st.button("🚀 SCAN BSJP", type="primary"):
        df1, df2 = scan_stocks(tickers, "BSJP", period, limit1, limit2)
        
        if df2.empty:
            st.warning("⚠️ No A/B grade BSJP setups found")
            if not df1.empty:
                with st.expander(f"📊 Stage 1: {len(df1)} Lower-Grade Candidates"):
                    st.dataframe(df1, use_container_width=True)
        else:
            st.markdown(f"### 🏆 TOP {len(df2)} BSJP PICKS")
            
            for idx, row in df2.iterrows():
                emoji = "🌙" if row['Grade'] in ['A+','A'] else "🔸"
                
                with st.expander(f"{emoji} **{row['Ticker']}** | Grade **{row['Grade']}** | Score: {row['Score']}/100", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"Rp {row['Price']:,.0f}")
                    col2.metric("Score", f"{row['Score']}")
                    col3.metric("Confidence", f"{row['Confidence']}%")
                    col4.metric("Grade", row['Grade'])
                    
                    st.success(f"""
                    **🎯 BUY before close:** Rp {row['Price']:,.0f} - {row['Entry']:,.0f}
                    **💰 Target:** Rp {row['TP1']:,.0f} (+8%)
                    **🛑 Stop Loss:** Rp {row['SL']:,.0f} (-6%)
                    **⏰ SELL tomorrow 09:30-10:30!**
                    """)
                    
                    st.markdown("**Analysis:**")
                    for k, v in row['Details'].items():
                        st.caption(f"• **{k}**: {v}")
            
            if not df1.empty:
                with st.expander(f"📊 Stage 1: All {len(df1)} Candidates"):
                    st.dataframe(df1, use_container_width=True)

elif "Bandar" in menu:
    st.markdown("### 🔮 Bandar Tracking - Wyckoff Smart Money")
    
    st.info("""
    **Wyckoff Methodology:**
    - 🟢 **ACCUMULATION** = BUY (Smart money loading)
    - 🚀 **MARKUP** = HOLD (Uptrend active)
    - 🔴 **DISTRIBUTION** = SELL (Smart money exiting)
    - ⚫ **MARKDOWN** = AVOID (Downtrend)
    - ⚪ **RANGING** = WAIT (No clear phase)
    
    Timeline: Weeks to months (swing/position trading)
    """)
    
    if st.button("🚀 SCAN BANDAR", type="primary"):
        df1, df2 = scan_stocks(tickers, "Bandar", period, limit1, limit2)
        
        if df2.empty:
            st.warning("⚠️ No strong A/B grade accumulation signals")
            if not df1.empty:
                with st.expander(f"📊 Stage 1: {len(df1)} Weaker Signals"):
                    st.dataframe(df1, use_container_width=True)
        else:
            st.markdown(f"### 🏆 TOP {len(df2)} WYCKOFF SIGNALS")
            
            for idx, row in df2.iterrows():
                phase = row['Details'].get('Phase', '')
                signal = row['Details'].get('Signal', '')
                
                if "Accumulation" in phase:
                    emoji = "🟢"
                    expanded = True
                elif "Markup" in phase:
                    emoji = "🚀"
                    expanded = True
                elif "Distribution" in phase:
                    emoji = "🔴"
                    expanded = False
                else:
                    emoji = "⚪"
                    expanded = False
                
                with st.expander(f"{emoji} **{row['Ticker']}** | {phase} | **{signal}** | Score: {row['Score']}", expanded=expanded):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"Rp {row['Price']:,.0f}")
                    col2.metric("Score", f"{row['Score']}")
                    col3.metric("Grade", row['Grade'])
                    col4.metric("Confidence", f"{row['Confidence']}%")
                    
                    if "Accumulation" in phase:
                        st.success(f"""
                        🟢 **SMART MONEY ACCUMULATING!**
                        
                        **Entry Zone:** Rp {row['Entry']:,.0f}
                        **TP1 (Sell 1/3):** Rp {row['TP1']:,.0f} (+8%)
                        **TP2 (Sell 1/3):** Rp {row['TP2']:,.0f} (+15%)
                        **Trail last 1/3 with 20 EMA**
                        **Stop Loss:** Rp {row['SL']:,.0f} (-6%)
                        
                        💎 **Best phase for entry - HIGH PROBABILITY**
                        """)
                    elif "Markup" in phase:
                        st.info(f"""
                        🚀 **UPTREND ACTIVE - Already moving**
                        
                        If you're in: Hold and trail stop
                        If not: Wait for pullback or next accumulation
                        """)
                    elif "Distribution" in phase:
                        st.error(f"""
                        🔴 **DANGER ZONE - Smart money exiting**
                        
                        **ACTION:** Sell if you're holding
                        **DO NOT:** Enter new positions
                        """)
                    else:
                        st.warning(f"""
                        ⚪ **NO CLEAR DIRECTION**
                        
                        Wait for accumulation phase to develop
                        Monitor for Wyckoff spring or upthrust
                        """)
                    
                    st.markdown("**Wyckoff Analysis:**")
                    for k, v in row['Details'].items():
                        st.caption(f"• **{k}**: {v}")
            
            if not df1.empty:
                with st.expander(f"📊 Stage 1: All {len(df1)} Candidates"):
                    st.dataframe(df1, use_container_width=True)

elif "SWING" in menu:
    st.markdown("### 🎯 SWING TRADER - Hold 3-5 Days")
    
    # Display last scan info
    display_last_scan_info()
    
    st.info("""
    **SWING TRADING Strategy:**
    - **Holding: 3-5 days**
    - **TP1 (+6%):** Day 2-3 target
    - **TP2 (+10%):** Day 4-5 target  
    - **TP3 (+15%):** Extended target (let runners go)
    - **SL (-5%):** Moderate stop loss
    - Focus: Trend continuation + momentum building
    - Entry: After breakout confirmed or pullback to EMA
    - Exit: Trail with 20 EMA for runners
    """)
    
    if st.button("🚀 START SWING SCAN", type="primary"):
        df1, df2 = scan_stocks(tickers, "Swing", period, limit1, limit2)
        
        if df2.empty:
            st.warning("⚠️ No A/B grade swing setups found")
            if not df1.empty:
                with st.expander(f"📊 Stage 1: {len(df1)} Lower-Grade Candidates"):
                    st.dataframe(df1, use_container_width=True)
        else:
            st.markdown(f"### 🏆 TOP {len(df2)} SWING PICKS")
            
            # Add download button
            create_csv_download(df2, "SWING")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Swing Picks", len(df2))
            col2.metric("Avg Score", f"{df2['Score'].mean():.0f}/100")
            col3.metric("Avg Confidence", f"{df2['Confidence'].mean():.0f}%")
            col4.metric("Grade A+/A", len(df2[df2['Grade'].isin(['A+','A'])]))
            
            for idx, row in df2.iterrows():
                emoji = "🎯" if row['Grade'] in ['A+','A'] else "🔹"
                
                with st.expander(f"{emoji} **{row['Ticker']}** | Grade **{row['Grade']}** | Score: {row['Score']}/100", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"Rp {row['Price']:,.0f}")
                    col2.metric("Score", f"{row['Score']}")
                    col3.metric("Confidence", f"{row['Confidence']}%")
                    col4.metric("Grade", row['Grade'])
                    
                    tp3_text = f"**TP3 (Runner):** Rp {row.get('TP3', 0):,.0f} (+15%)\n                    - Trail with 20 EMA\n                    " if 'TP3' in row and row['TP3'] else ""
                    
                    st.success(f"""
                    **🎯 SWING TRADING PLAN (3-5 Days):**
                    
                    **Entry:** Rp {row['Entry']:,.0f} (-1%)
                    
                    **TP1 (Day 2-3):** Rp {row['TP1']:,.0f} (+6%)
                    - Sell 30% position
                    
                    **TP2 (Day 4-5):** Rp {row['TP2']:,.0f} (+10%)
                    - Sell 40% position
                    
                    {tp3_text}
                    **Stop Loss:** Rp {row['SL']:,.0f} (-5%)
                    
                    ⏰ **Hold 3-5 days, let winners run!**
                    """)
                    
                    st.markdown("**Technical Analysis:**")
                    for k, v in row['Details'].items():
                        st.caption(f"• **{k}**: {v}")
            
            if not df1.empty:
                with st.expander(f"📊 Stage 1: All {len(df1)} Candidates"):
                    st.dataframe(df1, use_container_width=True)

elif "VALUE" in menu:
    st.markdown("### 💎 VALUE PLAYS - Undervalued Gems")
    
    # Display last scan info
    display_last_scan_info()
    
    st.info("""
    **VALUE INVESTING Strategy:**
    - **Target: Affordable stocks <Rp 1,000**
    - **Entry: Near support (oversold)**
    - **Holding: 5-10 days (patient)**
    - **TP1 (+15%):** First target
    - **TP2 (+25%):** Second target
    - **TP3 (+35%):** Moon target!
    - **SL (-7%):** Wider stop (reversal needs space)
    
    **Looking for:**
    - Stocks near EMA50 support or BB lower
    - RSI <40 (oversold/undervalued)
    - Volume picking up (smart money accumulating)
    - Reversal patterns forming
    """)
    
    if st.button("🚀 FIND VALUE PLAYS", type="primary"):
        df1, df2 = scan_stocks(tickers, "Value", period, limit1, limit2)
        
        if df2.empty:
            st.warning("⚠️ No quality value plays found")
            if not df1.empty:
                with st.expander(f"📊 Stage 1: {len(df1)} Potential Values"):
                    st.dataframe(df1, use_container_width=True)
        else:
            st.markdown(f"### 💎 TOP {len(df2)} VALUE PLAYS")
            
            # Add download button
            create_csv_download(df2, "VALUE")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Value Picks", len(df2))
            col2.metric("Avg Price", f"Rp {df2['Price'].mean():.0f}")
            col3.metric("Avg Score", f"{df2['Score'].mean():.0f}/100")
            col4.metric("Grade A/B+", len(df2[df2['Grade'].isin(['A','B+'])]))
            
            for idx, row in df2.iterrows():
                emoji = "💎" if row['Grade'] in ['A','B+'] else "💰"
                
                with st.expander(f"{emoji} **{row['Ticker']}** | Rp {row['Price']:,.0f} | Grade **{row['Grade']}** | Score: {row['Score']}/100", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"Rp {row['Price']:,.0f}", help="Affordable entry!")
                    col2.metric("Score", f"{row['Score']}")
                    col3.metric("Confidence", f"{row['Confidence']}%")
                    col4.metric("Grade", row['Grade'])
                    
                    st.success(f"""
                    💎 **VALUE PLAY SETUP (5-10 Days):**
                    
                    **Entry Zone:** Rp {row['Entry']:,.0f} (-2%)
                    *(Buy the dip at support)*
                    
                    **TP1:** Rp {row['TP1']:,.0f} (+15%)
                    - Sell 30% position
                    
                    **TP2:** Rp {row['TP2']:,.0f} (+25%)
                    - Sell 40% position
                    
                    **TP3 (Moon!):** Rp {row.get('TP3', 0):,.0f} (+35%)
                    - Let remaining 30% run!
                    
                    **Stop Loss:** Rp {row['SL']:,.0f} (-7%)
                    
                    ⏰ **Be patient - 5-10 days for reversal!**
                    💡 **This is a VALUE play - buy cheap, sell high!**
                    """)
                    
                    st.markdown("**Why Undervalued:**")
                    for k, v in row['Details'].items():
                        st.caption(f"• **{k}**: {v}")
            
            if not df1.empty:
                with st.expander(f"📊 Stage 1: All {len(df1)} Candidates"):
                    st.dataframe(df1, use_container_width=True)

elif "SPEED" in menu or "Elite" in menu:  # SPEED TRADER (default)
    st.markdown("### ⚡ SPEED TRADER - Quick 1-2 Days")
    
    # Display last scan info if available
    display_last_scan_info()
    
    st.info("""
    **SPEED TRADING Strategy:**
    - **Holding: 1-2 days MAX**
    - **TP1 (+4%):** Day 1 exit target
    - **TP2 (+7%):** Day 2 max target
    - **SL (-3%):** Tight stop loss
    - Focus: Short-term momentum + volume surge
    - Entry: Morning session (09:15-10:00)
    - Exit: Before market close or next morning
    """)
    
    if st.button("🚀 START SPEED SCAN", type="primary"):
        df1, df2 = scan_stocks(tickers, "General", period, limit1, limit2)
        
        if df2.empty:
            st.warning("⚠️ No A/B grade speed setups found")
            if not df1.empty:
                with st.expander(f"📊 Stage 1: {len(df1)} Lower-Grade Candidates"):
                    st.dataframe(df1, use_container_width=True)
        else:
            st.markdown(f"### 🏆 TOP {len(df2)} SPEED PICKS")
            
            # Add download button
            create_csv_download(df2, "SPEED")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Speed Picks", len(df2))
            col2.metric("Avg Score", f"{df2['Score'].mean():.0f}/100")
            col3.metric("Avg Confidence", f"{df2['Confidence'].mean():.0f}%")
            col4.metric("Grade A+/A", len(df2[df2['Grade'].isin(['A+','A'])]))
            
            for idx, row in df2.iterrows():
                emoji = "⚡" if row['Grade'] in ['A+','A'] else "🔹"
                
                with st.expander(f"{emoji} **{row['Ticker']}** | Grade **{row['Grade']}** | Score: {row['Score']}/100", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"Rp {row['Price']:,.0f}")
                    col2.metric("Score", f"{row['Score']}")
                    col3.metric("Confidence", f"{row['Confidence']}%")
                    col4.metric("Grade", row['Grade'])
                    
                    st.success(f"""
                    **⚡ SPEED TRADING PLAN (1-2 Days):**
                    
                    **Entry:** Rp {row['Entry']:,.0f} (buy dip -0.5%)
                    
                    **Day 1 Target:** Rp {row['TP1']:,.0f} (+4%)
                    - Jual 50% posisi
                    - Move SL to breakeven
                    
                    **Day 2 Target:** Rp {row['TP2']:,.0f} (+7%)
                    - Jual sisanya
                    - MAX hold: 2 hari!
                    
                    **Stop Loss:** Rp {row['SL']:,.0f} (-3%)
                    - Tight SL for quick cut
                    
                    ⏰ **EXIT DALAM 1-2 HARI!**
                    """)
                    
                    st.markdown("**Technical Analysis:**")
                    for k, v in row['Details'].items():
                        st.caption(f"• **{k}**: {v}")
            
            if not df1.empty:
                with st.expander(f"📊 Stage 1: All {len(df1)} Candidates"):
                    st.dataframe(df1, use_container_width=True)

else:  # Fallback
    st.error("Please select a trading strategy from the sidebar!")

st.markdown("---")
st.caption("🚀 IDX Power Screener v4.5 ENHANCED | Lock Screen Safe + IHSG Dashboard | Educational purposes only")
