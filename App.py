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

st.set_page_config(page_title="IDX Power Screener v4.0", page_icon="🎯", layout="wide")

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
    return 9 <= get_jakarta_time().hour < 10

def is_bsjp_time():
    return 14 <= get_jakarta_time().hour < 16

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
            'Close':q['close'],'Volume':q['volume'],'High':q['high'],'Low':q['low']
        }, index=pd.to_datetime(result['timestamp'], unit='s'))
        
        df = df.dropna()
        if len(df) < 50:
            return None
        
        df['EMA9'] = df['Close'].ewm(span=9).mean()
        df['EMA21'] = df['Close'].ewm(span=21).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean() if len(df)>=200 else df['Close'].ewm(span=len(df)).mean()
        
        delta = df['Close'].diff()
        gain = delta.where(delta>0,0).rolling(14).mean()
        loss = -delta.where(delta<0,0).rolling(14).mean()
        df['RSI'] = 100 - (100/(1+gain/loss))
        
        df['VOL_SMA'] = df['Volume'].rolling(20).mean()
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA']
        
        df['MOM_5D'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['MOM_20D'] = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100
        
        # OBV for Bandar
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        
        # BB for BSJP
        df['BB_MID'] = df['Close'].rolling(20).mean()
        df['BB_STD'] = df['Close'].rolling(20).std()
        df['BB_LOWER'] = df['BB_MID'] - 2*df['BB_STD']
        
        # Stoch for BPJS
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['STOCH_K'] = 100*(df['Close']-low14)/(high14-low14)
        
        return df
    except:
        return None

# ============= SCORING =============
def score_general(df):
    try:
        r = df.iloc[-1]
        if r['Close'] < r['EMA50'] and r['EMA50'] < r['EMA200']:
            return 0, {"Rejected": "Downtrend"}, 0, "D"
        
        score = 0
        details = {}
        
        if r['Close'] > r['EMA9'] > r['EMA21'] > r['EMA50'] > r['EMA200']:
            score += 40
            details['Trend'] = '🟢 Perfect uptrend'
        elif r['Close'] > r['EMA9'] > r['EMA21']:
            score += 25
            details['Trend'] = '🟡 Short uptrend'
        
        if 45 <= r['RSI'] <= 60:
            score += 30
            details['RSI'] = f'🟢 Sweet {r["RSI"]:.0f}'
        elif 40 <= r['RSI'] <= 70:
            score += 15
            details['RSI'] = f'🟡 OK {r["RSI"]:.0f}'
        
        vol = df['VOL_RATIO'].tail(5).mean()
        if vol > 1.5:
            score += 30
            details['Volume'] = f'🟢 Strong {vol:.1f}x'
        elif vol > 1.0:
            score += 15
            details['Volume'] = f'🟡 Normal {vol:.1f}x'
        
        grade = "A" if score >= 80 else "B" if score >= 60 else "C"
        conf = min(score, 100)
        
        return score, details, conf, grade
    except:
        return 0, {}, 0, "D"

def score_bpjs(df):
    try:
        r = df.iloc[-1]
        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "Downtrend"}, 0, "D"
        
        score = 0
        details = {}
        
        vol_pct = ((df['High']-df['Low'])/df['Low']*100).tail(5).mean()
        if 2 < vol_pct < 6:
            score += 35
            details['Volatility'] = f'🟢 IDEAL {vol_pct:.1f}%'
        
        if r['VOL_RATIO'] > 2.5:
            score += 35
            details['Volume'] = f'🟢 HUGE {r["VOL_RATIO"]:.1f}x'
        elif r['VOL_RATIO'] > 1.8:
            score += 20
            details['Volume'] = f'🟡 Strong {r["VOL_RATIO"]:.1f}x'
        
        if 30 < r['RSI'] < 45:
            score += 30
            details['RSI'] = f"🟢 Oversold {r['RSI']:.0f}"
        
        grade = "A" if score >= 80 else "B" if score >= 60 else "C"
        conf = min(score, 100)
        
        return score, details, conf, grade
    except:
        return 0, {}, 0, "D"

def score_bsjp(df):
    try:
        r = df.iloc[-1]
        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "Downtrend"}, 0, "D"
        
        score = 0
        details = {}
        
        bb_pos = (r['Close']-r['BB_LOWER'])/(r['Close'])*100
        if bb_pos < 3:
            score += 35
            details['BB'] = f'🟢 Extreme {bb_pos:.1f}%'
        
        gap = (r['Close']-df['Close'].iloc[-2])/df['Close'].iloc[-2]*100
        if -3 < gap < -0.5:
            score += 35
            details['Gap'] = f'🟢 Down {gap:.1f}%'
        
        if 30 < r['RSI'] < 50:
            score += 30
            details['RSI'] = f"🟢 Oversold {r['RSI']:.0f}"
        
        grade = "A" if score >= 80 else "B" if score >= 60 else "C"
        conf = min(score, 100)
        
        return score, details, conf, grade
    except:
        return 0, {}, 0, "D"

def score_bandar(df):
    try:
        r = df.iloc[-1]
        
        vol_ratio = df['Volume'].tail(10).mean() / df['Volume'].rolling(30).mean().iloc[-1]
        price_chg = (r['Close'] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100
        obv_trend = (df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / abs(df['OBV'].iloc[-20]) if df['OBV'].iloc[-20] != 0 else 0
        
        details = {}
        
        if vol_ratio > 1.4 and -3 < price_chg < 5 and obv_trend > 0.1:
            phase = "🟢 AKUMULASI"
            score = 95
            conf = 90
            details['Phase'] = 'Accumulation - BUY'
        elif price_chg > 5 and obv_trend > 0.1:
            phase = "🚀 MARKUP"
            score = 80
            conf = 75
            details['Phase'] = 'Markup - HOLD'
        elif vol_ratio > 1.5 and price_chg < -3:
            phase = "🔴 DISTRIBUSI"
            score = 10
            conf = 15
            details['Phase'] = 'Distribution - AVOID'
        else:
            phase = "⚪ SIDEWAYS"
            score = 50
            conf = 50
            details['Phase'] = 'Ranging - WAIT'
        
        details['Volume'] = f'{vol_ratio:.1f}x'
        details['Price'] = f'{price_chg:+.1f}%'
        
        grade = "A" if score >= 80 else "B" if score >= 60 else "C"
        
        return score, details, conf, grade
    except:
        return 0, {}, 0, "D"

# ============= PROCESS =============
def process_ticker(ticker, strategy, period):
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
        else:
            score, details, conf, grade = score_general(df)
        
        if score < 50:
            return None
        
        entry = round(price * 0.98, 0)
        tp1 = round(entry * 1.08, 0)
        sl = round(entry * 0.94, 0)
        
        return {
            "Ticker": ticker.replace('.JK',''),
            "Price": price,
            "Score": score,
            "Confidence": conf,
            "Grade": grade,
            "Entry": entry,
            "TP1": tp1,
            "SL": sl,
            "Details": details
        }
    except:
        return None

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
    
    df1 = pd.DataFrame(results).sort_values("Score", ascending=False).head(limit1)
    st.success(f"✅ Stage 1: Found {len(df1)} candidates")
    
    df2 = df1[df1['Grade'].isin(['A','B'])].head(limit2)
    st.success(f"🏆 Stage 2: {len(df2)} elite picks!")
    
    return df1, df2

# ============= UI =============
st.title("🎯 IDX Power Screener v4.0")
st.caption("2-Stage Filter: Scan All → Top 50 → Top 10 Elite")

tickers = load_tickers()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.info(f"📊 Total stocks: {len(tickers)}")
    
    jkt_time = get_jakarta_time()
    st.caption(f"🕐 Jakarta: {jkt_time.strftime('%H:%M WIB')}")
    
    st.markdown("---")
    
    menu = st.radio("📋 Strategy", [
        "🎯 Elite Screener",
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

# ============= MENU HANDLERS =============

if "Single Stock" in menu:
    st.markdown("### 🔍 Single Stock Analysis")
    
    selected = st.selectbox("Select Stock", [t.replace('.JK','') for t in tickers])
    strategy_single = st.selectbox("Strategy", ["General", "BPJS", "BSJP", "Bandar"])
    period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
    
    if st.button("🔍 ANALYZE", type="primary"):
        ticker_full = selected if selected.endswith('.JK') else f"{selected}.JK"
        result = process_ticker(ticker_full, strategy_single, period)
        
        if result is None:
            st.error("❌ Analysis failed or stock rejected")
        else:
            st.markdown(f"## {result['Ticker']}")
            st.markdown(f"**Grade {result['Grade']}**")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Price", f"Rp {result['Price']:,.0f}")
            col2.metric("Score", f"{result['Score']}/100")
            col3.metric("Confidence", f"{result['Confidence']}%")
            
            st.success(f"""
            **Entry:** Rp {result['Entry']:,.0f}
            **TP1 (+8%):** Rp {result['TP1']:,.0f}
            **SL (-6%):** Rp {result['SL']:,.0f}
            """)
            
            st.markdown("**Analysis:**")
            for k, v in result['Details'].items():
                st.caption(f"• {k}: {v}")

elif "BPJS" in menu:
    st.markdown("### ⚡ BPJS - Beli Pagi Jual Sore")
    
    if is_bpjs_time():
        st.success("✅ OPTIMAL TIME! (09:00-09:30 WIB)")
    else:
        st.warning("⏰ Best time: 09:00-09:30 WIB tomorrow")
    
    st.info("""
    **Strategy:**
    - Entry: 09:00-09:30 WIB
    - Exit: Same day 14:00-15:30
    - Target: 2-5% intraday
    - Focus: High volatility oversold stocks
    """)
    
    if st.button("🚀 SCAN BPJS", type="primary"):
        df1, df2 = scan_stocks(tickers, "BPJS", period, limit1, limit2)
        
        if df2.empty:
            st.warning("⚠️ No BPJS setups found")
            if not df1.empty:
                st.info(f"Stage 1 found {len(df1)} candidates")
                st.dataframe(df1)
        else:
            st.markdown(f"### 🏆 TOP {len(df2)} BPJS PICKS")
            
            for _, row in df2.iterrows():
                emoji = "⚡" if row['Grade']=='A' else "🔸"
                
                with st.expander(f"{emoji} {row['Ticker']} | Grade {row['Grade']} | Score: {row['Score']}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price", f"Rp {row['Price']:,.0f}")
                    col2.metric("Score", f"{row['Score']}/100")
                    col3.metric("Confidence", f"{row['Confidence']}%")
                    
                    st.success(f"""
                    **BUY NOW:** Rp {row['Price']:,.0f}
                    **Target:** Rp {row['TP1']:,.0f} (+8%)
                    **Stop Loss:** Rp {row['SL']:,.0f} (-6%)
                    **EXIT by 15:00 WIB!**
                    """)
                    
                    for k, v in row['Details'].items():
                        st.caption(f"• {k}: {v}")
            
            with st.expander(f"📊 Stage 1: {len(df1)} Candidates"):
                st.dataframe(df1)

elif "BSJP" in menu:
    st.markdown("### 🌙 BSJP - Beli Sore Jual Pagi")
    
    if is_bsjp_time():
        st.success("✅ OPTIMAL TIME! (14:00-15:30 WIB)")
    else:
        st.warning("⏰ Best time: 14:00-15:30 WIB")
    
    st.info("""
    **Strategy:**
    - Entry: 14:00-15:30 WIB (gap down stocks)
    - Exit: Next morning 09:30-10:30
    - Target: 2-4% gap recovery
    - Focus: Oversold near BB lower band
    """)
    
    if st.button("🚀 SCAN BSJP", type="primary"):
        df1, df2 = scan_stocks(tickers, "BSJP", period, limit1, limit2)
        
        if df2.empty:
            st.warning("⚠️ No BSJP setups found")
            if not df1.empty:
                st.info(f"Stage 1 found {len(df1)} candidates")
                st.dataframe(df1)
        else:
            st.markdown(f"### 🏆 TOP {len(df2)} BSJP PICKS")
            
            for _, row in df2.iterrows():
                emoji = "🌙" if row['Grade']=='A' else "🔸"
                
                with st.expander(f"{emoji} {row['Ticker']} | Grade {row['Grade']} | Score: {row['Score']}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price", f"Rp {row['Price']:,.0f}")
                    col2.metric("Score", f"{row['Score']}/100")
                    col3.metric("Confidence", f"{row['Confidence']}%")
                    
                    st.success(f"""
                    **BUY before close:** Rp {row['Price']:,.0f}
                    **Target:** Rp {row['TP1']:,.0f} (+8%)
                    **Stop Loss:** Rp {row['SL']:,.0f} (-6%)
                    **SELL tomorrow 09:30-10:30!**
                    """)
                    
                    for k, v in row['Details'].items():
                        st.caption(f"• {k}: {v}")
            
            with st.expander(f"📊 Stage 1: {len(df1)} Candidates"):
                st.dataframe(df1)

elif "Bandar" in menu:
    st.markdown("### 🔮 Bandar Tracking - Smart Money")
    
    st.info("""
    **Strategy:**
    - Detect: Wyckoff accumulation phases
    - 🟢 AKUMULASI = BUY (best entry)
    - 🚀 MARKUP = HOLD (let it run)
    - 🔴 DISTRIBUSI = AVOID (exit)
    - Timeline: Weeks to months
    """)
    
    if st.button("🚀 SCAN BANDAR", type="primary"):
        df1, df2 = scan_stocks(tickers, "Bandar", period, limit1, limit2)
        
        if df2.empty:
            st.warning("⚠️ No strong accumulation detected")
            if not df1.empty:
                st.info(f"Stage 1 found {len(df1)} candidates")
                st.dataframe(df1)
        else:
            st.markdown(f"### 🏆 TOP {len(df2)} BANDAR SIGNALS")
            
            for _, row in df2.iterrows():
                phase = row['Details'].get('Phase', '')
                
                if "AKUMULASI" in phase or "Accumulation" in phase:
                    emoji = "🟢"
                    color = "success"
                elif "MARKUP" in phase or "Markup" in phase:
                    emoji = "🚀"
                    color = "info"
                else:
                    emoji = "⚪"
                    color = "warning"
                
                with st.expander(f"{emoji} {row['Ticker']} | {phase} | Score: {row['Score']}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price", f"Rp {row['Price']:,.0f}")
                    col2.metric("Score", f"{row['Score']}/100")
                    col3.metric("Grade", row['Grade'])
                    
                    if "AKUMULASI" in phase or "Accumulation" in phase:
                        st.success(f"""
                        🟢 **SMART MONEY ACCUMULATING!**
                        
                        **Entry:** Rp {row['Entry']:,.0f}
                        **TP1:** Rp {row['TP1']:,.0f} (+8%)
                        **TP2:** Rp {int(row['TP1']*1.065):,.0f} (+15%)
                        **SL:** Rp {row['SL']:,.0f} (-6%)
                        
                        💎 Best phase for entry!
                        """)
                    elif "MARKUP" in phase:
                        st.info(f"""
                        🚀 **UPTREND ACTIVE**
                        
                        Already in markup phase
                        Hold with trailing stop
                        """)
                    else:
                        st.warning(f"""
                        ⚪ **NO CLEAR DIRECTION**
                        
                        Wait for better setup
                        """)
                    
                    st.markdown("**Wyckoff Analysis:**")
                    for k, v in row['Details'].items():
                        st.caption(f"• {k}: {v}")
            
            with st.expander(f"📊 Stage 1: {len(df1)} Candidates"):
                st.dataframe(df1)

else:  # Elite Screener
    st.markdown("### 🎯 Elite Screener - General Swing")
    
    st.info("""
    **Strategy:**
    - Multi-factor technical analysis
    - Timeline: 2-5 days swing trades
    - Focus: Strong trend + momentum + volume
    """)
    
    if st.button("🚀 START SCAN", type="primary"):
        df1, df2 = scan_stocks(tickers, "General", period, limit1, limit2)
        
        if df2.empty:
            st.warning("⚠️ No elite stocks found")
            if not df1.empty:
                st.info(f"Stage 1 found {len(df1)} candidates")
                st.dataframe(df1)
        else:
            st.markdown(f"### 🏆 TOP {len(df2)} ELITE PICKS")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Elite", len(df2))
            col2.metric("Avg Score", f"{df2['Score'].mean():.0f}")
            col3.metric("Avg Conf", f"{df2['Confidence'].mean():.0f}%")
            col4.metric("Grade A", len(df2[df2['Grade']=='A']))
            
            for _, row in df2.iterrows():
                emoji = "💎" if row['Grade']=='A' else "🔹" if row['Grade']=='B' else "⚪"
                
                with st.expander(f"{emoji} {row['Ticker']} | Grade {row['Grade']} | Score: {row['Score']}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price", f"Rp {row['Price']:,.0f}")
                    col2.metric("Score", f"{row['Score']}/100")
                    col3.metric("Confidence", f"{row['Confidence']}%")
                    
                    st.success(f"""
                    **Entry:** Rp {row['Entry']:,.0f}
                    **TP1 (Sell 1/3):** Rp {row['TP1']:,.0f} (+8%)
                    **TP2 (Sell 1/3):** Rp {int(row['TP1']*1.065):,.0f} (+15%)
                    **Trail last 1/3 with 20 EMA**
                    **Stop Loss:** Rp {row['SL']:,.0f} (-6%)
                    """)
                    
                    st.markdown("**Analysis:**")
                    for k, v in row['Details'].items():
                        st.caption(f"• {k}: {v}")
            
            with st.expander(f"📊 Stage 1: All {len(df1)} Candidates"):
                st.dataframe(df1, use_container_width=True)

st.markdown("---")
st.caption("🎯 IDX Power Screener v4.0 | Educational purposes only")
