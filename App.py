# app.py — v5.4 Real-Time + Auto-Refresh
#!/usr/bin/env python3
import os, math, time, json
import numpy as np, pandas as pd, requests, streamlit as st
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- auto-refresh component (3rd party) ----
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None  # ditangani di sidebar

st.set_page_config(page_title="IDX Power Screener v5.4 (RT + AutoRefresh)", page_icon="🚀", layout="wide")

# ================= CONSTS / UTILS =================
IDX_TICKS=[(0,200,1),(200,500,2),(500,2000,5),(2000,5000,10),(5000,float("inf"),25)]
SIGNAL_RANK={"Strong Buy":2,"Buy":1,"Hold":0,"Sell":-1}
REALTIME_TTL=15   # detik cache RT
SPARK_CHUNK=50    # symbols per request

def round_to_tick(price: float, mode="nearest")->int:
    if not price or not (price==price) or price<=0: return 0
    tick=1
    for lo,hi,t in IDX_TICKS:
        if lo<=price<hi: tick=t; break
    if mode=="floor": return int(math.floor(price/tick)*tick)
    if mode=="ceil":  return int(math.ceil(price/tick)*tick)
    return int(round(price/tick)*tick)

def normalize_ticker(t:str)->str:
    t=t.strip().upper()
    return t if t.endswith(".JK") else f"{t}.JK"

def format_idr(x: float)->str:
    try: return f"Rp {x:,.0f}".replace(",", ".")
    except: return "-"

def get_jakarta_time()->datetime: return datetime.now(timezone(timedelta(hours=7)))

# ================= SESSION =================
defaults = {
    "last_scan_results": None,          # (df2, df1)
    "last_scan_time": None,
    "last_scan_strategy": None,
    "last_scan_symbols": None,          # list[str] .JK
    "scan_count": 0,
    "last_period": "6mo",
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

# ================= IHSG WIDGET =================
@st.cache_data(ttl=180)
def fetch_ihsg_data():
    try:
        import yfinance as yf
        ihsg=yf.Ticker("^JKSE")
        hist=ihsg.history(period="1d", interval="1d", auto_adjust=False)
        if hist.empty: return None
        cur=float(hist["Close"].iloc[-1]); op=float(hist["Open"].iloc[-1])
        hi=float(hist["High"].iloc[-1]); lo=float(hist["Low"].iloc[-1])
        ch=cur-op; ch_pct=(ch/op)*100 if op else 0
        return {"price":cur,"change":ch,"change_pct":ch_pct,"high":hi,"low":lo,"status":"up" if ch>=0 else "down"}
    except: return None

def display_ihsg_widget():
    ihsg=fetch_ihsg_data()
    if not ihsg: st.info("📊 IHSG data temporarily unavailable"); return
    status_emoji="🟢" if ihsg["status"]=="up" else "🔴"
    status_text="BULLISH" if ihsg["status"]=="up" else "BEARISH"
    if ihsg["change_pct"]>1.5: condition,guide="🔥 Strong uptrend","✅ Excellent for SPEED/SWING"
    elif ihsg["change_pct"]>0.5: condition,guide="📈 Moderate uptrend","✅ Good for most"
    elif ihsg["change_pct"]>-0.5: condition,guide="➡️ Sideways","⚠️ Selective"
    elif ihsg["change_pct"]>-1.5: condition,guide="📉 Moderate downtrend","⚠️ Value only"
    else: condition,guide="🔻 Strong downtrend","❌ Sit out / very selective"
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1e3a8a,#1e40af);padding:15px;border-radius:10px;margin-bottom:20px;border-left:5px solid {"#22c55e" if ihsg['status']=="up" else "#ef4444"}'>
      <div style='display:flex;justify-content:space-between;align-items:center;'>
        <div><h3 style='margin:0;color:white;'>📊 MARKET OVERVIEW</h3>
        <p style='margin:5px 0;color:#e0e7ff'>Jakarta Composite Index</p></div>
        <div style='text-align:right'><h2 style='margin:0;color:white;'>{status_emoji} {ihsg['price']:,.2f}</h2>
        <p style='margin:5px 0;color:{"#22c55e" if ihsg["status"]=="up" else "#ef4444"};font-weight:bold'>{ihsg['change']:+,.2f} ({ihsg['change_pct']:+.2f}%)</p></div>
      </div>
      <div style='margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.2)'>
        <p style='margin:0;color:#e0e7ff'>High {ihsg['high']:,.2f} | Low {ihsg['low']:,.2f} | <b>{status_text}</b></p>
        <p style='margin:0;color:#fbbf24'>{condition}</p>
        <p style='margin:0;color:#a5b4fc'>{guide}</p>
        <p style='margin:0;color:#94a3b8;font-size:12px'>⏰ {datetime.now().strftime('%H:%M:%S')} WIB • Data may be delayed by Yahoo</p>
      </div>
    </div>""", unsafe_allow_html=True)

# ================= TICKERS =================
@st.cache_data(ttl=3600)
def load_tickers()->list[str]:
    try:
        if os.path.exists("idx_stocks.json"):
            with open("idx_stocks.json") as f: data=json.load(f)
            return [normalize_ticker(t) for t in data.get("tickers",[]) if t]
    except: pass
    return ["BBRI.JK","BBCA.JK","TLKM.JK","ASII.JK","ICBP.JK","INDF.JK"]

# ================= EOD FETCH (INDICATORS) =================
def _yahoo_chart_json(ticker:str, period:str)->dict|None:
    end=int(datetime.now().timestamp())
    days={"5d":5,"1mo":30,"3mo":90,"6mo":180,"1y":365}.get(period,180)
    start=end-(days*86400)
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params={"period1":start,"period2":end,"interval":"1d"}
    headers={"User-Agent":"Mozilla/5.0"}
    for i in range(3):
        try:
            r=requests.get(url,params=params,headers=headers,timeout=10)
            if r.status_code==200: return r.json()
        except: pass
        time.sleep(0.4*(i+1))
    return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker:str, period:str="6mo")->pd.DataFrame|None:
    try:
        data=_yahoo_chart_json(ticker,period)
        if not data: return None
        result=data.get("chart",{}).get("result",[None])[0]
        if not result: return None
        q=result["indicators"]["quote"][0]; ts=result.get("timestamp",[])
        if not ts or len(ts)!=len(q["close"]): return None
        df=pd.DataFrame({"Open":q["open"],"High":q["high"],"Low":q["low"],"Close":q["close"],"Volume":q["volume"]},
                        index=pd.to_datetime(ts,unit="s")).dropna()
        if len(df)<50: return None

        df["EMA9"]=df["Close"].ewm(span=9,adjust=False).mean()
        df["EMA21"]=df["Close"].ewm(span=21,adjust=False).mean()
        df["EMA50"]=df["Close"].ewm(span=50,adjust=False).mean()
        df["EMA200"]=df["Close"].ewm(span=min(len(df),200),adjust=False).mean()

        dlt=df["Close"].diff(); gain=dlt.clip(lower=0); loss=-dlt.clip(upper=0)
        ag=gain.ewm(alpha=1/14,adjust=False).mean(); al=loss.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan)
        rs=ag/al; df["RSI"]=100-(100/(1+rs))

        df["VOL_SMA20"]=df["Volume"].rolling(20).mean(); df["VOL_SMA50"]=df["Volume"].rolling(50).mean()
        df["VOL_RATIO"]=df["Volume"]/df["VOL_SMA20"].replace(0,np.nan)

        df["MOM_5D"]=df["Close"].pct_change(5)*100
        df["MOM_10D"]=df["Close"].pct_change(10)*100
        df["MOM_20D"]=df["Close"].pct_change(20)*100

        obv=[0]
        for i in range(1,len(df)):
            if df["Close"].iloc[i]>df["Close"].iloc[i-1]: obv.append(obv[-1]+(df["Volume"].iloc[i] or 0))
            elif df["Close"].iloc[i]<df["Close"].iloc[i-1]: obv.append(obv[-1]-(df["Volume"].iloc[i] or 0))
            else: obv.append(obv[-1])
        df["OBV"]=obv; df["OBV_EMA"]=pd.Series(df["OBV"]).ewm(span=10,adjust=False).mean()

        df["BB_MID"]=df["Close"].rolling(20).mean(); df["BB_STD"]=df["Close"].rolling(20).std()
        df["BB_UPPER"]=df["BB_MID"]+2*df["BB_STD"]; df["BB_LOWER"]=df["BB_MID"]-2*df["BB_STD"]
        df["BB_WIDTH"]=((df["BB_UPPER"]-df["BB_LOWER"])/df["BB_MID"])*100

        low14=df["Low"].rolling(14).min(); high14=df["High"].rolling(14).max()
        df["STOCH_K"]=100*(df["Close"]-low14)/(high14-low14); df["STOCH_D"]=df["STOCH_K"].rolling(3).mean()

        tr1=df["High"]-df["Low"]; tr2=(df["High"]-df["Close"].shift()).abs(); tr3=(df["Low"]-df["Close"].shift()).abs()
        tr=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1); df["ATR"]=tr.rolling(14).mean(); df["ATR_PCT"]=(df["ATR"]/df["Close"])*100

        ema12=df["Close"].ewm(span=12,adjust=False).mean(); ema26=df["Close"].ewm(span=26,adjust=False).mean()
        df["MACD"]=ema12-ema26; df["MACD_SIGNAL"]=df["MACD"].ewm(span=9,adjust=False).mean(); df["MACD_HIST"]=df["MACD"]-df["MACD_SIGNAL"]

        return df.replace([np.inf,-np.inf],np.nan).dropna().copy()
    except: return None

# ================= REAL-TIME (SPARK + QUOTE) =================
def _chunks(lst, n): 
    for i in range(0, len(lst), n): yield lst[i:i+n]

@st.cache_data(ttl=REALTIME_TTL)
def fetch_rt_bulk_spark(symbols: tuple)->dict:
    syms=list(symbols); out={}
    headers={"User-Agent":"Mozilla/5.0"}
    for chunk in _chunks(syms, SPARK_CHUNK):
        url="https://query1.finance.yahoo.com/v7/finance/spark"
        params={"symbols":",".join(chunk),"interval":"1m","range":"1d"}
        try:
            r=requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code!=200: continue
            items=r.json().get("spark",{}).get("result",[])
            for it in items:
                sym=it.get("symbol"); resp=it.get("response",[{}])[0]
                ts=resp.get("timestamp",[]); cl=resp.get("indicators",{}).get("close",[])
                if sym and ts and cl:
                    last=None; last_ts=None
                    for t,v in zip(reversed(ts), reversed(cl)):
                        if v is not None:
                            last=v; last_ts=t; break
                    if last is not None:
                        dt=datetime.fromtimestamp(last_ts, tz=timezone.utc).astimezone(timezone(timedelta(hours=7)))
                        out[sym]={"last":float(last),"last_time_wib":dt.strftime("%Y-%m-%d %H:%M:%S"),"source":"spark"}
        except: 
            continue
        time.sleep(0.05)
    return out

@st.cache_data(ttl=REALTIME_TTL)
def fetch_rt_bulk_quote(symbols: tuple)->dict:
    syms=list(symbols); out={}
    headers={"User-Agent":"Mozilla/5.0"}
    for chunk in _chunks(syms, SPARK_CHUNK):
        url="https://query1.finance.yahoo.com/v7/finance/quote"
        params={"symbols":",".join(chunk)}
        try:
            r=requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code!=200: continue
            for q in r.json().get("quoteResponse",{}).get("result",[]):
                sym=q.get("symbol"); last=q.get("regularMarketPrice"); prev=q.get("regularMarketPreviousClose")
                delay=q.get("exchangeDataDelayedBy",0); ts=q.get("regularMarketTime",None)
                if not sym: continue
                last_time_wib="-"
                if ts:
                    dt=datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(timezone(timedelta(hours=7)))
                    last_time_wib=dt.strftime("%Y-%m-%d %H:%M:%S")
                out[sym]={"last": float(last) if last is not None else None,
                          "prev_close": float(prev) if prev is not None else None,
                          "delay_min": int(delay) if delay is not None else 0,
                          "last_time_wib": last_time_wib,
                          "source":"quote"}
        except:
            continue
        time.sleep(0.05)
    return out

def get_realtime_quotes(symbols: list[str])->dict:
    t=tuple(symbols)
    rt_spark=fetch_rt_bulk_spark(t)
    rt_quote=fetch_rt_bulk_quote(t)
    out={}
    for s in symbols:
        a=rt_spark.get(s, {})
        b=rt_quote.get(s, {})
        out[s]={
            "last": a.get("last", b.get("last")),
            "prev_close": b.get("prev_close"),
            "delay_min": b.get("delay_min", 0),
            "last_time_wib": a.get("last_time_wib", b.get("last_time_wib","-")),
            "source": a.get("source","") or b.get("source","-"),
        }
    return out

def attach_rt_prices(df: pd.DataFrame)->pd.DataFrame:
    if df is None or df.empty or "Ticker" not in df: return df
    symbols=[f"{t}.JK" for t in df["Ticker"].astype(str).tolist()]
    rt=get_realtime_quotes(symbols)
    out=df.copy()
    out["Price"]=out["Ticker"].apply(lambda x: rt.get(f"{x}.JK",{}).get("last", np.nan)).fillna(out.get("Price", np.nan))
    out["PrevClose"]=out["Ticker"].apply(lambda x: rt.get(f"{x}.JK",{}).get("prev_close", np.nan))
    out["DelayMin"]=out["Ticker"].apply(lambda x: rt.get(f"{x}.JK",{}).get("delay_min", 0))
    out["LastTimeWIB"]=out["Ticker"].apply(lambda x: rt.get(f"{x}.JK",{}).get("last_time_wib","-"))
    return out

# ================= CHARTS =================
def create_chart(df:pd.DataFrame, ticker:str, days:int=60):
    try:
        d=df.tail(days).copy()
        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.03,row_heights=[0.6,0.2,0.2],
                          subplot_titles=(f"{ticker} - Price & EMAs","Volume","RSI"))
        fig.add_trace(go.Candlestick(x=d.index,open=d["Open"],high=d["High"],low=d["Low"],close=d["Close"],
                                     name="Price",increasing_line_color="#26a69a",decreasing_line_color="#ef5350"),row=1,col=1)
        colors={"EMA9":"#2196F3","EMA21":"#FF9800","EMA50":"#F44336","EMA200":"#9E9E9E"}
        for ema in ["EMA9","EMA21","EMA50","EMA200"]:
            fig.add_trace(go.Scatter(x=d.index,y=d[ema],name=ema,line=dict(color=colors[ema],width=1.4)),row=1,col=1)
        colors_vol=["#ef5350" if c<o else "#26a69a" for c,o in zip(d["Close"],d["Open"])]
        fig.add_trace(go.Bar(x=d.index,y=d["Volume"],name="Volume",marker_color=colors_vol,showlegend=False),row=2,col=1)
        fig.add_trace(go.Scatter(x=d.index,y=d["RSI"],name="RSI",line=dict(color="#9C27B0",width=2)),row=3,col=1)
        fig.add_hline(y=70,line_dash="dash",line_color="red",opacity=0.5,row=3,col=1)
        fig.add_hline(y=30,line_dash="dash",line_color="green",opacity=0.5,row=3,col=1)
        fig.add_hline(y=50,line_dash="dot",line_color="gray",opacity=0.3,row=3,col=1)
        fig.update_layout(height=700,template="plotly_dark",hovermode="x unified",xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        return fig
    except: return None

# ================= FILTERS / SCORING =================
def apply_liquidity_filter(df:pd.DataFrame):
    try:
        r=df.iloc[-1]; price=float(r["Close"]); vol_avg=float(df["Volume"].tail(20).mean())
        if price<50: return False,"Price too low"
        if vol_avg<500_000: return False,"Volume too low"
        if price*vol_avg<100_000_000: return False,"Turnover too low"
        return True,"Passed"
    except: return False,"Error"

def ema_alignment_score(r:pd.Series):
    pts=int(r["EMA9"]>r["EMA21"])+int(r["EMA21"]>r["EMA50"])+int(r["EMA50"]>r["EMA200"])+int(r["Close"]>r["EMA9"])
    return pts,{4:"🟢 Perfect",3:"🟡 Strong",2:"🟠 Moderate",1:"🔴 Weak",0:"🔴 Weak"}[pts]

def grade_from_score(score:int):
    if score>=85: return "A+",85
    if score>=75: return "A",75
    if score>=65: return "B+",65
    if score>=55: return "B",55
    if score>=45: return "C",45
    return "D",max(score,0)

def score_general(df):
    try:
        ok,why=apply_liquidity_filter(df)
        if not ok: return 0,{"Rejected":why},0,"F"
        r=df.iloc[-1]
        if r["Close"]<r["EMA50"]: return 0,{"Rejected":"Below EMA50"},0,"F"
        if r["Close"]<r["EMA21"]<r["EMA50"]<r["EMA200"]: return 0,{"Rejected":"Strong downtrend"},0,"F"
        score,det=0,{}
        mom20=float(r["MOM_20D"]); volr=float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 0.0
        if mom20<-8: return 0,{"Rejected":f"Strong negative momentum ({mom20:.1f}%)"},0,"F"
        penalty=1.0
        if -8<=mom20<-5: penalty=0.6; det["⚠️"]=f"Weak momentum {mom20:.1f}%"
        elif -5<=mom20<0: penalty=0.8; det["⚠️"]=f"Slight negative momentum {mom20:.1f}%"
        pts,label=ema_alignment_score(r); score+={4:40,3:25,2:10}.get(pts,0); det["Trend"]=label
        rsi=float(r["RSI"]) if not np.isnan(r["RSI"]) else 50.0
        if 50<=rsi<=65: score+=25; det["RSI"]=f"🟢 {rsi:.0f}"
        elif 45<=rsi<50: score+=20; det["RSI"]=f"🟡 {rsi:.0f}"
        elif 40<=rsi<45: score+=10; det["RSI"]=f"🟠 {rsi:.0f}"
        elif rsi>70: det["RSI"]=f"🔴 Overbought {rsi:.0f}"
        elif rsi<35: det["RSI"]=f"🔴 Oversold {rsi:.0f}"
        else: score+=5; det["RSI"]=f"⚪ {rsi:.0f}"
        if volr>2.0: score+=20; det["Vol"]=f"🟢 {volr:.1f}x"
        elif volr>1.5: score+=15; det["Vol"]=f"🟡 {volr:.1f}x"
        elif volr>1.0: score+=5;  det["Vol"]=f"🟠 {volr:.1f}x"
        else: det["Vol"]=f"🔴 {volr:.1f}x"
        m5,m10=float(r["MOM_5D"]),float(r["MOM_10D"])
        if m5>3 and m10>5: score+=15; det["Momentum"]=f"🟢 ST +{m5:.1f}%"
        elif m5>1 and m10>2: score+=10; det["Momentum"]=f"🟡 ST +{m5:.1f}%"
        elif m5>0: score+=5; det["Momentum"]=f"🟠 ST +{m5:.1f}%"
        elif mom20>5: score+=8; det["Momentum"]=f"🟡 20D +{mom20:.1f}%"
        score=int(score*penalty); grade,conf=grade_from_score(score)
        return score,det,conf,grade
    except Exception as e: return 0,{"Error":str(e)},0,"F"

# ================= TREND / SIGNAL / PLAN =================
def detect_trend(r:pd.Series)->str:
    p,e9,e21,e50,e200=map(float,[r["Close"],r["EMA9"],r["EMA21"],r["EMA50"],r["EMA200"]])
    if p>e9>e21>e50>e200: return "Strong Uptrend"
    if p>e50 and e9>e21>e50: return "Uptrend"
    if abs(p-e50)/p<0.03: return "Sideways"
    return "Downtrend"

def classify_signal(r:pd.Series, score:int, grade:str, trend:str)->str:
    rsi=float(r["RSI"]) if r.get("RSI")==r.get("RSI") else 50.0
    volr=float(r["VOL_RATIO"]) if r.get("VOL_RATIO")==r.get("VOL_RATIO") else 1.0
    m5=float(r["MOM_5D"]) if r.get("MOM_5D")==r.get("MOM_5D") else 0.0
    m20=float(r["MOM_20D"]) if r.get("MOM_20D")==r.get("MOM_20D") else 0.0
    if (trend in ["Strong Uptrend","Uptrend"] and grade in ["A+","A"] and volr>1.5 and 45<=rsi<=70 and m5>0 and m20>0): return "Strong Buy"
    if (trend in ["Strong Uptrend","Uptrend"] and grade in ["A+","A","B+"] and volr>1.0 and 40<=rsi<=75): return "Buy"
    if trend in ["Strong Uptrend","Uptrend"] and grade in ["A+","A","B+","B"]: return "Hold"
    if trend=="Sideways" and grade in ["B+","B","C"]: return "Hold"
    return "Sell"

def compute_trade_plan(df:pd.DataFrame, strategy:str, trend:str)->dict:
    r=df.iloc[-1]; price=float(r["Close"]); e21=float(r["EMA21"])
    entry=round_to_tick(price*0.995); tp1=round_to_tick(entry*1.04); tp2=round_to_tick(entry*1.07); tp3=None; sl=round_to_tick(entry*0.97)
    if trend in ["Strong Uptrend","Uptrend"] and e21<price:
        ema_entry=round_to_tick(e21*1.01)
        if price*0.9<ema_entry<price: entry=ema_entry
    if trend=="Downtrend": sl=round_to_tick(entry*0.96)
    return {"entry_ideal":entry,"entry_aggressive":round_to_tick(price),"tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl}

# ================= ANALYZE (tanpa RT) =================
def analyze_ticker_base(ticker:str, strategy:str, period:str)->dict|None:
    df=fetch_data(ticker,period)
    if df is None or df.empty: return None
    r=df.iloc[-1]
    score,details,conf,grade=score_general(df)  # ganti sesuai strategi jika perlu
    if grade not in ["A+","A","B+","B","C"]: return None
    trend=detect_trend(r); signal=classify_signal(r,score,grade,trend); plan=compute_trade_plan(df, strategy, trend)
    res={"Ticker":ticker.replace(".JK",""),
         "PriceEOD": float(r["Close"]),
         "Score":score,"Confidence":conf,"Grade":grade,
         "Trend":trend,"Signal":signal,
         "Entry":plan["entry_ideal"],"Entry_Aggressive":plan["entry_aggressive"],
         "TP1":plan["tp1"],"TP2":plan["tp2"],"SL":plan["sl"],
         "VolRatio": float(r["VOL_RATIO"]) if not np.isnan(r["VOL_RATIO"]) else 0.0,
         "Mom5": float(r["MOM_5D"]) if not np.isnan(r["MOM_5D"]) else 0.0,
         "Mom20": float(r["MOM_20D"]) if not np.isnan(r["MOM_20D"]) else 0.0,
         "RSI": float(r["RSI"]) if not np.isnan(r["RSI"]) else 50.0,
         "Details": details}
    if plan["tp3"]: res["TP3"]=plan["tp3"]
    return res

# ================= SCAN / MERGE REAL-TIME =================
def _is_active(row:pd.Series)->bool:
    return (row.get("VolRatio",0)>=1.1) or (row.get("Mom5",0)>0) or (row.get("Mom20",0)>0)

def process_ticker(t,strategy,period):
    try: return analyze_ticker_base(t, strategy, period)
    except: return None

def scan_stocks(tickers:list[str], strategy:str, period:str, limit1:int, limit2:int):
    st.info(f"🔍 **STAGE 1**: Scanning {len(tickers)} stocks…")
    results=[]; progress=st.progress(0); status=st.empty()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(process_ticker,t,strategy,period):t for t in tickers}
        done=0
        for f in as_completed(futs):
            done+=1; progress.progress(done/len(tickers))
            r=f.result()
            if r: results.append(r)
            status.text(f"📊 {done}/{len(tickers)} | Collected: {len(results)}")
            time.sleep(0.005)
    progress.empty(); status.empty()
    if not results: return pd.DataFrame(), pd.DataFrame()

    df=pd.DataFrame(results)

    mask_quality=df["Signal"].isin(["Strong Buy","Buy"]) & df["Trend"].isin(["Strong Uptrend","Uptrend"])
    mask_active=df.apply(_is_active,axis=1)
    df=df[mask_quality & mask_active].copy()
    if df.empty:
        st.warning("Tidak ada kandidat (Strong Buy/Buy + Uptrend + aktif).")
        return pd.DataFrame(), pd.DataFrame()

    # merge RT price (awal)
    symbols=[f"{t}.JK" for t in df["Ticker"].tolist()]
    rt_map=get_realtime_quotes(symbols)
    df["Price"]=df["Ticker"].apply(lambda x: rt_map.get(f"{x}.JK",{}).get("last", np.nan)).fillna(df["PriceEOD"])
    df["PrevClose"]=df["Ticker"].apply(lambda x: rt_map.get(f"{x}.JK",{}).get("prev_close", np.nan))
    df["DelayMin"]=df["Ticker"].apply(lambda x: rt_map.get(f"{x}.JK",{}).get("delay_min", 0))
    df["LastTimeWIB"]=df["Ticker"].apply(lambda x: rt_map.get(f"{x}.JK",{}).get("last_time_wib","-"))

    df["SignalRank"]=df["Signal"].map(SIGNAL_RANK).fillna(0)
    df.sort_values(["SignalRank","Score","Confidence","Mom20","VolRatio"], ascending=[False,False,False,False,False], inplace=True)
    df1=df.head(limit1).reset_index(drop=True)
    st.success(f"✅ Stage 1: {len(df1)} kandidat")

    df1["Stage1Rank"]=range(1,len(df1)+1)
    df2=df1.sort_values(["SignalRank","Score","Confidence","Mom20","VolRatio"], ascending=[False,False,False,False,False]).head(limit2).reset_index(drop=True)
    st.success(f"🏆 Stage 2: {len(df2)} elite picks")

    st.caption("ℹ️ Price = Yahoo Spark/Quote (may be delayed ~1–15m). Gunakan harga broker untuk eksekusi.")

    st.session_state.last_scan_results=(df2,df1)
    st.session_state.last_scan_time=datetime.now()
    st.session_state.last_scan_strategy=strategy
    st.session_state.last_scan_symbols=symbols
    st.session_state.last_period=period
    st.session_state.scan_count+=1
    return df1, df2

# ================= UI =================
st.title("🚀 IDX Power Screener v5.4 (RT + AutoRefresh)")
st.caption("Real-time display via Yahoo Spark/Quote • Indicators via EOD • Strict Buy-only filter")
display_ihsg_widget()
tickers=load_tickers()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.info(f"📊 Total stocks: {len(tickers)}")
    st.caption(f"🕐 Jakarta: {get_jakarta_time().strftime('%H:%M WIB')}")
    st.markdown("---")

    # Auto-refresh controls
    auto_refresh = st.checkbox("Auto refresh harga", value=True)
    interval_sec = st.slider("Interval refresh (detik)", 5, 60, 15, help="Hanya update harga; tidak re-scan indikator.")
    if auto_refresh:
        if st_autorefresh:
            st_autorefresh(interval=interval_sec*1000, key="rt_autorefresh")  # rerun page
        else:
            st.warning("Pasang package `streamlit-autorefresh` di requirements.txt untuk auto-refresh.")

    # Manual RT cache clear
    if st.button("🔄 Refresh Real-Time Sekarang"):
        fetch_rt_bulk_spark.clear(); fetch_rt_bulk_quote.clear()

    st.markdown("---")
    menu=st.radio("📋 Mode",["⚡ SPEED Trader (1-2d)","🎯 SWING Trader (3-5d)","💎 VALUE Plays (Undervalued)","🔍 Single Stock"], index=0)
    period=st.selectbox("Period indikator",["3mo","6mo","1y"],index={"3mo":0,"6mo":1,"1y":2}[st.session_state.last_period] if st.session_state.last_period else 1)
    limit1=st.selectbox("Stage 1 count",[50,100,150,200],index=0)
    limit2=st.selectbox("Stage 2 count",[10,20,30,40,50],index=0)

def show_table(df:pd.DataFrame,title:str):
    if df is None or df.empty: st.warning("Tidak ada hasil."); return
    view=df.copy()
    cols=[c for c in ["Ticker","Price","Signal","Trend","Score","Confidence","Mom20","VolRatio","LastTimeWIB","DelayMin","Entry","Entry_Aggressive","TP1","TP2","TP3","SL","Grade"] if c in view.columns]
    st.subheader(title)
    st.dataframe(view[cols], use_container_width=True, hide_index=True)
    exp=view.drop(columns=["Details"], errors="ignore")
    st.download_button("💾 Download Results (CSV)", data=exp.to_csv(index=False).encode("utf-8"), file_name=f"{title.replace(' ','_')}.csv", mime="text/csv")
    with st.expander("🔎 Detail per saham"):
        for _,row in view.iterrows():
            tp3=f" | TP3 {format_idr(row['TP3'])}" if "TP3" in row and pd.notna(row["TP3"]) else ""
            st.markdown(f"**{row['Ticker']}** • Price **{format_idr(row['Price'])}** @ {row.get('LastTimeWIB','-')} (delay ~{int(row.get('DelayMin',0))}m)")
            st.caption(f"Entry {format_idr(row['Entry'])} (Agg {format_idr(row['Entry_Aggressive'])}) | TP1 {format_idr(row['TP1'])} | TP2 {format_idr(row['TP2'])}{tp3} | SL {format_idr(row['SL'])} | Mom20 {row.get('Mom20',0):.2f}% | Vol {row.get('VolRatio',0):.2f}x")
            st.markdown("---")

def show_last_scan_with_rt():
    if not st.session_state.last_scan_results: 
        st.info("Belum ada hasil scan. Jalankan **RUN SCAN** dulu.")
        return
    df2, df1 = st.session_state.last_scan_results
    # re-attach RT prices setiap rerun
    df1r = attach_rt_prices(df1)
    df2r = attach_rt_prices(df2)
    show_table(df1r, f"Stage 1 – Top {len(df1r)} (auto-refreshed)")
    show_table(df2r, f"Stage 2 – Top {len(df2r)} (auto-refreshed)")

# ===== Views =====
if "Single Stock" in menu:
    st.markdown("### 🔍 Single Stock (Real-Time)")
    default=tickers[0].replace(".JK","") if tickers else "BBRI"
    sym=st.text_input("Symbol (tanpa .JK)", value=default).strip().upper()
    if st.button("🔍 ANALYZE", type="primary"):
        full=normalize_ticker(sym)
        with st.spinner(f"Analyzing {sym}..."):
            base=analyze_ticker_base(full, "General", period)
            if base is None: st.error("❌ Failed or rejected"); 
            else:
                rt=attach_rt_prices(pd.DataFrame([{"Ticker":base["Ticker"]}])).iloc[0].to_dict()
                last=rt.get("Price", np.nan); prev=np.nan  # prev optional
                df=fetch_data(full, period)
                chart=create_chart(df, sym)
                if chart: st.plotly_chart(chart, use_container_width=True)
                st.metric("Last (Yahoo)", f"{format_idr(last)}")
                st.caption(f"⏱️ {rt.get('LastTimeWIB','-')} WIB • Delayed ~{int(rt.get('DelayMin',0))}m • Yahoo Spark/Quote")
                st.markdown(f"**Signal:** {base['Signal']} | **Trend:** {base['Trend']} | **Grade:** {base['Grade']} | **Score:** {base['Score']}")
                st.success(f"Entry {format_idr(base['Entry'])} • Agg {format_idr(base['Entry_Aggressive'])} • TP1 {format_idr(base['TP1'])} • TP2 {format_idr(base['TP2'])} • SL {format_idr(base['SL'])}")
else:
    st.markdown("### 📈 Scanner")
    if st.button("🚀 RUN SCAN (Real-Time Strict)", type="primary", use_container_width=True):
        strategy="General"  # default; strategi lain bisa dipetakan
        df1,df2=scan_stocks(tickers, strategy, period, limit1, limit2)
        if df1.empty and df2.empty: st.error("❌ No candidates.")
        else:
            show_table(df1, f"Stage 1 – Top {limit1}")
            show_table(df2, f"Stage 2 – Top {limit2}")
    else:
        # jika sudah ada hasil sebelumnya, auto-refresh akan update harga di sini
        show_last_scan_with_rt()

st.markdown("---")
st.caption("⚠️ Broker apps = realtime exchange feed. Yahoo Spark/Quote bisa delay tipis. Auto-refresh hanya update harga; re-scan indikator tetap manual agar ringan.")
