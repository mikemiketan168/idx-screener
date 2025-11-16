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

# ================== BASIC CONFIG ==================
st.set_page_config(
    page_title="IDX Power Screener v5.0",
    page_icon="🚀",
    layout="wide"
)

# ============= SESSION STATE (LOCK SCREEN SAFE) =============
if 'last_scan_results' not in st.session_state:
    st.session_state.last_scan_results = None
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'last_scan_strategy' not in st.session_state:
    st.session_state.last_scan_strategy = None
if 'scan_count' not in st.session_state:
    st.session_state.scan_count = 0

# ============= IHSG MARKET WIDGET =============
@st.cache_data(ttl=180)
def fetch_ihsg_data():
    try:
        import yfinance as yf
        ihsg = yf.Ticker("^JKSE")
        hist = ihsg.history(period="1d")
        if hist.empty:
            return None
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
    except Exception:
        return None

def display_ihsg_widget():
    ihsg = fetch_ihsg_data()
    if not ihsg:
        st.info("📊 IHSG data temporarily unavailable")
        return

    status_emoji = "🟢" if ihsg['status'] == 'up' else "🔴"
    status_text = "BULLISH" if ihsg['status'] == 'up' else "BEARISH"

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
                border-left: 5px solid {"#22c55e" if ihsg['status']=="up" else "#ef4444"}'>
        <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div>
                <h3 style='margin:0;color:white;'>📊 MARKET OVERVIEW</h3>
                <p style='margin:5px 0;color:#e0e7ff;font-size:0.9em;'>
                    Jakarta Composite Index
                </p>
            </div>
            <div style='text-align:right;'>
                <h2 style='margin:0;color:white;'>
                    {status_emoji} {ihsg['price']:,.2f}
                </h2>
                <p style='margin:5px 0;color:{"#22c55e" if ihsg['status']=="up" else "#ef4444"};
                          font-size:1.1em;font-weight:bold;'>
                    {ihsg['change']:+,.2f} ({ihsg['change_pct']:+.2f}%)
                </p>
            </div>
        </div>
        <div style='margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.2);'>
            <p style='margin:3px 0;color:#e0e7ff;font-size:0.85em;'>
                📊 High: {ihsg['high']:,.2f} | Low: {ihsg['low']:,.2f} | Status: <strong>{status_text}</strong>
            </p>
            <p style='margin:3px 0;color:#fbbf24;font-size:0.9em;'>
                {condition}
            </p>
            <p style='margin:3px 0;color:#a5b4fc;font-size:0.85em;'>
                {guidance}
            </p>
            <p style='margin:5px 0 0 0;color:#94a3b8;font-size:0.75em;'>
                ⏰ Last update: {datetime.now().strftime('%H:%M:%S')} WIB | 🔄 Auto-refresh: 3 min
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============= TICKERS =============
def load_tickers():
    """
    Kalau kakak punya list IDX lengkap, boleh ganti isi fungsi ini
    dengan fungsi lama kakak. Logika lain tetap sama.
    """
    try:
        if os.path.exists("idx_stocks.json"):
            with open("idx_stocks.json", "r") as f:
                data = json.load(f)
            tickers = data.get("tickers", [])
            return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    except Exception:
        pass

    # fallback contoh (boleh ditambah sendiri)
    base = ["BBCA.JK","BBRI.JK","BMRI.JK","ASII.JK","TLKM.JK",
            "BBNI.JK","BRPT.JK","BUMI.JK","BUKA.JK","BREN.JK",
            "PGAS.JK","PTBA.JK","MDKA.JK","RAJA.JK","RATU.JK",
            "CDIA.JK","COAL.JK","PGUN.JK","TEBE.JK","COIN.JK"]
    return base

def get_jakarta_time():
    return datetime.now(timezone(timedelta(hours=7)))

def is_bpjs_time():
    jkt_hour = get_jakarta_time().hour
    return 9 <= jkt_hour < 10

def is_bsjp_time():
    jkt_hour = get_jakarta_time().hour
    return 14 <= jkt_hour < 16

# ============= CHART =============
def create_chart(df, ticker, period_days=30):
    try:
        df_chart = df.tail(period_days).copy()

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{ticker} - Price & EMAs', 'Volume', 'RSI')
        )

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

        colors = {'EMA9': '#2196F3', 'EMA21': '#FF9800',
                  'EMA50': '#F44336', 'EMA200': '#9E9E9E'}
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

        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart['RSI'],
                name='RSI',
                line=dict(color='#9C27B0', width=2)
            ),
            row=3, col=1
        )

        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)

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
    except Exception:
        return None

# ============= FETCH DATA =============
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, period="6mo"):
    try:
        end = int(datetime.now().timestamp())
        days = {"5d":5,"1mo":30,"3mo":90,"6mo":180,"1y":365}.get(period,180)
        start = end - (days*86400)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        r = requests.get(
            url,
            params={"period1":start,"period2":end,"interval":"1d"},
            headers={'User-Agent':'Mozilla/5.0'},
            timeout=10
        )
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
        df['BB_UPPER'] = df['BB_MID'] + 2*df['BB_STD']
        df['BB_LOWER'] = df['BB_MID'] - 2*df['BB_STD']
        df['BB_WIDTH'] = ((df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MID']) * 100

        # Stochastic
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['STOCH_K'] = 100*(df['Close']-low14)/(high14-low14)
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
    except Exception:
        return None

# ============= PREFILTER LIQUIDITY =============
def apply_liquidity_filter(df, ticker):
    try:
        r = df.iloc[-1]
        price = r['Close']
        vol_avg = df['Volume'].tail(20).mean()

        if price < 50:
            return False, "Price too low"

        if vol_avg < 500000:
            return False, "Volume too low"

        turnover = price * vol_avg
        if turnover < 100_000_000:
            return False, "Turnover too low"

        return True, "Passed"
    except Exception:
        return False, "Error"

# ============= SCORING (PERSIS DENGAN V4.5) =============
def score_general(df):
    try:
        r = df.iloc[-1]
        score = 0
        details = {}

        passed, reason = apply_liquidity_filter(df, "")
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"

        if r['Close'] < r['EMA21'] < r['EMA50'] < r['EMA200']:
            return 0, {"Rejected": "Strong downtrend"}, 0, "F"

        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "Below EMA50"}, 0, "F"

        mom_20d = r['MOM_20D']
        if mom_20d < -8:
            return 0, {"Rejected": f"Strong negative momentum ({mom_20d:.1f}%)"}, 0, "F"

        vol_ratio = r['VOL_RATIO']
        if vol_ratio < 1.0:
            return 0, {"Rejected": f"Insufficient volume ({vol_ratio:.1f}x)"}, 0, "F"

        if -8 <= mom_20d < -5:
            momentum_penalty = 0.6
            details['⚠️ Warning'] = f'Weak momentum {mom_20d:.1f}%'
        elif -5 <= mom_20d < 0:
            momentum_penalty = 0.8
            details['⚠️ Warning'] = f'Slight negative momentum {mom_20d:.1f}%'
        else:
            momentum_penalty = 1.0

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
            details['Trend'] = '🔴 Weak'

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
            details['RSI'] = f'🔴 Overbought {rsi:.0f}'
        elif rsi < 35:
            details['RSI'] = f'🔴 Oversold {rsi:.0f}'
        else:
            score += 5
            details['RSI'] = f'⚪ Neutral {rsi:.0f}'

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
            details['Volume'] = f'🔴 Weak {vol_ratio:.1f}x'

        mom_5d = r['MOM_5D']
        mom_10d = r['MOM_10D']

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
            score += 8
            details['Momentum'] = f'🟡 20D momentum +{mom_20d:.1f}%'

        score = int(score * momentum_penalty)

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
    try:
        r = df.iloc[-1]
        score = 0
        details = {}

        passed, reason = apply_liquidity_filter(df, "")
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"

        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "Below EMA50 - No uptrend"}, 0, "F"

        atr_pct = r['ATR_PCT']
        if 2.5 < atr_pct < 5:
            score += 35
            details['Volatility'] = f'🟢 PERFECT {atr_pct:.1f}%'
        elif 2 < atr_pct <= 2.5 or 5 <= atr_pct < 6:
            score += 20
            details['Volatility'] = f'🟡 Good {atr_pct:.1f}%'
        elif atr_pct >= 6:
            details['Volatility'] = f'🔴 Too high {atr_pct:.1f}%'
        else:
            details['Volatility'] = f'🔴 Too low {atr_pct:.1f}%'

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

        if 30 < r['RSI'] < 40:
            score += 20
            details['RSI'] = f"🟢 Oversold {r['RSI']:.0f}"
        elif 40 <= r['RSI'] < 50:
            score += 10
            details['RSI'] = f"🟡 OK {r['RSI']:.0f}"
        else:
            details['RSI'] = f"🔴 Not oversold {r['RSI']:.0f}"

        if r['STOCH_K'] < 30:
            score += 15
            details['Stoch'] = f"🟢 Oversold {r['STOCH_K']:.0f}"
        elif r['STOCH_K'] < 50:
            score += 5
            details['Stoch'] = f"🟡 OK {r['STOCH_K']:.0f}"

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
    except Exception:
        return 0, {}, 0, "F"

def score_bsjp(df):
    try:
        r = df.iloc[-1]
        score = 0
        details = {}

        passed, reason = apply_liquidity_filter(df, "")
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"

        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "No uptrend"}, 0, "F"

        bb_pos = ((r['Close'] - r['BB_LOWER']) / (r['BB_UPPER'] - r['BB_LOWER'])) * 100
        if bb_pos < 20:
            score += 35
            details['BB'] = f'🟢 Near lower {bb_pos:.0f}%'
        elif bb_pos < 35:
            score += 20
            details['BB'] = f'🟡 Below mid {bb_pos:.0f}%'
        else:
            details['BB'] = f'🔴 Too high {bb_pos:.0f}%'

        gap = ((r['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        if -4 < gap < -1:
            score += 30
            details['Gap'] = f'🟢 Perfect {gap:.1f}%'
        elif -1 <= gap < -0.3:
            score += 15
            details['Gap'] = f'🟡 Small {gap:.1f}%'
        elif gap < -4:
            details['Gap'] = f'🔴 Too large {gap:.1f}%'
        else:
            details['Gap'] = f'⚪ No gap {gap:.1f}%'

        if 25 < r['RSI'] < 40:
            score += 20
            details['RSI'] = f"🟢 Oversold {r['RSI']:.0f}"
        elif 40 <= r['RSI'] < 50:
            score += 10
            details['RSI'] = f"🟡 OK {r['RSI']:.0f}"

        if r['VOL_RATIO'] > 1.5:
            score += 15
            details['Volume'] = f"🟢 Strong {r['VOL_RATIO']:.1f}x"
        elif r['VOL_RATIO'] > 1.0:
            score += 5
            details['Volume'] = f"🟡 Normal {r['VOL_RATIO']:.1f}x"

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
    except Exception:
        return 0, {}, 0, "F"

def score_bandar(df):
    try:
        r = df.iloc[-1]
        score = 0
        details = {}

        passed, reason = apply_liquidity_filter(df, "")
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"

        vol_ratio_10d = df['Volume'].tail(10).mean() / df['VOL_SMA50'].iloc[-1]
        vol_ratio_recent = df['Volume'].tail(3).mean() / df['VOL_SMA20'].iloc[-1]

        price_chg_20d = ((r['Close'] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
        price_chg_5d = ((r['Close'] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100

        if df['OBV'].iloc[-20] != 0:
            obv_chg = ((df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / abs(df['OBV'].iloc[-20]))
        else:
            obv_chg = 0
        obv_vs_ema = df['OBV'].iloc[-1] > df['OBV_EMA'].iloc[-1]

        if (vol_ratio_10d > 1.3 and
            -2 < price_chg_20d < 5 and
            obv_chg > 0.05 and
            obv_vs_ema and
            r['Close'] > r['EMA50']):
            phase = "🟢 ACCUMULATION"
            score = 90
            conf = 85
            details['Phase'] = 'Accumulation Zone'
            details['Signal'] = 'STRONG BUY'
            details['Action'] = 'Enter position'
        elif (price_chg_20d > 8 and
              obv_chg > 0.1 and
              r['Close'] > r['EMA9'] > r['EMA21']):
            phase = "🚀 MARKUP"
            score = 75
            conf = 70
            details['Phase'] = 'Markup Phase'
            details['Signal'] = 'HOLD/TRAIL'
            details['Action'] = 'Trail stop'
        elif (vol_ratio_recent > 2.0 and
              price_chg_5d < -2 and
              (r['RSI'] > 70 or price_chg_20d > 15)):
            phase = "🔴 DISTRIBUTION"
            score = 20
            conf = 25
            details['Phase'] = 'Distribution Zone'
            details['Signal'] = 'SELL/AVOID'
            details['Action'] = 'Exit now'
        elif (r['Close'] < r['EMA50'] and obv_chg < -0.1):
            phase = "⚫ MARKDOWN"
            score = 10
            conf = 15
            details['Phase'] = 'Markdown Phase'
            details['Signal'] = 'AVOID'
            details['Action'] = 'Stay away'
        else:
            phase = "⚪ RANGING"
            score = 45
            conf = 40
            details['Phase'] = 'No clear phase'
            details['Signal'] = 'WAIT'
            details['Action'] = 'Monitor'

        details['Vol_10D'] = f'{vol_ratio_10d:.1f}x'
        details['Price_20D'] = f'{price_chg_20d:+.1f}%'
        details['OBV_Trend'] = f'{obv_chg*100:+.1f}%'

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
    try:
        r = df.iloc[-1]
        score = 0
        details = {}

        passed, reason = apply_liquidity_filter(df, "")
        if not passed:
            return 0, {"Rejected": reason}, 0, "F"

        if r['Close'] < r['EMA50']:
            return 0, {"Rejected": "Below EMA50"}, 0, "F"

        mom_10d = r['MOM_10D']
        mom_20d = r['MOM_20D']
        if mom_20d < -5:
            return 0, {"Rejected": f"Negative momentum ({mom_20d:.1f}%)"}, 0, "F"

        if r['VOL_RATIO'] < 1.0:
            return 0, {"Rejected": f"Weak volume ({r['VOL_RATIO']:.1f}x)"}, 0, "F"

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

        mom_10d = r['MOM_10D']
        mom_20d = r['MOM_20D']
        if mom_10d > 5 and mom_20d > 8:
            score += 20
            details['Momentum'] = f'🟢 Strong 10D:{mom_10d:.1f}% 20D:{mom_20d:.1f}%'
        elif mom_10d > 2 and mom_20d > 4:
            score += 15
            details['Momentum'] = f'🟡 Good 10D:{mom_10d:.1f}% 20D:{mom_20d:.1f}%'
        elif mom_10d > 0 and mom_20d > 0:
            score += 8
            details['Momentum'] = f'🟠 Positive 10D:{mom_10d:.1f}% 20D:{mom_20d:.1f}%'

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
    except Exception:
        return 0, {}, 0, "F"

def score_value(df):
    try:
        r = df.iloc[-1]
        price = r['Close']
        score = 0
        details = {}

        if price >= 1000:
            return 0, {"Rejected": "Price too high (not undervalued)"}, 0, "F"

        vol_avg = df['Volume'].tail(20).mean()
        if vol_avg < 300000:
            return 0, {"Rejected": "Volume too low"}, 0, "F"

        if r['Close'] < r['EMA200']:
            return 0, {"Rejected": "Below 200 EMA"}, 0, "F"

        dist_to_ema50 = ((r['Close'] - r['EMA50']) / r['EMA50']) * 100
        bb_position = ((r['Close'] - r['BB_LOWER']) / (r['BB_UPPER'] - r['BB_LOWER'])) * 100

        if -5 < dist_to_ema50 < 3:
            score += 20
            details['Support'] = f'🟢 Near EMA50 ({dist_to_ema50:+.1f}%)'
        elif bb_position < 25:
            score += 20
            details['Support'] = f'🟢 Near BB lower ({bb_position:.0f}%)'
        elif -10 < dist_to_ema50 < 5:
            score += 10
            details['Support'] = f'🟡 Approaching EMA50'

        rsi = r['RSI']
        if rsi < 35:
            score += 20
            details['RSI'] = f'🟢 Oversold {rsi:.0f}'
        elif rsi < 45:
            score += 10
            details['RSI'] = f'🟡 Low {rsi:.0f}'

        vol_trend = df['Volume'].tail(5).mean() / df['VOL_SMA20'].iloc[-1]
        if vol_trend > 1.3:
            score += 20
            details['Volume'] = f'🟢 Increasing {vol_trend:.1f}x'
        elif vol_trend > 1.0:
            score += 10
            details['Volume'] = f'🟡 Normal {vol_trend:.1f}x'

        green_candles = sum(1 for i in range(-5, 0) if df['Close'].iloc[i] > df['Open'].iloc[i])
        if green_candles >= 3:
            score += 15
            details['Pattern'] = f'🟢 {green_candles}/5 green candles'
        elif green_candles >= 2:
            score += 8
            details['Pattern'] = f'🟡 {green_candles}/5 green candles'

        if price < 500:
            score += 25
            details['Price'] = f'🟢 Very affordable Rp {price:.0f}'
        elif price < 750:
            score += 15
            details['Price'] = f'🟡 Affordable Rp {price:.0f}'
        else:
            score += 5
            details['Price'] = f'🟠 Rp {price:.0f}'

        if score >= 75:
            grade, conf = "A", 75
        elif score >= 65:
            grade, conf = "B+", 70
        elif score >= 50:
            grade, conf = "B", 60
        else:
            grade, conf = "C", max(score, 0)
        return score, details, conf, grade
    except Exception:
        return 0, {}, 0, "F"

# ============= TREND, SIGNAL & TRADE PLAN =============
def detect_trend(last_row):
    price = float(last_row['Close'])
    ema9  = float(last_row['EMA9'])
    ema21 = float(last_row['EMA21'])
    ema50 = float(last_row['EMA50'])
    ema200 = float(last_row['EMA200'])

    if price > ema9 > ema21 > ema50 > ema200:
        return "Strong Uptrend"
    elif price > ema50 and ema9 > ema21 > ema50:
        return "Uptrend"
    elif abs(price - ema50) / price < 0.03:
        return "Sideways"
    else:
        return "Downtrend"

def classify_signal(last_row, score, gradedef classify_signal(last_row, score, grade, trend):
    rsi        = float(last_row['RSI'])
    vol_ratio  = float(last_row['VOL_RATIO'])
    mom_5d     = float(last_row['MOM_5D'])
    mom_20d    = float(last_row['MOM_20D'])

    # ================================
    # 1) UPTREND LOGIC
    # ================================
    if trend in ["Strong Uptrend", "Uptrend"]:
        # STRONG BUY (semua sangat sehat)
        if (
            grade in ["A+", "A"]
            and vol_ratio > 1.5
            and 45 <= rsi <= 70
            and mom_5d > 0
            and mom_20d > 0
        ):
            return "Strong Buy"

        # BUY NORMAL
        if (
            grade in ["A+", "A", "B+"]
            and vol_ratio > 1.0
            and 40 <= rsi <= 75
            and mom_20d >= 0
        ):
            return "Buy"

        # TREND BAGUS TAPI OVERBOUGHT / MELEMAH
        if rsi > 75 or mom_5d < 0 or vol_ratio < 1.0:
            return "Take Profit / Wait"

        # DEFAULT (UPTREND tapi belum entry)
        return "Hold"

    # ================================
    # 2) SIDEWAYS
    # ================================
    if trend == "Sideways":
        if grade in ["A+", "A", "B+"]:
            return "Buy on Dip"
        return "Hold"

    # ================================
    # 3) DOWNTREND
    # ================================
    if trend == "Downtrend":
        # REVERSAL SPEKULATIF
        if grade in ["A+", "A"] and rsi < 35 and mom_20d > -5:
            return "Speculative Buy"
        return "Sell"

    # Fallback jika trend tidak dikenal
    return "Hold"

def compute_trade_plan(df, strategy, trend):
    r = df.iloc[-1]
    price = float(r['Close'])
    ema21 = float(r['EMA21'])
    ema50 = float(r['EMA50'])

    if strategy == "Swing":
        entry_ideal = round(price * 0.99, 0)
        tp1 = round(entry_ideal * 1.06, 0)
        tp2 = round(entry_ideal * 1.10, 0)
        tp3 = round(entry_ideal * 1.15, 0)
        sl  = round(entry_ideal * 0.95, 0)
    elif strategy == "Value":
        entry_ideal = round(price * 0.98, 0)
        tp1 = round(entry_ideal * 1.15, 0)
        tp2 = round(entry_ideal * 1.25, 0)
        tp3 = round(entry_ideal * 1.35, 0)
        sl  = round(entry_ideal * 0.93, 0)
    else:
        entry_ideal = round(price * 0.995, 0)
        tp1 = round(entry_ideal * 1.04, 0)
        tp2 = round(entry_ideal * 1.07, 0)
        tp3 = None
        sl  = round(entry_ideal * 0.97, 0)

    if trend in ["Strong Uptrend", "Uptrend"] and ema21 < price:
        ema_entry = round(ema21 * 1.01, 0)
        if price * 0.9 < ema_entry < price:
            entry_ideal = ema_entry

    if trend == "Downtrend":
        sl = round(entry_ideal * 0.96, 0)

    entry_aggressive = round(price, 0)

    return {
        "entry_ideal": entry_ideal,
        "entry_aggressive": entry_aggressive,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
    }

# ============= PROCESS TICKER =============
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
        elif strategy == "Swing":
            score, details, conf, grade = score_swing(df)
        elif strategy == "Value":
            score, details, conf, grade = score_value(df)
        else:
            score, details, conf, grade = score_general(df)

        if grade not in ['A+','A','B+','B','C']:
            return None

        last_row = df.iloc[-1]
        trend = detect_trend(last_row)
        signal = classify_signal(last_row, score, grade, trend)
        plan = compute_trade_plan(df, strategy, trend)

        result = {
            "Ticker": ticker.replace('.JK',''),
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
            "SL": plan["sl"],
            "Details": details
        }
        if plan["tp3"]:
            result["TP3"] = plan["tp3"]

        return result
    except Exception:
        return None

# ============= SESSION SAVE =============
def save_scan_to_session(df2, df1, strategy):
    st.session_state.last_scan_results = (df2, df1)
    st.session_state.last_scan_time = datetime.now()
    st.session_state.last_scan_strategy = strategy
    st.session_state.scan_count += 1

def display_last_scan_info():
    if st.session_state.last_scan_results:
        df2, df1 = st.session_state.last_scan_results
        time_ago = datetime.now() - st.session_state.last_scan_time
        mins_ago = int(time_ago.total_seconds() / 60)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg,#064e3b 0%,#065f46 100%);
                    padding:12px;border-radius:8px;margin-bottom:15px;
                    border-left:4px solid #10b981;'>
            <p style='margin:0;color:white;font-weight:bold;'>
                📂 LAST SCAN RESULTS
            </p>
            <p style='margin:5px 0 0 0;color:#d1fae5;font-size:0.9em;'>
                Strategy: {st.session_state.last_scan_strategy} |
                Time: {st.session_state.last_scan_time.strftime('%H:%M:%S')} ({mins_ago} min ago) |
                Found: {len(df2)} Grade A+/A/B picks
            </p>
        </div>
        """, unsafe_allow_html=True)
        return True
    return False

def create_csv_download(df, strategy):
    if not df.empty:
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
            mime="text/csv"
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

    df1 = pd.DataFrame(results).sort_values("Score", ascending=False).head(limit1)
    st.success(f"✅ Stage 1: Found {len(df1)} candidates (Avg score: {df1['Score'].mean():.0f})")

    df2 = df1[df1['Grade'].isin(['A+','A','B+','B'])].head(limit2)
    st.success(f"🏆 Stage 2: {len(df2)} elite picks (Avg conf: {df2['Confidence'].mean():.0f}%)")

    save_scan_to_session(df2, df1, strategy)
    return df1, df2

# ============= UI HEADER =============
st.title("🚀 IDX Power Screener v5.0 STOCKBOT")
st.caption("Trend + Signal + Entry Ideal/Agresif + TP1–TP3 + SL | Lock Screen Safe")

display_ihsg_widget()
tickers = load_tickers()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.info(f"📊 Total stocks: {len(tickers)}")
    jkt_time = get_jakarta_time()
    st.caption(f"🕐 Jakarta: {jkt_time.strftime('%H:%M WIB')}")

    st.markdown("---")
    menu = st.radio(
        "📋 Strategy",
        [
            "⚡ SPEED Trader (1-2d)",
            "🎯 SWING Trader (3-5d)",
            "💎 VALUE Plays (Undervalued)",
            "⚡ BPJS (Day Trading)",
            "🌙 BSJP (Overnight)",
            "🔮 Bandar Tracking",
            "🔍 Single Stock"
        ]
    )

    st.markdown("---")

    if "Single" not in menu:
        period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
        st.markdown("### 🎯 Filtering")
        limit1 = st.slider("Stage 1: Top N", 20, 100, 50, 10)
        limit2 = st.slider("Stage 2: Elite", 5, 30, 10, 5)
        st.caption(f"Scan {len(tickers)} → Top {limit1} → Elite {limit2}")

    st.markdown("---")
    st.caption("v5.0 STOCKBOT - Trend + Signal + Entry Plan")

# ============= MENU HANDLERS =============
if "Single Stock" in menu:
    st.markdown("### 🔍 Single Stock Analysis")

    selected = st.selectbox("Select Stock", [t.replace('.JK','') for t in tickers])
    strategy_single = st.selectbox("Strategy", ["General", "BPJS", "BSJP", "Bandar", "Swing", "Value"])
    period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)

    if st.button("🔍 ANALYZE", type="primary"):
        ticker_full = selected if selected.endswith('.JK') else f"{selected}.JK"
        with st.spinner(f"Analyzing {selected}..."):
            df = fetch_data(ticker_full, period)
            if df is None:
                st.error("❌ Failed to fetch data")
            else:
                result = process_ticker(ticker_full
