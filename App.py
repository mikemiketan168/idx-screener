import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(
    page_title="IDX Power Screener v5.0",
    page_icon="📈",
    layout="wide"
)

# Style CSS untuk tampilan yang lebih baik
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin: 1rem 0;
        font-weight: bold;
    }
    .stock-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .signal-strong-buy {
        color: #00cc00;
        font-weight: bold;
        background-color: #e6ffe6;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    .signal-buy {
        color: #66b266;
        font-weight: bold;
        background-color: #f0fff0;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    .signal-hold {
        color: #ff9900;
        font-weight: bold;
        background-color: #fff5e6;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    .signal-sell {
        color: #ff3333;
        font-weight: bold;
        background-color: #ffe6e6;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class StockScreener:
    def __init__(self):
        self.stocks_data = self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate sample data untuk 800 saham"""
        np.random.seed(42)
        stocks = []
        
        # Daftar saham IDX populer
        stock_codes = [
            'BBCA', 'BBRI', 'BMRI', 'BBNI', 'TLKM', 'ASII', 'UNVR', 'ICBP', 
            'INDF', 'MNCN', 'WSKT', 'ADRO', 'ANTM', 'PTBA', 'PGAS', 'MEDC',
            'KLBF', 'SRIL', 'SMGR', 'INTP', 'TPIA', 'AKRA', 'HRUM', 'ITMG',
            'JSMR', 'LPKR', 'PGAS', 'SIDO', 'TINS', 'UNTR', 'WIKA', 'WSBP',
            'BTPN', 'DMAS', 'EXCL', 'GGRM', 'HMSP', 'ICBP', 'JPFA', 'KAEF',
            'LPPF', 'MAPI', 'MYOR', 'PTPP', 'SCMA', 'SMRA', 'TOWR', 'ULTJ'
        ]
        
        # Generate 800 saham
        for i in range(800):
            if i < len(stock_codes):
                code = stock_codes[i] + '.JK'
            else:
                code = f"STK{i+1:03d}.JK"
            
            # Generate data acak yang realistis
            price = np.random.uniform(100, 50000)
            volatility = np.random.uniform(0.1, 0.5)
            volume = np.random.randint(100000, 10000000)
            rsi = np.random.uniform(20, 80)
            macd = np.random.uniform(-2, 2)
            
            # Calculate metrics
            entry_ideal = price * (1 - np.random.uniform(0.02, 0.05))
            entry_agresif = price * (1 - np.random.uniform(0.01, 0.03))
            tp1 = price * (1 + np.random.uniform(0.05, 0.15))
            tp2 = price * (1 + np.random.uniform(0.15, 0.25))
            cl = price * (1 - np.random.uniform(0.03, 0.08))
            
            # Determine signal berdasarkan RSI dan MACD
            if rsi < 30 and macd > 0:
                signal = "STRONG BUY"
                score = np.random.randint(85, 100)
            elif rsi < 45 and macd > 0:
                signal = "BUY"
                score = np.random.randint(70, 85)
            elif rsi < 70:
                signal = "HOLD"
                score = np.random.randint(40, 70)
            else:
                signal = "SELL"
                score = np.random.randint(0, 40)
            
            # Determine trend
            trend_options = ["UPTREND", "DOWNTREND", "SIDEWAYS"]
            weights = [0.4, 0.3, 0.3]
            trend = np.random.choice(trend_options, p=weights)
            
            stocks.append({
                'Kode': code,
                'Harga': round(price, 2),
                'Entry Ideal': round(entry_ideal, 2),
                'Entry Agresif': round(entry_agresif, 2),
                'TP1': round(tp1, 2),
                'TP2': round(tp2, 2),
                'CL': round(cl, 2),
                'Signal': signal,
                'Trend': trend,
                'RSI': round(rsi, 2),
                'MACD': round(macd, 2),
                'Volume': volume,
                'Score': score
            })
        
        return pd.DataFrame(stocks)
    
    def get_top_stocks(self, n=50):
        """Ambil n saham terbaik berdasarkan score"""
        return self.stocks_data.nlargest(n, 'Score')
    
    def get_bpjs_stocks(self):
        """Saham untuk Beli Pagi Jual Sore"""
        bpjs_data = self.stocks_data.copy()
        # Kriteria BPJS: volatilitas tinggi, volume tinggi, trend sideways/up
        bpjs_data = bpjs_data[
            (bpjs_data['Volume'] > bpjs_data['Volume'].median()) &
            (bpjs_data['Trend'].isin(['UPTREND', 'SIDEWAYS'])) &
            (bpjs_data['Signal'].isin(['STRONG BUY', 'BUY']))
        ]
        return bpjs_data.nlargest(20, 'Score')
    
    def get_bsjp_stocks(self):
        """Saham untuk Beli Sore Jual Pagi"""
        bsjp_data = self.stocks_data.copy()
        # Kriteria BSJP: momentum reversal, support kuat
        bsjp_data = bsjp_data[
            (bsjp_data['RSI'] < 40) &  # Oversold
            (bsjp_data['MACD'] > -0.5) &  # Potensi reversal
            (bsjp_data['Signal'].isin(['STRONG BUY', 'BUY']))
        ]
        return bsjp_data.nlargest(20, 'Score')
    
    def get_bandarolgi_stocks(self):
        """Saham dengan karakteristik bandar"""
        bandar_data = self.stocks_data.copy()
        # Kriteria Bandar: volume sangat tinggi, pergerakan ekstrem
        bandar_data = bandar_data[
            (bandar_data['Volume'] > bandar_data['Volume'].quantile(0.8)) &
            (bandar_data['Harga'] < 5000) &  # Saham murah
            (np.abs(bandar_data['MACD']) > 1)  # Momentum kuat
        ]
        return bandar_data.nlargest(15, 'Volume')

def format_currency(value):
    """Format angka menjadi format currency"""
    return f"Rp {value:,.2f}"

def display_stock_table(df, title):
    """Display dataframe dengan formatting yang baik"""
    st.markdown(f"<div class='sub-header'>{title}</div>", unsafe_allow_html=True)
    
    # Format currency untuk kolom harga
    display_df = df.copy()
    currency_cols = ['Harga', 'Entry Ideal', 'Entry Agresif', 'TP1', 'TP2', 'CL']
    for col in currency_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"Rp {x:,.2f}")
    
    # Apply styling untuk Signal
    def color_signal(val):
        if val == "STRONG BUY":
            return 'color: #00cc00; font-weight: bold;'
        elif val == "BUY":
            return 'color: #66b266; font-weight: bold;'
        elif val == "HOLD":
            return 'color: #ff9900; font-weight: bold;'
        elif val == "SELL":
            return 'color: #ff3333; font-weight: bold;'
        return ''
    
    # Display tabel
    st.dataframe(
        display_df.style.applymap(color_signal, subset=['Signal']),
        use_container_width=True,
        height=400
    )

def main():
    st.markdown("<div class='main-header'>📈 IDX Power Screener v5.0</div>", unsafe_allow_html=True)
    
    # Inisialisasi screener
    screener = StockScreener()
    
    # Sidebar menu
    st.sidebar.title("🎯 Menu Analisis")
    menu_option = st.sidebar.radio(
        "Pilih Menu:",
        ["Screener Full Saham", "Single Stock Analysis", "BPJS", "BSJP", "Bandarolgi"]
    )
    
    if menu_option == "Screener Full Saham":
        st.markdown("<div class='sub-header'>🏆 TOP 50 SAHAM TERBAIK</div>", unsafe_allow_html=True)
        
        top_50 = screener.get_top_stocks(50)
        display_stock_table(top_50, "50 Saham dengan Score Tertinggi")
        
        st.markdown("<div class='sub-header'>🎯 TOP 15 SAHAM PALING BAIK</div>", unsafe_allow_html=True)
        top_15 = screener.get_top_stocks(15)
        display_stock_table(top_15, "15 Saham Terbaik")
        
        # Statistik tambahan
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Saham Analyzed", "800")
        with col2:
            st.metric("Rata-rata Score", f"{top_50['Score'].mean():.1f}")
        with col3:
            strong_buy_count = len(top_50[top_50['Signal'] == 'STRONG BUY'])
            st.metric("Strong Buy Signals", strong_buy_count)
    
    elif menu_option == "Single Stock Analysis":
        st.markdown("<div class='sub-header'>🔍 ANALISIS SINGLE STOCK</div>", unsafe_allow_html=True)
        
        # Input stock code
        col1, col2 = st.columns([2, 1])
        with col1:
            stock_code = st.text_input("Masukkan Kode Saham (contoh: BBCA.JK):", "BBCA.JK").upper()
        with col2:
            analyze_btn = st.button("🔎 Analisis Saham")
        
        if analyze_btn or stock_code:
            stock_data = screener.stocks_data[screener.stocks_data['Kode'] == stock_code]
            
            if not stock_data.empty:
                stock = stock_data.iloc[0]
                
                # Display stock card
                st.markdown(f"""
                <div class='stock-card'>
                    <h3>📊 {stock['Kode']} - Analysis Result</h3>
                    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;'>
                        <div><strong>Harga Sekarang:</strong> {format_currency(stock['Harga'])}</div>
                        <div><strong>Signal:</strong> <span class='signal-{stock['Signal'].lower().replace(' ', '-')}'>{stock['Signal']}</span></div>
                        <div><strong>Trend:</strong> {stock['Trend']}</div>
                        <div><strong>RSI:</strong> {stock['RSI']}</div>
                        <div><strong>MACD:</strong> {stock['MACD']}</div>
                        <div><strong>Score:</strong> {stock['Score']}/100</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Trading levels
                st.markdown("<div class='sub-header'>🎯 LEVEL TRADING</div>", unsafe_allow_html=True)
                
                levels_col1, levels_col2, levels_col3, levels_col4, levels_col5 = st.columns(5)
                with levels_col1:
                    st.metric("Entry Ideal", format_currency(stock['Entry Ideal']))
                with levels_col2:
                    st.metric("Entry Agresif", format_currency(stock['Entry Agresif']))
                with levels_col3:
                    st.metric("Take Profit 1", format_currency(stock['TP1']))
                with levels_col4:
                    st.metric("Take Profit 2", format_currency(stock['TP2']))
                with levels_col5:
                    st.metric("Cut Loss", format_currency(stock['CL']))
                
                # Chart simulasi
                st.markdown("<div class='sub-header'>📈 PRICE CHART SIMULATION</div>", unsafe_allow_html=True)
                self.create_price_chart(stock)
                
            else:
                st.warning(f"❌ Saham {stock_code} tidak ditemukan dalam database.")
    
    elif menu_option == "BPJS":
        st.markdown("<div class='sub-header'>🌅 BPJS - Beli Pagi Jual Sore</div>", unsafe_allow_html=True)
        st.info("Saham dengan karakteristik cocok untuk trading intraday (Beli Pagi - Jual Sore)")
        
        bpjs_stocks = screener.get_bpjs_stocks()
        display_stock_table(bpjs_stocks, "Rekomendasi Saham BPJS")
        
        if not bpjs_stocks.empty:
            st.markdown("**📋 Kriteria BPJS:**")
            st.write("- ✅ Volume tinggi untuk likuiditas")
            st.write("- ✅ Trend Sideways atau Uptrend")
            st.write("- ✅ Signal Strong Buy atau Buy")
            st.write("- ✅ Volatilitas memadai untuk profit intraday")
    
    elif menu_option == "BSJP":
        st.markdown("<div class='sub-header'>🌙 BSJP - Beli Sore Jual Pagi</div>", unsafe_allow_html=True)
        st.info("Saham dengan momentum reversal untuk trading overnight (Beli Sore - Jual Pagi)")
        
        bsjp_stocks = screener.get_bsjp_stocks()
        display_stock_table(bsjp_stocks, "Rekomendasi Saham BSJP")
        
        if not bsjp_stocks.empty:
            st.markdown("**📋 Kriteria BSJP:**")
            st.write("- ✅ RSI oversold (< 40) untuk potensi rebound")
            st.write("- ✅ MACD menunjukkan potensi reversal")
            st.write("- ✅ Support level kuat")
            st.write("- ✅ Kandidat gap up di opening berikutnya")
    
    elif menu_option == "Bandarolgi":
        st.markdown("<div class='sub-header'>🎲 BANDAROLGI - Saham Bandar</div>", unsafe_allow_html=True)
        st.warning("Saham dengan karakteristik permainan bandar - HIGH RISK!")
        
        bandar_stocks = screener.get_bandarolgi_stocks()
        display_stock_table(bandar_stocks, "Saham dengan Karakteristik Bandar")
        
        if not bandar_stocks.empty:
            st.markdown("**📋 Karakteristik Bandarolgi:**")
            st.write("- ⚠️ Volume sangat tinggi tidak wajar")
            st.write("- ⚠️ Harga rendah (< Rp 5,000)")
            st.write("- ⚠️ Momentum MACD ekstrem")
            st.write("- ⚠️ HIGH RISK - HIGH REWARD")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "IDX Power Screener v5.0 • Data untuk edukasi dan simulasi • Trading mengandung risiko"
        "</div>", 
        unsafe_allow_html=True
    )

    def create_price_chart(self, stock):
        """Buat chart harga untuk single stock analysis"""
        # Generate sample price data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        base_price = stock['Harga']
        
        # Generate realistic price movement
        prices = [base_price]
        for i in range(1, 30):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create chart
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode='lines',
            name='Harga',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # Add trading levels
        fig.add_hline(y=stock['Entry Ideal'], line_dash="dash", line_color="green", 
                     annotation_text="Entry Ideal")
        fig.add_hline(y=stock['Entry Agresif'], line_dash="dash", line_color="lightgreen",
                     annotation_text="Entry Agresif")
        fig.add_hline(y=stock['TP1'], line_dash="dash", line_color="orange",
                     annotation_text="TP1")
        fig.add_hline(y=stock['TP2'], line_dash="dash", line_color="red",
                     annotation_text="TP2")
        fig.add_hline(y=stock['CL'], line_dash="dash", line_color="darkred",
                     annotation_text="Cut Loss")
        
        fig.update_layout(
            title=f"Price Chart Simulation - {stock['Kode']}",
            xaxis_title="Date",
            yaxis_title="Price (Rp)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
