import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

# --- Page Config ---
st.set_page_config(
    page_title="Pro Stock Trader & Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clean Financial UI
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e222d;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2962ff;
        margin-bottom: 10px;
    }
    .buy-box {
        background-color: rgba(38, 166, 154, 0.15);
        border: 1px solid #26a69a;
        padding: 16px;
        border-radius: 8px;
        color: #26a69a;
    }
    .sell-box {
        background-color: rgba(239, 83, 80, 0.15);
        border: 1px solid #ef5350;
        padding: 16px;
        border-radius: 8px;
        color: #ef5350;
    }
    .wait-box {
        background-color: rgba(255, 179, 0, 0.15);
        border: 1px solid #ffb300;
        padding: 16px;
        border-radius: 8px;
        color: #ffb300;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.title("⚡ Control Panel")
st.sidebar.caption("Trade Setup & Risk Settings")

# Stock Input
raw_symbol = st.sidebar.text_input("Stock Symbol (NSE / BSE)", value="RELIANCE").upper().strip()

# Auto-format symbol for Indian market
if "." in raw_symbol:
    clean_symbol = raw_symbol.split(".")[0]
    exchange = "BSE" if ".BO" in raw_symbol else "NSE"
    yf_symbol = raw_symbol
else:
    clean_symbol = raw_symbol
    exchange = "NSE"
    yf_symbol = f"{raw_symbol}.NS"

# Trade Mode & Capital
trade_mode = st.sidebar.radio("Trading Mode", ["Intraday Trading", "Long-Term / Swing"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Capital & Risk Management")
trading_capital = st.sidebar.number_input("Account Balance (₹)", value=50000, step=5000, min_value=1000)
risk_per_trade_pct = st.sidebar.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.5)

# --- TradingView Full Interactive Chart ---
st.subheader(f"📊 {exchange}:{clean_symbol} Live Chart")

tv_widget_html = f"""
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container" style="height:580px;width:100%;">
  <div id="tradingview_chart_div" style="height:calc(100% - 32px);width:100%;"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget(
  {{
    "autosize": true,
    "symbol": "{exchange}:{clean_symbol}",
    "interval": "{"5" if trade_mode == "Intraday Trading" else "D"}",
    "timezone": "Asia/Kolkata",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "enable_publishing": false,
    "hide_top_toolbar": false,
    "hide_legend": false,
    "save_image": true,
    "container_id": "tradingview_chart_div"
  }}
  );
  </script>
</div>
<!-- TradingView Widget END -->
"""
components.html(tv_widget_html, height=600)

# --- Data Fetching & Calculation ---
st.write("---")

@st.cache_data(ttl=60)
def load_market_data(ticker, mode):
    interval = "5m" if mode == "Intraday Trading" else "1d"
    period = "1mo" if mode == "Intraday Trading" else "1y"
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception:
        return pd.DataFrame()

df = load_market_data(yf_symbol, trade_mode)

# Fallback check for newly listed stocks without .NS
if df.empty or len(df) < 5:
    fallback_symbol = f"{clean_symbol}.BO"
    df = load_market_data(fallback_symbol, trade_mode)

if df.empty or len(df) < 15:
    st.error(f"⚠️ `{clean_symbol}` ka data load nahi ho paya. Kripya symbol verify karein ya confirm karein ki share active traded hai.")
else:
    # Technical Indicators
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # RSI
    diff = df['Close'].diff()
    gain = diff.clip(lower=0).rolling(window=14).mean()
    loss = (-diff.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR (Average True Range)
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    current_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2])
    change_pts = current_price - prev_price
    change_pct = (change_pts / prev_price) * 100
    current_rsi = float(df['RSI'].iloc[-1])
    current_ema20 = float(df['EMA_20'].iloc[-1])
    current_atr = float(df['ATR'].iloc[-1])

    # Top Snapshot Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LTP (Last Price)", f"₹{current_price:,.2f}", f"{change_pts:+.2f} ({change_pct:+.2f}%)")
    c2.metric("RSI (14)", f"{current_rsi:.1f}")
    c3.metric("EMA 20", f"₹{current_ema20:,.2f}")
    c4.metric("ATR (Volatility Range)", f"₹{current_atr:,.2f}")

    # --- Signal Engine ---
    signal_score = 0
    rationales = []

    if current_price > current_ema20:
        signal_score += 1
        rationales.append("✅ Price 20-period EMA ke upar trade kar raha hai (Bullish Trend).")
    else:
        signal_score -= 1
        rationales.append("🔻 Price 20-period EMA ke neeche chal raha hai (Bearish Trend).")

    if current_rsi > 55:
        signal_score += 1
        rationales.append("✅ RSI 55 ke upar hai, buying momentum ban raha hai.")
    elif current_rsi < 45:
        signal_score -= 1
        rationales.append("🔻 RSI 45 ke neeche hai, selling pressure dikh raha hai.")
    else:
        rationales.append("⚖️ RSI neutral zone (45-55) me sideways move kar raha hai.")

    if df['Close'].iloc[-1] > df['Open'].iloc[-1]:
        signal_score += 1
        rationales.append("✅ Latest candle bullish (Green) close hui hai.")
    else:
        signal_score -= 1
        rationales.append("🔻 Latest candle bearish (Red) close hui hai.")

    # Determine Setup
    if signal_score >= 2:
        trade_action = "BUY / CALL"
        box_class = "buy-box"
    elif signal_score <= -2:
        trade_action = "SELL / PUT"
        box_class = "sell-box"
    else:
        trade_action = "WAIT (No High-Probability Setup)"
        box_class = "wait-box"

    st.markdown(f"""
        <div class="{box_class}">
            <h3 style="margin:0;">Signal: {trade_action} | Strength Score: {signal_score}/3</h3>
        </div>
    """, unsafe_allow_html=True)

    # --- Trade Plan (Entry, SL, Targets & Risk Qty) ---
    st.subheader("📋 Systematic Trade Plan")

    max_risk_inr = trading_capital * (risk_per_trade_pct / 100.0)
    sl_distance = max(current_atr * 1.5, current_price * 0.005)

    if trade_action == "BUY / CALL":
        entry_price = current_price
        sl_price = max(0.05, entry_price - sl_distance)
        t1_price = entry_price + (sl_distance * 1.5)
        t2_price = entry_price + (sl_distance * 2.5)
        shares_qty = int(max_risk_inr / sl_distance) if sl_distance > 0 else 0

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Recommended Entry", f"₹{entry_price:,.2f}")
        p2.metric("Stop-Loss (SL)", f"₹{sl_price:,.2f}", f"-₹{sl_distance:,.2f}", delta_color="inverse")
        p3.metric("Target 1 (1:1.5)", f"₹{t1_price:,.2f}", f"+₹{(t1_price - entry_price):,.2f}")
        p4.metric("Target 2 (1:2.5)", f"₹{t2_price:,.2f}", f"+₹{(t2_price - entry_price):,.2f}")

        st.info(f"💡 **Position Sizing:** Total Risk = **₹{max_risk_inr:,.2f}** | Recommended Quantity = **{shares_qty} Shares** (Capital Required ≈ ₹{(shares_qty * entry_price):,.2f})")

    elif trade_action == "SELL / PUT":
        entry_price = current_price
        sl_price = entry_price + sl_distance
        t1_price = max(0.05, entry_price - (sl_distance * 1.5))
        t2_price = max(0.05, entry_price - (sl_distance * 2.5))
        shares_qty = int(max_risk_inr / sl_distance) if sl_distance > 0 else 0

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Sell Entry", f"₹{entry_price:,.2f}")
        p2.metric("Stop-Loss (SL)", f"₹{sl_price:,.2f}", f"+₹{sl_distance:,.2f}", delta_color="inverse")
        p3.metric("Target 1 (1:1.5)", f"₹{t1_price:,.2f}", f"-₹{(entry_price - t1_price):,.2f}")
        p4.metric("Target 2 (1:2.5)", f"₹{t2_price:,.2f}", f"-₹{(entry_price - t2_price):,.2f}")

        st.info(f"💡 **Position Sizing:** Total Risk = **₹{max_risk_inr:,.2f}** | Recommended Quantity = **{shares_qty} Shares**")

    else:
        st.warning("Market abhi neutral / consolidate zone me hai. Jab tak score +2 ya -2 na ho, fresh trade lene se bachein.")

    # Logic Reasons
    with st.expander("🔍 Dekhein Signal ke peeche ke Technical Reasons"):
        for reason in rationales:
            st.write(reason)
