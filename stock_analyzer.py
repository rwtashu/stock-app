"""
Stock Analyzer (Intraday + Long Term)
Run:
    pip install streamlit yfinance plotly pandas numpy
    streamlit run stock_analyzer.py

Symbols: NSE -> RELIANCE.NS, TCS.NS | BSE -> RELIANCE.BO | US -> AAPL
Educational tool only. Not financial advice.
"""
import math
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Stock Analyzer – Intraday & Long Term")

# ---------------- Sidebar ----------------
with st.sidebar:
    symbol = st.text_input("Symbol (e.g. RELIANCE.NS, AAPL)", "RELIANCE.NS").upper().strip()
    mode = st.radio("Mode", ["Intraday", "Long Term"])
    capital = st.number_input("Capital", min_value=1000, value=100000, step=10000)
    risk_pct = st.slider("Risk per trade (%)", 0.5, 3.0, 1.0, 0.5)
    run = st.button("Analyze", type="primary")

CFG = {
    "Intraday": dict(interval="5m", period="5d", fast=9, slow=21, sl_mult=1.5, bars=300),
    "Long Term": dict(interval="1d", period="3y", fast=50, slow=200, sl_mult=2.5, bars=250),
}


# ---------------- Data ----------------
@st.cache_data(ttl=300, show_spinner=False)
def load(sym, interval, period):
    df = yf.download(sym, interval=interval, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ---------------- Indicators ----------------
def add_indicators(df, fast, slow, intraday):
    df = df.copy()
    c = df["Close"]
    df["EMA_F"] = c.ewm(span=fast, adjust=False).mean()
    df["EMA_S"] = c.ewm(span=slow, adjust=False).mean()

    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_SIG"] = macd.ewm(span=9, adjust=False).mean()

    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - c.shift()).abs(), (df["Low"] - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    if intraday:
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        day = pd.Series(df.index.date, index=df.index)
        df["VWAP"] = (tp * df["Volume"]).groupby(day).cumsum() / df["Volume"].groupby(day).cumsum()
    return df


# ---------------- Candlestick patterns ----------------
BULLISH = ["Hammer", "Bullish Engulfing", "Morning Star"]
BEARISH = ["Shooting Star", "Bearish Engulfing", "Evening Star"]


def detect_patterns(df):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    green, red = c > o, c < o

    p = pd.DataFrame(index=df.index)
    p["Doji"] = (body / rng) < 0.1
    p["Hammer"] = (lower >= 2 * body) & (upper <= body) & (body / rng > 0.05)
    p["Shooting Star"] = (upper >= 2 * body) & (lower <= body) & (body / rng > 0.05)
    p["Bullish Engulfing"] = red.shift() & green & (c >= o.shift()) & (o <= c.shift())
    p["Bearish Engulfing"] = green.shift() & red & (c <= o.shift()) & (o >= c.shift())
    big1 = body.shift(2) > body.rolling(10).mean().shift(2)
    small = body.shift(1) < body.shift(2) * 0.5
    p["Morning Star"] = red.shift(2) & big1 & small & green & (c > (o.shift(2) + c.shift(2)) / 2)
    p["Evening Star"] = green.shift(2) & big1 & small & red & (c < (o.shift(2) + c.shift(2)) / 2)
    return p.fillna(False)


# ---------------- Support / Resistance ----------------
def sr_levels(df, price, atr, n=5, lookback=250):
    d = df.tail(lookback)
    lo = d["Low"][d["Low"] == d["Low"].rolling(2 * n + 1, center=True).min()]
    hi = d["High"][d["High"] == d["High"].rolling(2 * n + 1, center=True).max()]
    sup, res = lo[lo < price], hi[hi > price]
    support = float(sup.max()) if len(sup) else price - 2 * atr
    resistance = float(res.min()) if len(res) else price + 2 * atr
    return support, resistance


# ---------------- Signal engine ----------------
def build_signal(df, pats, intraday, sl_mult, capital, risk_pct, daily_atr):
    last = df.iloc[-1]
    price, atr = float(last["Close"]), float(last["ATR"])
    score, reasons = 0, []

    if last["EMA_F"] > last["EMA_S"] and price > last["EMA_S"]:
        score += 2; reasons.append("Trend up (fast EMA > slow EMA, price above both)")
    elif last["EMA_F"] < last["EMA_S"] and price < last["EMA_S"]:
        score -= 2; reasons.append("Trend down (fast EMA < slow EMA, price below both)")
    else:
        reasons.append("Trend mixed / sideways")

    if price > last["EMA_F"]:
        score += 1
    else:
        score -= 1

    rsi = float(last["RSI"])
    if rsi > 70:
        score -= 1; reasons.append(f"RSI {rsi:.0f}: overbought, pullback risk")
    elif rsi < 30:
        score += 1; reasons.append(f"RSI {rsi:.0f}: oversold, bounce possible")
    elif rsi > 55:
        score += 1; reasons.append(f"RSI {rsi:.0f}: bullish momentum")
    elif rsi < 45:
        score -= 1; reasons.append(f"RSI {rsi:.0f}: bearish momentum")
    else:
        reasons.append(f"RSI {rsi:.0f}: neutral")

    if last["MACD"] > last["MACD_SIG"]:
        score += 1; reasons.append("MACD above signal line")
    else:
        score -= 1; reasons.append("MACD below signal line")

    if intraday and "VWAP" in df:
        if price > last["VWAP"]:
            score += 1; reasons.append("Price above VWAP")
        else:
            score -= 1; reasons.append("Price below VWAP")

    recent = pats.tail(3)
    for name in BULLISH:
        if recent[name].any():
            score += 1; reasons.append(f"Bullish pattern: {name}")
    for name in BEARISH:
        if recent[name].any():
            score -= 1; reasons.append(f"Bearish pattern: {name}")
    if recent["Doji"].iloc[-1]:
        reasons.append("Doji on last candle: indecision")

    support, resistance = sr_levels(df, price, atr)
    reasons.append(f"Nearest support {support:.2f}, resistance {resistance:.2f}")

    if score >= 3:
        side = "BUY"
    elif score <= -3:
        side = "SELL"
    else:
        side = "WAIT"

    plan = None
    if side in ("BUY", "SELL"):
        if side == "BUY":
            risk = min(max(price - (support - 0.25 * atr), 0.8 * atr), sl_mult * atr)
            sl, t1, t2 = price - risk, price + 1.5 * risk, price + 2.5 * risk
        else:
            risk = min(max((resistance + 0.25 * atr) - price, 0.8 * atr), sl_mult * atr)
            sl, t1, t2 = price + risk, price - 1.5 * risk, price - 2.5 * risk
        qty = max(math.floor(capital * risk_pct / 100 / risk), 0)
        plan = dict(entry=price, sl=sl, t1=t1, t2=t2, risk=risk, qty=qty)

    return dict(side=side, score=score, reasons=reasons, plan=plan,
                support=support, resistance=resistance, price=price, rsi=rsi)


# ---------------- Chart ----------------
def make_chart(df, intraday, sig):
    fmt = "%d-%b %H:%M" if intraday else "%d-%b-%y"
    x = df.index.strftime(fmt)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=x, open=df["Open"], high=df["High"], low=df["Low"],
                                 close=df["Close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["EMA_F"], name="Fast EMA", line=dict(width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["EMA_S"], name="Slow EMA", line=dict(width=1.2)), row=1, col=1)
    if intraday and "VWAP" in df:
        fig.add_trace(go.Scatter(x=x, y=df["VWAP"], name="VWAP", line=dict(width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["RSI"], name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)

    fig.add_hline(y=sig["support"], line_dash="dash", line_color="green", annotation_text="Support", row=1, col=1)
    fig.add_hline(y=sig["resistance"], line_dash="dash", line_color="red", annotation_text="Resistance", row=1, col=1)
    if sig["plan"]:
        p = sig["plan"]
        for label, val in [("Entry", p["entry"]), ("SL", p["sl"]), ("T1", p["t1"]), ("T2", p["t2"])]:
            fig.add_hline(y=val, line_dash="solid", line_width=1, annotation_text=label, row=1, col=1)

    fig.update_xaxes(type="category", nticks=10, rangeslider_visible=False)
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return fig


# ---------------- Main ----------------
if run or symbol:
    cfg = CFG[mode]
    intraday = mode == "Intraday"

    with st.spinner("Data la raha hu..."):
        raw = load(symbol, cfg["interval"], cfg["period"])
        daily = load(symbol, "1d", "1y")

    if raw.empty or daily.empty or len(raw) < 60:
        st.error("Data nahi mila. Symbol check karo (NSE ke liye .NS lagao, jaise RELIANCE.NS).")
        st.stop()

    df = add_indicators(raw, cfg["fast"], cfg["slow"], intraday).dropna(subset=["EMA_F", "EMA_S", "RSI", "ATR"])
    daily = add_indicators(daily, 20, 50, False)
    daily_atr = float(daily["ATR"].iloc[-1])
    pats = detect_patterns(df)
    sig = build_signal(df, pats, intraday, cfg["sl_mult"], capital, risk_pct, daily_atr)
    price = sig["price"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{price:.2f}")
    c2.metric("Signal", sig["side"])
    c3.metric("Score", f"{sig['score']} (range -7 to +7)")
    c4.metric("RSI", f"{sig['rsi']:.0f}")

    st.plotly_chart(make_chart(df.tail(cfg["bars"]), intraday, sig), use_container_width=True)

    st.subheader("Trade plan")
    if sig["plan"]:
        p = sig["plan"]
        if (not intraday) and sig["side"] == "SELL":
            st.warning("Long term me SELL signal = fresh buy avoid karo / holding ho to exit ya stop-loss lagao.")
        st.table(pd.DataFrame({
            "Level": ["Entry", "Stop-loss", "Target 1 (1:1.5)", "Target 2 (1:2.5)", "Risk per share", "Quantity (as per risk %)"],
            "Value": [f"{p['entry']:.2f}", f"{p['sl']:.2f}", f"{p['t1']:.2f}", f"{p['t2']:.2f}",
                      f"{p['risk']:.2f}", p["qty"]],
        }))
        if intraday:
            st.caption("Intraday: din ke end tak position band karo. SELL = short selling.")
    else:
        st.info(f"Abhi clear setup nahi hai (WAIT). Buy ke liye support {sig['support']:.2f} ke paas "
                f"reversal candle ya resistance {sig['resistance']:.2f} ke upar breakout ka wait karo.")

    st.subheader("Kitna upar / niche ja sakta hai (approx, ATR based)")
    if intraday:
        rows = [("Aaj ka typical range", 1)]
    else:
        rows = [("~1 hafta (5 din)", 5), ("~1 mahina (21 din)", 21), ("~3 mahine (63 din)", 63)]
    out = []
    for name, days in rows:
        move = daily_atr * math.sqrt(days)
        out.append({"Period": name, "Upar (approx)": f"{price + move:.2f}",
                    "Niche (approx)": f"{price - move:.2f}", "Move ±": f"{move:.2f}"})
    st.table(pd.DataFrame(out))
    st.caption("Ye volatility based andaza hai, guarantee nahi. Price iske bahar bhi ja sakta hai.")

    st.subheader("Kyu ye signal?")
    for r in sig["reasons"]:
        st.write("• " + r)

    st.subheader("Recent candlestick patterns (last 10 candles)")
    recent = pats.tail(10)
    found = [(str(ts), n) for ts, row in recent.iterrows() for n in recent.columns if row[n]]
    if found:
        st.table(pd.DataFrame(found, columns=["Time", "Pattern"]))
    else:
        st.write("Koi major pattern nahi mila.")

    st.caption("⚠️ Sirf educational tool hai, financial advice nahi. Pehle paper trading me test karo.")
