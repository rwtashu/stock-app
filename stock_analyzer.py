""" Stock Analyzer v2 - TradingView-style chart + Intraday / Long Term analysis Run locally: streamlit run stock_analyzer.py Educational tool only. Not financial advice. """
import html
import json
import math

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")
st.markdown(
    "<style>.block-container{padding-top:1.2rem;padding-bottom:2rem}</style>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ config
TF = {  # label: (yfinance interval, period)
    "1m": ("1m", "7d"),
    "5m": ("5m", "1mo"),
    "15m": ("15m", "1mo"),
    "30m": ("30m", "1mo"),
    "1h": ("60m", "6mo"),
    "1D": ("1d", "5y"),
    "1W": ("1wk", "10y"),
    "1M": ("1mo", "max"),
}
INTRADAY_TF = {"1m", "5m", "15m", "30m", "1h"}
CFG_INTRA = dict(fast=9, slow=21, sl_mult=1.5)
CFG_LONG = dict(fast=50, slow=200, sl_mult=2.5)
BULLISH = ["Hammer", "Bullish Engulfing", "Morning Star"]
BEARISH = ["Shooting Star", "Bearish Engulfing", "Evening Star"]


# ------------------------------------------------------------------ data
@st.cache_data(ttl=300, show_spinner=False)
def load(sym, interval, period):
    try:
        df = yf.download(sym, interval=interval, period=period,
                         auto_adjust=True, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df[~df.index.duplicated(keep="last")]


@st.cache_data(ttl=86400, show_spinner=False)
def get_name(sym):
    try:
        info = yf.Ticker(sym).info
        return info.get("longName") or info.get("shortName") or sym
    except Exception:
        return sym


@st.cache_data(ttl=600, show_spinner=False)
def search_symbols(q, india_only):
    quotes = []
    try:
        quotes = yf.Search(q, max_results=15).quotes
    except Exception:
        quotes = []
    if not quotes:
        try:
            r = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": q, "quotesCount": 15, "newsCount": 0},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=8,
            )
            quotes = r.json().get("quotes", [])
        except Exception:
            quotes = []
    out, seen = [], set()
    for x in quotes:
        sym = x.get("symbol")
        if not sym or sym in seen:
            continue
        if x.get("quoteType") not in ("EQUITY", "ETF", "INDEX", None):
            continue
        if india_only and not (sym.endswith((".NS", ".BO")) or sym.startswith("^")):
            continue
        name = x.get("longname") or x.get("shortname") or sym
        exch = x.get("exchDisp") or x.get("exchange") or ""
        out.append({"symbol": sym, "name": name, "label": f"{name} · {sym} · {exch}"})
        seen.add(sym)
    return out


# ------------------------------------------------------------------ indicators
def true_range(df):
    c = df["Close"]
    return pd.concat(
        [df["High"] - df["Low"], (df["High"] - c.shift()).abs(), (df["Low"] - c.shift()).abs()],
        axis=1,
    ).max(axis=1)


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
    df["ATR"] = true_range(df).ewm(alpha=1 / 14, adjust=False).mean()
    if intraday:
        df["VWAP"] = vwap(df)
    return df


def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = pd.Series(df.index.date, index=df.index)
    return (tp * df["Volume"]).groupby(day).cumsum() / df["Volume"].groupby(day).cumsum().replace(0, np.nan)


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
    return p.fillna(False).astype(bool)


def sr_levels(df, price, atr, n=5, lookback=250):
    d = df.tail(lookback)
    lo = d["Low"][d["Low"] == d["Low"].rolling(2 * n + 1, center=True).min()]
    hi = d["High"][d["High"] == d["High"].rolling(2 * n + 1, center=True).max()]
    sup, res = lo[lo < price], hi[hi > price]
    support = float(sup.max()) if len(sup) else price - 2 * atr
    resistance = float(res.min()) if len(res) else price + 2 * atr
    return support, resistance


# ------------------------------------------------------------------ signal engine
def build_signal(df, pats, intraday, sl_mult, capital, risk_pct):
    last = df.iloc[-1]
    price, atr = float(last["Close"]), float(last["ATR"])
    state = {"score": 0}
    reasons = []

    def add(cat, text, s):
        state["score"] += s
        reasons.append((cat, text, s))

    if last["EMA_F"] > last["EMA_S"] and price > last["EMA_S"]:
        add("Trend", "Trend upar hai (fast EMA > slow EMA, price dono ke upar)", 2)
    elif last["EMA_F"] < last["EMA_S"] and price < last["EMA_S"]:
        add("Trend", "Trend niche hai (fast EMA < slow EMA, price dono ke niche)", -2)
    else:
        add("Trend", "Trend mixed / sideways hai", 0)
    if price > last["EMA_F"]:
        add("Trend", "Price fast EMA ke upar", 1)
    else:
        add("Trend", "Price fast EMA ke niche", -1)
    if intraday and "VWAP" in df and not pd.isna(last["VWAP"]):
        if price > last["VWAP"]:
            add("Trend", "Price VWAP ke upar (buyers strong)", 1)
        else:
            add("Trend", "Price VWAP ke niche (sellers strong)", -1)

    rsi = float(last["RSI"])
    if rsi > 70:
        add("Momentum", f"RSI {rsi:.0f}: overbought, pullback ka risk", -1)
    elif rsi < 30:
        add("Momentum", f"RSI {rsi:.0f}: oversold, bounce ho sakta hai", 1)
    elif rsi > 55:
        add("Momentum", f"RSI {rsi:.0f}: bullish momentum", 1)
    elif rsi < 45:
        add("Momentum", f"RSI {rsi:.0f}: bearish momentum", -1)
    else:
        add("Momentum", f"RSI {rsi:.0f}: neutral", 0)
    if last["MACD"] > last["MACD_SIG"]:
        add("Momentum", "MACD signal line ke upar", 1)
    else:
        add("Momentum", "MACD signal line ke niche", -1)

    recent = pats.tail(3)
    found_any = False
    for name in BULLISH:
        if recent[name].any():
            add("Candlestick", f"Bullish pattern: {name}", 1)
            found_any = True
    for name in BEARISH:
        if recent[name].any():
            add("Candlestick", f"Bearish pattern: {name}", -1)
            found_any = True
    if recent["Doji"].iloc[-1]:
        add("Candlestick", "Aakhri candle Doji hai: market me confusion", 0)
        found_any = True
    if not found_any:
        add("Candlestick", "Pichhli 3 candles me koi major pattern nahi", 0)

    support, resistance = sr_levels(df, price, atr)
    add("Levels", f"Nearest support {support:,.2f} | resistance {resistance:,.2f}", 0)

    score = state["score"]
    side = "BUY" if score >= 3 else "SELL" if score <= -3 else "WAIT"

    plan = None
    if side == "BUY":
        risk = min(max(price - (support - 0.25 * atr), 0.8 * atr), sl_mult * atr)
        plan = dict(entry=price, sl=price - risk, t1=price + 1.5 * risk, t2=price + 2.5 * risk, risk=risk)
    elif side == "SELL":
        risk = min(max((resistance + 0.25 * atr) - price, 0.8 * atr), sl_mult * atr)
        plan = dict(entry=price, sl=price + risk, t1=price - 1.5 * risk, t2=price - 2.5 * risk, risk=risk)
    if plan:
        plan["qty"] = max(math.floor(capital * risk_pct / 100 / plan["risk"]), 0)

    return dict(side=side, score=score, reasons=reasons, plan=plan, support=support,
                resistance=resistance, price=price, rsi=rsi)


def analyze(df, intraday, cfg, capital, risk_pct):
    if df is None or len(df) < 30:
        return None
    n = len(df)
    fast = min(cfg["fast"], max(5, n // 4))
    slow = min(cfg["slow"], max(10, n // 2))
    d = add_indicators(df, fast, slow, intraday).dropna(subset=["EMA_F", "EMA_S", "RSI", "ATR", "MACD"])
    if len(d) < 20:
        return None
    pats = detect_patterns(d)
    sig = build_signal(d, pats, intraday, cfg["sl_mult"], capital, risk_pct)
    sig.update(pats=pats, fast=fast, slow=slow, bars=n)
    return sig


# ------------------------------------------------------------------ chart (TradingView lightweight-charts)
def to_times(idx, daily):
    if daily:
        return list(idx.strftime("%Y-%m-%d"))
    naive = idx.tz_localize(None) if idx.tz is not None else idx
    secs = (naive - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    return [int(x) for x in secs]


def line_series(times, series, color, name, width=1):
    data = [{"time": t, "value": round(float(v), 2)} for t, v in zip(times, series.tolist()) if not pd.isna(v)]
    return {"name": name, "color": color, "width": width, "data": data}


CHART_HTML = """ <div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1117;border-radius:8px;padding:6px"> <div id="legend" style="color:#d1d4dc;font-size:13px;padding:4px 6px;min-height:38px;line-height:1.5"></div> <div id="chart" style="width:100%;height:480px"></div> <div style="display:flex;gap:8px;padding:8px 4px 2px 4px;flex-wrap:wrap"> <button class="b" onclick="zoom(0.65)">＋ Zoom in</button> <button class="b" onclick="zoom(1.5)">－ Zoom out</button> <button class="b" onclick="chart.timeScale().fitContent()">Fit all</button> <button class="b" onclick="chart.timeScale().scrollToRealTime()">Latest ▶</button> </div> </div> <style>.b{background:#1f2430;color:#d1d4dc;border:1px solid #2a2e39;border-radius:6px;padding:7px 12px;font-size:13px;cursor:pointer}.b:active{background:#2a2e39}</style> <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script> <script> const D = __DATA__; const el = document.getElementById('chart'); const legend = document.getElementById('legend'); if (typeof LightweightCharts === 'undefined') { el.innerHTML = '<div style="color:#ef5350;padding:20px">Chart library load nahi hui. Internet check karke page refresh karo.</div>'; } else { const chart = LightweightCharts.createChart(el, { width: el.clientWidth, height: 480, layout: { background: { type: 'solid', color: '#0e1117' }, textColor: '#d1d4dc' }, grid: { vertLines: { color: '#1b1f2b' }, horzLines: { color: '#1b1f2b' } }, crosshair: { mode: LightweightCharts.CrosshairMode.Normal }, rightPriceScale: { borderColor: '#2a2e39' }, timeScale: { borderColor: '#2a2e39', timeVisible: D.intraday, secondsVisible: false, rightOffset: 6 }, handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }, handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true }, localization: { priceFormatter: p => p.toFixed(2) } }); const candles = chart.addCandlestickSeries({ upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350' }); candles.setData(D.candles); if (D.volume.length) { const vol = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: '' }); vol.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } }); vol.setData(D.volume); } (D.lines || []).forEach(l => { const s = chart.addLineSeries({ color: l.color, lineWidth: l.width, priceLineVisible: false, lastValueVisible: false, title: l.name }); s.setData(l.data); }); (D.levels || []).forEach(l => { candles.createPriceLine({ price: l.price, color: l.color, lineWidth: 1, lineStyle: l.style, axisLabelVisible: true, title: l.title }); }); const f = v => (+v).toFixed(2); function setLegend(c) { if (!c) return; const col = c.close >= c.open ? '#26a69a' : '#ef5350'; legend.innerHTML = '<b style="font-size:15px;color:#fff">' + D.title + '</b> <span style="color:#9aa0a6">· ' + D.tf + '</span><br>' + 'O <span style="color:' + col + '">' + f(c.open) + '</span> &nbsp;H <span style="color:' + col + '">' + f(c.high) + '</span> &nbsp;L <span style="color:' + col + '">' + f(c.low) + '</span> &nbsp;C <span style="color:' + col + '">' + f(c.close) + '</span>'; } const n = D.candles.length; const lastBar = D.candles[n - 1]; setLegend(lastBar); chart.subscribeCrosshairMove(p => { const c = p && p.seriesData ? p.seriesData.get(candles) : null; setLegend(c || lastBar); }); chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - 120), to: n + 8 }); function zoom(factor) { const ts = chart.timeScale(); const r = ts.getVisibleLogicalRange(); if (!r) return; const mid = (r.from + r.to) / 2, half = (r.to - r.from) / 2 * factor; ts.setVisibleLogicalRange({ from: mid - half, to: mid + half }); } window.zoom = zoom; window.chart = chart; window.addEventListener('resize', () => chart.applyOptions({ width: el.clientWidth })); } </script> """


def render_chart(df, tf, title, overlays, sig):
    daily = tf not in INTRADAY_TF
    times = to_times(df.index, daily)
    o, h, l, c, v = (df[k].tolist() for k in ["Open", "High", "Low", "Close", "Volume"])
    candles = [
        {"time": t, "open": round(a, 2), "high": round(b, 2), "low": round(d, 2), "close": round(e, 2)}
        for t, a, b, d, e in zip(times, o, h, l, c)
    ]
    volume = []
    if "Volume" in overlays:
        volume = [
            {"time": t, "value": float(x),
             "color": "rgba(38,166,154,0.45)" if e >= a else "rgba(239,83,80,0.45)"}
            for t, x, a, e in zip(times, v, o, c)
        ]
    lines = []
    if "EMA 20/50" in overlays:
        lines.append(line_series(times, df["Close"].ewm(span=20, adjust=False).mean(), "#2962ff", "EMA20"))
        if len(df) >= 60:
            lines.append(line_series(times, df["Close"].ewm(span=50, adjust=False).mean(), "#ff9800", "EMA50"))
    if "VWAP" in overlays and not daily:
        lines.append(line_series(times, vwap(df), "#ab47bc", "VWAP"))

    levels = []
    if sig:
        if "Support/Resistance" in overlays:
            levels.append(dict(price=round(sig["support"], 2), color="#26a69a", style=2, title="Support"))
            levels.append(dict(price=round(sig["resistance"], 2), color="#ef5350", style=2, title="Resistance"))
        if "Trade levels" in overlays and sig["plan"]:
            p = sig["plan"]
            levels += [
                dict(price=round(p["entry"], 2), color="#2962ff", style=0, title="Entry"),
                dict(price=round(p["sl"], 2), color="#ef5350", style=0, title="SL"),
                dict(price=round(p["t1"], 2), color="#26a69a", style=0, title="T1"),
                dict(price=round(p["t2"], 2), color="#26a69a", style=0, title="T2"),
            ]
    payload = dict(title=html.escape(title), tf=tf, intraday=not daily,
                   candles=candles, volume=volume, lines=lines, levels=levels)
    page = CHART_HTML.replace("__DATA__", json.dumps(payload).replace("</", "<\\/"))
    components.html(page, height=640)


# ------------------------------------------------------------------ analysis tab renderer
ICON = {1: "🟢", -1: "🔴", 0: "⚪"}


def render_tab(sig, kind, daily_atr, cur, empty_msg):
    if sig is None:
        st.info(empty_msg)
        return
    intraday = kind == "intraday"
    side = sig["side"]
    if intraday:
        label = {"BUY": "BUY (Long)", "SELL": "SELL (Short)", "WAIT": "WAIT"}[side]
    else:
        label = {"BUY": "BUY / ACCUMULATE", "SELL": "AVOID / EXIT", "WAIT": "WAIT"}[side]
    box = {"BUY": st.success, "SELL": st.error, "WAIT": st.warning}[side]
    box(f"### {label}\nScore **{sig['score']}** (BUY ≥ +3, SELL ≤ −3) · RSI **{sig['rsi']:.0f}**")

    price = sig["price"]
    st.markdown("#### 📍 Trade plan")
    if sig["plan"]:
        p = sig["plan"]
        rows = [
            ("Entry", f"{cur}{p['entry']:,.2f}"),
            ("Stop-loss", f"{cur}{p['sl']:,.2f}"),
            ("Target 1 (1:1.5)", f"{cur}{p['t1']:,.2f}"),
            ("Target 2 (1:2.5)", f"{cur}{p['t2']:,.2f}"),
            ("Risk per share", f"{cur}{p['risk']:,.2f}"),
            ("Quantity (risk % ke hisaab se)", str(p["qty"])),
        ]
        st.table(pd.DataFrame(rows, columns=["Level", "Value"]).set_index("Level"))
        if intraday:
            st.caption("Intraday: din ke end tak position band karo. SELL = short selling.")
        elif side == "SELL":
            st.caption("Long term me SELL = fresh buy avoid karo, holding ho to exit ya stop-loss lagao.")
    else:
        st.write(f"Abhi clear setup nahi hai. **{cur}{sig['support']:,.2f}** ke paas reversal candle ya "
                 f"**{cur}{sig['resistance']:,.2f}** ke upar breakout ka wait karo.")

    st.markdown("#### 📏 Kitna upar / niche ja sakta hai (approx)")
    periods = [("Aaj ka typical range", 1)] if intraday else [
        ("~1 hafta (5 din)", 5), ("~1 mahina (21 din)", 21), ("~3 mahine (63 din)", 63)]
    rows = []
    for name, days in periods:
        mv = daily_atr * math.sqrt(days)
        rows.append((name, f"{cur}{price + mv:,.2f}", f"{cur}{price - mv:,.2f}", f"±{mv:,.2f}"))
    st.table(pd.DataFrame(rows, columns=["Period", "Upar", "Niche", "Move"]).set_index("Period"))
    st.caption("ATR (volatility) par based andaza hai, guarantee nahi.")

    st.markdown("#### 🔎 Signal kyu bana")
    for cat in ["Trend", "Momentum", "Candlestick", "Levels"]:
        items = [r for r in sig["reasons"] if r[0] == cat]
        if items:
            st.markdown(f"**{cat}**")
            for _, text, s in items:
                st.markdown(f"{ICON[s]} {text}")

    with st.expander("🕯️ Recent candlestick patterns (last 10 candles)"):
        rec = sig["pats"].tail(10)
        found = [(str(ts)[:16], n) for ts, row in rec.iterrows() for n in rec.columns if row[n]]
        if found:
            st.table(pd.DataFrame(found, columns=["Time", "Pattern"]).set_index("Time"))
        else:
            st.write("Koi major pattern nahi mila.")


# ================================================================== UI
if "symbol" not in st.session_state:
    st.session_state.symbol = "RELIANCE.NS"
    st.session_state.names = {}

st.title("📈 Stock Analyzer")

# ---- search
sc1, sc2 = st.columns([4, 1])
q = sc1.text_input("Search", placeholder="🔍 Stock ka naam ya symbol likho (reliance, tata motors, zomato...)",
                   label_visibility="collapsed")
india_only = sc2.checkbox("Sirf India", value=True)
if q.strip():
    results = search_symbols(q.strip(), india_only)
    if results:
        labels = [r["label"] for r in results]
        pick = st.selectbox("Search results (yahan se stock chuno)", labels)
        r = results[labels.index(pick)]
        st.session_state.symbol = r["symbol"]
        st.session_state.names[r["symbol"]] = r["name"]
    else:
        st.info("Search me kuch nahi mila. 'Sirf India' band karke dekho, ya symbol seedha use karo "
                "(NSE: NAME.NS, BSE: NAME.BO).")
        if st.button(f"'{q.strip().upper()}' ko seedha symbol ki tarah use karo"):
            st.session_state.symbol = q.strip().upper()
symbol = st.session_state.symbol

with st.expander("⚙️ Settings (capital, risk, refresh)"):
    capital = st.number_input("Capital", min_value=1000, value=100000, step=10000)
    risk_pct = st.slider("Risk per trade (%)", 0.5, 3.0, 1.0, 0.5)
    if st.button("🔄 Data refresh"):
        st.cache_data.clear()
        st.rerun()

# ---- timeframe + overlays
tf = st.radio("Timeframe", list(TF.keys()), index=5, horizontal=True, label_visibility="collapsed")
overlays = st.multiselect(
    "Chart par dikhao",
    ["EMA 20/50", "VWAP", "Volume", "Support/Resistance", "Trade levels"],
    default=["EMA 20/50", "Volume", "Support/Resistance", "Trade levels"],
)

# ---- data
with st.spinner("Data load ho raha hai..."):
    chart_df = load(symbol, *TF[tf])
    d_df = load(symbol, "1d", "5y")
    i_df = load(symbol, "5m", "5d")

if d_df.empty:
    st.error("Is symbol ka data nahi mila. Symbol check karo (NSE ke liye .NS, BSE ke liye .BO). "
             "Naya listing hai to thoda ruk kar 'Data refresh' try karo.")
    st.stop()
if chart_df.empty:
    st.warning(f"{tf} timeframe ka data is stock ke liye available nahi hai. Koi aur timeframe chuno.")

name = st.session_state.names.get(symbol) or get_name(symbol)
cur = "₹" if symbol.endswith((".NS", ".BO")) or symbol.startswith(("^NSE", "^BSE")) else ""

# ---- stock header
last = d_df.iloc[-1]
prev = d_df.iloc[-2] if len(d_df) > 1 else last
price = float(last["Close"])
chg = price - float(prev["Close"])
pct = chg / float(prev["Close"]) * 100 if float(prev["Close"]) else 0.0
wk = d_df.tail(252)
st.subheader(name)
st.caption(f"{symbol} · {len(d_df)} trading din ka data")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Price", f"{cur}{price:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
m2.metric("Day Low – High", f"{cur}{float(last['Low']):,.2f} – {float(last['High']):,.2f}")
m3.metric("52W Low – High", f"{cur}{float(wk['Low'].min()):,.2f} – {float(wk['High'].max()):,.2f}")
m4.metric("Volume", f"{int(last['Volume']):,}")
if len(d_df) < 60:
    st.info("🆕 Is stock ki history bahut kam hai (naya listing). Long-term signal kam reliable hoga.")

# ---- analysis
daily_atr = float(true_range(d_df).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
long_sig = analyze(d_df, False, CFG_LONG, capital, risk_pct)
intra_sig = analyze(i_df, True, CFG_INTRA, capital, risk_pct)

# ---- chart
if not chart_df.empty:
    chart_sig = intra_sig if tf in INTRADAY_TF else long_sig
    render_chart(chart_df, tf, f"{name} ({symbol})", overlays, chart_sig)
    st.caption("👆 Ungli se drag = scroll · 2 ungli se pinch = zoom · neeche ke buttons se bhi zoom kar sakte ho. "
               "Trade levels chart ke timeframe ke hisaab se Intraday (1m–1h) ya Long Term (1D+) signal se aate hain.")

# ---- tabs
tab1, tab2 = st.tabs(["⚡ Intraday", "📅 Long Term"])
with tab1:
    st.caption("5-minute candles (last 5 din) par based. Din ke andar ka trade.")
    render_tab(intra_sig, "intraday", daily_atr, cur,
               "Intraday data available nahi ya bahut kam hai (market band ya naya listing ho sakta hai).")
with tab2:
    st.caption("Daily candles (last 5 saal tak) par based. Hafton/mahino ki position.")
    render_tab(long_sig, "long", daily_atr, cur,
               "Long-term signal ke liye kam se kam 30 trading din ka data chahiye. Stock bahut naya hai.")

st.caption("⚠️ Sirf educational tool hai, financial advice nahi. Pehle paper trading me test karo.")
