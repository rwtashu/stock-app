"""
Stock Analyzer v5.2 - Smooth Pro Edition
- Clean Mobile-First UI (Streamlit Sidebar controls + Top metric cards)
- Fast non-blinking background feed
- Accurate Breakout / Swing Target & SL levels
- Permanent historical trade tracking on chart
Run: streamlit run stock_analyzer.py
"""
import copy
import html
import inspect
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import feed
import storage
import trades as tr

st.set_page_config(
    page_title="Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ STYLING
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; padding: 6px 16px; font-weight: 600; }
    div[data-testid="stExpander"] { border-radius: 8px; border: 1px solid #262730; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ CONFIG
TF = {
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
CFG_INTRA = dict(fast=9, slow=21)
CFG_LONG = dict(fast=50, slow=200)
BULLISH = ["Hammer", "Bullish Engulfing", "Morning Star"]
BEARISH = ["Shooting Star", "Bearish Engulfing", "Evening Star"]
GREEN, RED, AMBER, BLUE = "#00e676", "#ff1744", "#ffb300", "#2979ff"
PAGE_CHART, PAGE_WL, PAGE_J, PAGE_DATA = "📊 Chart", "⭐ Watchlist", "📒 Journal", "💾 Data & Price"
MAX_WL = 25
IST = timezone(timedelta(hours=5, minutes=30))


def f2(x):
    return round(float(x), 2)


def ist_now():
    return datetime.now(IST)


def is_indian(sym):
    return sym.endswith((".NS", ".BO")) or sym.startswith(("^NSE", "^BSE", "^CNX"))


def market_open(sym):
    now = ist_now()
    mins, wd = now.hour * 60 + now.minute, now.weekday()
    if sym.endswith(("-USD", "=X")):
        return True
    if is_indian(sym):
        return wd < 5 and 9 * 60 + 15 <= mins <= 15 * 60 + 35
    return (wd <= 4 and mins >= 18 * 60 + 45) or (1 <= wd <= 5 and mins <= 2 * 60 + 45)


# ------------------------------------------------------------------ BACKGROUND HUB
class Hub:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=6)
        self.lock = threading.Lock()
        self.data = {}
        self.busy = set()

    def _run(self, key, fn):
        try:
            v = fn()
            ok = v is not None and not (isinstance(v, pd.DataFrame) and v.empty)
        except Exception:
            v, ok = None, False
        with self.lock:
            cur = self.data.get(key) or {}
            if ok:
                self.data[key] = dict(v=v, t=time.time(), fail=0, next=0.0)
            else:
                fail = cur.get("fail", 0) + 1
                self.data[key] = dict(
                    v=cur.get("v"),
                    t=cur.get("t", 0.0),
                    fail=fail,
                    next=time.time() + min(120, 5 * 2**fail),
                )
            self.busy.discard(key)
            if len(self.data) > 100:
                for k in sorted(self.data, key=lambda k_: self.data[k_]["t"])[:20]:
                    self.data.pop(k, None)

    def get(self, key, fn, ttl, wait=False):
        now = time.time()
        with self.lock:
            e = self.data.get(key)
            if key not in self.busy and (e is None or (now - e["t"] > ttl and now >= e.get("next", 0.0))):
                self.busy.add(key)
                self.pool.submit(self._run, key, fn)
        if wait and (e is None or e.get("v") is None):
            t0 = time.time()
            while time.time() - t0 < 25:
                with self.lock:
                    e = self.data.get(key)
                    if e is not None and key not in self.busy:
                        break
                time.sleep(0.05)
        with self.lock:
            e = self.data.get(key)
        if e is None or e.get("v") is None:
            return None, None, False
        return e["v"], time.time() - e["t"], e.get("fail", 0) > 0


@st.cache_resource
def get_hub():
    return Hub()


def secret(name):
    try:
        v = st.secrets.get(name)
        return str(v).strip() if v else None
    except Exception:
        return None


def groww_cfg():
    return dict(api_key=secret("GROWW_API_KEY"), secret=secret("GROWW_API_SECRET"))


def fetch_candles(sym, interval, period):
    df = yf.Ticker(sym).history(interval=interval, period=period, auto_adjust=True, actions=False)
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if interval in ("1d", "1wk", "1mo") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[~df.index.duplicated(keep="last")]


def hist(sym, interval, period, ttl, wait=True):
    v, _, stale = get_hub().get(
        ("h", sym, interval, period),
        lambda: fetch_candles(sym, interval, period),
        ttl,
        wait,
    )
    return (v if v is not None else pd.DataFrame()), stale


def live_quote(sym, ttl=3.0, wait=False):
    cfg = groww_cfg()
    v, _, _ = get_hub().get(
        ("q", sym, bool(cfg["api_key"])),
        lambda: feed.get_quote(sym, cfg, ttl=0.5),
        ttl,
        wait,
    )
    return v


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
        quotes = yf.Search(q, max_results=10).quotes
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


# ------------------------------------------------------------------ INDICATORS & ANALYSIS
def true_range(df):
    c = df["Close"]
    return pd.concat(
        [df["High"] - df["Low"], (df["High"] - c.shift()).abs(), (df["Low"] - c.shift()).abs()],
        axis=1,
    ).max(axis=1)


def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = pd.Series(df.index.date, index=df.index)
    vol = df["Volume"]
    return (tp * vol).groupby(day).cumsum() / vol.groupby(day).cumsum().replace(0, np.nan)


def adx_series(df, n=14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr_ = true_range(df).ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / atr_
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


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
    df["ADX"] = adx_series(df)
    if intraday:
        df["VWAP"] = vwap(df)
    return df


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
    p["Morning Star"] = red.shift(2) & (body.shift(1) < body.shift(2) * 0.5) & green & (c > (o.shift(2) + c.shift(2)) / 2)
    p["Evening Star"] = green.shift(2) & (body.shift(1) < body.shift(2) * 0.5) & red & (c < (o.shift(2) + c.shift(2)) / 2)
    return p.fillna(False).astype(bool)


def pivots(d, k):
    h, l = d["High"].to_numpy(float), d["Low"].to_numpy(float)
    n = len(d)
    hs, ls = [], []
    for i in range(k, n - k):
        if h[i] >= h[i - k : i + k + 1].max():
            hs.append((i, float(h[i])))
        if l[i] <= l[i - k : i + k + 1].min():
            ls.append((i, float(l[i])))
    return hs, ls


def sr_levels(df, price, atr, n=5, lookback=200):
    d = df.tail(lookback)
    lo = d["Low"][d["Low"] == d["Low"].rolling(2 * n + 1, center=True).min()]
    hi = d["High"][d["High"] == d["High"].rolling(2 * n + 1, center=True).max()]
    sup, res = lo[lo < price], hi[hi > price]
    support = float(sup.max()) if len(sup) else price - 1.5 * atr
    resistance = float(res.min()) if len(res) else price + 1.5 * atr
    return support, resistance


# ------------------------------------------------------------------ TRADE ACCURACY ENGINE
def make_plan(side, entry, support, resistance, daily_atr, intraday, rr, capital, risk_pct):
    if side == "BUY":
        sl = round(support - (0.05 * daily_atr), 2)
        risk = entry - sl
        if risk <= (0.15 * daily_atr):
            risk = 0.40 * daily_atr if intraday else 1.0 * daily_atr
            sl = round(entry - risk, 2)
        t1 = round(entry + (risk * rr), 2)
        t2 = round(entry + (risk * (rr + 1)), 2)
    else:
        sl = round(resistance + (0.05 * daily_atr), 2)
        risk = sl - entry
        if risk <= (0.15 * daily_atr):
            risk = 0.40 * daily_atr if intraday else 1.0 * daily_atr
            sl = round(entry + risk, 2)
        t1 = round(entry - (risk * rr), 2)
        t2 = round(entry - (risk * (rr + 1)), 2)

    qty = math.floor((capital * (risk_pct / 100)) / risk) if risk > 0 else 0
    qty = int(max(min(qty, math.floor(capital / entry)), 0))

    return dict(
        side=side,
        entry=f2(entry),
        risk=f2(risk),
        rr=rr,
        sl=f2(sl),
        t1=f2(t1),
        t2=f2(t2),
        qty=qty,
    )


def build_signal(df, pats, intraday, daily_atr, rr, capital, risk_pct):
    last = df.iloc[-1]
    price, atr = float(last["Close"]), float(last["ATR"])
    score = 0
    reasons = []

    def log(c, t, s):
        nonlocal score
        score += s
        reasons.append((c, t, s))

    if last["EMA_F"] > last["EMA_S"] and price > last["EMA_S"]:
        log("Trend", "Fast EMA > Slow EMA (Bullish)", 3)
    elif last["EMA_F"] < last["EMA_S"] and price < last["EMA_S"]:
        log("Trend", "Fast EMA < Slow EMA (Bearish)", -3)

    if intraday and "VWAP" in df and not pd.isna(last["VWAP"]):
        log("Trend", "Price > VWAP", 2 if price > last["VWAP"] else -2)

    adx = float(last["ADX"]) if "ADX" in df and not pd.isna(last["ADX"]) else 20
    rsi = float(last["RSI"])

    if 52 <= rsi <= 68:
        log("Momentum", f"RSI Bullish Zone ({rsi:.0f})", 2)
    elif 32 <= rsi <= 48:
        log("Momentum", f"RSI Bearish Zone ({rsi:.0f})", -2)

    if last["MACD"] > last["MACD_SIG"]:
        log("Momentum", "MACD Above Signal", 1)
    else:
        log("Momentum", "MACD Below Signal", -1)

    vol = df["Volume"]
    vol_ratio = 1.0
    if len(vol) > 20 and float(vol.iloc[-21:-1].mean()) > 0:
        vol_ratio = float(vol.iloc[-1]) / float(vol.iloc[-21:-1].mean())
        if vol_ratio >= 1.5:
            log("Volume", f"High Volume Surge ({vol_ratio:.1f}x)", 2 if last["Close"] > last["Open"] else -2)

    support, resistance = sr_levels(df, price, atr)
    log("Levels", f"Nearest S: {support:,.2f} | R: {resistance:,.2f}", 0)

    # Filter out sideways chopped markets
    is_sideways = adx < 18 or vol_ratio < 0.75
    if score >= 6 and not is_sideways:
        side, strength = "BUY", "Strong"
    elif score <= -6 and not is_sideways:
        side, strength = "SELL", "Strong"
    else:
        side, strength = "WAIT", "Neutral"

    plans = {
        s: make_plan(s, price, support, resistance, daily_atr, intraday, rr, capital, risk_pct)
        for s in ("BUY", "SELL")
    }
    cond = {
        "BUY": make_plan("BUY", resistance + 0.05 * daily_atr, support, resistance, daily_atr, intraday, rr, capital, risk_pct),
        "SELL": make_plan("SELL", support - 0.05 * daily_atr, support, resistance, daily_atr, intraday, rr, capital, risk_pct),
    }

    return dict(
        side=side,
        score=score,
        strength=strength,
        reasons=reasons,
        plan=plans[side] if side != "WAIT" else None,
        plans=plans,
        cond=cond,
        support=support,
        resistance=resistance,
        price=price,
        rsi=rsi,
        adx=adx,
        atr=daily_atr,
    )


def analyze(df, intraday, cfg, daily_atr, rr, capital, risk_pct):
    if df is None or len(df) < 30 or not daily_atr > 0:
        return None
    d = add_indicators(df, cfg["fast"], cfg["slow"], intraday).dropna(subset=["EMA_F", "EMA_S", "RSI", "ATR", "MACD"])
    if len(d) < 20:
        return None
    return build_signal(d, detect_patterns(d), intraday, daily_atr, rr, capital, risk_pct)


# ------------------------------------------------------------------ SMOOTH PLOTLY CHART
PLOT_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
}


def build_figure(df, tf, name, symbol, overlays, sig, pos_mode, view, trades_draw=None):
    n = len(df)
    labs = [x.strftime("%d %b %H:%M" if tf in INTRADAY_TF else "%d %b %y") for x in df.index]
    o, h, l, c, v = (
        df["Open"].to_numpy(float),
        df["High"].to_numpy(float),
        df["Low"].to_numpy(float),
        df["Close"].to_numpy(float),
        df["Volume"].to_numpy(float),
    )

    has_vol = "Volume" in overlays and float(np.nansum(v)) > 0
    fig = make_subplots(
        rows=2 if has_vol else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.82, 0.18] if has_vol else [1.0],
    )

    # Smooth candles styling
    fig.add_trace(
        go.Candlestick(
            x=labs,
            open=o,
            high=h,
            low=l,
            close=c,
            name="Price",
            showlegend=False,
            increasing=dict(line=dict(color=GREEN, width=1), fillcolor=GREEN),
            decreasing=dict(line=dict(color=RED, width=1), fillcolor=RED),
        ),
        row=1,
        col=1,
    )

    # Indicators
    if "EMA 20/50" in overlays:
        fig.add_trace(go.Scatter(x=labs, y=df["Close"].ewm(span=20).mean().round(2), mode="lines", name="EMA 20", line=dict(color="#2979ff", width=1.5)), row=1, col=1)
        if n >= 50:
            fig.add_trace(go.Scatter(x=labs, y=df["Close"].ewm(span=50).mean().round(2), mode="lines", name="EMA 50", line=dict(color="#ff9100", width=1.5)), row=1, col=1)

    if "VWAP" in overlays and tf in INTRADAY_TF:
        fig.add_trace(go.Scatter(x=labs, y=vwap(df).round(2), mode="lines", name="VWAP", line=dict(color="#e040fb", width=1.5, dash="dot")), row=1, col=1)

    if has_vol:
        vcol = ["rgba(0, 230, 118, 0.45)" if cc >= oo else "rgba(255, 23, 68, 0.45)" for oo, cc in zip(o, c)]
        fig.add_trace(go.Bar(x=labs, y=v, marker_color=vcol, name="Volume", showlegend=False), row=2, col=1)

    # Levels
    if sig and "Support/Resistance" in overlays:
        fig.add_hline(y=f2(sig["support"]), line_dash="dash", line_color=GREEN, line_width=1.2, annotation_text="Support", annotation_position="bottom right", row=1, col=1)
        fig.add_hline(y=f2(sig["resistance"]), line_dash="dash", line_color=RED, line_width=1.2, annotation_text="Resistance", annotation_position="top right", row=1, col=1)

    # Permanent Historical & Active Position Boxes
    vis = min(int(view) if view != "All" else n, n)
    lo_v, hi_v = float(np.nanmin(l[-vis:])), float(np.nanmax(h[-vis:]))

    for t in trades_draw or []:
        idx_match = [i for i, x in enumerate(df.index) if str(x)[:16] >= str(t["entry_ts"])[:16]]
        x0 = idx_match[0] if idx_match else 0
        x1 = n - 1 + 8 if t["status"] != "CLOSED" else min(n - 1, x0 + t.get("bars", 10))

        e, sl_, t1, t2 = f2(t["entry"]), f2(t["sl0"]), f2(t["t1"]), f2(t["t2"])
        lo_v, hi_v = min(lo_v, sl_, t2), max(hi_v, sl_, t2)
        closed = t["status"] == "CLOSED"
        opac = 0.08 if closed else 0.22

        fig.add_shape(type="rect", x0=x0, x1=x1, y0=e, y1=t2, fillcolor=f"rgba(0,230,118,{opac})", line_width=0, layer="below", row=1, col=1)
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=e, y1=sl_, fillcolor=f"rgba(255,23,68,{opac})", line_width=0, layer="below", row=1, col=1)
        fig.add_shape(type="line", x0=x0, x1=x1, y0=e, y1=e, line=dict(color=BLUE, width=1.5), row=1, col=1)

        if closed:
            is_win = (t.get("pnl") or 0) > 0
            tag_text = f"🎯 TARGET HIT (+{t.get('r_mult', 0):.1f}R)" if is_win else f"🛑 SL HIT ({t.get('r_mult', 0):.1f}R)"
            fig.add_annotation(
                x=x1,
                y=t.get("exit_price", e),
                text=tag_text,
                showarrow=False,
                font=dict(color=GREEN if is_win else RED, size=11),
                bgcolor="#161b22",
                row=1,
                col=1,
            )

    fig.update_xaxes(range=[n - vis - 0.5, n + 8], type="category", showgrid=False)
    fig.update_yaxes(side="right", gridcolor="#1f242c", range=[lo_v * 0.995, hi_v * 1.005])
    fig.update_layout(
        template="plotly_dark",
        height=540 if has_vol else 480,
        margin=dict(l=6, r=6, t=20, b=6),
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#0b0e14",
        hovermode="x unified",
        dragmode="pan",
    )
    return fig


# ------------------------------------------------------------------ STATE & PERSISTENCE
if "db_ready" not in st.session_state:
    st.session_state.trades = []
    st.session_state.wl = ["RELIANCE.NS", "TATAMOTORS.NS", "TCS.NS"]
    st.session_state.names = {}
    st.session_state.symbol = "TATAMOTORS.NS"
    st.session_state.db_ready = True

# ------------------------------------------------------------------ SIDEBAR (CLEAN CONTROLS)
with st.sidebar:
    st.header("⚙️ App Controls")
    search_q = st.text_input("🔍 Search Symbol", placeholder="TATA, RELIANCE, NIFTY...")
    if search_q.strip():
        res = search_symbols(search_q.strip(), True)
        if res:
            chosen = st.selectbox("Results", [r["label"] for r in res])
            if st.button("Open Stock"):
                st.session_state.symbol = chosen.split(" · ")[1]
                st.rerun()

    st.divider()
    live_toggle = st.toggle("🔴 Live Auto-Refresh", value=True)
    refresh_sec = st.selectbox("Interval", [10, 15, 30, 60], index=1)
    auto_paper = st.toggle("🤖 Auto Paper-Trading", value=True)

    with st.expander("Risk Management"):
        capital = st.number_input("Capital (₹)", min_value=5000, value=100000, step=10000)
        risk_pct = st.slider("Risk Per Trade (%)", 0.25, 3.0, 1.0, step=0.25)
        rr = st.slider("Target R:R", 1.0, 4.0, 2.0, step=0.5)

    if st.button("🧹 Clear All Cache"):
        st.cache_data.clear()
        get_hub().data.clear()
        st.success("Refreshed!")

# ------------------------------------------------------------------ MAIN PAGE HEADER
sym = st.session_state.symbol
name = get_name(sym)
cur = "₹" if is_indian(sym) else "$"

col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.subheader(f"{name} ({sym})")
with col_head2:
    if sym in st.session_state.wl:
        if st.button("❌ Remove Watchlist"):
            st.session_state.wl.remove(sym)
            st.rerun()
    else:
        if st.button("⭐ Add Watchlist"):
            st.session_state.wl.append(sym)
            st.rerun()

# Real-time Metrics Card Row
q = live_quote(sym, ttl=2.0, wait=True)
d_df, _ = hist(sym, "1d", "5y", 120, wait=True)
price = q["price"] if q else (float(d_df["Close"].iloc[-1]) if not d_df.empty else 0.0)
prev = q.get("prev_close") if q else (float(d_df["Close"].iloc[-2]) if len(d_df) > 1 else price)
chg = price - prev
pct = (chg / prev) * 100 if prev else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Price", f"{cur}{price:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
m2.metric("Day Low - High", f"{cur}{q.get('low', 0):,.2f} - {cur}{q.get('high', 0):,.2f}" if q else "—")
m3.metric("Volume", f"{int(q.get('volume', 0)):,}" if q else "—")
m4.metric("Status", "🟢 Market Open" if market_open(sym) else "⚪ Closed")

# ------------------------------------------------------------------ CHART & TIMEFRAMES
tf_col, opt_col = st.columns([3, 1])
with tf_col:
    tf = st.radio("Timeframe", list(TF.keys()), index=1, horizontal=True, label_visibility="collapsed")
with opt_col:
    chart_view = st.selectbox("Bars Visible", ["60", "120", "200", "All"], index=1, label_visibility="collapsed")

i_df, _ = hist(sym, "5m", "1mo", refresh_sec if live_toggle else 300)
chart_df = d_df if tf == "1D" else (i_df if tf == "5m" else hist(sym, *TF[tf], 300)[0])
daily_atr = float(true_range(d_df).ewm(alpha=1 / 14).mean().iloc[-1]) if not d_df.empty else 1.0

sig = analyze(chart_df, tf in INTRADAY_TF, CFG_INTRA if tf in INTRADAY_TF else CFG_LONG, daily_atr, rr, capital, risk_pct)

# Trade History and Active trades
active_draw = [t for t in st.session_state.trades if t.get("symbol") == sym]

fig = build_figure(
    chart_df,
    tf,
    name,
    sym,
    ["EMA 20/50", "Volume", "Support/Resistance"],
    sig,
    "Auto",
    chart_view,
    trades_draw=active_draw,
)
st.plotly_chart(fig, config=PLOT_CONFIG, use_container_width=True)

# ------------------------------------------------------------------ SIGNALS & DECISION TABS
t_tab1, t_tab2 = st.tabs(["⚡ Trade Setup & Levels", "📒 History & Hits"])

with t_tab1:
    if sig:
        status_box = st.success if sig["side"] == "BUY" else (st.error if sig["side"] == "SELL" else st.warning)
        status_box(f"**Action:** {sig['side']} | **Strength:** {sig['strength']} | **Score:** {sig['score']:+d} | **RSI:** {sig['rsi']:.0f}")

        if sig["plan"]:
            p = sig["plan"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Entry Trigger", f"{cur}{p['entry']:,.2f}")
            c2.metric("Stop Loss", f"{cur}{p['sl']:,.2f}")
            c3.metric("Target 1 (1:2)", f"{cur}{p['t1']:,.2f}")
            c4.metric("Qty", str(p["qty"]))

            if st.button("🔒 Lock Paper Trade Now"):
                st.session_state.trades.append(
                    dict(
                        symbol=sym,
                        entry=p["entry"],
                        sl0=p["sl"],
                        t1=p["t1"],
                        t2=p["t2"],
                        entry_ts=chart_df.index[-1],
                        status="OPEN",
                        pnl=0,
                        r_mult=0,
                        bars=0,
                    )
                )
                st.toast("Trade locked successfully on chart!")
                st.rerun()
        else:
            st.info("Market range-bound / sideways hai. Breakout confirm hone ka intezaar karein.")
            st.write(f"📈 **Long Trigger:** {cur}{sig['cond']['BUY']['entry']:,.2f} ke upar close.")
            st.write(f"📉 **Short Trigger:** {cur}{sig['cond']['SELL']['entry']:,.2f} ke niche close.")

with t_tab2:
    if st.session_state.trades:
        st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True)
    else:
        st.caption("Abhi koi executed ya closed trade history mein nahi hai.")
