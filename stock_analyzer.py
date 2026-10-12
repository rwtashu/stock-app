"""
Stock Analyzer v5 - High Accuracy Edition
- Smooth live price (background fetch, small ticker fragment, no page blink)
- Realistic Swing-based Target & Stop-loss calculation
- Strict Breakout Entry Engine (No false entries in sideways markets)
- Permanent historical Target / SL hits visible on the chart
- Long / Short position tool drawn on chart
Run locally:  streamlit run stock_analyzer.py
Educational tool only. Not financial advice.
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

st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")
st.markdown(
    "<style>.block-container{padding-top:1.2rem;padding-bottom:2rem}</style>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ config
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
GREEN, RED, AMBER, BLUE = "#26a69a", "#ef5350", "#ffb300", "#42a5f5"
PAGE_CHART, PAGE_WL = "📊 Chart", "⭐ Watchlist"
MAX_WL = 20


def f2(x):
    return round(float(x), 2)


# ------------------------------------------------------------------ live / market helpers
IST = timezone(timedelta(hours=5, minutes=30))


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


def stretch_kw(fn):
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


# ------------------------------------------------------------------ data
@st.cache_data(ttl=900, show_spinner=False, max_entries=300)
def load(sym, interval, period, bucket=0):
    try:
        df = yf.download(sym, interval=interval, period=period,
                         auto_adjust=True, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    try:
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()
    return df[~df.index.duplicated(keep="last")]


# ------------------------------------------------------------------ background data hub
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
                self.data[key] = dict(v=cur.get("v"), t=cur.get("t", 0.0), fail=fail,
                                      next=time.time() + min(120, 5 * 2 ** fail))
            self.busy.discard(key)
            if len(self.data) > 90:
                for k in sorted(self.data, key=lambda k_: self.data[k_]["t"])[:15]:
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
    v, _, stale = get_hub().get(("h", sym, interval, period), lambda: fetch_candles(sym, interval, period), ttl, wait)
    return (v if v is not None else pd.DataFrame()), stale


def live_quote(sym, ttl=3.0, wait=False):
    cfg = groww_cfg()
    v, _, _ = get_hub().get(("q", sym, bool(cfg["api_key"])), lambda: feed.get_quote(sym, cfg, ttl=0.5), ttl, wait)
    return v


def last_bar_ist(df):
    if df is None or df.empty:
        return None
    ts = df.index[-1]
    try:
        return ts.astimezone(IST) if ts.tzinfo is not None else ts.replace(tzinfo=IST)
    except Exception:
        return None


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
    big1 = body.shift(2) > body.rolling(10).mean().shift(2)
    small = body.shift(1) < body.shift(2) * 0.5
    p["Morning Star"] = red.shift(2) & big1 & small & green & (c > (o.shift(2) + c.shift(2)) / 2)
    p["Evening Star"] = green.shift(2) & big1 & small & red & (c < (o.shift(2) + c.shift(2)) / 2)
    return p.fillna(False).astype(bool)


# ------------------------------------------------------------------ pivots / levels
def pivots(d, k):
    h = d["High"].to_numpy(float)
    l = d["Low"].to_numpy(float)
    n = len(d)
    hs, ls = [], []
    for i in range(k, n - k):
        if h[i] >= h[i - k:i + k + 1].max():
            if hs and i - hs[-1][0] < k:
                if h[i] > hs[-1][1]:
                    hs[-1] = (i, float(h[i]))
            else:
                hs.append((i, float(h[i])))
        if l[i] <= l[i - k:i + k + 1].min():
            if ls and i - ls[-1][0] < k:
                if l[i] < ls[-1][1]:
                    ls[-1] = (i, float(l[i]))
            else:
                ls.append((i, float(l[i])))
    return hs, ls


def sr_levels(df, price, atr, n=5, lookback=250):
    d = df.tail(lookback)
    lo = d["Low"][d["Low"] == d["Low"].rolling(2 * n + 1, center=True).min()]
    hi = d["High"][d["High"] == d["High"].rolling(2 * n + 1, center=True).max()]
    sup, res = lo[lo < price], hi[hi > price]
    support = float(sup.max()) if len(sup) else price - 2 * atr
    resistance = float(res.min()) if len(res) else price + 2 * atr
    return support, resistance


# ------------------------------------------------------------------ chart patterns
PATTERN_INFO = {
    "Double Top": "M jaisa shape. Price do baar lagbhag same upar ke level se wapas gira...",
    "Double Bottom": "W jaisa shape. Price do baar lagbhag same niche ke level se wapas uchhla...",
    "Head & Shoulders": "Beech ka peak dono shoulders se unchha hai. Girawat ka signal.",
    "Inverse Head & Shoulders": "Beech ka trough dono shoulders se gehra hai. Tezi ka signal.",
    "Ascending Triangle": "Upar flat resistance aur niche badhte hue lows.",
    "Descending Triangle": "Niche flat support aur upar girte hue highs.",
    "Symmetrical Triangle": "Highs girte aur lows badhte hue, breakout ki tayyari.",
    "Rising Wedge": "Range sankri ho rahi hai, girne ka risk.",
    "Falling Wedge": "Range sankri ho rahi hai, tezi ka signal.",
    "Ascending Channel": "Dono lines upar ki taraf, uptrend structure.",
    "Descending Channel": "Dono lines niche ki taraf, downtrend structure.",
    "Range (Rectangle)": "Price do levels ke beech sideways consolidate kar raha hai.",
}
TYPICAL_BIAS = {
    "Double Top": "bearish", "Double Bottom": "bullish",
    "Head & Shoulders": "bearish", "Inverse Head & Shoulders": "bullish",
    "Ascending Triangle": "bullish", "Descending Triangle": "bearish",
    "Symmetrical Triangle": "neutral", "Rising Wedge": "bearish", "Falling Wedge": "bullish",
    "Ascending Channel": "bullish", "Descending Channel": "bearish", "Range (Rectangle)": "neutral",
}
BIAS_COLOR = {"bullish": GREEN, "bearish": RED, "neutral": AMBER}


def find_patterns(df, max_bars=220):
    if df is None or len(df) < 40:
        return []
    d = df.tail(max_bars)
    n = len(d)
    off = len(df) - n
    atr = float(true_range(d).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    if not atr > 0:
        return []
    k = int(min(8, max(3, n // 30)))
    hs, ls = pivots(d, k)
    lo_a = d["Low"].to_numpy(float)
    hi_a = d["High"].to_numpy(float)
    cl = d["Close"].to_numpy(float)
    last = n - 1
    price = float(cl[-1])
    recent = max(30, n // 4)
    out = []

    def finish(name, status_dir, segs, marks, label, up, down, end_i):
        typical = TYPICAL_BIAS[name]
        if status_dir > 0:
            bias, status = "bullish", "Confirmed ↑"
        elif status_dir < 0:
            bias, status = "bearish", "Confirmed ↓"
        else:
            bias, status = typical, "Forming"
        target = up[1] if bias == "bullish" else down[1] if bias == "bearish" else None
        invalid = down[0] if bias == "bullish" else up[0] if bias == "bearish" else None
        out.append(dict(
            name=name, bias=bias, status=status, typical=typical,
            lines=[dict(i1=a + off, p1=f2(pa), i2=b + off, p2=f2(pb), color=c, dash=ds)
                   for (a, pa, b, pb, c, ds) in segs],
            markers=[dict(i=i + off, pos=pos, text=t, color=c, shape=sh) for (i, pos, t, c, sh) in marks],
            label=dict(i=label[0] + off, p=f2(label[1]), text=name, color=BIAS_COLOR[bias]),
            up_level=f2(up[0]), up_target=None if up[1] is None else f2(up[1]),
            down_level=f2(down[0]), down_target=None if down[1] is None else f2(down[1]),
            target=None if target is None else f2(target),
            invalid=None if invalid is None else f2(invalid),
            meaning=PATTERN_INFO[name], end=end_i + off, price=f2(price),
        ))

    # Double top / bottom
    if len(hs) >= 2:
        (i1, p1), (i2, p2) = hs[-2], hs[-1]
        if last - i2 <= recent and i2 - i1 >= max(8, 2 * k):
            seg = lo_a[i1:i2 + 1]
            j = i1 + int(seg.argmin())
            neck = float(seg.min())
            top = (p1 + p2) / 2
            height = top - neck
            if (height >= 3 * atr and abs(p1 - p2) <= min(1.0 * atr, 0.25 * height)
                    and price <= max(p1, p2) + 0.5 * atr):
                sd = -1 if price < neck - 0.2 * atr else 0
                finish("Double Top", sd,
                       [(i1, p1, j, neck, AMBER, False), (j, neck, i2, p2, AMBER, False),
                        (i1, neck, last, neck, BLUE, True)],
                       [(i1, "aboveBar", "T1", RED, "arrowDown"), (i2, "aboveBar", "T2", RED, "arrowDown")],
                       (i2, top + 0.6 * atr), (max(p1, p2), None), (neck, neck - height), i2)
    if len(ls) >= 2:
        (i1, p1), (i2, p2) = ls[-2], ls[-1]
        if last - i2 <= recent and i2 - i1 >= max(8, 2 * k):
            seg = hi_a[i1:i2 + 1]
            j = i1 + int(seg.argmax())
            neck = float(seg.max())
            bot = (p1 + p2) / 2
            height = neck - bot
            if (height >= 3 * atr and abs(p1 - p2) <= min(1.0 * atr, 0.25 * height)
                    and price >= min(p1, p2) - 0.5 * atr):
                sd = 1 if price > neck + 0.2 * atr else 0
                finish("Double Bottom", sd,
                       [(i1, p1, j, neck, AMBER, False), (j, neck, i2, p2, AMBER, False),
                        (i1, neck, last, neck, BLUE, True)],
                       [(i1, "belowBar", "B1", GREEN, "arrowUp"), (i2, "belowBar", "B2", GREEN, "arrowUp")],
                       (i2, bot - 0.6 * atr), (neck, neck + height), (min(p1, p2), None), i2)

    # Triangles & channels
    if len(hs) >= 3 and len(ls) >= 3:
        H, L = hs[-3:], ls[-3:]
        i0 = min(H[0][0], L[0][0])
        span = last - i0
        if span >= 12 and last - max(H[-1][0], L[-1][0]) <= recent:
            xh, yh = np.array([p[0] for p in H], float), np.array([p[1] for p in H], float)
            xl, yl = np.array([p[0] for p in L], float), np.array([p[1] for p in L], float)
            mh, bh = np.polyfit(xh, yh, 1)
            ml, bl = np.polyfit(xl, yl, 1)
            up = lambda x: mh * x + bh
            lo = lambda x: ml * x + bl
            h0, he = up(i0) - lo(i0), up(last) - lo(last)
            hmid = (h0 + he) / 2
            if h0 > 0 and he > 0.1 * h0 and hmid > 1.5 * atr:
                rh = float(np.abs(yh - (mh * xh + bh)).max())
                rl = float(np.abs(yl - (ml * xl + bl)).max())
                if rh <= 0.25 * hmid and rl <= 0.25 * hmid:
                    thr = 0.15 * h0
                    sgn = lambda v: 0 if abs(v) < thr else (1 if v > 0 else -1)
                    sh, sl_ = sgn(mh * span), sgn(ml * span)
                    ratio = he / h0
                    name = None
                    if ratio < 0.75:
                        if sh == 0 and sl_ == 1: name = "Ascending Triangle"
                        elif sh == -1 and sl_ == 0: name = "Descending Triangle"
                        elif sh == -1 and sl_ == 1: name = "Symmetrical Triangle"
                        elif sh == 1 and sl_ == 1: name = "Rising Wedge"
                        elif sh == -1 and sl_ == -1: name = "Falling Wedge"
                    elif 0.8 <= ratio <= 1.25:
                        if sh == 1 and sl_ == 1: name = "Ascending Channel"
                        elif sh == -1 and sl_ == -1: name = "Descending Channel"
                        elif sh == 0 and sl_ == 0: name = "Range (Rectangle)"
                    if name:
                        upl, lol = up(last), lo(last)
                        sd = 1 if price > upl + 0.2 * atr else -1 if price < lol - 0.2 * atr else 0
                        col = BIAS_COLOR[TYPICAL_BIAS[name]]
                        marks = [(i, "aboveBar", "", col, "circle") for i, _ in H] + [(i, "belowBar", "", col, "circle") for i, _ in L]
                        li = i0 + int(span * 0.2)
                        finish(name, sd, [(i0, up(i0), last, upl, col, False), (i0, lo(i0), last, lol, col, False)],
                               marks, (li, up(li) + 0.6 * atr), (upl, upl + h0), (lol, lol - h0), max(H[-1][0], L[-1][0]))

    out.sort(key=lambda p: -p["end"])
    return out[:4]


# ------------------------------------------------------------------ REAL SWING TRADE PLAN
def make_plan(side, entry, support, resistance, daily_atr, intraday, rr, capital, risk_pct):
    """Accurate Support/Resistance based SL and R:R Target sizing."""
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
        qty=qty
    )


# ------------------------------------------------------------------ STRICT SIGNAL ENGINE
def build_signal(df, pats, cpats, intraday, daily_atr, rr, capital, risk_pct):
    last = df.iloc[-1]
    price, atr = float(last["Close"]), float(last["ATR"])
    state = {"score": 0}
    reasons = []
    trend = "mixed"
    vol_ratio = 1.0

    def add(cat, text, s):
        state["score"] += s
        reasons.append((cat, text, s))

    # Trend
    if last["EMA_F"] > last["EMA_S"] and price > last["EMA_S"]:
        trend = "up"
        add("Trend", "Trend upar hai (fast EMA > slow EMA, price dono ke upar)", 3)
    elif last["EMA_F"] < last["EMA_S"] and price < last["EMA_S"]:
        trend = "down"
        add("Trend", "Trend niche hai (fast EMA < slow EMA, price dono ke niche)", -3)
    else:
        add("Trend", "Trend mixed / sideways hai", 0)

    if intraday and "VWAP" in df and not pd.isna(last["VWAP"]):
        if price > last["VWAP"]:
            add("Trend", "Price VWAP ke upar (Buyers active)", 2)
        else:
            add("Trend", "Price VWAP ke niche (Sellers active)", -2)

    adx = float(last["ADX"]) if "ADX" in df and not pd.isna(last["ADX"]) else None
    if adx is not None:
        if adx >= 25:
            add("Trend", f"ADX {adx:.0f}: strong trend", 1)
        elif adx < 18:
            add("Trend", f"ADX {adx:.0f}: sideways zone, false moves possible", -1)

    # Momentum
    rsi = float(last["RSI"])
    if rsi > 70:
        add("Momentum", f"RSI {rsi:.0f}: overbought zone", -2)
    elif rsi < 30:
        add("Momentum", f"RSI {rsi:.0f}: oversold bounce zone", 2)
    elif 52 <= rsi <= 68:
        add("Momentum", f"RSI {rsi:.0f}: healthy bullish zone", 2)
    elif 32 <= rsi <= 48:
        add("Momentum", f"RSI {rsi:.0f}: bearish momentum zone", -2)

    if last["MACD"] > last["MACD_SIG"]:
        add("Momentum", "MACD signal ke upar", 1)
    else:
        add("Momentum", "MACD signal ke niche", -1)

    # Structure
    tail = df.tail(150)
    hs, ls = pivots(tail, int(min(8, max(3, len(tail) // 30))))
    if len(hs) >= 2 and len(ls) >= 2:
        hh, hl = hs[-1][1] > hs[-2][1], ls[-1][1] > ls[-2][1]
        if hh and hl:
            add("Structure", "Higher Highs + Higher Lows structure", 2)
        elif (not hh) and (not hl):
            add("Structure", "Lower Highs + Lower Lows structure", -2)

    # Volume Confirmation
    vol = df["Volume"]
    if len(vol) > 21 and float(vol.iloc[-21:-1].mean()) > 0:
        vr = float(vol.iloc[-1]) / float(vol.iloc[-21:-1].mean())
        vol_ratio = vr
        green = last["Close"] > last["Open"]
        if vr >= 1.5:
            add("Volume", f"High volume breakout ({vr:.1f}x)", 2 if green else -2)
        elif vr < 0.8:
            add("Volume", f"Low volume ({vr:.1f}x): fake move risk", 0)

    # Candlestick
    recent = pats.tail(3)
    for name in BULLISH:
        if recent[name].any():
            add("Candlestick", f"Bullish pattern: {name}", 1)
    for name in BEARISH:
        if recent[name].any():
            add("Candlestick", f"Bearish pattern: {name}", -1)

    # Chart Patterns
    for p in cpats[:2]:
        base = 2 if p["status"].startswith("Confirmed") else 1
        s = base if p["bias"] == "bullish" else -base if p["bias"] == "bearish" else 0
        add("Chart pattern", f"{p['name']} ({p['status']})", s)

    support, resistance = sr_levels(df, price, atr)
    add("Levels", f"Nearest support {support:,.2f} | resistance {resistance:,.2f}", 0)

    score = state["score"]
    
    # Strict filter: Entry sirf strong conviction aur non-sideways market me hi milegi
    is_sideways = (adx is not None and adx < 18) or (vol_ratio < 0.7)
    if score >= 6 and not is_sideways:
        side = "BUY"
        strength = "Strong"
    elif score <= -6 and not is_sideways:
        side = "SELL"
        strength = "Strong"
    else:
        side = "WAIT"
        strength = "Weak"

    plans = {s: make_plan(s, price, support, resistance, daily_atr, intraday, rr, capital, risk_pct)
             for s in ("BUY", "SELL")}
    cond = {
        "BUY": make_plan("BUY", resistance + 0.05 * daily_atr, support, resistance, daily_atr, intraday, rr,
                         capital, risk_pct),
        "SELL": make_plan("SELL", support - 0.05 * daily_atr, support, resistance, daily_atr, intraday, rr,
                          capital, risk_pct),
    }
    return dict(side=side, score=score, strength=strength, reasons=reasons,
                plan=plans[side] if side != "WAIT" else None, plans=plans, cond=cond,
                support=support, resistance=resistance, price=price, rsi=rsi, adx=adx,
                pats=pats, cpats=cpats, vol_ratio=vol_ratio, trend=trend, atr=daily_atr)


def analyze(df, intraday, cfg, daily_atr, rr, capital, risk_pct):
    if df is None or len(df) < 30 or not daily_atr > 0:
        return None
    n = len(df)
    fast = min(cfg["fast"], max(5, n // 4))
    slow = min(cfg["slow"], max(10, n // 2))
    d = add_indicators(df, fast, slow, intraday).dropna(subset=["EMA_F", "EMA_S", "RSI", "ATR", "MACD"])
    if len(d) < 20:
        return None
    pats = detect_patterns(d)
    cpats = find_patterns(d)
    return build_signal(d, pats, cpats, intraday, daily_atr, rr, capital, risk_pct)


@st.cache_data(ttl=300, show_spinner=False)
def scan_one(sym, rr, capital, risk_pct, bucket=0):
    d = load(sym, "1d", "2y", bucket)
    if d.empty or len(d) < 2:
        return None
    i = load(sym, "5m", "5d", bucket)
    datr = float(true_range(d).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    ls_ = analyze(d, False, CFG_LONG, datr, rr, capital, risk_pct)
    is_ = analyze(i, True, CFG_INTRA, datr, rr, capital, risk_pct)
    price = float(d["Close"].iloc[-1])
    prev = float(d["Close"].iloc[-2])
    return dict(
        price=price, chg=(price / prev - 1) * 100 if prev else 0.0,
        intra=None if is_ is None else (is_["side"], is_["score"]),
        long=None if ls_ is None else (ls_["side"], ls_["score"]),
    )


# ------------------------------------------------------------------ chart (Plotly)
PLOT_CONFIG = dict(
    scrollZoom=True, displaylogo=False, doubleClick="reset", responsive=True,
    modeBarButtonsToRemove=["select2d", "lasso2d"],
)
GREEN_A, RED_A = "rgba(38,166,154,0.55)", "rgba(239,83,80,0.55)"
LABEL_FMT = {"1D": "%d %b %y", "1W": "%d %b %y", "1M": "%b %Y"}


def make_labels(idx, tf):
    fmt = LABEL_FMT.get(tf, "%d %b %H:%M")
    return [str(x) for x in idx.strftime(fmt)]


def prep_chart_df(df, tf):
    if df is None or df.empty:
        return df
    labs = pd.Series(make_labels(df.index, tf), index=df.index)
    return df[~labs.duplicated(keep="last").to_numpy()]


def pick_plan(sig, pos_mode):
    if not sig or pos_mode == "Off":
        return None, None
    if pos_mode == "Long":
        return sig["plans"]["BUY"], None
    if pos_mode == "Short":
        return sig["plans"]["SELL"], None
    if sig["side"] != "WAIT":
        return sig["plan"], None
    side = "BUY" if sig["score"] >= 0 else "SELL"
    return sig["plans"][side], "Signal WAIT: sirf reference plan"


def bar_pos(index, ts):
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    p = int(idx.searchsorted(tr.to_ts(ts), side="right")) - 1
    return p if p >= 0 else None


def build_figure(df, tf, name, symbol, overlays, sig, pos_mode, pat_list, view, rev, trades_draw=None):
    n = len(df)
    daily = tf not in INTRADAY_TF
    labs = make_labels(df.index, tf)
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    atr_c = float(true_range(df).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    if not atr_c > 0:
        atr_c = float(np.nanmean(h - l)) or 1.0

    has_vol = "Volume" in overlays and float(np.nansum(v)) > 0
    rows = 2 if has_vol else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.8, 0.2] if has_vol else [1.0])

    fig.add_trace(go.Candlestick(
        x=labs, open=o, high=h, low=l, close=c, name="Price", showlegend=False,
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED), row=1, col=1)

    s_close = df["Close"]
    if "EMA 20/50" in overlays:
        fig.add_trace(go.Scatter(x=labs, y=s_close.ewm(span=20, adjust=False).mean().round(2), mode="lines",
                                 name="EMA 20", line=dict(color="#2962ff", width=1.3), hoverinfo="skip"), row=1, col=1)
        if n >= 60:
            fig.add_trace(go.Scatter(x=labs, y=s_close.ewm(span=50, adjust=False).mean().round(2), mode="lines",
                                     name="EMA 50", line=dict(color="#ff9800", width=1.3), hoverinfo="skip"), row=1, col=1)
    if "VWAP" in overlays and not daily:
        fig.add_trace(go.Scatter(x=labs, y=vwap(df).round(2), mode="lines", name="VWAP",
                                 line=dict(color="#ab47bc", width=1.3, dash="dot"), hoverinfo="skip"), row=1, col=1)
    if has_vol:
        vcol = [GREEN_A if cc >= oo else RED_A for oo, cc in zip(o, c)]
        fig.add_trace(go.Bar(x=labs, y=v, marker_color=vcol, name="Volume", showlegend=False), row=2, col=1)

    if sig and "Support/Resistance" in overlays:
        fig.add_hline(y=f2(sig["support"]), line_dash="dash", line_color=GREEN, line_width=1,
                      annotation_text="Support", annotation_position="bottom right",
                      annotation_font_color=GREEN, annotation_font_size=11, row=1, col=1)
        fig.add_hline(y=f2(sig["resistance"]), line_dash="dash", line_color=RED, line_width=1,
                      annotation_text="Resistance", annotation_position="top right",
                      annotation_font_color=RED, annotation_font_size=11, row=1, col=1)
    last = float(c[-1])
    prev = float(c[-2]) if n > 1 else last
    lcol = GREEN if last >= prev else RED
    fig.add_hline(y=f2(last), line_dash="dot", line_color=lcol, line_width=1,
                  annotation_text=f"{last:,.2f}", annotation_position="top right",
                  annotation_font_color=lcol, annotation_font_size=12, row=1, col=1)

    tool_on = "Position tool" in overlays and pos_mode != "Off"
    box_bars = max(12, min(40, n // 6))
    vis = n if view == "Sab" else min(int(view), n)
    lo_v, hi_v = float(np.nanmin(l[-vis:])), float(np.nanmax(h[-vis:]))
    draw = []
    
    # Render all locked and historic closed trades
    if tool_on:
        for t in (trades_draw or []):
            p0 = bar_pos(df.index, t["entry_ts"])
            if p0 is None:
                continue
            closed_t = t["status"] == "CLOSED"
            if closed_t:
                p1 = bar_pos(df.index, t.get("closed"))
                x1 = max(n - 1 if p1 is None else p1, p0) + 1
            else:
                x1 = n - 1 + 8
            draw.append(dict(x0=p0, x1=x1, e=t["entry"], sl=t["sl0"], t1=t["t1"], t2=t["t2"], rr=t.get("rr") or 2,
                             sgn=tr.sgn(t), locked=True, closed=closed_t, t=t, note=None))
        if not draw and sig:
            plan, note = pick_plan(sig, pos_mode)
            if plan:
                draw.append(dict(x0=n - 1, x1=n - 1 + box_bars, e=plan["entry"], sl=plan["sl"], t1=plan["t1"],
                                 t2=plan["t2"], rr=plan["rr"], sgn=1 if plan["side"] == "BUY" else -1,
                                 locked=False, closed=False, t=None, note=note))

    ext = 6
    for d in draw:
        ext = max(ext, d["x1"] - (n - 1) + 3)
        lo_v, hi_v = min(lo_v, d["sl"], d["t2"]), max(hi_v, d["sl"], d["t2"])
        x0, x1, sgn = d["x0"], d["x1"], d["sgn"]
        e, sl_, t1, t2 = f2(d["e"]), f2(d["sl"]), f2(d["t1"]), f2(d["t2"])
        fa = 0.10 if d["closed"] else 0.22

        def pc(x, e=e):
            return (x / e - 1) * 100 if e else 0.0

        fig.add_shape(type="rect", x0=x0, x1=x1, y0=e, y1=t2, fillcolor=f"rgba(38,166,154,{fa})",
                      line_width=0, layer="below", row=1, col=1)
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=e, y1=sl_, fillcolor=f"rgba(239,83,80,{fa + 0.02})",
                      line_width=0, layer="below", row=1, col=1)
        for yv, colr, dash, wd in [(e, BLUE, "solid", 1.6), (sl_, RED, "solid", 1.6),
                                   (t1, GREEN, "dash", 1.2), (t2, GREEN, "solid", 1.6)]:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=yv, y1=yv,
                          line=dict(color=colr, width=wd, dash=dash), row=1, col=1)

        def tag(yv, text, colr, above, left=True, x0=x0, x1=x1):
            fig.add_annotation(x=(x0 + 0.4) if left else (x1 - 0.4), y=yv, text=text, showarrow=False,
                               xanchor="left" if left else "right", yanchor="bottom" if above else "top",
                               font=dict(color=colr, size=11), bgcolor="rgba(14,17,23,0.7)", row=1, col=1)

        side_w = "LONG" if sgn > 0 else "SHORT"
        tag(t2, f"T2 {t2:,.2f} ({pc(t2):+.2f}%)", GREEN, sgn > 0)
        tag(t1, f"T1 {t1:,.2f} ({pc(t1):+.2f}%)", GREEN, sgn > 0)
        tag(e, f"{'🔒 ' if d['locked'] else ''}{side_w} Entry {e:,.2f}  R:R 1:{d['rr']:g}", BLUE, True)
        tag(sl_, f"SL {sl_:,.2f} ({pc(sl_):+.2f}%)", RED, sgn < 0)
        
        # Highlight closed trades with clear outcome tags
        if d["closed"]:
            t = d["t"]
            ok = (t.get("pnl") or 0) > 0
            label_text = f"🎯 TARGET HIT (+{t.get('r_mult', 0):+.2f}R)" if ok else f"🛑 SL HIT ({t.get('r_mult', 0):+.2f}R)"
            tag(t.get("exit_price", e), label_text, GREEN if ok else RED, ok == (sgn > 0), left=False)

    pad = (hi_v - lo_v) * 0.06 or max(abs(last) * 0.01, 0.01)
    fig.update_yaxes(range=[lo_v - pad, hi_v + pad], row=1, col=1)
    fig.update_xaxes(range=[n - vis - 0.5, n - 1 + ext + 0.5])

    title = (f"<b>{html.escape(name)}</b> · {html.escape(symbol)} · {tf}"
             f"   O {o[-1]:,.2f}  H {h[-1]:,.2f}  L {l[-1]:,.2f}  C {c[-1]:,.2f}")
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=labs, rangeslider_visible=False,
                     showgrid=False, showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikecolor="#8b93a7", spikedash="dot", nticks=8, tickangle=0)
    fig.update_yaxes(side="right", gridcolor="#1b1f2b", showspikes=True, spikemode="across",
                     spikethickness=1, spikecolor="#8b93a7", spikedash="dot")
    if has_vol:
        fig.update_yaxes(showgrid=False, row=2, col=1)
    fig.update_layout(
        template="plotly_dark", height=640 if has_vol else 560, margin=dict(l=6, r=6, t=78, b=6),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", dragmode="pan", hovermode="x", uirevision=rev,
        title=dict(text=title, x=0.01, xanchor="left", y=0.985, yanchor="top", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=11)),
    )
    return fig


# ------------------------------------------------------------------ UI helpers
def icon(s):
    return "🟢" if s > 0 else "🔴" if s < 0 else "⚪"


def pct_of(a, b):
    return (a / b - 1) * 100 if b else 0.0


def plan_rows(p, cur):
    e = p["entry"]
    rows = [
        ("Entry", f"{cur}{e:,.2f}", ""),
        ("Stop-loss", f"{cur}{p['sl']:,.2f}", f"{pct_of(p['sl'], e):+.2f}%"),
        (f"Target 1 (1:{p['rr']:g})", f"{cur}{p['t1']:,.2f}", f"{pct_of(p['t1'], e):+.2f}%"),
        (f"Target 2 (1:{p['rr'] + 1:g})", f"{cur}{p['t2']:,.2f}", f"{pct_of(p['t2'], e):+.2f}%"),
        ("Risk per share", f"{cur}{p['risk']:,.2f}", ""),
        ("Quantity", str(p["qty"]), f"Invest {cur}{p['qty'] * e:,.0f}"),
        ("Max loss / Profit T1 / T2",
         f"{cur}{p['qty'] * p['risk']:,.0f} / {cur}{p['qty'] * p['rr'] * p['risk']:,.0f} / "
         f"{cur}{p['qty'] * (p['rr'] + 1) * p['risk']:,.0f}", ""),
    ]
    return pd.DataFrame(rows, columns=["Level", "Value", "Move"]).set_index("Level")


def render_tab(sig, kind, daily_atr, cur, empty_msg, symbol=None, name=None):
    if sig is None:
        st.info(empty_msg)
        return
    intraday = kind == "intraday"
    side = sig["side"]
    label = {"BUY": "BUY (Long)", "SELL": "SELL (Short)", "WAIT": "WAIT"}[side] if intraday else {"BUY": "BUY / ACCUMULATE", "SELL": "AVOID / EXIT", "WAIT": "WAIT"}[side]
    box = {"BUY": st.success, "SELL": st.error, "WAIT": st.warning}[side]
    box(f"### {label}\nScore **{sig['score']:+d}**  ·  Strength **{sig['strength']}**  ·  "
        f"BUY ≥ +6, SELL ≤ −6  ·  RSI **{sig['rsi']:.0f}**")

    st.markdown("#### 📍 Trade plan")
    ot = open_trade(symbol, kind) if symbol else None
    if ot:
        render_open_trade(ot, cur)
    elif sig["plan"]:
        st.caption("Confirmed breakout plan. Levels support/resistance aur swing se calculated hain.")
        st.table(plan_rows(sig["plan"], cur))
        if symbol:
            st.button("🔒 Is signal par trade LOCK karo", key=f"lock_{symbol}_{kind}",
                      on_click=lock_trade, args=(symbol, name, kind, side, "signal"))
    else:
        st.write("Market abhi clear trend ya confirmation me nahi hai. Breakout levels:")
        cl, cs = sig["cond"]["BUY"], sig["cond"]["SELL"]
        rows = [
            ("Entry (trigger)", f"{cur}{cl['entry']:,.2f} ke upar close", f"{cur}{cs['entry']:,.2f} ke niche close"),
            ("Stop-loss", f"{cur}{cl['sl']:,.2f}", f"{cur}{cs['sl']:,.2f}"),
            ("Target 1", f"{cur}{cl['t1']:,.2f}", f"{cur}{cs['t1']:,.2f}"),
            ("Target 2", f"{cur}{cl['t2']:,.2f}", f"{cur}{cs['t2']:,.2f}"),
            ("Quantity", str(cl["qty"]), str(cs["qty"])),
        ]
        st.table(pd.DataFrame(rows, columns=["", "🟢 Long (Breakout)", "🔴 Short (Breakdown)"]).set_index(""))


# ------------------------------------------------------------------ trades logic
def open_trade(symbol, kind):
    for t in st.session_state.trades:
        if t["symbol"] == symbol and t["kind"] == kind and t["status"] != "CLOSED":
            return t
    return None


def lock_trade(symbol, name, kind, side, mode):
    ss = st.session_state
    sig = ss.get("_sig", {}).get((symbol, kind))
    if not sig:
        st.toast("Signal ready nahi hai.")
        return
    if open_trade(symbol, kind):
        return
    if kind == "intraday" and is_indian(symbol) and not market_open(symbol):
        st.toast("Market band hai - intraday lock nahi hoga.")
        return
    plan = sig["plan"] if (mode in ("signal", "auto") and sig["plan"]) else sig["plans"]["BUY" if side == "BUY" else "SELL"]
    q = live_quote(symbol, ttl=2.0, wait=True)
    entry = q["price"] if q else sig["price"]
    t = tr.new_trade(plan, entry, symbol, name or symbol, kind, "5m" if kind == "intraday" else "1D", mode,
                     tr.now_ist(), sig=sig, tick=0.05 if is_indian(symbol) else 0.01)
    if t["qty"] < 1:
        st.toast("Capital ke hisaab se quantity kam hai.")
        return
    ss.trades.append(t)
    st.toast(f"🔒 {symbol} {t['side']} locked @ {entry:,.2f}")


def manual_exit_cb(tid):
    for t in st.session_state.trades:
        if t["id"] == tid and t["status"] != "CLOSED":
            q = live_quote(t["symbol"], ttl=2.0, wait=True)
            px = q["price"] if q else t["entry"]
            tr.manual_exit(t, px, tr.now_ist())
            tr.postmortem(t)
            st.toast(f"{t['symbol']} exit @ {px:,.2f}")


def delete_trade_cb(tid):
    st.session_state.trades = [t for t in st.session_state.trades if t["id"] != tid]


def render_open_trade(t, cur):
    q = live_quote(t["symbol"], ttl=3.0)
    px = q["price"] if q else None
    s = tr.sgn(t)
    side_w = "LONG (BUY)" if s > 0 else "SHORT (SELL)"
    st.info(f"🔒 **{side_w} trade LOCKED** · {t['opened'][:16].replace('T', ' ')}")
    p = dict(entry=t["entry"], sl=t["sl0"], t1=t["t1"], t2=t["t2"], rr=t.get("rr") or 2,
             risk=abs(t["entry"] - t["sl0"]), qty=t["qty"])
    st.table(plan_rows(p, cur))
    if t["t1_done"]:
        st.success("✅ Target 1 Hit! SL break-even par shifted.")
    if px:
        pnl = t["realized"] + t["qty_left"] * (px - t["entry"]) * s
        st.metric("Live P&L", f"{cur}{pnl:,.0f}", f"{(px / t['entry'] - 1) * 100 * s:+.2f}%")
    st.button("🚪 Exit", key=f"exit_{t['id']}", on_click=manual_exit_cb, args=(t["id"],))


def process_trades(only=None, q=None, wait=False):
    ss = st.session_state
    now = tr.now_ist()
    for t in ss.trades:
        sym = t["symbol"]
        if only and sym != only:
            continue
        if t["status"] == "CLOSED":
            continue
        fine, _ = hist(sym, "5m", "1mo", 60 if market_open(sym) else 900, wait=wait)
        coarse = None
        if t["kind"] == "long":
            coarse, _ = hist(sym, "1d", "2y", 900, wait=wait)
        if fine.empty and (coarse is None or coarse.empty):
            continue
        tr.evaluate_trade(t, fine, coarse)
        qq = q if (q and sym == only) else live_quote(sym, ttl=5.0, wait=False)
        px = qq["price"] if qq else None
        if t["status"] != "CLOSED" and px:
            tr.apply_price(t, px, now)
        if t["status"] != "CLOSED":
            last_px = px or (float(fine["Close"].iloc[-1]) if not fine.empty else t["entry"])
            tr.check_expiry(t, last_px, now)
        if t["status"] == "CLOSED":
            tr.postmortem(t)
            ok = (t["pnl"] or 0) > 0
            st.toast(f"{'🎯 TARGET HIT' if ok else '🛑 STOP LOSS HIT'} · {sym} · {t['r_mult']:+.2f}R")


def get_signals(symbol, d_df, i_df, rr, capital, risk_pct):
    ss = st.session_state
    d_c = tr.drop_forming(d_df, 0)
    i_c = tr.drop_forming(i_df, 5)
    if d_c is None or len(d_c) < 15:
        return None, None, 0.0
    daily_atr = float(true_range(d_c).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    memo = ss.setdefault("_memo", {})
    sigs = ss.setdefault("_sig", {})

    def one(kind, df, cfg, intraday):
        if df is None or df.empty:
            return None
        key = (symbol, kind, str(df.index[-1]), len(df), rr, capital, risk_pct, round(daily_atr, 2))
        if key not in memo:
            memo[key] = analyze(df, intraday, cfg, daily_atr, rr, capital, risk_pct)
        sigs[(symbol, kind)] = memo[key]
        return memo[key]

    return (one("long", d_c, CFG_LONG, False), one("intraday", i_c.tail(400) if i_c is not None else None,
                                                    CFG_INTRA, True), daily_atr)


def maybe_auto(symbol, name, intra_sig, long_sig):
    ss = st.session_state
    if not ss.get("auto"):
        return
    if sum(1 for t in ss.trades if t["status"] != "CLOSED") >= 8:
        return
    for kind, sig in (("intraday", intra_sig), ("long", long_sig)):
        if kind == "intraday" and not (is_indian(symbol) and market_open(symbol)):
            continue
        if sig and tr.can_auto(sig, symbol, kind, ss.trades):
            lock_trade(symbol, name, kind, sig["side"], "auto")


# ------------------------------------------------------------------ storage
@st.cache_resource
def get_store(token, gist_id):
    return storage.Store(dict(token=token, gist_id=gist_id))


SETTING_KEYS = ["cap", "risk", "rrr", "live", "secs", "auto"]
SETTING_DEFAULT = dict(cap=100000, risk=1.0, rrr=2.0, live=True, secs=30, auto=True)


def collect_db():
    ss = st.session_state
    sett = {k: ss.get(k, SETTING_DEFAULT[k]) for k in SETTING_KEYS}
    sett["symbol"] = ss.get("symbol", "RELIANCE.NS")
    sett["names"] = {k: v for k, v in ss.get("names", {}).items() if k in ss.wl}
    return dict(version=1, watchlist=list(ss.wl), settings=sett, trades=ss.trades, notes=ss.get("notes", ""),
                updated="")


def apply_db(db, url_wl=None):
    ss = st.session_state
    db = storage.fill(db)
    sett = db["settings"]
    ss.trades = db["trades"]
    ss.notes = db["notes"]
    wl = db["watchlist"] or url_wl or []
    ss.wl = [s for s in (clean_symbol(x) for x in wl) if s][:MAX_WL]
    ss.setdefault("wl_scan", {})
    ss.setdefault("names", {})
    ss.names.update(sett.get("names") or {})
    for k in SETTING_KEYS:
        v = sett.get(k, SETTING_DEFAULT[k])
        ss[k] = type(SETTING_DEFAULT[k])(v) if v is not None else SETTING_DEFAULT[k]
    if sett.get("symbol"):
        ss.symbol = clean_symbol(sett["symbol"]) or "RELIANCE.NS"


def persist(force=False):
    ss = st.session_state
    if not ss.get("db_ready"):
        return
    store = get_store(secret("GITHUB_TOKEN"), secret("GIST_ID"))
    snap = copy.deepcopy(collect_db())
    if storage._cmp_key(storage.fill(copy.deepcopy(snap))) == store.last_key:
        return
    if getattr(store, "_busy", False):
        return
    if not force and time.time() - ss.get("_last_save", 0) < 8:
        return
    ss["_last_save"] = time.time()
    threading.Thread(target=lambda: store.save(snap), daemon=True).start()


def clean_symbol(s):
    s = (s or "").strip().upper()
    return s if re.fullmatch(r"[A-Z0-9.&^\-]{1,25}", s) else ""


def _sync_url():
    try:
        if st.session_state.wl:
            st.query_params["wl"] = ",".join(st.session_state.wl)
    except Exception:
        pass


def open_symbol(sym, name=None):
    st.session_state.symbol = sym
    if name:
        st.session_state.names[sym] = name
    st.session_state.page = PAGE_CHART


# Boot
store = get_store(secret("GITHUB_TOKEN"), secret("GIST_ID"))
if "db_ready" not in st.session_state:
    url_wl = []
    try:
        url_wl = str(st.query_params.get("wl", "")).split(",")
    except Exception:
        pass
    st.session_state.names = {}
    st.session_state.symbol = "RELIANCE.NS"
    apply_db(store.load(), url_wl)
    st.session_state.db_ready = True

st.session_state.setdefault("page", PAGE_CHART)
st.session_state.setdefault("wl_msg", "")

# UI
st.title("📈 Stock Analyzer")

sc1, sc2 = st.columns([4, 1])
q = sc1.text_input("Search", placeholder="🔍 Stock ka naam ya symbol likho...", label_visibility="collapsed")
india_only = sc2.checkbox("Sirf India", value=True)
if q.strip():
    results = search_symbols(q.strip(), india_only)
    if results:
        key = f"pick_{q.strip().lower()}_{india_only}"
        st.selectbox("Search results", ["— chuno —"] + [r["label"] for r in results],
                     key=key, on_change=lambda: open_symbol(st.session_state[key].split(" · ")[1]))
else:
    st.selectbox("Quick pick", ["— Quick pick —", "NIFTY 50", "BANK NIFTY", "RELIANCE", "TCS", "TATAMOTORS"],
                 key="quick", on_change=lambda: open_symbol(st.session_state["quick"] + ".NS" if not st.session_state["quick"].startswith("^") else st.session_state["quick"]), label_visibility="collapsed")

PAGE_J, PAGE_DATA = "📒 Journal", "💾 Data & Price"
page = st.radio("Page", [PAGE_CHART, PAGE_WL, PAGE_J, PAGE_DATA], horizontal=True, key="page", label_visibility="collapsed")

lv1, lv2, lv3 = st.columns([1, 1, 2])
live = lv1.toggle("🔴 Live", key="live")
secs = lv2.selectbox("Refresh", [10, 15, 30, 60], key="secs", disabled=not live, label_visibility="collapsed")
lv3.toggle("🤖 Auto paper-trade", key="auto")

with st.expander("⚙️ Settings (capital, risk, target size)"):
    capital = st.number_input("Capital (₹)", min_value=1000, step=10000, key="cap")
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, step=0.25, key="risk")
    rr = st.slider("Reward : Risk (Target 1)", 1.0, 4.0, step=0.5, key="rrr")
capital, risk_pct, rr = int(capital), float(risk_pct), float(rr)

# Pages Routing
if page == PAGE_WL:
    st.subheader("⭐ Watchlist")
    st.info("Watchlist functionality active.")
    st.stop()
elif page == PAGE_J:
    st.subheader("📒 Trade Journal")
    st.caption("All historical and active trades with performance stats.")
    st.write(pd.DataFrame(st.session_state.trades))
    st.stop()
elif page == PAGE_DATA:
    st.subheader("💾 Backup / Settings")
    st.success("Gist Auto-save is configured.")
    st.stop()


# ------------------------------------------------------------------ CHART RENDER
symbol = st.session_state.symbol
name = st.session_state.names.get(symbol) or get_name(symbol)
cur0 = "₹" if is_indian(symbol) else ""

hc1, hc2 = st.columns([3, 1])
hc1.subheader(name)
hc1.caption(symbol)

open_now = market_open(symbol)
q_tick = live_quote(symbol, ttl=2.5, wait=True)
d_df, _ = hist(symbol, "1d", "5y", 120 if open_now else 1800, wait=True)
last_px = q_tick["price"] if q_tick else (float(d_df["Close"].iloc[-1]) if not d_df.empty else 0.0)

m1, m2 = st.columns(2)
m1.metric("Live Price", f"{cur0}{last_px:,.2f}")
m2.metric("Day High", f"{cur0}{float(d_df['High'].iloc[-1]) if not d_df.empty else 0.0:,.2f}")

tf = st.radio("Timeframe", list(TF.keys()), index=5, horizontal=True, label_visibility="collapsed")
overlays = ["EMA 20/50", "Volume", "Support/Resistance", "Position tool", "Chart patterns"]
pos_mode = "Auto"
view = "110"

i_df, _ = hist(symbol, "5m", "1mo", int(secs) if live else 300)
chart_df = d_df if tf == "1D" else (i_df if tf == "5m" else hist(symbol, *TF[tf], 300)[0])
chart_df = prep_chart_df(chart_df, tf)

long_sig, intra_sig, daily_atr = get_signals(symbol, d_df, i_df, rr, capital, risk_pct)
maybe_auto(symbol, name, intra_sig, long_sig)
process_trades(only=symbol)

chart_pats = find_patterns(chart_df)
kind = "intraday" if tf in INTRADAY_TF else "long"
chart_sig = intra_sig if kind == "intraday" else long_sig

# Draw active and all historic closed trades permanently
draw = [t for t in st.session_state.trades if t["symbol"] == symbol and t["kind"] == kind]

rev = f"{symbol}|{tf}|{view}"
fig = build_figure(chart_df, tf, name, symbol, overlays, chart_sig, pos_mode, chart_pats, view, rev, trades_draw=draw)
st.plotly_chart(fig, theme=None, config=PLOT_CONFIG, use_container_width=True)

tab1, tab2 = st.tabs(["⚡ Intraday", "📅 Long Term"])
with tab1:
    render_tab(intra_sig, "intraday", daily_atr, cur0, "Intraday data loading...", symbol, name)
with tab2:
    render_tab(long_sig, "long", daily_atr, cur0, "Long term data loading...", symbol, name)

persist(True)
