"""
Stock Analyzer v5
- Smooth live price (background fetch, small ticker fragment, no page blink)
- Locked Long/Short trades (levels never move until target / SL), journal with auto "why wrong" notes
- Saved watchlist / settings / trades (GitHub Gist), optional exact Groww price
- Auto chart patterns drawn on chart (+ explanation and expected move)
- Long / Short position tool (Entry, Stop-loss, Targets) drawn on chart
- Watchlist (saved in the page URL), search incl. new listings
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
    """Rough 'is the market trading now' check (holidays are not known)."""
    now = ist_now()
    mins, wd = now.hour * 60 + now.minute, now.weekday()
    if sym.endswith(("-USD", "=X")):  # crypto / forex
        return True
    if is_indian(sym):
        return wd < 5 and 9 * 60 + 15 <= mins <= 15 * 60 + 35
    # US-like hours (about 19:00 - 02:45 IST)
    return (wd <= 4 and mins >= 18 * 60 + 45) or (1 <= wd <= 5 and mins <= 2 * 60 + 45)


def stretch_kw(fn):
    """Full-width kwargs that work on both old and new Streamlit versions."""
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
    """`bucket` only changes the cache key, so data is re-fetched when the bucket (time slot) changes."""
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


# ------------------------------------------------------------------ background data hub (no blocking, no blink)
class Hub:
    """Fetches prices / candles in background threads. The UI always gets the last good value instantly,
    so a refresh never makes the page wait (that wait was the 'blink')."""

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
        """-> (value or None, age_seconds or None, is_stale). Blocks only if wait=True AND nothing is cached yet."""
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
    """-> (DataFrame, is_stale). Never blocks once data has been loaded one time."""
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
    """Single / multi candle patterns (bool DataFrame)."""
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
    """Swing highs / lows: lists of (position, price)."""
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
    "Double Top": "M jaisa shape. Price do baar lagbhag same upar ke level se wapas gira, yaani buyers us level ke upar price tika nahi paaye. Ye upar se palatne (reversal) ka signal hai.",
    "Double Bottom": "W jaisa shape. Price do baar lagbhag same niche ke level se wapas uchhla, yaani sellers us level ke niche nahi le ja paaye. Ye niche se palatne (reversal) ka signal hai.",
    "Head & Shoulders": "Beech ka peak (head) dono shoulders se unchha hai. Upar ki taraf buyers thak rahe hain. Neckline tootne par aksar bada girawat aata hai.",
    "Inverse Head & Shoulders": "Beech ka gaddha (head) dono shoulders se gehra hai. Niche ki taraf sellers thak rahe hain. Neckline ke upar close hone par aksar tezi aati hai.",
    "Ascending Triangle": "Upar flat resistance aur niche badhte hue lows. Buyers har baar upar se khareed rahe hain. Aksar upar breakout hota hai.",
    "Descending Triangle": "Niche flat support aur upar girte hue highs. Sellers har baar niche se bech rahe hain. Aksar niche breakdown hota hai.",
    "Symmetrical Triangle": "Highs girte aur lows badhte hue, price sankra ho raha hai. Market tay nahi kar paa raha. Jis taraf breakout hoga, usi taraf tez move aata hai.",
    "Rising Wedge": "Price upar ja raha hai lekin range sankri ho rahi hai, yaani upar jaane ki taakat kam ho rahi hai. Aksar niche breakdown (bearish) aata hai.",
    "Falling Wedge": "Price niche ja raha hai lekin range sankri ho rahi hai, yaani girne ki taakat kam ho rahi hai. Aksar upar breakout (bullish) aata hai.",
    "Ascending Channel": "Dono lines upar ki taraf hain, price channel ke andar badh raha hai (uptrend). Neeche wali line ke paas khareedna aur upar wali line ke paas profit book karna common hai. Neeche tootna trend badalne ka signal hai.",
    "Descending Channel": "Dono lines niche ki taraf hain, price channel ke andar gir raha hai (downtrend). Upar wali line ke paas becha jata hai. Upar tootna trend badalne ka signal hai.",
    "Range (Rectangle)": "Price do flat lines ke beech ghoom raha hai. Support ke paas khareed aur resistance ke paas bechna (range trading) common hai. Breakout par badi move aati hai.",
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
    """Detect chart patterns on the last bars. Positions are indices into `df`."""
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
        """status_dir: +1 breakout up confirmed, -1 breakdown confirmed, 0 forming."""
        typical = TYPICAL_BIAS[name]
        if status_dir > 0:
            bias, status = "bullish", "Confirmed ↑"
        elif status_dir < 0:
            bias, status = "bearish", "Confirmed ↓"
        else:
            bias, status = typical, "Forming"
        target = None
        invalid = None
        if bias == "bullish":
            target = up[1]
            invalid = down[0]
        elif bias == "bearish":
            target = down[1]
            invalid = up[0]
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

    # ---- Double top / bottom
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

    # ---- Head & Shoulders / Inverse
    if len(hs) >= 3:
        (a, pa), (b, pb), (c, pc) = hs[-3:]
        hh = pb - (pa + pc) / 2
        if last - c <= recent and pb > max(pa, pc) and hh >= 1.5 * atr and abs(pa - pc) <= 0.5 * hh:
            s1 = lo_a[a:b + 1]
            s2 = lo_a[b:c + 1]
            t1 = a + int(s1.argmin())
            t2 = b + int(s2.argmin())
            v1, v2 = float(s1.min()), float(s2.min())
            slope = (v2 - v1) / max(t2 - t1, 1)
            neck_last = v2 + slope * (last - t2)
            neck_head = v1 + slope * (b - t1)
            height = pb - neck_head
            if height > 0 and price <= pb:
                sd = -1 if price < neck_last - 0.2 * atr else 0
                finish("Head & Shoulders", sd,
                       [(a, pa, t1, v1, AMBER, False), (t1, v1, b, pb, AMBER, False),
                        (b, pb, t2, v2, AMBER, False), (t2, v2, c, pc, AMBER, False),
                        (t1, v1, last, neck_last, BLUE, True)],
                       [(a, "aboveBar", "LS", RED, "arrowDown"), (b, "aboveBar", "Head", RED, "arrowDown"),
                        (c, "aboveBar", "RS", RED, "arrowDown")],
                       (b, pb + 0.6 * atr), (pb, None), (neck_last, neck_last - height), c)
    if len(ls) >= 3:
        (a, pa), (b, pb), (c, pc) = ls[-3:]
        hh = (pa + pc) / 2 - pb
        if last - c <= recent and pb < min(pa, pc) and hh >= 1.5 * atr and abs(pa - pc) <= 0.5 * hh:
            s1 = hi_a[a:b + 1]
            s2 = hi_a[b:c + 1]
            t1 = a + int(s1.argmax())
            t2 = b + int(s2.argmax())
            v1, v2 = float(s1.max()), float(s2.max())
            slope = (v2 - v1) / max(t2 - t1, 1)
            neck_last = v2 + slope * (last - t2)
            neck_head = v1 + slope * (b - t1)
            height = neck_head - pb
            if height > 0 and price >= pb:
                sd = 1 if price > neck_last + 0.2 * atr else 0
                finish("Inverse Head & Shoulders", sd,
                       [(a, pa, t1, v1, AMBER, False), (t1, v1, b, pb, AMBER, False),
                        (b, pb, t2, v2, AMBER, False), (t2, v2, c, pc, AMBER, False),
                        (t1, v1, last, neck_last, BLUE, True)],
                       [(a, "belowBar", "LS", GREEN, "arrowUp"), (b, "belowBar", "Head", GREEN, "arrowUp"),
                        (c, "belowBar", "RS", GREEN, "arrowUp")],
                       (b, pb - 0.6 * atr), (neck_last, neck_last + height), (pb, None), c)

    # ---- Triangles / wedges / channels / range
    if len(hs) >= 3 and len(ls) >= 3:
        H, L = hs[-3:], ls[-3:]
        i0 = min(H[0][0], L[0][0])
        span = last - i0
        if span >= 12 and last - max(H[-1][0], L[-1][0]) <= recent:
            xh = np.array([p[0] for p in H], float)
            yh = np.array([p[1] for p in H], float)
            xl = np.array([p[0] for p in L], float)
            yl = np.array([p[1] for p in L], float)
            mh, bh = np.polyfit(xh, yh, 1)
            ml, bl = np.polyfit(xl, yl, 1)

            def up(x):
                return mh * x + bh

            def lo(x):
                return ml * x + bl

            h0 = up(i0) - lo(i0)
            he = up(last) - lo(last)
            hmid = (h0 + he) / 2
            if h0 > 0 and he > 0.1 * h0 and hmid > 1.5 * atr:
                rh = float(np.abs(yh - (mh * xh + bh)).max())
                rl = float(np.abs(yl - (ml * xl + bl)).max())
                if rh <= 0.25 * hmid and rl <= 0.25 * hmid:
                    dh, dl = mh * span, ml * span
                    thr = 0.15 * h0

                    def sgn(v):
                        return 0 if abs(v) < thr else (1 if v > 0 else -1)

                    sh, sl_ = sgn(dh), sgn(dl)
                    ratio = he / h0
                    name = None
                    if ratio < 0.75:
                        if sh == 0 and sl_ == 1:
                            name = "Ascending Triangle"
                        elif sh == -1 and sl_ == 0:
                            name = "Descending Triangle"
                        elif sh == -1 and sl_ == 1:
                            name = "Symmetrical Triangle"
                        elif sh == 1 and sl_ == 1:
                            name = "Rising Wedge"
                        elif sh == -1 and sl_ == -1:
                            name = "Falling Wedge"
                    elif 0.8 <= ratio <= 1.25:
                        if sh == 1 and sl_ == 1:
                            name = "Ascending Channel"
                        elif sh == -1 and sl_ == -1:
                            name = "Descending Channel"
                        elif sh == 0 and sl_ == 0:
                            name = "Range (Rectangle)"
                    if name:
                        upl, lol = up(last), lo(last)
                        sd = 1 if price > upl + 0.2 * atr else -1 if price < lol - 0.2 * atr else 0
                        col = BIAS_COLOR[TYPICAL_BIAS[name]]
                        marks = [(i, "aboveBar", "", col, "circle") for i, _ in H] + \
                                [(i, "belowBar", "", col, "circle") for i, _ in L]
                        li = i0 + int(span * 0.2)
                        finish(name, sd,
                               [(i0, up(i0), last, upl, col, False), (i0, lo(i0), last, lol, col, False)],
                               marks, (li, up(li) + 0.6 * atr),
                               (upl, upl + h0), (lol, lol - h0), max(H[-1][0], L[-1][0]))

    out.sort(key=lambda p: -p["end"])
    return out[:4]


# ------------------------------------------------------------------ trade plan
def make_plan(side, entry, support, resistance, daily_atr, intraday, rr, capital, risk_pct):
    """Entry / SL / targets. Risk is sized from the DAILY ATR so targets are meaningful."""
    lo_m, hi_m = (0.30, 0.60) if intraday else (1.0, 2.5)
    if side == "BUY":
        struct = entry - (support - 0.1 * daily_atr)
    else:
        struct = (resistance + 0.1 * daily_atr) - entry
    risk = min(max(struct, lo_m * daily_atr), hi_m * daily_atr)
    sgn = 1 if side == "BUY" else -1
    plan = dict(
        side=side, entry=entry, risk=risk, rr=rr,
        sl=entry - sgn * risk,
        t1=entry + sgn * rr * risk,
        t2=entry + sgn * (rr + 1) * risk,
    )
    qty = math.floor(capital * risk_pct / 100 / risk) if risk > 0 else 0
    plan["qty"] = int(max(min(qty, math.floor(capital / entry)), 0))
    return plan


# ------------------------------------------------------------------ signal engine
def build_signal(df, pats, cpats, intraday, daily_atr, rr, capital, risk_pct):
    last = df.iloc[-1]
    price, atr = float(last["Close"]), float(last["ATR"])
    state = {"score": 0}
    reasons = []
    trend = "mixed"
    vol_ratio = None

    def add(cat, text, s):
        state["score"] += s
        reasons.append((cat, text, s))

    # Trend
    if last["EMA_F"] > last["EMA_S"] and price > last["EMA_S"]:
        trend = "up"
        add("Trend", "Trend upar hai (fast EMA > slow EMA, price dono ke upar)", 2)
    elif last["EMA_F"] < last["EMA_S"] and price < last["EMA_S"]:
        trend = "down"
        add("Trend", "Trend niche hai (fast EMA < slow EMA, price dono ke niche)", -2)
    else:
        add("Trend", "Trend mixed / sideways hai", 0)
    add("Trend", "Price fast EMA ke upar" if price > last["EMA_F"] else "Price fast EMA ke niche",
        1 if price > last["EMA_F"] else -1)
    if intraday and "VWAP" in df and not pd.isna(last["VWAP"]):
        if price > last["VWAP"]:
            add("Trend", "Price VWAP ke upar (buyers strong)", 1)
        else:
            add("Trend", "Price VWAP ke niche (sellers strong)", -1)
    adx = float(last["ADX"]) if "ADX" in df and not pd.isna(last["ADX"]) else None
    if adx is not None:
        if adx >= 25:
            add("Trend", f"ADX {adx:.0f}: trend mazboot hai", 0)
        elif adx < 18:
            add("Trend", f"ADX {adx:.0f}: trend kamzor / sideways, breakout trade me dhyan rakho", 0)
        else:
            add("Trend", f"ADX {adx:.0f}: trend madhyam", 0)

    # Momentum
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

    # Market structure (swing highs / lows)
    tail = df.tail(150)
    hs, ls = pivots(tail, int(min(8, max(3, len(tail) // 30))))
    if len(hs) >= 2 and len(ls) >= 2:
        hh, hl = hs[-1][1] > hs[-2][1], ls[-1][1] > ls[-2][1]
        if hh and hl:
            add("Structure", "Higher High + Higher Low (uptrend structure)", 2)
        elif (not hh) and (not hl):
            add("Structure", "Lower High + Lower Low (downtrend structure)", -2)
        else:
            add("Structure", "Swing highs/lows mixed (structure clear nahi)", 0)

    # Volume
    vol = df["Volume"]
    if len(vol) > 21 and float(vol.iloc[-21:-1].mean()) > 0:
        vr = float(vol.iloc[-1]) / float(vol.iloc[-21:-1].mean())
        vol_ratio = vr
        green = last["Close"] > last["Open"]
        if vr >= 1.5:
            add("Volume", f"Volume {vr:.1f}x average, {'green' if green else 'red'} candle ke saath",
                1 if green else -1)
        elif vr < 0.6:
            add("Volume", f"Volume kam hai ({vr:.1f}x average): move par bharosa kam", 0)

    # Candlestick
    recent = pats.tail(3)
    found = False
    for name in BULLISH:
        if recent[name].any():
            add("Candlestick", f"Bullish pattern: {name}", 1)
            found = True
    for name in BEARISH:
        if recent[name].any():
            add("Candlestick", f"Bearish pattern: {name}", -1)
            found = True
    if recent["Doji"].iloc[-1]:
        add("Candlestick", "Aakhri candle Doji hai: market me confusion", 0)
        found = True
    if not found:
        add("Candlestick", "Pichhli 3 candles me koi major pattern nahi", 0)

    # Chart patterns
    for p in cpats[:2]:
        base = 2 if p["status"].startswith("Confirmed") else 1
        s = base if p["bias"] == "bullish" else -base if p["bias"] == "bearish" else 0
        add("Chart pattern", f"{p['name']} ({p['status']}), jhukav {p['bias']}", s)

    support, resistance = sr_levels(df, price, atr)
    add("Levels", f"Nearest support {support:,.2f} | resistance {resistance:,.2f}", 0)

    score = state["score"]
    side = "BUY" if score >= 4 else "SELL" if score <= -4 else "WAIT"
    strength = "Strong" if abs(score) >= 7 else "Moderate" if abs(score) >= 4 else "Weak"

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


# ------------------------------------------------------------------ chart (Plotly, runs natively inside Streamlit)
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
    """The category x-axis needs unique labels, so drop candles whose label repeats."""
    if df is None or df.empty:
        return df
    labs = pd.Series(make_labels(df.index, tf), index=df.index)
    return df[~labs.duplicated(keep="last").to_numpy()]


def pick_plan(sig, pos_mode):
    """Which long/short plan to draw: returns (plan, note) or (None, None)."""
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
    """Position of the candle that contains time `ts` (None if before the first candle)."""
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
        atr_c = float(np.nanmean(h - l))
    if not atr_c > 0:
        atr_c = 1.0

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
                                 name="EMA 20", line=dict(color="#2962ff", width=1.3), hoverinfo="skip"),
                      row=1, col=1)
        if n >= 60:
            fig.add_trace(go.Scatter(x=labs, y=s_close.ewm(span=50, adjust=False).mean().round(2), mode="lines",
                                     name="EMA 50", line=dict(color="#ff9800", width=1.3), hoverinfo="skip"),
                          row=1, col=1)
    if "VWAP" in overlays and not daily:
        fig.add_trace(go.Scatter(x=labs, y=vwap(df).round(2), mode="lines", name="VWAP",
                                 line=dict(color="#ab47bc", width=1.3, dash="dot"), hoverinfo="skip"),
                      row=1, col=1)
    if has_vol:
        vcol = [GREEN_A if cc >= oo else RED_A for oo, cc in zip(o, c)]
        fig.add_trace(go.Bar(x=labs, y=v, marker_color=vcol, name="Volume", showlegend=False), row=2, col=1)

    # ---- support / resistance, last price line
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

    # ---- chart patterns + candle patterns
    groups = {}

    def add_mark(i, pos, text, color, shape):
        if not (0 <= i < n):
            return
        sym = {"arrowUp": "triangle-up", "arrowDown": "triangle-down"}.get(shape, "circle")
        y = h[i] + 0.35 * atr_c if pos == "aboveBar" else l[i] - 0.35 * atr_c
        g = groups.setdefault((color, sym, pos), ([], [], []))
        g[0].append(labs[i])
        g[1].append(f2(y))
        g[2].append(text)

    shown = pat_list[:3] if "Chart patterns" in overlays else []
    for p in shown:
        for ln in p["lines"]:
            i1, i2 = int(ln["i1"]), int(ln["i2"])
            if 0 <= i1 < n and 0 <= i2 < n and i1 != i2:
                fig.add_trace(go.Scatter(
                    x=[labs[i1], labs[i2]], y=[ln["p1"], ln["p2"]], mode="lines", showlegend=False,
                    line=dict(color=ln["color"], width=2, dash="dot" if ln["dash"] else "solid"),
                    hoverinfo="skip"), row=1, col=1)
        for m in p["markers"]:
            add_mark(int(m["i"]), m["pos"], m["text"], m["color"], m["shape"])
        lb = p["label"]
        if 0 <= int(lb["i"]) < n:
            fig.add_annotation(x=labs[int(lb["i"])], y=lb["p"], text="<b>" + html.escape(lb["text"]) + "</b>",
                               showarrow=False, font=dict(color=lb["color"], size=12),
                               bgcolor="rgba(14,17,23,0.75)", row=1, col=1)
    if "Candle patterns" in overlays:
        cp = detect_patterns(df).tail(40)
        base = n - len(cp)
        for k, (_, row) in enumerate(cp.iterrows()):
            for nm in BULLISH:
                if row[nm]:
                    add_mark(base + k, "belowBar", nm.replace("Bullish ", ""), GREEN, "arrowUp")
            for nm in BEARISH:
                if row[nm]:
                    add_mark(base + k, "aboveBar", nm.replace("Bearish ", ""), RED, "arrowDown")
    for (color, sym, pos), (xs, ys, txt) in groups.items():
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=txt, showlegend=False, hoverinfo="skip",
            textposition="top center" if pos == "aboveBar" else "bottom center",
            textfont=dict(color=color, size=10), marker=dict(symbol=sym, size=9, color=color)),
            row=1, col=1)

    # ---- long / short position tool (locked trades first, else a reference plan)
    tool_on = "Position tool" in overlays and pos_mode != "Off"
    box_bars = max(12, min(40, n // 6))
    vis = n if view == "Sab" else min(int(view), n)
    lo_v, hi_v = float(np.nanmin(l[-vis:])), float(np.nanmax(h[-vis:]))
    draw = []
    if tool_on:
        for t in (trades_draw or []):
            p0 = bar_pos(df.index, t["entry_ts"])
            if p0 is None:
                continue
            closed_t = t["status"] == "CLOSED"
            if closed_t:
                p1 = bar_pos(df.index, t["closed"])
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
        if d["note"]:
            tag(min(sl_, t2), d["note"], AMBER, False, left=False)
        if d["closed"]:
            t = d["t"]
            ok = (t["pnl"] or 0) > 0
            tag(t["exit_price"], f"{'✔' if ok else '✖'} {tr.RES_TEXT.get(t['result'], t['result'])} "
                f"{t['r_mult']:+.2f}R", GREEN if ok else RED, ok == (sgn > 0), left=False)

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
    if intraday:
        label = {"BUY": "BUY (Long)", "SELL": "SELL (Short)", "WAIT": "WAIT"}[side]
    else:
        label = {"BUY": "BUY / ACCUMULATE", "SELL": "AVOID / EXIT", "WAIT": "WAIT"}[side]
    box = {"BUY": st.success, "SELL": st.error, "WAIT": st.warning}[side]
    box(f"### {label}\nScore **{sig['score']:+d}**  ·  Strength **{sig['strength']}**  ·  "
        f"BUY ≥ +4, SELL ≤ −4  ·  RSI **{sig['rsi']:.0f}**")

    st.markdown("#### 📍 Trade plan")
    ot = open_trade(symbol, kind) if symbol else None
    if ot:
        render_open_trade(ot, cur)
    elif sig["plan"]:
        st.caption("Reference plan: last COMPLETED candle se bana hai, live price ke saath badalta nahi.")
        st.table(plan_rows(sig["plan"], cur))
        if intraday:
            st.caption("Intraday: din ke end tak position band karo. SELL = short selling. "
                       "Target/SL aaj ki volatility (daily ATR) se bane hain.")
        elif side == "SELL":
            st.caption("Long term me SELL = fresh buy avoid karo, holding ho to exit ya stop-loss lagao.")
        if symbol:
            st.button("🔒 Is signal par trade LOCK karo (live price par)", key=f"lock_{symbol}_{kind}",
                      on_click=lock_trade, args=(symbol, name, kind, side, "signal"),
                      help="Entry/SL/Target tab tak chart par ek hi jagah rahenge jab tak target ya SL hit na ho.")
    else:
        st.write("Abhi clear signal nahi hai, isliye **breakout ka wait** karo. Conditional plans:")
        cl, cs = sig["cond"]["BUY"], sig["cond"]["SELL"]
        rows = [
            ("Entry (trigger)", f"{cur}{cl['entry']:,.2f} ke upar", f"{cur}{cs['entry']:,.2f} ke niche"),
            ("Stop-loss", f"{cur}{cl['sl']:,.2f}", f"{cur}{cs['sl']:,.2f}"),
            ("Target 1", f"{cur}{cl['t1']:,.2f}", f"{cur}{cs['t1']:,.2f}"),
            ("Target 2", f"{cur}{cl['t2']:,.2f}", f"{cur}{cs['t2']:,.2f}"),
            ("Quantity", str(cl["qty"]), str(cs["qty"])),
        ]
        st.table(pd.DataFrame(rows, columns=["", "🟢 Long (upar breakout)", "🔴 Short (niche breakdown)"]).set_index(""))
        st.caption("Candle trigger ke upar/niche close ho tabhi entry lo, sirf touch par nahi.")
        if symbol:
            with st.expander("✋ Khud decision lo (manual entry lock)"):
                m1, m2 = st.columns(2)
                m1.button("🟢 Long lock", key=f"mlong_{symbol}_{kind}", on_click=lock_trade,
                          args=(symbol, name, kind, "BUY", "manual"), **stretch_kw(st.button))
                m2.button("🔴 Short lock", key=f"mshort_{symbol}_{kind}", on_click=lock_trade,
                          args=(symbol, name, kind, "SELL", "manual"), **stretch_kw(st.button))

    st.markdown("#### 📏 Kitna upar / niche ja sakta hai (approx)")
    price = sig["price"]
    periods = [("Aaj ka typical range", 1)] if intraday else [
        ("~1 hafta (5 din)", 5), ("~1 mahina (21 din)", 21), ("~3 mahine (63 din)", 63)]
    rows = []
    for name, days in periods:
        mv = daily_atr * math.sqrt(days)
        rows.append((name, f"{cur}{price + mv:,.2f}", f"{cur}{price - mv:,.2f}", f"±{mv:,.2f}"))
    st.table(pd.DataFrame(rows, columns=["Period", "Upar", "Niche", "Move"]).set_index("Period"))
    st.caption("ATR (volatility) par based andaza hai, guarantee nahi.")

    st.markdown("#### 🔎 Signal kyu bana")
    for cat in ["Trend", "Structure", "Momentum", "Volume", "Candlestick", "Chart pattern", "Levels"]:
        items = [r for r in sig["reasons"] if r[0] == cat]
        if items:
            st.markdown(f"**{cat}**")
            for _, text, s in items:
                st.markdown(f"{icon(s)} {text}")


def render_patterns(pat_list, cur, tf):
    st.markdown(f"#### 🧩 Chart patterns ({tf} chart par)")
    if not pat_list:
        st.info("Is timeframe par abhi koi clear chart pattern (triangle, double top/bottom, head & shoulders, "
                "wedge, channel) nahi mila. Doosra timeframe try karo.")
        return
    for p in pat_list:
        em = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}[p["bias"]]
        with st.container(border=True):
            st.markdown(f"**{em} {p['name']}** · {p['status']} · jhukav: **{p['bias']}**")
            st.write(p["meaning"])
            if p["status"].startswith("Confirmed"):
                if p["target"] is not None:
                    st.write(f"🎯 Expected move: **{cur}{p['target']:,.2f}** tak "
                             f"({pct_of(p['target'], p['price']):+.1f}%)")
                if p["invalid"] is not None:
                    side = "niche" if p["bias"] == "bullish" else "upar"
                    st.write(f"🛑 Pattern fail agar price {cur}{p['invalid']:,.2f} se {side} close ho.")
            else:
                parts = []
                if p["up_target"] is not None:
                    parts.append(f"**{cur}{p['up_level']:,.2f}** ke upar close → upar move, "
                                 f"target ~{cur}{p['up_target']:,.2f} ({pct_of(p['up_target'], p['price']):+.1f}%)")
                else:
                    parts.append(f"**{cur}{p['up_level']:,.2f}** ke upar close → pattern fail")
                if p["down_target"] is not None:
                    parts.append(f"**{cur}{p['down_level']:,.2f}** ke niche close → niche move, "
                                 f"target ~{cur}{p['down_target']:,.2f} ({pct_of(p['down_target'], p['price']):+.1f}%)")
                else:
                    parts.append(f"**{cur}{p['down_level']:,.2f}** ke niche close → pattern fail")
                st.write("⏳ Abhi pattern ban raha hai (confirm nahi): " + "; ".join(parts) + ".")
    st.caption("Pattern auto-detection ek andaza hai, confirmation (close + volume) ka wait zaroor karo.")


# ================================================================== trades (locked levels) + signals
def open_trade(symbol, kind):
    for t in st.session_state.trades:
        if t["symbol"] == symbol and t["kind"] == kind and t["status"] != "CLOSED":
            return t
    return None


def lock_trade(symbol, name, kind, side, mode):
    """Freeze entry / SL / targets. They never change after this until the trade closes."""
    ss = st.session_state
    sig = ss.get("_sig", {}).get((symbol, kind))
    if not sig:
        st.toast("Signal ready nahi hai, thoda ruko.")
        return
    if open_trade(symbol, kind):
        return
    if kind == "intraday" and is_indian(symbol) and not market_open(symbol):
        st.toast("Market band hai - intraday trade abhi lock nahi hoga.")
        return
    plan = sig["plan"] if (mode in ("signal", "auto") and sig["plan"]) else sig["plans"]["BUY" if side == "BUY" else "SELL"]
    q = live_quote(symbol, ttl=2.0, wait=True)
    entry = q["price"] if q else sig["price"]
    t = tr.new_trade(plan, entry, symbol, name or symbol, kind, "5m" if kind == "intraday" else "1D", mode,
                     tr.now_ist(), sig=sig, tick=0.05 if is_indian(symbol) else 0.01)
    if t["qty"] < 1:
        st.toast("Capital/risk ke hisaab se quantity 0 aa rahi hai. Settings me capital badhao.")
        return
    ss.trades.append(t)
    st.toast(f"🔒 {symbol} {t['side']} lock @ {entry:,.2f}")


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


def note_cb(tid):
    for t in st.session_state.trades:
        if t["id"] == tid:
            t["note"] = st.session_state.get(f"note_{tid}", "")


def render_open_trade(t, cur):
    q = live_quote(t["symbol"], ttl=3.0)
    px = q["price"] if q else None
    s = tr.sgn(t)
    side_w = "LONG (BUY)" if s > 0 else "SHORT (SELL)"
    st.info(f"🔒 **{side_w} trade LOCKED** · {t['opened'][:16].replace('T', ' ')} · mode: {t['mode']}\n\n"
            "Ye levels ab price ke saath nahi badlenge. Sirf target / stop-loss hit hone par trade band hoga.")
    p = dict(entry=t["entry"], sl=t["sl0"], t1=t["t1"], t2=t["t2"], rr=t.get("rr") or 2,
             risk=abs(t["entry"] - t["sl0"]), qty=t["qty"])
    st.table(plan_rows(p, cur))
    if t["t1_done"]:
        st.success(f"✅ Target 1 hit. Stop-loss ab entry ({cur}{t['entry']:,.2f}) par (break-even); aadhi qty book.")
    if px:
        pnl = t["realized"] + t["qty_left"] * (px - t["entry"]) * s
        st.metric("Live P&L", f"{cur}{pnl:,.0f}", f"{(px / t['entry'] - 1) * 100 * s:+.2f}%")
    st.button("🚪 Abhi exit karo (manual)", key=f"exit_{t['id']}", on_click=manual_exit_cb, args=(t["id"],))


def process_trades(only=None, q=None, wait=False):
    """Replay candles for every open trade from its ORIGINAL levels, then apply the live tick."""
    ss = st.session_state
    now = tr.now_ist()
    for t in ss.trades:
        sym = t["symbol"]
        if only and sym != only:
            continue
        if t["status"] == "CLOSED":
            if t.get("after") is None and (t.get("pnl") or 0) < 0 and t.get("closed"):
                fine, _ = hist(sym, "5m", "1mo", 900, wait=False)
                tr.post_exit_check(t, fine)
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
            st.toast(f"{'🎯' if ok else '🛑'} {sym} {tr.RES_TEXT.get(t['result'], t['result'])} · {t['r_mult']:+.2f}R")


def get_signals(symbol, d_df, i_df, rr, capital, risk_pct):
    """Signals from COMPLETED candles only, memoised per candle: they stay constant inside a candle."""
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
            while len(memo) > 24:
                memo.pop(next(iter(memo)))
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


# ================================================================== save / load
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
    if ss.secs not in (10, 15, 30, 60):
        ss.secs = 30
    if sett.get("symbol"):
        ss.symbol = clean_symbol(sett["symbol"]) or "RELIANCE.NS"


def _save_bg(store, snap):
    store._busy = True
    try:
        store.save(snap)
    finally:
        store._busy = False


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
    threading.Thread(target=_save_bg, args=(store, snap), daemon=True).start()


# ================================================================== state / callbacks
def clean_symbol(s):
    s = (s or "").strip().upper()
    return s if re.fullmatch(r"[A-Z0-9.&^\-]{1,25}", s) else ""


def _sync_url():
    try:
        wl = st.session_state.wl
        if wl:
            st.query_params["wl"] = ",".join(wl)
        elif "wl" in st.query_params:
            del st.query_params["wl"]
    except Exception:
        pass


def open_symbol(sym, name=None):
    st.session_state.symbol = sym
    if name:
        st.session_state.names[sym] = name
    st.session_state.page = PAGE_CHART


def pick_cb(key, results):
    val = st.session_state.get(key)
    for r in results:
        if r["label"] == val:
            open_symbol(r["symbol"], r["name"])
            return


def quick_cb():
    val = st.session_state.get("quick")
    qp = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
    if val and val != "— Quick pick —":
        open_symbol(qp.get(val, val + ".NS"))


def wl_add(sym):
    if sym and sym not in st.session_state.wl and len(st.session_state.wl) < MAX_WL:
        st.session_state.wl.append(sym)
        _sync_url()


def wl_remove(sym):
    if sym in st.session_state.wl:
        st.session_state.wl.remove(sym)
        st.session_state.wl_scan.pop(sym, None)
        _sync_url()


def wl_add_input():
    raw = st.session_state.get("wl_input", "")
    sym = clean_symbol(raw)
    if not sym:
        st.session_state.wl_msg = "Symbol sahi nahi hai."
        return
    if not sym.startswith("^") and "." not in sym:
        sym += ".NS"
    if load(sym, "1d", "1mo").empty:
        st.session_state.wl_msg = f"{sym} ka data nahi mila. Symbol check karo (NSE: NAME.NS, BSE: NAME.BO)."
        return
    if len(st.session_state.wl) >= MAX_WL:
        st.session_state.wl_msg = f"Watchlist me max {MAX_WL} stocks."
        return
    wl_add(sym)
    st.session_state.wl_msg = f"{sym} watchlist me add ho gaya."
    st.session_state.wl_input = ""


def restore_cb():
    f = st.session_state.get("restore_file")
    try:
        apply_db(json.loads(f.getvalue().decode("utf-8")))
        _sync_url()
        st.session_state.restore_msg = "✅ Backup restore ho gaya."
        st.session_state._last_save = 0
    except Exception as e:
        st.session_state.restore_msg = f"❌ Restore fail: {str(e)[:100]}"


# ------------------------------------------------------------------ boot: load saved data once per session
store = get_store(secret("GITHUB_TOKEN"), secret("GIST_ID"))
if "db_ready" not in st.session_state:
    url_wl = []
    qp_s = ""
    try:
        url_wl = str(st.query_params.get("wl", "")).split(",")
        qp_s = clean_symbol(st.query_params.get("s", ""))
    except Exception:
        pass
    st.session_state.names = {}
    st.session_state.symbol = "RELIANCE.NS"
    apply_db(store.load(), url_wl)
    if qp_s:
        st.session_state.symbol = qp_s
    st.session_state.db_ready = True
st.session_state.setdefault("page", PAGE_CHART)
st.session_state.setdefault("wl_msg", "")

# ================================================================== UI
st.title("📈 Stock Analyzer")

sc1, sc2 = st.columns([4, 1])
q = sc1.text_input("Search", placeholder="🔍 Stock ka naam ya symbol likho (reliance, zomato, tata motors...)",
                   label_visibility="collapsed")
india_only = sc2.checkbox("Sirf India", value=True)
if q.strip():
    results = search_symbols(q.strip(), india_only)
    if results:
        key = f"pick_{q.strip().lower()}_{india_only}"
        st.selectbox("Search results (yahan se stock chuno)", ["— chuno —"] + [r["label"] for r in results],
                     key=key, on_change=pick_cb, args=(key, results))
    else:
        st.info("Search me kuch nahi mila. 'Sirf India' band karke dekho, ya symbol seedha use karo "
                "(NSE: NAME.NS, BSE: NAME.BO).")
        sym_try = clean_symbol(q)
        if sym_try:
            st.button(f"'{sym_try}' ko seedha symbol ki tarah kholo", on_click=open_symbol, args=(sym_try,))
else:
    st.selectbox("Quick pick", ["— Quick pick —", "NIFTY 50", "BANK NIFTY", "SENSEX", "RELIANCE", "TCS",
                                "HDFCBANK", "INFY", "SBIN", "TATAMOTORS", "ITC", "ICICIBANK"],
                 key="quick", on_change=quick_cb, label_visibility="collapsed")

PAGE_J, PAGE_DATA = "📒 Journal", "💾 Data & Price"
page = st.radio("Page", [PAGE_CHART, PAGE_WL, PAGE_J, PAGE_DATA], horizontal=True, key="page",
                label_visibility="collapsed")

lv1, lv2, lv3 = st.columns([1, 1, 2])
live = lv1.toggle("🔴 Live", key="live", help="Market chalu ho tab price aur chart apne aap chalte rehte hain.")
secs = lv2.selectbox("Refresh", [10, 15, 30, 60], key="secs", format_func=lambda s: f"{s} sec",
                     label_visibility="collapsed", disabled=not live, help="Chart / signal refresh. Price har 3 sec me chalta hai.")
lv3.toggle("🤖 Auto paper-trade", key="auto",
           help="Strong signal aane par software khud trade lock karta hai (sirf practice, asli order nahi). "
                "Ye tabhi chalta hai jab app khula ho.")

with st.expander("⚙️ Settings (capital, risk, target size)"):
    capital = st.number_input("Capital (₹)", min_value=1000, step=10000, key="cap")
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, step=0.25, key="risk")
    rr = st.slider("Reward : Risk (Target 1)", 1.0, 4.0, step=0.5, key="rrr",
                   help="2.0 matlab Target 1 stop-loss se 2 guna door, Target 2 us se 3 guna door.")
    if st.button("🔄 Abhi refresh karo (cache saaf)"):
        st.cache_data.clear()
        get_hub().data.clear()
        st.rerun()
capital, risk_pct, rr = int(capital), float(risk_pct), float(rr)

_sv = store
if _sv.mode == "gist":
    if _sv.error:
        st.caption(f"⚠️ Save problem: {_sv.error}")
    elif _sv.saved_at:
        st.caption(f"💾 Saved ✓ {_sv.saved_at:%H:%M:%S}")
    else:
        st.caption("💾 Auto-save on (Gist)")
else:
    st.caption("⚠️ Data permanent save nahi ho raha (GITHUB_TOKEN secret nahi mila). 'Data & Price' page dekho.")
persist(True)


# ------------------------------------------------------------------ WATCHLIST PAGE
def page_watchlist():
    st.subheader("⭐ Watchlist")
    ic1, ic2 = st.columns([4, 1])
    ic1.text_input("Symbol add karo", key="wl_input", placeholder="Symbol (jaise TCS, IRCTC.NS, ^NSEI)",
                   label_visibility="collapsed")
    ic2.button("➕ Add", on_click=wl_add_input, **stretch_kw(st.button))
    if st.session_state.wl_msg:
        st.caption(st.session_state.wl_msg)
    wl = st.session_state.wl
    if not wl:
        st.info("Watchlist khali hai. Upar symbol add karo, ya kisi stock ka chart khol kar "
                "'⭐ Watchlist me add' dabao.")
    else:
        if st.button("🔄 Sabka signal scan karo"):
            bar = st.progress(0.0)
            for k, s in enumerate(wl):
                st.session_state.wl_scan[s] = scan_one(s, rr, capital, risk_pct, int(time.time() // 300))
                bar.progress((k + 1) / len(wl))
            bar.empty()
        for s in wl:
            r = st.session_state.wl_scan.get(s)
            cur_s = "₹" if is_indian(s) else ""
            with st.container(border=True):
                st.markdown(f"**{st.session_state.names.get(s) or s}**  ·  `{s}`")
                if r:
                    ib = "—" if r["intra"] is None else f"{icon(r['intra'][1])} {r['intra'][0]} ({r['intra'][1]:+d})"
                    lb = "—" if r["long"] is None else f"{icon(r['long'][1])} {r['long'][0]} ({r['long'][1]:+d})"
                    st.write(f"{cur_s}{r['price']:,.2f} ({r['chg']:+.2f}%)  ·  ⚡ Intraday: {ib}  ·  📅 Long: {lb}")
                else:
                    st.caption("Scan nahi hua. Upar 'Sabka signal scan karo' dabao.")
                b1, b2 = st.columns(2)
                b1.button("📊 Chart kholo", key=f"open_{s}", on_click=open_symbol, args=(s,),
                          **stretch_kw(st.button))
                b2.button("❌ Hatao", key=f"rm_{s}", on_click=wl_remove, args=(s,), **stretch_kw(st.button))
        st.caption("💾 Watchlist automatically save hoti hai (Gist) aur dobara waisi hi milti hai.")
    st.caption("⚠️ Sirf educational tool hai, financial advice nahi.")


# ------------------------------------------------------------------ JOURNAL PAGE
def trades_csv(trades):
    cols = ["opened", "closed", "symbol", "side", "kind", "mode", "entry", "sl0", "t1", "t2", "qty", "exit_price",
            "result", "pnl", "r_mult", "mfe", "mae", "tags"]
    rows = [{c: (";".join(t.get(c, [])) if c == "tags" else t.get(c)) for c in cols} for t in trades]
    return pd.DataFrame(rows, columns=cols).to_csv(index=False).encode("utf-8")


def page_journal():
    ss = st.session_state
    process_trades(wait=len([t for t in ss.trades if t["status"] != "CLOSED"]) <= 4)
    st.subheader("📒 Trade Journal")
    st.caption("Har trade yahan note hota hai. Galat (loss) trade ka reason software khud likhta hai. "
               "Neeche 'Claude ko dikhao' box ka text copy karke mujhe bhejo, main strategy sudhar dunga.")
    trades = ss.trades
    sx = tr.stats(trades)
    if sx["n"]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", sx["n"], f"{sx['wins']}W / {sx['losses']}L")
        c2.metric("Win rate", f"{sx['win_rate']:.0f}%")
        c3.metric("Net P&L", f"{sx['pnl']:,.0f}")
        c4.metric("Avg R", f"{sx['avg_r']:+.2f}")
        if sx["mistakes"]:
            st.markdown("**Sabse common galtiyan**")
            st.table(pd.DataFrame([(tr.TAG_NAMES.get(k, k), v) for k, v in sx["mistakes"]],
                                  columns=["Galti", "Kitni baar"]).set_index("Galti"))
            top = sx["mistakes"][0]
            if top[1] >= 2:
                st.warning(f"⚠️ Sabse zyada galti: **{tr.TAG_NAMES.get(top[0], top[0])}** ({top[1]} baar). "
                           "Isse bacho.")
    else:
        st.info("Abhi koi band trade nahi hai. Chart page par signal par '🔒 trade LOCK' karo ya Auto paper-trade on rakho.")

    opens = [t for t in trades if t["status"] != "CLOSED"]
    if opens:
        st.markdown("#### 🟡 Chalte hue trades")
        for t in opens:
            cur = "₹" if is_indian(t["symbol"]) else ""
            qd = live_quote(t["symbol"], ttl=5.0)
            s = tr.sgn(t)
            pnl = t["realized"] + t["qty_left"] * ((qd["price"] if qd else t["entry"]) - t["entry"]) * s
            with st.container(border=True):
                st.markdown(f"**{t['symbol']}** {t['side']} ({t['kind']}) · entry {cur}{t['entry']:,.2f} · "
                            f"SL {cur}{t['sl0']:,.2f} · T1 {cur}{t['t1']:,.2f} · T2 {cur}{t['t2']:,.2f}")
                st.caption(f"Live P&L {cur}{pnl:,.0f} · {t['status']} · {t['opened'][:16].replace('T', ' ')}")
                st.button("🚪 Exit karo", key=f"jx_{t['id']}", on_click=manual_exit_cb, args=(t["id"],))

    losses = [t for t in reversed(tr.closed_trades(trades)) if t["pnl"] < 0]
    wins = [t for t in reversed(tr.closed_trades(trades)) if t["pnl"] >= 0]
    st.markdown(f"#### ❌ Galat decisions ({len(losses)})")
    for t in losses:
        cur = "₹" if is_indian(t["symbol"]) else ""
        with st.expander(f"❌ {t['symbol']} {t['side']} · {t['opened'][:10]} · {t['pnl']:,.0f} ({t['r_mult']:+.2f}R) · "
                         f"{tr.RES_TEXT.get(t['result'], t['result'])}"):
            st.write(f"Entry {cur}{t['entry']:,.2f} → Exit {cur}{t['exit_price']:,.2f} · SL {cur}{t['sl0']:,.2f} · "
                     f"T1 {cur}{t['t1']:,.2f} · T2 {cur}{t['t2']:,.2f} · {t['bars']} candles")
            st.markdown("**Kyun galat hua:**")
            for w in t.get("why", []):
                st.markdown(f"- {w}")
            if t.get("entry_reasons"):
                with st.container():
                    st.caption("Entry ke waqt signal ke karan: " + " | ".join(t["entry_reasons"][:6]))
            st.text_input("Meri note", value=t.get("note", ""), key=f"note_{t['id']}", on_change=note_cb,
                          args=(t["id"],))
            st.button("🗑 Hatao", key=f"del_{t['id']}", on_click=delete_trade_cb, args=(t["id"],))
    if wins:
        with st.expander(f"✅ Sahi trades ({len(wins)})"):
            for t in wins:
                st.markdown(f"✅ {t['symbol']} {t['side']} · {t['opened'][:10]} · {t['pnl']:,.0f} "
                            f"({t['r_mult']:+.2f}R) · {tr.RES_TEXT.get(t['result'], t['result'])}")

    st.markdown("#### 📝 Meri general notes")
    st.text_area("Notes", key="notes", height=140, label_visibility="collapsed",
                 placeholder="Apni seekh / observation yahan likho...")
    st.markdown("#### 📤 Claude ko dikhao")
    text = tr.summary_text(trades, ss.get("notes", ""))
    st.code(text, language=None)
    st.caption("Upar box ke corner me copy button hai. Copy karke chat me bhej do.")
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Text", text.encode("utf-8"), "journal.txt", **stretch_kw(st.download_button))
    d2.download_button("⬇️ CSV", trades_csv(trades), "trades.csv", **stretch_kw(st.download_button))
    d3.download_button("⬇️ JSON", json.dumps(trades, indent=1, ensure_ascii=False).encode("utf-8"), "trades.json",
                       **stretch_kw(st.download_button))


# ------------------------------------------------------------------ DATA & PRICE PAGE
def page_data():
    ss = st.session_state
    st.subheader("💾 Data save")
    if store.mode == "gist":
        st.success(f"GitHub Gist se save ho raha hai. Gist id: `{(store.gist_id or 'pehli save par banegi')[:12]}`")
        if store.error:
            st.error(store.error)
        if store.saved_at:
            st.caption(f"Aakhri save: {store.saved_at:%d %b %H:%M:%S} IST")
        st.button("💾 Abhi save karo", on_click=lambda: ss.update(_last_save=0))
    else:
        st.warning("Abhi data sirf is app ke temporary storage me hai (restart par ud jata hai). Permanent save ke liye:\n\n"
                   "1. GitHub → Settings → Developer settings → Personal access tokens → **Tokens (classic)** → "
                   "Generate new token, sirf **gist** ticked.\n"
                   "2. Streamlit app → ⋮ → Settings → **Secrets** me likho:\n\n"
                   "`GITHUB_TOKEN = \"ghp_xxxxxxxx\"`\n\n3. Save karke app Reboot karo.")
    st.markdown("**Backup / Restore**")
    st.download_button("⬇️ Poora backup (JSON)", json.dumps(collect_db(), indent=1, ensure_ascii=False).encode("utf-8"),
                       "stock_app_backup.json")
    st.file_uploader("Backup file chuno", type=["json"], key="restore_file")
    if ss.get("restore_file") is not None:
        st.button("♻️ Restore karo", on_click=restore_cb)
    if ss.get("restore_msg"):
        st.caption(ss.restore_msg)

    st.subheader("💹 Price accuracy")
    cfg = groww_cfg()
    if feed.groww_configured(cfg):
        st.success("Groww API keys mili hain: NSE stocks ka price Groww se aayega (exchange wala exact price).")
    else:
        st.info("Abhi price Yahoo (free) se aa raha hai - ye Groww se kuch sec/min late ya thoda alag ho sakta hai. "
                "Bilkul Groww jaisa price chahiye to Groww Trade API (paid subscription) lo, phir Secrets me:\n\n"
                "`GROWW_API_KEY = \"...\"`\n\n`GROWW_API_SECRET = \"...\"`")
    sym = ss.get("symbol", "RELIANCE.NS")
    if st.button(f"🔍 {sym} ka price test karo"):
        r = feed.get_quote(sym, cfg, ttl=0)
        if r:
            st.write(f"Price **{r['price']:,.2f}** · source **{r['source']}** · "
                     f"exchange time: {r.get('mkt_ts') or 'pata nahi'} · {r.get('note') or ''}")
        else:
            st.error("Price nahi mila. " + feed.groww_error())
    if feed.groww_error():
        st.caption(f"Groww: {feed.groww_error()}")
    st.caption("⚠️ Sirf educational tool hai, financial advice nahi.")


if page == PAGE_WL:
    page_watchlist()
    persist(True)
    st.stop()
if page == PAGE_J:
    page_journal()
    persist(True)
    st.stop()
if page == PAGE_DATA:
    page_data()
    persist(True)
    st.stop()


# ------------------------------------------------------------------ CHART PAGE
def frag(fn, every, key):
    """Fragment that re-runs by itself every `every` seconds (None = no timer)."""
    if not hasattr(st, "fragment"):
        return fn
    try:
        return st.fragment(run_every=every, key=key)(fn)
    except TypeError:
        return st.fragment(run_every=every)(fn)


def ticker_view(symbol, cur, live, was_open):
    """Small fragment: only the price numbers. Updates every ~3 s without touching chart / tables."""
    is_open = market_open(symbol)
    if live and is_open != was_open:
        st.rerun()
    q = live_quote(symbol, ttl=2.5, wait=True)
    d_df, _ = hist(symbol, "1d", "5y", 120 if is_open else 1800, wait=True)
    if q is None and d_df.empty:
        st.error("Is symbol ka price/data nahi mila. Symbol check karo (NSE: .NS, BSE: .BO).")
        return
    last = d_df.iloc[-1] if not d_df.empty else None
    price = q["price"] if q else float(last["Close"])
    prev = (q or {}).get("prev_close")
    if not prev and len(d_df) > 1:
        prev = float(d_df["Close"].iloc[-2])
    chg = price - prev if prev else 0.0
    pct = chg / prev * 100 if prev else 0.0
    lo = (q or {}).get("low") or (float(last["Low"]) if last is not None else price)
    hi = (q or {}).get("high") or (float(last["High"]) if last is not None else price)
    vol = (q or {}).get("volume") or (float(last["Volume"]) if last is not None else 0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"{cur}{price:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
    m2.metric("Day Low – High", f"{cur}{lo:,.2f} – {hi:,.2f}")
    if not d_df.empty:
        wk = d_df.tail(252)
        m3.metric("52W Low – High", f"{cur}{float(wk['Low'].min()):,.2f} – {float(wk['High'].max()):,.2f}")
    m4.metric("Volume", f"{int(vol):,}")
    now = ist_now()
    if q:
        src = "✅ Groww (exact)" if q.get("source") == "Groww" else "Yahoo (free)"
        mt = q.get("mkt_ts")
        delay = (now - mt).total_seconds() if mt else None
        tick = f" · price {int(delay)} sec purana" if (delay is not None and is_open and delay < 600) else ""
        state = "🟢 LIVE" if (live and is_open) else "⏸ Live band" if is_open else "⚪ Market band"
        st.caption(f"{state} · {now:%H:%M:%S} IST · source: {src}{tick}")
        if is_open and q.get("source") != "Groww" and delay is not None and delay > 90 and is_indian(symbol):
            st.warning(f"Yahoo ka price ~{delay / 60:.0f} min late hai. Groww jaisa exact price ke liye "
                       "'Data & Price' page dekho (Groww API).")
        if q.get("note"):
            st.caption(f"ℹ️ {q['note']}")
    process_trades(only=symbol, q=q)
    persist()


def chart_page(symbol, name, tf, overlays, pos_mode, view, capital, risk_pct, rr, secs, live):
    """Chart + signals. Runs as its own fragment; reads cached data so it is fast (no blink)."""
    is_open = market_open(symbol)
    now_s = time.time()
    fast = max(int(secs), 10) if (live and is_open) else 300 if is_open else 1800
    slow = 120 if is_open else 1800
    first = st.session_state.get("_seen") != (symbol, tf)
    with (st.spinner("Data load ho raha hai...") if first else nullcontext()):
        d_df, stale_d = hist(symbol, "1d", "5y", slow)
        i_df, stale_i = hist(symbol, "5m", "1mo", fast)
        if tf == "1D":
            chart_df, stale_c = d_df, stale_d
        elif tf == "5m":
            chart_df, stale_c = i_df, stale_i
        else:
            chart_df, stale_c = hist(symbol, *TF[tf], fast)
    st.session_state["_seen"] = (symbol, tf)
    if d_df.empty:
        st.error("Is symbol ka data nahi mila. Symbol check karo (NSE ke liye .NS, BSE ke liye .BO). "
                 "Naya listing hai to thoda ruk kar 'Abhi refresh karo' try karo.")
        return
    cur = "₹" if is_indian(symbol) else ""
    ts = last_bar_ist(i_df if not i_df.empty else d_df)
    st.caption(f"Chart/signal refresh: {ist_now():%H:%M:%S} IST · aakhri candle "
               f"{ts:%d %b %H:%M} IST" if ts is not None else "")
    if stale_d or stale_c or stale_i:
        st.warning("Naya data nahi aa paya (Yahoo ne rok diya ho sakta hai), isliye pichhla data dikh raha hai.")
    if len(d_df) < 60:
        st.info("🆕 Is stock ki history bahut kam hai (naya listing). Long-term signal kam reliable hoga.")

    long_sig, intra_sig, daily_atr = get_signals(symbol, d_df, i_df, rr, capital, risk_pct)
    maybe_auto(symbol, name, intra_sig, long_sig)
    process_trades(only=symbol)

    chart_pats = []
    chart_df = prep_chart_df(chart_df, tf)
    if chart_df is None or chart_df.empty or len(chart_df) < 2:
        st.warning(f"{tf} timeframe ka data is stock ke liye available nahi hai. Koi aur timeframe chuno.")
    else:
        chart_pats = find_patterns(chart_df)
        kind = "intraday" if tf in INTRADAY_TF else "long"
        chart_sig = intra_sig if kind == "intraday" else long_sig
        now_i = tr.now_ist()
        draw = []
        for t in st.session_state.trades:
            if t["symbol"] != symbol or t["kind"] != kind:
                continue
            if t["status"] != "CLOSED":
                draw.append(t)
            elif t.get("closed") and (now_i - tr.to_ts(t["closed"])) < pd.Timedelta(hours=6 if kind == "intraday" else 72):
                draw.append(t)
        rev = f"{symbol}|{tf}|{view}|{len(chart_df) // 30}"
        fig = build_figure(chart_df, tf, name, symbol, overlays, chart_sig, pos_mode, chart_pats, view, rev,
                           trades_draw=draw)
        st.plotly_chart(fig, theme=None, config=PLOT_CONFIG, key=f"chart_{symbol}_{tf}",
                        **stretch_kw(st.plotly_chart))
        st.caption("👆 Ungli se drag = scroll · 2 ungli se pinch = zoom · double tap = reset. "
                   "🔒 wala box LOCKED trade hai: wo apni jagah rehta hai jab tak target / SL hit na ho. "
                   "🟩 profit zone, 🟥 risk zone.")
        render_patterns(chart_pats, cur, tf)

    tab1, tab2 = st.tabs(["⚡ Intraday", "📅 Long Term"])
    with tab1:
        st.caption("5-minute candles (completed) par based. Din ke andar ka trade.")
        render_tab(intra_sig, "intraday", daily_atr, cur,
                   "Intraday data available nahi ya bahut kam hai (market band ya naya listing ho sakta hai).",
                   symbol, name)
    with tab2:
        st.caption("Daily candles (completed) par based. Hafton/mahino ki position.")
        render_tab(long_sig, "long", daily_atr, cur,
                   "Long-term signal ke liye kam se kam 30 trading din ka data chahiye. Stock bahut naya hai.",
                   symbol, name)
    st.caption("⚠️ Sirf educational tool hai, financial advice nahi. Pehle paper trading me test karo.")
    persist()


symbol = st.session_state.symbol
try:
    st.query_params["s"] = symbol
    _sync_url()
except Exception:
    pass
name = st.session_state.names.get(symbol) or get_name(symbol)
cur0 = "₹" if is_indian(symbol) else ""

hc1, hc2 = st.columns([3, 1])
hc1.subheader(name)
hc1.caption(symbol)
if symbol in st.session_state.wl:
    hc2.button("❌ Watchlist se hatao", key=f"wl_btn_{symbol}", on_click=wl_remove, args=(symbol,),
               **stretch_kw(st.button))
else:
    hc2.button("⭐ Watchlist me add", key=f"wl_btn_{symbol}", on_click=wl_add, args=(symbol,),
               **stretch_kw(st.button))

open_now = market_open(symbol)
frag(ticker_view, 3 if (live and open_now) else None, "ticker")(symbol, cur0, live, open_now)

tf = st.radio("Timeframe", list(TF.keys()), index=5, horizontal=True, label_visibility="collapsed")
oc1, oc2, oc3 = st.columns([3, 1, 1])
overlays = oc1.multiselect(
    "Chart par dikhao",
    ["EMA 20/50", "VWAP", "Volume", "Support/Resistance", "Position tool", "Chart patterns", "Candle patterns"],
    default=["EMA 20/50", "Volume", "Support/Resistance", "Position tool", "Chart patterns"],
)
pos_mode = oc2.selectbox("Position tool", ["Auto", "Long", "Short", "Off"],
                         help="Auto = signal ke hisaab se. Locked trade hamesha apni jagah dikhta hai.")
view = oc3.selectbox("Chart view", ["50", "110", "250", "Sab"], index=1,
                     help="Chart shuru me kitni candles dikhaye. Baad me ungli se zoom/scroll kar sakte ho.")

every = int(secs) if (live and open_now) else (300 if open_now else None)
frag(chart_page, every, "live_chart")(symbol, name, tf, overlays, pos_mode, view, capital, risk_pct, rr, secs, live)
persist(True)
