""" Stock Analyzer v3 - TradingView-style chart (scroll / pinch zoom / timeframes) - Auto chart patterns drawn on chart (+ explanation and expected move) - Long / Short position tool (Entry, Stop-loss, Targets) drawn on chart - Watchlist (saved in the page URL), search incl. new listings Run locally: streamlit run stock_analyzer.py Educational tool only. Not financial advice. """
import html
import json
import math
import re

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
CFG_INTRA = dict(fast=9, slow=21)
CFG_LONG = dict(fast=50, slow=200)
BULLISH = ["Hammer", "Bullish Engulfing", "Morning Star"]
BEARISH = ["Shooting Star", "Bearish Engulfing", "Evening Star"]
GREEN, RED, AMBER, BLUE = "#26a69a", "#ef5350", "#ffb300", "#42a5f5"
PAGE_CHART, PAGE_WL = "📊 Chart", "⭐ Watchlist"
MAX_WL = 20


def f2(x):
    return round(float(x), 2)


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
    try:
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()
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

    def add(cat, text, s):
        state["score"] += s
        reasons.append((cat, text, s))

    # Trend
    if last["EMA_F"] > last["EMA_S"] and price > last["EMA_S"]:
        add("Trend", "Trend upar hai (fast EMA > slow EMA, price dono ke upar)", 2)
    elif last["EMA_F"] < last["EMA_S"] and price < last["EMA_S"]:
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
                pats=pats, cpats=cpats)


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
def scan_one(sym, rr, capital, risk_pct):
    d = load(sym, "1d", "2y")
    if d.empty or len(d) < 2:
        return None
    i = load(sym, "5m", "5d")
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


CHART_HTML = """ <div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1117;border-radius:8px;padding:6px"> <div id="legend" style="color:#d1d4dc;font-size:13px;padding:4px 6px;min-height:56px;line-height:1.5"></div> <div id="wrap" style="position:relative;width:100%"> <div id="chart" style="width:100%;height:500px"></div> <canvas id="ov" style="position:absolute;left:0;top:0;pointer-events:none"></canvas> </div> <div style="display:flex;gap:8px;padding:8px 4px 2px 4px;flex-wrap:wrap"> <button class="b" onclick="zoom(0.65)">＋ Zoom in</button> <button class="b" onclick="zoom(1.5)">－ Zoom out</button> <button class="b" onclick="fitAll()">Fit all</button> <button class="b" onclick="latest()">Latest ▶</button> </div> </div> <style>.b{background:#1f2430;color:#d1d4dc;border:1px solid #2a2e39;border-radius:6px;padding:7px 12px;font-size:13px;cursor:pointer}.b:active{background:#2a2e39}</style> <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script> <script> const D = __DATA__; const el = document.getElementById('chart'); const ov = document.getElementById('ov'); const legend = document.getElementById('legend'); const H = 500; function zoom() {} function fitAll() {} function latest() {} if (typeof LightweightCharts === 'undefined') { el.innerHTML = '<div style="color:#ef5350;padding:20px">Chart library load nahi hui. Internet check karke page refresh karo.</div>'; } else { const n = D.candles.length; const P = D.pos; const chart = LightweightCharts.createChart(el, { width: el.clientWidth, height: H, layout: { background: { type: 'solid', color: '#0e1117' }, textColor: '#d1d4dc' }, grid: { vertLines: { color: '#1b1f2b' }, horzLines: { color: '#1b1f2b' } }, crosshair: { mode: LightweightCharts.CrosshairMode.Normal }, rightPriceScale: { borderColor: '#2a2e39' }, timeScale: { borderColor: '#2a2e39', timeVisible: D.intraday, secondsVisible: false, rightOffset: 6 }, handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }, handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true }, localization: { priceFormatter: p => p.toFixed(2) } }); const candles = chart.addCandlestickSeries({ upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350', autoscaleInfoProvider: (orig) => { const r = orig(); if (!r || !P) return r; try { const vr = chart.timeScale().getVisibleLogicalRange(); if (vr && vr.to >= n - 1) { r.priceRange.minPrice = Math.min(r.priceRange.minPrice, P.lo); r.priceRange.maxPrice = Math.max(r.priceRange.maxPrice, P.hi); } } catch (e) {} return r; } }); candles.setData(D.candles); if (D.volume.length) { const vol = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: '' }); vol.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } }); vol.setData(D.volume); } (D.lines || []).forEach(l => { const s = chart.addLineSeries({ color: l.color, lineWidth: l.width, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }); s.setData(l.data); }); (D.segs || []).forEach(sg => { const t1 = D.candles[sg.i1], t2 = D.candles[sg.i2]; if (!t1 || !t2 || sg.i2 <= sg.i1) return; const s = chart.addLineSeries({ color: sg.color, lineWidth: 2, lineStyle: sg.dash ? 2 : 0, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }); s.setData([{ time: t1.time, value: sg.p1 }, { time: t2.time, value: sg.p2 }]); }); (D.levels || []).forEach(l => { candles.createPriceLine({ price: l.price, color: l.color, lineWidth: 1, lineStyle: l.style, axisLabelVisible: true, title: l.title }); }); if ((D.markers || []).length) { const mk = D.markers.filter(m => D.candles[m.i]).map(m => ({ time: D.candles[m.i].time, position: m.pos, color: m.color, shape: m.shape, text: m.text, size: 1 })); mk.sort((a, b) => (a.time < b.time ? -1 : a.time > b.time ? 1 : 0)); candles.setMarkers(mk); } const f = v => (+v).toFixed(2); const fmt = v => (+v).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); const pct = (a, b) => ((a / b - 1) * 100); const patLine = (D.patnames && D.patnames.length) ? '<br><span style="color:#ffb300">▣ ' + D.patnames.join(' · ') + '</span>' : ''; function setLegend(c) { if (!c) return; const col = c.close >= c.open ? '#26a69a' : '#ef5350'; legend.innerHTML = '<b style="font-size:15px;color:#fff">' + D.title + '</b> <span style="color:#9aa0a6">· ' + D.tf + '</span><br>' + 'O <span style="color:' + col + '">' + f(c.open) + '</span> &nbsp;H <span style="color:' + col + '">' + f(c.high) + '</span> &nbsp;L <span style="color:' + col + '">' + f(c.low) + '</span> &nbsp;C <span style="color:' + col + '">' + f(c.close) + '</span>' + patLine; } const lastBar = D.candles[n - 1]; setLegend(lastBar); chart.subscribeCrosshairMove(p => { const c = p && p.seriesData ? p.seriesData.get(candles) : null; setLegend(c || lastBar); }); const ext = P ? P.bars + 4 : 8; chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - 110), to: n + ext }); // ---------- overlay canvas: position tool + pattern labels const dpr = window.devicePixelRatio || 1; let sig = ''; function sizeCv() { const w = el.clientWidth; ov.width = Math.round(w * dpr); ov.height = Math.round(H * dpr); ov.style.width = w + 'px'; ov.style.height = H + 'px'; sig = ''; } function priceW() { try { return chart.priceScale('right').width(); } catch (e) { return 56; } } function box(ctx, x, y, w, h, color) { ctx.fillStyle = color; ctx.fillRect(x, y, w, h); } function hline(ctx, x1, x2, y, color, dash, lw) { ctx.strokeStyle = color; ctx.lineWidth = lw || 1; ctx.setLineDash(dash ? [5, 4] : []); ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x2, y); ctx.stroke(); ctx.setLineDash([]); } function text(ctx, s, x, y, color, bold) { ctx.font = (bold ? 'bold ' : '') + '11px -apple-system,Segoe UI,Roboto,sans-serif'; const w = ctx.measureText(s).width; ctx.fillStyle = 'rgba(14,17,23,0.72)'; ctx.fillRect(x - 2, y - 11, w + 4, 14); ctx.fillStyle = color; ctx.fillText(s, x, y); } function draw() { const w = el.clientWidth; const ctx = ov.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, H); const ts = chart.timeScale(); const maxX = w - priceW(); ctx.save(); ctx.beginPath(); ctx.rect(0, 0, maxX, H - 28); ctx.clip(); if (P) { const x1 = ts.logicalToCoordinate(n - 1), x2 = ts.logicalToCoordinate(n - 1 + P.bars); const yE = candles.priceToCoordinate(P.entry), yS = candles.priceToCoordinate(P.sl); const y1 = candles.priceToCoordinate(P.t1), y2 = candles.priceToCoordinate(P.t2); if ([x1, x2, yE, yS, y1, y2].every(v => v !== null && isFinite(v))) { const wd = Math.max(x2 - x1, 20); box(ctx, x1, Math.min(yE, y2), wd, Math.abs(y2 - yE), 'rgba(38,166,154,0.20)'); box(ctx, x1, Math.min(yE, yS), wd, Math.abs(yS - yE), 'rgba(239,83,80,0.22)'); hline(ctx, x1, x1 + wd, y2, '#26a69a', false, 1.5); hline(ctx, x1, x1 + wd, y1, '#26a69a', true, 1); hline(ctx, x1, x1 + wd, yE, '#42a5f5', false, 1.5); hline(ctx, x1, x1 + wd, yS, '#ef5350', false, 1.5); const lx = Math.max(x1 + 6, 4); const sgn = P.side === 'BUY' ? 1 : -1; text(ctx, 'T2 ' + fmt(P.t2) + ' (' + (pct(P.t2, P.entry)).toFixed(2) + '%)', lx, y2 + (sgn > 0 ? -5 : 14), '#26a69a', true); text(ctx, 'T1 ' + fmt(P.t1) + ' (' + (pct(P.t1, P.entry)).toFixed(2) + '%)', lx, y1 + (sgn > 0 ? -5 : 14), '#26a69a', false); text(ctx, (P.side === 'BUY' ? 'LONG' : 'SHORT') + ' Entry ' + fmt(P.entry) + ' R:R 1:' + P.rr, lx, yE + (sgn > 0 ? -5 : 14), '#42a5f5', true); text(ctx, 'SL ' + fmt(P.sl) + ' (' + (pct(P.sl, P.entry)).toFixed(2) + '%)', lx, yS + (sgn > 0 ? 14 : -5), '#ef5350', true); if (P.note) text(ctx, P.note, lx, Math.min(yS, y2) - 18, '#ffb300', false); } } (D.labels || []).forEach(lb => { const x = ts.logicalToCoordinate(lb.i), y = candles.priceToCoordinate(lb.p); if (x !== null && y !== null && isFinite(x) && isFinite(y)) text(ctx, lb.text, x, y, lb.color, true); }); ctx.restore(); } function loop() { const ts = chart.timeScale(); const ref = lastBar.close; const key = [el.clientWidth, ts.logicalToCoordinate(n - 1), ts.logicalToCoordinate(n + 10), candles.priceToCoordinate(ref), candles.priceToCoordinate(ref * 1.1), priceW()].join('|'); if (key !== sig) { sig = key; draw(); } requestAnimationFrame(loop); } sizeCv(); requestAnimationFrame(loop); window.zoom = function (factor) { const ts = chart.timeScale(); const r = ts.getVisibleLogicalRange(); if (!r) return; const mid = (r.from + r.to) / 2, half = (r.to - r.from) / 2 * factor; ts.setVisibleLogicalRange({ from: mid - half, to: mid + half }); }; window.fitAll = function () { chart.timeScale().fitContent(); }; window.latest = function () { chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - 110), to: n + ext }); }; window.chart = chart; window.addEventListener('resize', () => { chart.applyOptions({ width: el.clientWidth }); sizeCv(); }); } </script> """


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return str(o)


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


def render_chart(df, tf, title, overlays, sig, pos_mode, pat_list):
    daily = tf not in INTRADAY_TF
    times = to_times(df.index, daily)
    n = len(df)
    o, h, l, c, v = (df[k].tolist() for k in ["Open", "High", "Low", "Close", "Volume"])
    candles = [
        {"time": t, "open": round(a, 2), "high": round(b, 2), "low": round(d, 2), "close": round(e, 2)}
        for t, a, b, d, e in zip(times, o, h, l, c)
    ]
    volume = []
    if "Volume" in overlays and any(x > 0 for x in v):
        volume = [
            {"time": t, "value": float(x),
             "color": "rgba(38,166,154,0.45)" if e >= a else "rgba(239,83,80,0.45)"}
            for t, x, a, e in zip(times, v, o, c)
        ]
    lines = []
    if "EMA 20/50" in overlays:
        lines.append(line_series(times, df["Close"].ewm(span=20, adjust=False).mean(), "#2962ff", "EMA20"))
        if n >= 60:
            lines.append(line_series(times, df["Close"].ewm(span=50, adjust=False).mean(), "#ff9800", "EMA50"))
    if "VWAP" in overlays and not daily:
        lines.append(line_series(times, vwap(df), "#ab47bc", "VWAP"))

    levels = []
    if sig and "Support/Resistance" in overlays:
        levels.append(dict(price=f2(sig["support"]), color=GREEN, style=2, title="Support"))
        levels.append(dict(price=f2(sig["resistance"]), color=RED, style=2, title="Resistance"))

    segs, markers, labels, patnames = [], [], [], []
    if "Chart patterns" in overlays:
        for p in pat_list[:3]:
            patnames.append(f"{p['name']} ({p['status']})")
            segs += [s for s in p["lines"] if 0 <= s["i1"] < n and 0 <= s["i2"] < n]
            markers += [m for m in p["markers"] if 0 <= m["i"] < n]
            if 0 <= p["label"]["i"] < n:
                labels.append(p["label"])
    if "Candle patterns" in overlays:
        cp = detect_patterns(df).tail(40)
        base = n - len(cp)
        for k, (_, row) in enumerate(cp.iterrows()):
            for name in BULLISH:
                if row[name]:
                    markers.append(dict(i=base + k, pos="belowBar", text=name.replace("Bullish ", ""),
                                        color=GREEN, shape="arrowUp"))
            for name in BEARISH:
                if row[name]:
                    markers.append(dict(i=base + k, pos="aboveBar", text=name.replace("Bearish ", ""),
                                        color=RED, shape="arrowDown"))

    pos = None
    if "Position tool" in overlays:
        plan, note = pick_plan(sig, pos_mode)
        if plan:
            pos = dict(side=plan["side"], entry=f2(plan["entry"]), sl=f2(plan["sl"]), t1=f2(plan["t1"]),
                       t2=f2(plan["t2"]), rr=plan["rr"], bars=max(12, min(40, n // 6)), note=note,
                       lo=f2(min(plan["sl"], plan["t2"])), hi=f2(max(plan["sl"], plan["t2"])))
    payload = dict(title=html.escape(title), tf=tf, intraday=not daily, candles=candles, volume=volume,
                   lines=lines, levels=levels, segs=segs, markers=markers, labels=labels,
                   patnames=patnames, pos=pos)
    page = CHART_HTML.replace("__DATA__", json.dumps(payload, default=_json_default).replace("</", "<\\/"))
    components.html(page, height=700)


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
    box(f"### {label}\nScore **{sig['score']:+d}** · Strength **{sig['strength']}** · "
        f"BUY ≥ +4, SELL ≤ −4 · RSI **{sig['rsi']:.0f}**")

    st.markdown("#### 📍 Trade plan")
    if sig["plan"]:
        st.table(plan_rows(sig["plan"], cur))
        if intraday:
            st.caption("Intraday: din ke end tak position band karo. SELL = short selling. "
                       "Target/SL aaj ki volatility (daily ATR) se bane hain.")
        elif side == "SELL":
            st.caption("Long term me SELL = fresh buy avoid karo, holding ho to exit ya stop-loss lagao.")
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


if "symbol" not in st.session_state:
    qp_s = clean_symbol(st.query_params.get("s", "")) if hasattr(st, "query_params") else ""
    st.session_state.symbol = qp_s or "RELIANCE.NS"
    st.session_state.names = {}
if "wl" not in st.session_state:
    raw = st.query_params.get("wl", "") if hasattr(st, "query_params") else ""
    st.session_state.wl = [s for s in (clean_symbol(x) for x in str(raw).split(",")) if s][:MAX_WL]
    st.session_state.wl_scan = {}
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

page = st.radio("Page", [PAGE_CHART, PAGE_WL], horizontal=True, key="page", label_visibility="collapsed")

with st.expander("⚙️ Settings (capital, risk, target size)"):
    capital = st.number_input("Capital (₹)", min_value=1000, value=100000, step=10000)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25)
    rr = st.slider("Reward : Risk (Target 1)", 1.0, 4.0, 2.0, 0.5,
                   help="2.0 matlab Target 1 stop-loss se 2 guna door, Target 2 us se 3 guna door.")
    if st.button("🔄 Data refresh"):
        st.cache_data.clear()
        st.rerun()

# ------------------------------------------------------------------ WATCHLIST PAGE
if page == PAGE_WL:
    st.subheader("⭐ Watchlist")
    ic1, ic2 = st.columns([4, 1])
    ic1.text_input("Symbol add karo", key="wl_input", placeholder="Symbol (jaise TCS, IRCTC.NS, ^NSEI)",
                   label_visibility="collapsed")
    ic2.button("➕ Add", on_click=wl_add_input, use_container_width=True)
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
                st.session_state.wl_scan[s] = scan_one(s, rr, capital, risk_pct)
                bar.progress((k + 1) / len(wl))
            bar.empty()
        for s in wl:
            r = st.session_state.wl_scan.get(s)
            cur_s = "₹" if s.endswith((".NS", ".BO")) or s.startswith(("^NSE", "^BSE")) else ""
            with st.container(border=True):
                st.markdown(f"**{st.session_state.names.get(s) or s}** · `{s}`")
                if r:
                    ib = "—" if r["intra"] is None else f"{icon(r['intra'][1])} {r['intra'][0]} ({r['intra'][1]:+d})"
                    lb = "—" if r["long"] is None else f"{icon(r['long'][1])} {r['long'][0]} ({r['long'][1]:+d})"
                    st.write(f"{cur_s}{r['price']:,.2f} ({r['chg']:+.2f}%) · ⚡ Intraday: {ib} · 📅 Long: {lb}")
                else:
                    st.caption("Scan nahi hua. Upar 'Sabka signal scan karo' dabao.")
                b1, b2 = st.columns(2)
                b1.button("📊 Chart kholo", key=f"open_{s}", on_click=open_symbol, args=(s,),
                          use_container_width=True)
                b2.button("❌ Hatao", key=f"rm_{s}", on_click=wl_remove, args=(s,), use_container_width=True)
        st.caption("💡 Watchlist is page ke link (URL) me save hoti hai. Is page ko bookmark / 'Add to Home screen' "
                   "kar lo, to har baar wahi watchlist khulegi.")
    st.caption("⚠️ Sirf educational tool hai, financial advice nahi.")
    st.stop()

# ------------------------------------------------------------------ CHART PAGE
symbol = st.session_state.symbol
try:
    st.query_params["s"] = symbol
except Exception:
    pass

tf = st.radio("Timeframe", list(TF.keys()), index=5, horizontal=True, label_visibility="collapsed")
oc1, oc2 = st.columns([3, 1])
overlays = oc1.multiselect(
    "Chart par dikhao",
    ["EMA 20/50", "VWAP", "Volume", "Support/Resistance", "Position tool", "Chart patterns", "Candle patterns"],
    default=["EMA 20/50", "Volume", "Support/Resistance", "Position tool", "Chart patterns"],
)
pos_mode = oc2.selectbox("Position tool", ["Auto", "Long", "Short", "Off"],
                         help="Auto = strategy ke signal ke hisaab se Long ya Short. Long/Short = jo chaho wo dikhao.")

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

last = d_df.iloc[-1]
prev = d_df.iloc[-2] if len(d_df) > 1 else last
price = float(last["Close"])
chg = price - float(prev["Close"])
pct = chg / float(prev["Close"]) * 100 if float(prev["Close"]) else 0.0
wk = d_df.tail(252)
hc1, hc2 = st.columns([3, 1])
hc1.subheader(name)
hc1.caption(f"{symbol} · {len(d_df)} trading din ka data")
if symbol in st.session_state.wl:
    hc2.button("❌ Watchlist se hatao", on_click=wl_remove, args=(symbol,), use_container_width=True)
else:
    hc2.button("⭐ Watchlist me add", on_click=wl_add, args=(symbol,), use_container_width=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Price", f"{cur}{price:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
m2.metric("Day Low – High", f"{cur}{float(last['Low']):,.2f} – {float(last['High']):,.2f}")
m3.metric("52W Low – High", f"{cur}{float(wk['Low'].min()):,.2f} – {float(wk['High'].max()):,.2f}")
m4.metric("Volume", f"{int(last['Volume']):,}")
if len(d_df) < 60:
    st.info("🆕 Is stock ki history bahut kam hai (naya listing). Long-term signal kam reliable hoga.")

daily_atr = float(true_range(d_df).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
long_sig = analyze(d_df, False, CFG_LONG, daily_atr, rr, capital, risk_pct)
intra_sig = analyze(i_df, True, CFG_INTRA, daily_atr, rr, capital, risk_pct)

chart_pats = []
if not chart_df.empty:
    chart_pats = find_patterns(chart_df)
    chart_sig = intra_sig if tf in INTRADAY_TF else long_sig
    render_chart(chart_df, tf, f"{name} ({symbol})", overlays, chart_sig, pos_mode, chart_pats)
    st.caption("👆 Ungli se drag = scroll · 2 ungli se pinch = zoom · neeche ke buttons se bhi zoom. "
               "Position tool (Long/Short box) 1m–1h chart par Intraday signal se, 1D+ par Long Term signal se banta hai: "
               "🟩 profit zone, 🟥 risk zone.")

if not chart_df.empty:
    render_patterns(chart_pats, cur, tf)

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
