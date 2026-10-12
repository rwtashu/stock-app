"""
Technical Indicators, Chart Patterns & Signal Engine
"""
import math
import numpy as np
import pandas as pd

# Constants & Configurations
CFG_INTRA = dict(fast=9, slow=21)
CFG_LONG = dict(fast=50, slow=200)
BULLISH = ["Hammer", "Bullish Engulfing", "Morning Star"]
BEARISH = ["Shooting Star", "Bearish Engulfing", "Evening Star"]
GREEN, RED, AMBER, BLUE = "#26a69a", "#ef5350", "#ffb300", "#42a5f5"

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

def f2(x):
    return round(float(x), 2)

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

    if len(hs) >= 2:
        (i1, p1), (i2, p2) = hs[-2], hs[-1]
        if last - i2 <= recent and i2 - i1 >= max(8, 2 * k):
            seg = lo_a[i1:i2 + 1]
            j = i1 + int(seg.argmin())
            neck = float(seg.min())
            top = (p1 + p2) / 2
            height = top - neck
            if height >= 3 * atr and abs(p1 - p2) <= min(1.0 * atr, 0.25 * height) and price <= max(p1, p2) + 0.5 * atr:
                sd = -1 if price < neck - 0.2 * atr else 0
                finish("Double Top", sd,
                       [(i1, p1, j, neck, AMBER, False), (j, neck, i2, p2, AMBER, False), (i1, neck, last, neck, BLUE, True)],
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
            if height >= 3 * atr and abs(p1 - p2) <= min(1.0 * atr, 0.25 * height) and price >= min(p1, p2) - 0.5 * atr:
                sd = 1 if price > neck + 0.2 * atr else 0
                finish("Double Bottom", sd,
                       [(i1, p1, j, neck, AMBER, False), (j, neck, i2, p2, AMBER, False), (i1, neck, last, neck, BLUE, True)],
                       [(i1, "belowBar", "B1", GREEN, "arrowUp"), (i2, "belowBar", "B2", GREEN, "arrowUp")],
                       (i2, bot - 0.6 * atr), (neck, neck + height), (min(p1, p2), None), i2)

    if len(hs) >= 3:
        (a, pa), (b, pb), (c, pc) = hs[-3:]
        hh = pb - (pa + pc) / 2
        if last - c <= recent and pb > max(pa, pc) and hh >= 1.5 * atr and abs(pa - pc) <= 0.5 * hh:
            s1, s2 = lo_a[a:b + 1], lo_a[b:c + 1]
            t1, t2 = a + int(s1.argmin()), b + int(s2.argmin())
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
                       [(a, "aboveBar", "LS", RED, "arrowDown"), (b, "aboveBar", "Head", RED, "arrowDown"), (c, "aboveBar", "RS", RED, "arrowDown")],
                       (b, pb + 0.6 * atr), (pb, None), (neck_last, neck_last - height), c)
    if len(ls) >= 3:
        (a, pa), (b, pb), (c, pc) = ls[-3:]
        hh = (pa + pc) / 2 - pb
        if last - c <= recent and pb < min(pa, pc) and hh >= 1.5 * atr and abs(pa - pc) <= 0.5 * hh:
            s1, s2 = hi_a[a:b + 1], hi_a[b:c + 1]
            t1, t2 = a + int(s1.argmax()), b + int(s2.argmax())
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
                       [(a, "belowBar", "LS", GREEN, "arrowUp"), (b, "belowBar", "Head", GREEN, "arrowUp"), (c, "belowBar", "RS", GREEN, "arrowUp")],
                       (b, pb - 0.6 * atr), (neck_last, neck_last + height), (pb, None), c)

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

def make_plan(side, entry, support, resistance, daily_atr, intraday, rr, capital, risk_pct):
    lo_m, hi_m = (0.30, 0.60) if intraday else (1.0, 2.5)
    struct = entry - (support - 0.1 * daily_atr) if side == "BUY" else (resistance + 0.1 * daily_atr) - entry
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

    if last["EMA_F"] > last["EMA_S"] and price > last["EMA_S"]:
        trend = "up"
        add("Trend", "Trend upar hai (fast EMA > slow EMA, price dono ke upar)", 2)
    elif last["EMA_F"] < last["EMA_S"] and price < last["EMA_S"]:
        trend = "down"
        add("Trend", "Trend niche hai (fast EMA < slow EMA, price dono ke niche)", -2)
    else:
        add("Trend", "Trend mixed / sideways hai", 0)
    add("Trend", "Price fast EMA ke upar" if price > last["EMA_F"] else "Price fast EMA ke niche", 1 if price > last["EMA_F"] else -1)

    if intraday and "VWAP" in df and not pd.isna(last["VWAP"]):
        add("Trend", "Price VWAP ke upar (buyers strong)" if price > last["VWAP"] else "Price VWAP ke niche (sellers strong)", 1 if price > last["VWAP"] else -1)

    adx = float(last["ADX"]) if "ADX" in df and not pd.isna(last["ADX"]) else None
    if adx is not None:
        add("Trend", f"ADX {adx:.0f}: trend mazboot hai" if adx >= 25 else (f"ADX {adx:.0f}: trend kamzor / sideways" if adx < 18 else f"ADX {adx:.0f}: trend madhyam"), 0)

    rsi = float(last["RSI"])
    if rsi > 70: add("Momentum", f"RSI {rsi:.0f}: overbought, pullback ka risk", -1)
    elif rsi < 30: add("Momentum", f"RSI {rsi:.0f}: oversold, bounce ho sakta hai", 1)
    elif rsi > 55: add("Momentum", f"RSI {rsi:.0f}: bullish momentum", 1)
    elif rsi < 45: add("Momentum", f"RSI {rsi:.0f}: bearish momentum", -1)
    else: add("Momentum", f"RSI {rsi:.0f}: neutral", 0)

    add("Momentum", "MACD signal line ke upar" if last["MACD"] > last["MACD_SIG"] else "MACD signal line ke niche", 1 if last["MACD"] > last["MACD_SIG"] else -1)

    tail = df.tail(150)
    hs, ls = pivots(tail, int(min(8, max(3, len(tail) // 30))))
    if len(hs) >= 2 and len(ls) >= 2:
        hh, hl = hs[-1][1] > hs[-2][1], ls[-1][1] > ls[-2][1]
        if hh and hl: add("Structure", "Higher High + Higher Low (uptrend structure)", 2)
        elif (not hh) and (not hl): add("Structure", "Lower High + Lower Low (downtrend structure)", -2)
        else: add("Structure", "Swing highs/lows mixed (structure clear nahi)", 0)

    vol = df["Volume"]
    if len(vol) > 21 and float(vol.iloc[-21:-1].mean()) > 0:
        vr = float(vol.iloc[-1]) / float(vol.iloc[-21:-1].mean())
        vol_ratio = vr
        green = last["Close"] > last["Open"]
        if vr >= 1.5: add("Volume", f"Volume {vr:.1f}x average, {'green' if green else 'red'} candle ke saath", 1 if green else -1)
        elif vr < 0.6: add("Volume", f"Volume kam hai ({vr:.1f}x average): move par bharosa kam", 0)

    recent = pats.tail(3)
    found = False
    for nm in BULLISH:
        if recent[nm].any(): add("Candlestick", f"Bullish pattern: {nm}", 1); found = True
    for nm in BEARISH:
        if recent[nm].any(): add("Candlestick", f"Bearish pattern: {nm}", -1); found = True
    if recent["Doji"].iloc[-1]: add("Candlestick", "Aakhri candle Doji hai: confusion", 0); found = True
    if not found: add("Candlestick", "Pichhli 3 candles me koi major pattern nahi", 0)

    for p in cpats[:2]:
        base = 2 if p["status"].startswith("Confirmed") else 1
        s = base if p["bias"] == "bullish" else -base if p["bias"] == "bearish" else 0
        add("Chart pattern", f"{p['name']} ({p['status']}), jhukav {p['bias']}", s)

    support, resistance = sr_levels(df, price, atr)
    add("Levels", f"Nearest support {support:,.2f} | resistance {resistance:,.2f}", 0)

    score = state["score"]
    side = "BUY" if score >= 4 else "SELL" if score <= -4 else "WAIT"
    strength = "Strong" if abs(score) >= 7 else "Moderate" if abs(score) >= 4 else "Weak"

    plans = {s: make_plan(s, price, support, resistance, daily_atr, intraday, rr, capital, risk_pct) for s in ("BUY", "SELL")}
    cond = {
        "BUY": make_plan("BUY", resistance + 0.05 * daily_atr, support, resistance, daily_atr, intraday, rr, capital, risk_pct),
        "SELL": make_plan("SELL", support - 0.05 * daily_atr, support, resistance, daily_atr, intraday, rr, capital, risk_pct),
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
