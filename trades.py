"""trades.py - trade lock (levels never move), candle replay, auto post-mortem ("galat decision kyun hua"), stats.
Pure logic: no Streamlit here, so it is easy to test.
"""
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
EOD_MIN = 15 * 60 + 15  # intraday square-off 15:15 IST
LONG_MAX_DAYS = 60  # long-term trade is closed (expired) after this many calendar days

RES_TEXT = {"T2": "Target 2 hit", "T1_BE": "T1 hit, baaki break-even par band", "SL": "Stop-loss hit",
            "EOD": "Din ke end par square-off", "EXPIRED": "Time khatam (expire)", "MANUAL": "Manual exit"}


# ---------------------------------------------------------------- helpers
def now_ist():
    return datetime.now(IST)


def to_ts(x):
    """ISO string / datetime / Timestamp -> tz-aware (IST) pandas Timestamp."""
    t = pd.Timestamp(x)
    return t.tz_localize(IST) if t.tzinfo is None else t.tz_convert(IST)


def iso(x):
    return to_ts(x).isoformat(timespec="seconds")


def _r(x, tick=0.05):
    return round(round(float(x) / tick) * tick, 2) if tick else round(float(x), 2)


def sgn(t):
    return 1 if t["side"] == "BUY" else -1


# ---------------------------------------------------------------- create
def new_trade(plan, entry_price, symbol, name, kind, tf, mode, entry_ts, sig=None, qty=None, tick=0.05):
    """Lock a trade. Distances (risk/targets) come from the plan, but are re-based on the REAL entry price."""
    shift = float(entry_price) - float(plan["entry"])
    lv = {k: _r(float(plan[k]) + shift, tick) for k in ("sl", "t1", "t2")}
    entry = _r(entry_price, tick)
    sig = sig or {}
    reasons = [f"{c}: {t} ({s:+d})" for c, t, s in (sig.get("reasons") or []) if s != 0][:12]
    ctx = {k: sig.get(k) for k in ("score", "strength", "rsi", "adx", "vol_ratio", "trend", "atr",
                                   "support", "resistance")}
    ctx = {k: (round(float(v), 2) if isinstance(v, (int, float)) else v) for k, v in ctx.items()}
    return dict(
        id=uuid.uuid4().hex[:8], symbol=symbol, name=name, side=plan["side"], kind=kind, tf=tf, mode=mode,
        entry=entry, sl0=lv["sl"], sl=lv["sl"], t1=lv["t1"], t2=lv["t2"], rr=plan.get("rr"),
        qty=int(qty if qty is not None else plan.get("qty", 0)), opened=iso(now_ist()), entry_ts=iso(entry_ts),
        status="OPEN", t1_done=False, realized=0.0, qty_left=int(qty if qty is not None else plan.get("qty", 0)),
        result=None, exit_price=None, closed=None, mfe=0.0, mae=0.0, bars=0, ambiguous=False,
        pnl=None, r_mult=None, ctx=ctx, entry_reasons=reasons, why=[], tags=[], note="", after=None,
    )


def reset_state(t):
    t.update(sl=t["sl0"], status="OPEN", t1_done=False, realized=0.0, qty_left=t["qty"], mfe=0.0, mae=0.0, bars=0,
             ambiguous=False)


# ---------------------------------------------------------------- state machine
def _close(t, result, price, ts):
    price = float(price)
    s = sgn(t)
    t["result"], t["exit_price"], t["status"] = result, round(price, 2), "CLOSED"
    t["closed"] = iso(ts)
    t["pnl"] = round(t["realized"] + t["qty_left"] * (price - t["entry"]) * s, 2)
    risk = abs(t["entry"] - t["sl0"]) * max(t["qty"], 1)
    t["r_mult"] = round(t["pnl"] / risk, 2) if risk else 0.0
    t["qty_left"] = 0
    return t


def apply_bar(t, o, h, l, c, ts):
    """Feed ONE price bar after entry (a tick is a bar with o=h=l=c). Pessimistic: if SL and target are both
    inside one candle we assume SL came first. Returns True if the trade was closed on this bar."""
    if t["status"] == "CLOSED":
        return True
    s = sgn(t)
    t["bars"] += 1
    fav = (h - t["entry"]) if s > 0 else (t["entry"] - l)
    adv = (t["entry"] - l) if s > 0 else (h - t["entry"])
    t["mfe"] = round(max(t["mfe"], fav), 2)
    t["mae"] = round(max(t["mae"], adv), 2)
    sl_hit = l <= t["sl"] if s > 0 else h >= t["sl"]
    t1_hit = h >= t["t1"] if s > 0 else l <= t["t1"]
    t2_hit = h >= t["t2"] if s > 0 else l <= t["t2"]
    gap_sl = (o <= t["sl"]) if s > 0 else (o >= t["sl"])
    if sl_hit:
        if not t["t1_done"] and (t1_hit or t2_hit):
            t["ambiguous"] = True
        px = o if gap_sl else t["sl"]  # a gap through the SL fills at the open (slippage)
        _close(t, "T1_BE" if t["t1_done"] else "SL", px, ts)
        return True
    if t2_hit:
        if not t["t1_done"] and t["qty"] >= 2:
            half = t["qty"] // 2
            t["realized"] += half * (t["t1"] - t["entry"]) * s
            t["qty_left"] -= half
            t["t1_done"] = True
        _close(t, "T2", t["t2"], ts)
        return True
    if t1_hit and not t["t1_done"]:
        t["t1_done"] = True
        t["status"] = "T1"
        if t["qty"] >= 2:
            half = t["qty"] // 2
            t["realized"] += half * (t["t1"] - t["entry"]) * s
            t["qty_left"] -= half
        t["sl"] = t["entry"]  # break-even after T1
    return False


def apply_price(t, price, ts):
    return apply_bar(t, price, price, price, price, ts)


def expire(t, price, ts, result="EOD"):
    if t["status"] != "CLOSED":
        _close(t, result, price, ts)
    return True


def manual_exit(t, price, ts):
    return expire(t, price, ts, "MANUAL")


def check_expiry(t, price, ts):
    """Intraday: square off at 15:15 IST. Long: after LONG_MAX_DAYS days. Returns True if closed."""
    if t["status"] == "CLOSED":
        return True
    ts, op = to_ts(ts), to_ts(t["opened"])
    if t["kind"] == "intraday":
        past_day = ts.date() > op.date()
        if past_day or (ts.date() == op.date() and ts.hour * 60 + ts.minute >= EOD_MIN):
            return expire(t, price, ts, "EOD")
    elif (ts - op).days >= LONG_MAX_DAYS:
        return expire(t, price, ts, "EXPIRED")
    return False


def drop_forming(df, minutes, now=None):
    """Remove the candle that is still forming, so signals are computed on CLOSED candles only (no repaint).
    minutes=0 means daily candles (today's bar is dropped while it is still today)."""
    if df is None or df.empty:
        return df
    now = to_ts(now or now_ist())
    last = to_ts(df.index[-1])
    if minutes:
        return df.iloc[:-1] if last + pd.Timedelta(minutes=minutes) > now else df
    return df.iloc[:-1] if last.date() == now.date() and now.hour * 60 + now.minute < 16 * 60 else df


# ---------------------------------------------------------------- replay on candles
def _bars_after(df, after_ts):
    if df is None or df.empty:
        return []
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    m = idx > after_ts
    if not m.any():
        return []
    sub = df[m]
    return list(zip(idx[m], sub["Open"].astype(float), sub["High"].astype(float),
                    sub["Low"].astype(float), sub["Close"].astype(float)))


def evaluate_trade(t, fine_df, coarse_df=None, step_min=5):
    """Re-play all candles after entry from the ORIGINAL levels (deterministic) and update the trade.
    fine_df = e.g. 5m candles, coarse_df = daily candles (used only for days older than the fine data)."""
    if t["status"] == "CLOSED":
        return t
    reset_state(t)
    ets = to_ts(t["entry_ts"])
    bars = _bars_after(fine_df, ets)
    if coarse_df is not None and not coarse_df.empty:
        # daily bars are stamped at midnight; use only whole days that the fine data does not cover
        has_fine = fine_df is not None and not fine_df.empty
        first_fine = to_ts(fine_df.index[0]).normalize() if has_fine else None
        old = [b for b in _bars_after(coarse_df, ets.normalize()) if first_fine is None or b[0] < first_fine]
        bars = old + bars
    for ts, o, h, l, c in bars:
        if t["kind"] == "intraday":
            tsx = ts
            if tsx.date() > to_ts(t["opened"]).date() or tsx.hour * 60 + tsx.minute >= EOD_MIN:
                expire(t, o, tsx, "EOD")
                break
        if apply_bar(t, o, h, l, c, ts + pd.Timedelta(minutes=step_min if t["kind"] == "intraday" else 0)):
            break
    return t


# ---------------------------------------------------------------- auto entry (paper trade)
def can_auto(sig, symbol, kind, trades, now=None, min_score=6, cooldown_min=30):
    """Auto paper-trade gate: strong signal, no open trade in this symbol+kind, cooldown after the last exit."""
    if not sig or sig.get("side") == "WAIT" or not sig.get("plan") or abs(sig.get("score", 0)) < min_score:
        return False
    if sig["plan"].get("qty", 0) < 1:
        return False
    now = to_ts(now or now_ist())
    if kind == "intraday" and (now.hour * 60 + now.minute < 9 * 60 + 30 or now.hour * 60 + now.minute >= 14 * 60 + 45):
        return False  # skip opening noise and the last 30 minutes
    for t in trades:
        if t["symbol"] == symbol and t["kind"] == kind:
            if t["status"] != "CLOSED":
                return False
            if t.get("closed") and (now - to_ts(t["closed"])) < pd.Timedelta(
                    minutes=cooldown_min if kind == "intraday" else 24 * 60):
                return False
    return True


# ---------------------------------------------------------------- post-mortem
def _tag(tags, why, tag, text):
    tags.append(tag)
    why.append(text)


def postmortem(t):
    """Rule based 'why did it go wrong'. Fills t['why'] (text) and t['tags'] (short keys) for losing trades."""
    if t.get("status") != "CLOSED" or t.get("pnl") is None:
        return t
    why, tags = [], []
    if t["pnl"] >= 0:
        t["why"], t["tags"] = [], []
        return t
    if abs(t.get("r_mult") or 0) < 0.1:
        t["why"] = ["Lagbhag break-even (bahut chhota loss, slippage jitna) - ise galat decision nahi maana."]
        t["tags"] = ["breakeven"]
        return t
    s = sgn(t)
    ctx = t.get("ctx") or {}
    risk = abs(t["entry"] - t["sl0"]) or 1e-9
    d1 = abs(t["t1"] - t["entry"]) or 1e-9
    side_word = "BUY" if s > 0 else "SELL"

    if t["mfe"] < 0.25 * risk:
        _tag(tags, why, "wrong_direction", "Entry ke baad price kabhi sahi direction me nahi gaya (MFE bahut kam). "
             "Entry ka timing ya direction galat tha.")
    elif t["mfe"] >= 0.6 * d1 and t["result"] in ("SL", "T1_BE", "EOD"):
        _tag(tags, why, "reversal_near_target", f"Price T1 ke {t['mfe'] / d1 * 100:.0f}% tak gaya aur palat gaya. "
             "Profit book/trail nahi hua - T1 thoda paas rakho ya aadhe par SL entry par le aao.")
    trend = ctx.get("trend")
    if trend and ((s > 0 and trend == "down") or (s < 0 and trend == "up")):
        _tag(tags, why, "against_trend", f"{side_word} trade bade trend ({trend}) ke KHILAF tha.")
    vr = ctx.get("vol_ratio")
    if isinstance(vr, (int, float)) and vr < 0.8:
        _tag(tags, why, "low_volume", f"Entry par volume kam tha ({vr:.1f}x average) - move me dum nahi tha.")
    rsi = ctx.get("rsi")
    if isinstance(rsi, (int, float)) and ((s > 0 and rsi > 68) or (s < 0 and rsi < 32)):
        _tag(tags, why, "extreme_rsi", f"RSI {rsi:.0f} par entry: {'overbought me BUY' if s > 0 else 'oversold me SELL'} "
             "- pullback ka risk tha.")
    adx = ctx.get("adx")
    if isinstance(adx, (int, float)) and adx < 20:
        _tag(tags, why, "sideways", f"ADX {adx:.0f}: market sideways tha, trend-trade yahan kam chalte hain.")
    sc = ctx.get("score")
    if isinstance(sc, (int, float)) and abs(sc) < 6:
        _tag(tags, why, "weak_signal", f"Signal kamzor tha (score {sc:+.0f}) - sirf Strong (7+) signal lo.")
    atr = ctx.get("atr")
    intra = t["kind"] == "intraday"
    if isinstance(atr, (int, float)) and atr > 0 and risk < (0.35 if intra else 1.2) * atr:
        _tag(tags, why, "tight_sl", f"SL bahut tight tha (risk = daily ATR ka {risk / atr * 100:.0f}%) - normal noise me hit ho gaya.")
    lvl = ctx.get("resistance") if s > 0 else ctx.get("support")
    if isinstance(lvl, (int, float)) and isinstance(atr, (int, float)) and atr > 0 and abs(lvl - t["entry"]) < (0.25 if intra else 0.5) * atr:
        _tag(tags, why, "near_level", f"Entry {'resistance' if s > 0 else 'support'} ({lvl:,.2f}) ke bilkul paas thi - "
             "wahan price aksar ruk/palat jaata hai.")
    if t["kind"] == "intraday":
        et = to_ts(t["opened"])
        m = et.hour * 60 + et.minute
        if m >= 14 * 60 + 30:
            _tag(tags, why, "late_entry", "Din ke aakhir me entry li - target tak pahunchne ka time kam tha.")
        elif m < 9 * 60 + 30:
            _tag(tags, why, "opening_noise", "Market open hote hi (9:15-9:30) entry li - shuru ki volatility me SL hit hona aam hai.")
    if t["result"] == "SL" and t["exit_price"] is not None and abs(t["exit_price"] - t["sl0"]) > 0.2 * risk:
        _tag(tags, why, "gap_slippage", f"SL {t['sl0']:,.2f} tha par exit {t['exit_price']:,.2f} par hua (gap/slippage).")
    if t["result"] in ("EOD", "EXPIRED"):
        _tag(tags, why, "no_follow_through", "Target tak move aaya hi nahi; square-off/expiry par loss me band hua.")
    if t.get("ambiguous"):
        _tag(tags, why, "ambiguous_bar", "Ek hi candle me SL aur target dono touch hue; software ne conservative tareeke se "
             "SL pehle maana (asli result alag ho sakta hai).")
    if t["result"] == "MANUAL":
        _tag(tags, why, "manual_exit", "Aapne manual exit kiya - plan ke SL/target se pehle nikal gaye.")
    if t.get("after") == "reversed":
        _tag(tags, why, "stop_hunt", "SL hit hone ke BAAD price wapas target ki taraf gaya (stop-hunt). SL thoda door/structure ke peeche rakho.")
    if not why:
        why = ["Koi clear galti nahi mili - ye normal loss lagta hai (har strategy ke kuch trades haarte hain). "
               "SL ka rule follow hua, wahi sabse zaroori hai."]
        tags = ["normal_loss"]
    t["why"], t["tags"] = why, tags
    return t


def post_exit_check(t, df, max_bars=12):
    """After a loss: did price go to T1 soon after the exit? (stop-hunt check). Sets t['after'] once."""
    if t.get("status") != "CLOSED" or t.get("after") is not None or t.get("pnl", 0) >= 0 or not t.get("closed"):
        return False
    bars = _bars_after(df, to_ts(t["closed"]))
    if len(bars) < max_bars:
        return False
    s = sgn(t)
    hit = any((b[2] >= t["t1"]) if s > 0 else (b[3] <= t["t1"]) for b in bars[:max_bars])
    t["after"] = "reversed" if hit else "ok"
    postmortem(t)
    return True


# ---------------------------------------------------------------- stats + share text
def closed_trades(trades):
    return [t for t in trades if t.get("status") == "CLOSED" and t.get("pnl") is not None]


def stats(trades):
    cl = closed_trades(trades)
    if not cl:
        return dict(n=0)
    wins = [t for t in cl if t["pnl"] > 0]
    loss = [t for t in cl if t["pnl"] < 0]
    gw, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in loss)
    mistakes = Counter(tag for t in loss for tag in t.get("tags", []) if tag not in ("normal_loss", "breakeven"))
    return dict(n=len(cl), wins=len(wins), losses=len(loss), win_rate=len(wins) / len(cl) * 100,
                pnl=sum(t["pnl"] for t in cl), avg_r=sum(t["r_mult"] for t in cl) / len(cl),
                pf=(gw / gl) if gl else None, mistakes=mistakes.most_common())


TAG_NAMES = {
    "wrong_direction": "Direction/timing galat", "reversal_near_target": "Target ke paas se palta",
    "against_trend": "Trend ke khilaf", "low_volume": "Kam volume", "extreme_rsi": "RSI extreme par entry",
    "sideways": "Sideways market", "weak_signal": "Kamzor signal", "tight_sl": "SL bahut tight",
    "near_level": "Support/resistance ke paas entry", "late_entry": "Der se entry", "opening_noise": "Opening noise",
    "gap_slippage": "Gap/slippage", "no_follow_through": "Move aaya hi nahi", "ambiguous_bar": "Ek candle me SL+target",
    "manual_exit": "Manual exit", "stop_hunt": "Stop-hunt", "normal_loss": "Normal loss", "breakeven": "Break-even",
}


def summary_text(trades, notes="", limit=25):
    """Plain text you can copy and paste to Claude to improve the strategy."""
    st_ = stats(trades)
    out = ["STOCK APP - TRADE JOURNAL SUMMARY", f"Date: {now_ist():%d %b %Y %H:%M} IST"]
    if st_["n"] == 0:
        out.append("Abhi koi band trade nahi hai.")
    else:
        pf = f"{st_['pf']:.2f}" if st_["pf"] is not None else "n/a"
        out.append(f"Trades: {st_['n']} | Win {st_['wins']} | Loss {st_['losses']} | Win rate {st_['win_rate']:.0f}% | "
                   f"Net P&L {st_['pnl']:,.0f} | Avg R {st_['avg_r']:+.2f} | Profit factor {pf}")
        if st_["mistakes"]:
            out.append("Sabse common galtiyan: " + ", ".join(f"{TAG_NAMES.get(k, k)} x{v}" for k, v in st_["mistakes"]))
    bad = [t for t in closed_trades(trades) if t["pnl"] < 0][-limit:]
    for i, t in enumerate(bad, 1):
        ctx = t.get("ctx") or {}
        out.append(f"\n#{i} {t['symbol']} {t['side']} ({t['kind']}, {t['tf']}, {t['mode']}) {t['opened'][:16]}")
        out.append(f"  Entry {t['entry']} SL {t['sl0']} T1 {t['t1']} T2 {t['t2']} qty {t['qty']} | "
                   f"Exit {t['exit_price']} ({RES_TEXT.get(t['result'], t['result'])}) | P&L {t['pnl']:,.2f} ({t['r_mult']:+.2f}R)")
        out.append(f"  MFE {t['mfe']:.2f} MAE {t['mae']:.2f} bars {t['bars']} | score {ctx.get('score')} rsi {ctx.get('rsi')} "
                   f"adx {ctx.get('adx')} vol {ctx.get('vol_ratio')} trend {ctx.get('trend')}")
        for w in t.get("why", []):
            out.append(f"  - {w}")
        if t.get("note"):
            out.append(f"  Meri note: {t['note']}")
    if notes.strip():
        out.append("\nMERE NOTES:\n" + notes.strip())
    return "\n".join(out)
