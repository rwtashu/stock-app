"""feed.py - live price.
Groww (exact exchange price, needs the PAID Groww Trade API) if configured, otherwise Yahoo (free, can be delayed).
"""
import hashlib
import time
from datetime import datetime, timedelta, timezone

import requests
import yfinance as yf

IST = timezone(timedelta(hours=5, minutes=30))
INDEX_MAP = {"^NSEI": "NIFTY", "^NSEBANK": "BANKNIFTY"}
_cache = {}
_groww = {"client": None, "exp": 0.0, "err": "", "retry": 0.0}


def ist_now():
    return datetime.now(IST)


def groww_symbol(sym):
    """'RELIANCE.NS' -> ('NSE', 'RELIANCE'); None if Groww cannot be used for this symbol."""
    s = (sym or "").upper()
    if s in INDEX_MAP:
        return "NSE", INDEX_MAP[s]
    if s.endswith(".NS"):
        return "NSE", s[:-3]
    return None


def groww_configured(cfg):
    cfg = cfg or {}
    return bool(cfg.get("api_key") and cfg.get("secret"))


def groww_error():
    return _groww["err"]


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except Exception:
        return None


GROWW = "https://api.groww.in/v1"


def _groww_token(cfg):
    """Daily access token from API key + secret (REST, no extra package). Cached ~5 hours."""
    now = time.time()
    if _groww["client"] is not None and now < _groww["exp"]:
        return _groww["client"]
    if now < _groww["retry"]:
        return None
    try:
        ts = str(int(now))
        checksum = hashlib.sha256((cfg["secret"] + ts).encode()).hexdigest()
        r = requests.post(f"{GROWW}/token/api/access",
                          headers={"Authorization": f"Bearer {cfg['api_key']}", "Accept": "application/json",
                                   "Content-Type": "application/json", "X-API-VERSION": "1.0"},
                          json={"key_type": "approval", "checksum": checksum, "timestamp": ts}, timeout=10)
        j = r.json() if r.content else {}
        tok = j.get("token") or (j.get("payload") or {}).get("token")
        if r.status_code == 200 and tok:
            _groww.update(client=tok, exp=now + 5 * 3600, err="")
            return tok
        _groww["err"] = f"Groww login fail ({r.status_code}): {r.text[:140]}"
    except Exception as e:
        _groww["err"] = f"Groww login fail: {str(e)[:140]}"
    _groww["retry"] = now + 120
    return None


def _groww_quote(token, sym):
    m = groww_symbol(sym)
    if not m:
        return None
    exch, ts_ = m
    r = requests.get(f"{GROWW}/live-data/quote",
                     headers={"Authorization": f"Bearer {token}", "Accept": "application/json", "X-API-VERSION": "1.0"},
                     params={"exchange": exch, "segment": "CASH", "trading_symbol": ts_}, timeout=6)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:100]}")
    j = r.json()
    q = j.get("payload", j) if isinstance(j, dict) else {}
    price = _num(q.get("last_price"))
    if not price:
        return None
    chg, pct = _num(q.get("day_change")), _num(q.get("day_change_perc"))
    ohlc = q.get("ohlc") or {}
    mt = _num(q.get("last_trade_time"))
    mkt = datetime.fromtimestamp(mt / 1000 if mt > 1e11 else mt, IST) if mt else None
    return dict(price=price, prev_close=None if chg is None else price - chg, change=chg, pct=pct,
                high=_num(ohlc.get("high")), low=_num(ohlc.get("low")), open=_num(ohlc.get("open")),
                volume=_num(q.get("volume")), mkt_ts=mkt)


def _yahoo_quote(sym):
    """Yahoo chart endpoint: price + the exchange timestamp of that price (so the real delay is visible)."""
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                         params={"interval": "1m", "range": "1d"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        m = r.json()["chart"]["result"][0]["meta"]
        price = _num(m.get("regularMarketPrice"))
        if price:
            prev = _num(m.get("chartPreviousClose")) or _num(m.get("previousClose"))
            mt = _num(m.get("regularMarketTime"))
            return dict(price=price, prev_close=prev, change=None if prev is None else price - prev,
                        pct=None if not prev else (price / prev - 1) * 100, high=_num(m.get("regularMarketDayHigh")),
                        low=_num(m.get("regularMarketDayLow")), open=None, volume=_num(m.get("regularMarketVolume")),
                        mkt_ts=datetime.fromtimestamp(mt, IST) if mt else None)
    except Exception:
        pass
    fi = yf.Ticker(sym).fast_info

    def g(*keys):
        for k in keys:
            try:
                v = _num(fi[k])
                if v is not None:
                    return v
            except Exception:
                pass
        return None

    price = g("last_price", "lastPrice")
    if not price:
        return None
    prev = g("previous_close", "previousClose", "regularMarketPreviousClose")
    return dict(price=price, prev_close=prev, change=None if prev is None else price - prev,
                pct=None if not prev else (price / prev - 1) * 100, high=g("day_high", "dayHigh"),
                low=g("day_low", "dayLow"), open=g("open"), volume=g("last_volume", "lastVolume"), mkt_ts=None)


def get_quote(sym, cfg=None, ttl=2.0):
    """Latest price dict: price, prev_close, change, pct, high, low, volume, ts, source ('Groww'/'Yahoo'), exact, note."""
    use_groww = groww_configured(cfg) and groww_symbol(sym) is not None
    key = (sym, use_groww)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    out, note = None, ""
    if use_groww:
        client = _groww_token(cfg)
        if client is not None:
            try:
                out = _groww_quote(client, sym)
                if out:
                    out.update(source="Groww", exact=True)
            except Exception as e:
                note = f"Groww quote fail: {str(e)[:120]}"
                _groww["exp"] = 0.0  # login again next time
        else:
            note = _groww["err"]
    if out is None:
        try:
            out = _yahoo_quote(sym)
        except Exception as e:
            out = None
            note = note or f"Yahoo fail: {str(e)[:100]}"
        if out:
            out.update(source="Yahoo", exact=False)
    if out:
        out["ts"] = ist_now()
        out["note"] = note
        _cache[key] = (time.time(), out)
        return out
    if hit:  # keep showing the last known price rather than nothing
        old = dict(hit[1])
        old["note"] = (note + " | " if note else "") + "naya price nahi aaya, pichhla dikh raha hai"
        return old
    return None
