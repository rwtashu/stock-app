"""storage.py - saves watchlist / settings / trades / notes so they come back next time.
Order: secret GitHub Gist (permanent, does NOT redeploy the app) -> local file (lost on app restart).
"""
import copy
import json
import os
from datetime import datetime, timedelta, timezone

import requests

IST = timezone(timedelta(hours=5, minutes=30))
LOCAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.json")
API = "https://api.github.com"
MAX_TRADES = 300


def default_db():
    return {"version": 1, "watchlist": [], "settings": {}, "trades": [], "notes": "", "updated": ""}


def fill(d):
    """Make sure all keys exist with the right types."""
    base = default_db()
    if isinstance(d, dict):
        for k in base:
            if k in d and isinstance(d[k], type(base[k])):
                base[k] = d[k]
    return base


def _cmp_key(data):
    d = dict(data)
    d["updated"] = ""
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


GIST_FILE = "stock_app_data.json"
GIST_DESC = "stock-app-data (auto saved)"


class Store:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.token = cfg.get("token")
        self.gist_id = cfg.get("gist_id") or None
        self.error = ""
        self.ok = True  # False if Gist is configured but loading failed -> we refuse to overwrite
        self.last_key = None
        self.saved_at = None

    @property
    def mode(self):
        return "gist" if self.token else "file"

    def _h(self):
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"}

    def _find(self):
        for page in (1, 2, 3):
            r = requests.get(f"{API}/gists", headers=self._h(), params={"per_page": 100, "page": page}, timeout=15)
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code}: {r.text[:120]}")
            items = r.json()
            for g in items:
                if GIST_FILE in (g.get("files") or {}):
                    return g["id"]
            if len(items) < 100:
                break
        return None

    def load(self):
        if self.mode == "gist":
            try:
                if not self.gist_id:
                    self.gist_id = self._find()
                if self.gist_id:
                    r = requests.get(f"{API}/gists/{self.gist_id}", headers=self._h(), timeout=15)
                    if r.status_code != 200:
                        raise RuntimeError(f"{r.status_code}: {r.text[:120]}")
                    f = (r.json().get("files") or {}).get(GIST_FILE) or {}
                    raw = f.get("content")
                    if f.get("truncated") and f.get("raw_url"):
                        raw = requests.get(f["raw_url"], timeout=15).text
                    data = fill(json.loads(raw or "{}"))
                    self.last_key, self.error, self.ok = _cmp_key(data), "", True
                    return data
                self.last_key, self.error, self.ok = None, "", True  # first run: nothing saved yet
                return self._from_file() or default_db()
            except Exception as e:
                self.error = f"Gist se data load nahi hua: {str(e)[:140]}"
                self.ok = False
                return self._from_file() or default_db()
        data = self._from_file() or default_db()
        self.last_key = _cmp_key(data)
        return data

    def _from_file(self):
        try:
            if os.path.exists(LOCAL_FILE):
                with open(LOCAL_FILE, encoding="utf-8") as f:
                    return fill(json.load(f))
        except Exception:
            pass
        return None

    def save(self, data):
        data = fill(copy.deepcopy(data))
        data["trades"] = data["trades"][-MAX_TRADES:]
        key = _cmp_key(data)
        if key == self.last_key:
            return True
        data["updated"] = datetime.now(IST).isoformat(timespec="seconds")
        text = json.dumps(data, indent=1, ensure_ascii=False)
        try:
            with open(LOCAL_FILE, "w", encoding="utf-8") as f:
                f.write(text)
            local_ok = True
        except Exception:
            local_ok = False
        if self.mode != "gist":
            self.last_key, self.saved_at = key, datetime.now(IST)
            return local_ok
        if not self.ok:
            self.error = "Gist se purana data load nahi hua tha, isliye save roka gaya (purana data overwrite na ho)."
            return False
        try:
            body = {"files": {GIST_FILE: {"content": text}}}
            if self.gist_id:
                r = requests.patch(f"{API}/gists/{self.gist_id}", headers=self._h(), json=body, timeout=20)
            else:
                body.update(description=GIST_DESC, public=False)
                r = requests.post(f"{API}/gists", headers=self._h(), json=body, timeout=20)
            if r.status_code in (200, 201):
                self.gist_id = r.json().get("id", self.gist_id)
                self.last_key, self.saved_at, self.error = key, datetime.now(IST), ""
                return True
            self.error = f"Gist save fail ({r.status_code}): {r.text[:150]}"
        except Exception as e:
            self.error = f"Gist save fail: {str(e)[:120]}"
        return False
