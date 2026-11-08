# -*- coding: utf-8 -*-
"""
ส่วนคำนวณของเว็บสแกนหุ้น (ใช้ร่วมกับ scan.py บน GitHub Actions) — EMA 12/26/50/100/200/800 + RSI 14 + RSI-based MA
ข้อมูลฟรีจาก Yahoo Finance (ผ่าน yfinance)

วิธีรัน:  python app.py   แล้วเปิด http://localhost:5000
"""
import json
import os
import pickle
import threading
import time
import traceback
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

EMA_LENS = [12, 26, 50, 100, 200, 800]
HISTORY_PERIOD = "5y"      # EMA 800 ต้องใช้ข้อมูลอย่างน้อย ~3.2 ปี
BATCH = 80                 # จำนวนหุ้นต่อการดึงข้อมูล 1 ครั้ง
UA = {"User-Agent": "Mozilla/5.0 (stock-scanner; personal use)"}

# ---------------------------------------------------------------------------
# รายชื่อหุ้น (universe)
# ---------------------------------------------------------------------------
THAI_LARGE = """ADVANC AOT AWC BANPU BBL BCP BDMS BEM BGRIM BH BJC BTS CBG CCET CENTEL
COM7 CPALL CPF CPN CRC DELTA EGCO GLOBAL GPSC GULF HMPRO IVL KBANK KKP KTB KTC LH
MINT MTC OR OSP PTT PTTEP PTTGC RATCH SCB SCC SCGP TISCO TLI TOP TRUE TTB WHA ITC""".split()

THEME_LIST = """CEG VST GEV TLN SMR NVDA AVGO TSM ASML MRVL MSFT AMZN GOOGL IBM LLY NVO ISRG
CRWD PANW ZS TSLA SYM MP IONQ QNT RGTI QBTS QUBT RKLB ASTS""".split()


def _wiki_tables(page):
    url = f"https://en.wikipedia.org/wiki/{page}"
    html = requests.get(url, headers=UA, timeout=30).text
    return pd.read_html(StringIO(html))


def _find_col(tables, names, min_rows=20):
    """หาตารางที่มีคอลัมน์ชื่อหุ้น (รองรับหัวตาราง 2 ชั้นและชื่อคอลัมน์ที่เปลี่ยนไป)"""
    flat = []
    for t in tables:
        if len(t) < min_rows:
            continue
        t = t.copy()
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" ".join(str(x) for x in c if "Unnamed" not in str(x)).strip()
                         for c in t.columns]
        t.columns = [str(c).strip() for c in t.columns]
        flat.append(t)
    for n in names:                      # ชื่อตรงตัวก่อน
        for t in flat:
            for c in t.columns:
                if c.lower() == n.lower():
                    return t, c
    for key in ("ticker", "symbol", "code", "epic"):   # แล้วค่อยหาแบบมีคำนี้อยู่
        for t in flat:
            for c in t.columns:
                if key in c.lower():
                    return t, c
    raise ValueError(f"ไม่พบคอลัมน์ {names}")


def _sp500():
    t, c = _find_col(_wiki_tables("List_of_S%26P_500_companies"), ["Symbol"])
    return [str(s).replace(".", "-") for s in t[c].dropna()]


def _nasdaq100():
    # รายชื่อย้ายไปอยู่หน้า "List of NASDAQ-100 companies" แล้ว (ลองหน้าเดิมเป็นสำรอง)
    last = None
    for page in ("List_of_NASDAQ-100_companies", "Nasdaq-100"):
        try:
            t, c = _find_col(_wiki_tables(page), ["Ticker", "Symbol"])
            return [str(s).replace(".", "-") for s in t[c].dropna()]
        except Exception as e:
            last = e
    raise last


def _dax():
    t, c = _find_col(_wiki_tables("DAX"), ["Ticker", "Ticker symbol"])
    out = []
    for s in t[c].dropna():
        s = str(s).strip()
        out.append(s if "." in s else s + ".DE")
    return out


def _ftse100():
    t, c = _find_col(_wiki_tables("FTSE_100_Index"), ["Ticker", "EPIC"])
    out = []
    for s in t[c].dropna():
        s = str(s).strip().rstrip(".")
        out.append(s.replace(".", "-") + ".L")
    return out


def _hangseng():
    t, c = _find_col(_wiki_tables("Hang_Seng_Index"), ["Ticker", "Stock code", "Code"])
    out = []
    for s in t[c].dropna():
        digits = "".join(ch for ch in str(s) if ch.isdigit())
        if digits:
            out.append(digits[-4:].zfill(4) + ".HK")
    return out


def _csi300():
    """หุ้นจีนแผ่นดินใหญ่ CSI 300 (ตลาดเซี่ยงไฮ้ .SS / เซินเจิ้น .SZ)"""
    t, c = _find_col(_wiki_tables("CSI_300_Index"), ["Ticker"])
    out = []
    for s in t[c].dropna():
        s = str(s).upper()
        code = "".join(ch for ch in s if ch.isdigit())
        if len(code) != 6:
            continue
        if "SZ" in s:
            out.append(code + ".SZ")
        elif "SS" in s or "SH" in s:
            out.append(code + ".SS")
        else:   # เดาจากรหัส: 6xxxxx = เซี่ยงไฮ้, 0/3xxxxx = เซินเจิ้น
            out.append(code + (".SS" if code.startswith("6") else ".SZ"))
    return out


DR_FILE = os.path.join(BASE, "thai_dr_list.json")


def load_dr():
    """รายชื่อ DR ในตลาดหุ้นไทย -> หุ้นแม่ใน Yahoo (ไฟล์ thai_dr_list.json ดึงจาก set.or.th)"""
    try:
        with open(DR_FILE, encoding="utf-8") as f:
            return json.load(f)["items"]
    except Exception:
        return []


def _thai_dr():
    items = load_dr()
    if not items:
        raise ValueError("ไม่พบไฟล์ thai_dr_list.json")
    return [x["y"] for x in items]


UNIVERSES = {
    "theme":    ("หุ้นตามธีมที่คุยกัน (~30)", lambda: THEME_LIST),
    "thai":     ("หุ้นไทยตัวใหญ่ (~50)", lambda: [s + ".BK" for s in THAI_LARGE]),
    "nasdaq100": ("Nasdaq-100 (สหรัฐฯ)", _nasdaq100),
    "sp500":    ("S&P 500 (สหรัฐฯ)", _sp500),
    "hangseng": ("Hang Seng (ฮ่องกง)", _hangseng),
    "csi300":   ("หุ้นจีน CSI 300 (เซี่ยงไฮ้/เซินเจิ้น)", _csi300),
    "thai_dr":  ("หุ้นแม่ของ DR ในตลาดไทย (~360)", _thai_dr),
    "dax":      ("DAX 40 (เยอรมนี)", _dax),
    "ftse100":  ("FTSE 100 (อังกฤษ)", _ftse100),
}


def get_universe(key):
    """คืนรายชื่อหุ้น เก็บแคชไว้ 7 วัน"""
    path = os.path.join(CACHE_DIR, f"universe_{key}.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 7 * 86400:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    tickers = list(dict.fromkeys(UNIVERSES[key][1]()))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tickers, f)
    return tickers


# ---------------------------------------------------------------------------
# ข้อมูลราคา
# ---------------------------------------------------------------------------
def _cache_path(t):
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in t)
    return os.path.join(CACHE_DIR, f"px_{safe}.pkl")


def _fresh(path, hours=6):
    return os.path.exists(path) and time.time() - os.path.getmtime(path) < hours * 3600


def _clean(df):
    if df is None or len(df) == 0:
        return None
    df = df.rename(columns=str.title)
    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        return None
    df = df[need].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df if len(df) else None


def download_prices(tickers, progress=None):
    """คืน dict ticker -> DataFrame(OHLCV). ใช้แคชถ้าดึงมาไม่เกิน 6 ชม."""
    import yfinance as yf

    out, todo = {}, []
    for t in tickers:
        p = _cache_path(t)
        if _fresh(p):
            try:
                with open(p, "rb") as f:
                    out[t] = pickle.load(f)
                continue
            except Exception:
                pass
        todo.append(t)

    done = len(tickers) - len(todo)
    if progress:
        progress(done, len(tickers))

    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        try:
            raw = yf.download(chunk, period=HISTORY_PERIOD, interval="1d",
                              group_by="ticker", auto_adjust=False,
                              threads=True, progress=False)
        except Exception:
            raw = None
        for t in chunk:
            df = None
            try:
                if raw is not None and len(raw):
                    if isinstance(raw.columns, pd.MultiIndex):
                        if t in raw.columns.get_level_values(0):
                            df = raw[t]
                        elif t in raw.columns.get_level_values(1):
                            df = raw.xs(t, axis=1, level=1)
                    else:
                        df = raw
                df = _clean(df)
            except Exception:
                df = None
            if df is not None:
                out[t] = df
                with open(_cache_path(t), "wb") as f:
                    pickle.dump(df, f)
        done += len(chunk)
        if progress:
            progress(done, len(tickers))
    return out


# ---------------------------------------------------------------------------
# อินดิเคเตอร์ (สูตรแบบเดียวกับ TradingView)
# ---------------------------------------------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close, n=14):
    d = close.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma(-d.clip(upper=0), n)
    rs = up / dn
    out = 100 - 100 / (1 + rs)
    return out.mask((dn == 0) & up.notna(), 100.0)


def ma(s, n, kind):
    kind = (kind or "SMA").upper()
    if kind == "EMA":
        return s.ewm(span=n, adjust=False, min_periods=n).mean()
    if kind == "RMA":
        return rma(s, n)
    if kind == "WMA":
        w = np.arange(1, n + 1)
        return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    return s.rolling(n).mean()


def indicators(df, rsi_len=14, ma_len=14, ma_kind="SMA"):
    c = df["Close"]
    res = {f"ema{n}": ema(c, n) for n in EMA_LENS}
    res["rsi"] = rsi(c, rsi_len)
    res["rsima"] = ma(res["rsi"], ma_len, ma_kind)
    return res


LOOKBACK_52W = 252   # จำนวนวันทำการใน 52 สัปดาห์


def lows_highs(df, lb=LOOKBACK_52W):
    """New low / New high แบบ 52 สัปดาห์
    New low  = วันที่ Low ต่ำกว่า Low ต่ำสุดของ 252 วันก่อนหน้า
    New high = วันที่ High สูงกว่า High สูงสุดของ 252 วันก่อนหน้า
    คืน (วันที่ New low ล่าสุด, วันที่ New high ล่าสุดที่เกิดหลัง New low นั้น)"""
    lo, hi = df["Low"], df["High"]
    is_nl = lo < lo.shift(1).rolling(lb, min_periods=lb).min()
    is_nh = hi > hi.shift(1).rolling(lb, min_periods=lb).max()
    nl_dates = df.index[is_nl.fillna(False).values]
    nl = nl_dates[-1] if len(nl_dates) else None
    nh_mask = is_nh.fillna(False)
    if nl is not None:
        nh_mask = nh_mask & (df.index > nl)
    nh_dates = df.index[nh_mask.values]
    nh = nh_dates[-1] if len(nh_dates) else None
    return nl, nh


def _f(x, nd=2):
    try:
        x = float(x)
        return None if np.isnan(x) else round(x, nd)
    except Exception:
        return None


