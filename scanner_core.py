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


SET_DR_API = "https://www.set.or.th/api/set/dr/search?tradeDateType={t}&lang=en"
DR_FIX = {"DCVFMVN30 ETF": "E1VFVN30.VN", "DCVFMVN DIAMOND ETF": "FUEVFVND.VN"}   # ไม่มีรหัสในวงเล็บ


def _dr_suffix(ex):
    rules = [("Hong Kong", "HK"), ("Shanghai", "SS"), ("Shenzhen", "SZ"), ("Tokyo", "T"),
             ("Singapore", "SI"), ("Taiwan", "TW"), ("Paris", "PA"), ("Amsterdam", "AS"),
             ("Milan", "MI"), ("Deutsche", "DE"), ("Copenhagen", "CO"), ("Hochiminh", "VN")]
    for key, sfx in rules:
        if key in (ex or ""):
            return sfx
    return "US"


def dr_items_from_set(data):
    """แปลงผลจาก API ของ set.or.th เป็นรายการ หุ้นแม่ (ชื่อย่อ Yahoo) -> DR ในไทย"""
    import re
    groups = {}
    for d in data:
        u = d.get("underlying")
        if not u:
            continue
        g = groups.setdefault(u, {"ex": d.get("underlyingExchange", ""), "name": d.get("underlyingName") or u,
                                  "cls": d.get("underlyingClassName") or "", "drs": []})
        g["drs"].append(d["symbol"])
    items = []
    for u, g in groups.items():
        m = re.search(r"\(([^()]+)\)\s*$", g["name"])
        code = m.group(1).strip() if m else u
        sfx = _dr_suffix(g["ex"])
        if u in DR_FIX:
            y = DR_FIX[u]
        elif sfx == "US":
            y = code.replace(".", "-")
        elif sfx == "HK":
            y = "".join(ch for ch in code.replace(".HK", "") if ch.isdigit()).zfill(4) + ".HK"
        elif not m:
            print(f"DR: ข้าม {u} (ไม่รู้รหัสหุ้นแม่)")
            continue
        else:
            y = code.replace(" ", "") + "." + sfx
        items.append({"y": y, "u": u, "name": re.sub(r"\s*\([^()]*\)\s*$", "", g["name"]).strip(),
                      "ex": sfx, "etf": "Collective" in g["cls"], "drs": sorted(g["drs"])})
    return sorted(items, key=lambda x: x["u"])


DR_QUOTES = {}   # ราคา DR ล่าสุดจาก set.or.th: {"PETROCN80": {"px":..,"chg":..,"val":..,"iss":..,"ratio":..}}


def refresh_dr_list():
    """ดึงรายชื่อ DR ล่าสุดจาก set.or.th แล้วเขียนทับ thai_dr_list.json
    คืน (จำนวน DR ใหม่, รายชื่อ DR ใหม่, รายชื่อ DR ที่หายไป) หรือ None ถ้าดึงไม่ได้ (ใช้ไฟล์เดิมต่อ)"""
    from datetime import date
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                       "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
                         "Accept": "application/json, text/plain, */*",
                         "Referer": "https://www.set.or.th/en/market/product/dr/marketdata"})
    try:
        sess.get("https://www.set.or.th/en/market/product/dr/marketdata", timeout=30)
        got = {}
        for t in ("C", "P"):          # C = วันทำการปัจจุบัน/ถัดไป, P = วันทำการก่อนหน้า (มีราคาปิดแน่นอน)
            try:
                r = sess.get(SET_DR_API.format(t=t), timeout=30)
                r.raise_for_status()
                got[t] = r.json().get("data") or []
            except Exception as e:
                print(f"DR: ดึงชุด {t} ไม่ได้ ({e})")
                got[t] = []
        # รายชื่อ: รวมทั้งสองชุด (DR ใหม่จะโผล่ในชุด C ก่อน)
        merged = {d["symbol"]: d for d in got["P"]}
        merged.update({d["symbol"]: d for d in got["C"] if d.get("underlying")})
        data = list(merged.values())
    except Exception as e:
        print(f"DR: ดึงจาก set.or.th ไม่ได้ ({e}) — ใช้รายชื่อเดิม")
        return None
    if len(data) < 100:               # กันกรณีได้หน้าเปล่า/ถูกบล็อก
        print(f"DR: ได้ข้อมูลแค่ {len(data)} รายการ — ใช้รายชื่อเดิม")
        return None
    items = dr_items_from_set(data)
    cur = {d["symbol"]: d for d in got["C"]}
    prev = {d["symbol"]: d for d in got["P"]}
    for d in data:
        c = cur.get(d["symbol"]) or {}
        if c.get("last"):              # มีการซื้อขายแล้ววันนี้ (เช่น รอบกลางคืน) ใช้ราคาล่าสุด
            d = {**d, **{k: c.get(k) for k in ("last", "percentChange", "totalValue")}}
        elif d["symbol"] in prev:
            d = prev[d["symbol"]]
        px = d.get("last") or d.get("prior")
        DR_QUOTES[d["symbol"]] = {
            "px": px, "chg": round(d["percentChange"], 2) if d.get("percentChange") is not None else None,
            "val": round((d.get("totalValue") or 0) / 1e6, 2),          # ล้านบาท
            "iss": d.get("issuer"), "ratio": d.get("conversionRatio"),
        }
    old = {d for x in load_dr() for d in x["drs"]}
    new = {d for x in items for d in x["drs"]}
    with open(DR_FILE, "w", encoding="utf-8") as f:
        json.dump({"source": SET_DR_API.format(t="C"), "fetched": date.today().isoformat(), "items": items},
                  f, ensure_ascii=False)
    added, removed = sorted(new - old), sorted(old - new)
    print(f"DR: อัปเดตแล้ว {len(new)} ตัว (ใหม่ {len(added)}, หายไป {len(removed)})")
    return len(added), added, removed


# ---------------------------------------------------------------------------
# หุ้นจีนที่จดทะเบียน 2 ตลาด (A-share เซี่ยงไฮ้/เซินเจิ้น  <->  H-share ฮ่องกง)
# ใช้ประกบคู่ เช่น 601857.SS (PetroChina A) กับ 0857.HK (H) ซึ่งมี DR ไทย PETROCN80
# ---------------------------------------------------------------------------
AH_FILE = os.path.join(BASE, "ah_pairs.json")
AH_MANUAL = {   # H -> A  (หุ้นที่มี DR ไทย ตรวจไว้เอง ใช้ก่อนข้อมูลจาก Wikidata)
    "0857.HK": "601857.SS", "1398.HK": "601398.SS", "2318.HK": "601318.SS", "2899.HK": "601899.SS",
    "6690.HK": "600690.SS", "1211.HK": "002594.SZ", "3750.HK": "300750.SZ", "3968.HK": "600036.SS",
    "0300.HK": "000333.SZ", "2238.HK": "601238.SS", "1772.HK": "002460.SZ", "6680.HK": "300748.SZ",
    "0941.HK": "600941.SS", "0981.HK": "688981.SS", "1347.HK": "688347.SS", "2359.HK": "603259.SS",
    "6869.HK": "601869.SS", "3986.HK": "603986.SS", "6809.HK": "688008.SS", "2475.HK": "002475.SZ",
    "2476.HK": "300476.SZ",
}
WIKIDATA_AH = """SELECT ?hk ?cn ?cnEx WHERE {
  ?item p:P414 ?s1 . ?s1 ps:P414 wd:Q496672 ; pq:P249 ?hk .
  ?item p:P414 ?s2 . ?s2 ps:P414 ?cnEx ; pq:P249 ?cn . VALUES ?cnEx { wd:Q739514 wd:Q1132622 }
  FILTER NOT EXISTS { ?s1 pq:P582 ?e1 } FILTER NOT EXISTS { ?s2 pq:P582 ?e2 } }"""


def load_ah_pairs():
    """คืน dict H-share -> A-share  (ดึงจาก Wikidata ทุกครั้ง ถ้าไม่ได้ใช้ไฟล์เดิม แล้วทับด้วยรายการที่ตรวจเอง)"""
    pairs = {}
    try:
        r = requests.get("https://query.wikidata.org/sparql", params={"query": WIKIDATA_AH, "format": "json"},
                         headers={"User-Agent": "stock-scanner/1.0 (personal research; github.com/ukritprop31-max)",
                                  "Accept": "application/sparql-results+json"}, timeout=60)
        r.raise_for_status()
        for b in r.json()["results"]["bindings"]:
            hk = "".join(ch for ch in b["hk"]["value"] if ch.isdigit())
            cn = "".join(ch for ch in b["cn"]["value"] if ch.isdigit())
            if not hk or len(cn) != 6 or hk.startswith("46"):      # 46xx = หุ้นบุริมสิทธิ์
                continue
            sfx = ".SS" if b["cnEx"]["value"].endswith("Q739514") else ".SZ"
            pairs[hk.zfill(4) + ".HK"] = cn + sfx
        if len(pairs) > 30:
            with open(AH_FILE, "w", encoding="utf-8") as f:
                json.dump(pairs, f, ensure_ascii=False, indent=0, sort_keys=True)
        print(f"A/H: Wikidata {len(pairs)} คู่")
    except Exception as e:
        print(f"A/H: ดึงจาก Wikidata ไม่ได้ ({e}) — ใช้ไฟล์เดิม")
        try:
            with open(AH_FILE, encoding="utf-8") as f:
                pairs = json.load(f)
        except Exception:
            pairs = {}
    manual_a = set(AH_MANUAL.values())
    pairs = {h: a for h, a in pairs.items() if a not in manual_a}   # รายการที่ตรวจเองชนะ
    pairs.update(AH_MANUAL)
    return pairs


def _thai_dr():
    items = load_dr()
    if not items:
        raise ValueError("ไม่พบไฟล์ thai_dr_list.json")
    return [x["y"] for x in items]


NAMES = {}   # ชื่อบริษัท (เติมระหว่างโหลดรายชื่อ) ใช้แสดงในหน้าเว็บ

SET100_FALLBACK = """AAV ADVANC AEONTS AMATA AOT AP AURA AWC BA BAM BANPU BBL BCH BCP BCPG BDMS BEM BGRIM BH BJC
BLA BTG BTS CBG CCET CENTEL CHG CK COM7 CPALL CPF CPN CRC DELTA DOHOME EA EGCO ERW GFPT GLOBAL GPSC GST
GULF GUNKUL HANA HMPRO ICHI IRPC IVL JMART JMT KBANK KCE KKP KTB KTC LH M MEGA MINT MOSHI MRDIYT MTC OR
OSP PLANB PR9 PRM PTG PTT PTTEP PTTGC QH RATCH RCL SAWAD SCB SCC SCGP SIRI SPALI SPRC STA STECON STGT
TASCO TCAP TFG THAI TIDLOR TISCO TLI TOA TOP TRUE TTB TU VGI WHA WHAUP""".split()   # ณ 10 พ.ย. 2026


def _set100():
    """หุ้นใน SET100 ดึงจาก set.or.th ทุกครั้ง (ถ้าไม่ได้ใช้รายชื่อสำรองด้านบน)"""
    try:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                           "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
                             "Accept": "application/json, text/plain, */*",
                             "Referer": "https://www.set.or.th/en/market/index/set100/overview"})
        sess.get("https://www.set.or.th/en/market/index/set100/overview", timeout=30)
        r = sess.get("https://www.set.or.th/api/set/index/set100/composition?lang=en", timeout=30)
        r.raise_for_status()
        infos = r.json()["composition"]["stockInfos"]
        if len(infos) < 90:
            raise ValueError(f"ได้แค่ {len(infos)} ตัว")
        out = []
        for x in infos:
            t = x["symbol"].replace(" ", "") + ".BK"
            out.append(t)
            if x.get("nameEN"):
                NAMES[t] = x["nameEN"].replace(" PUBLIC COMPANY LIMITED", "").title()
        print(f"SET100: ดึงจาก set.or.th {len(out)} ตัว")
        return out
    except Exception as e:
        print(f"SET100: ดึงจาก set.or.th ไม่ได้ ({e}) — ใช้รายชื่อสำรอง")
        return [s + ".BK" for s in SET100_FALLBACK]


TH_INDEXES = {"SET": "ดัชนี SET", "SET50": "ดัชนี SET50", "SET100": "ดัชนี SET100",
              "SETHD": "ดัชนี SETHD (ปันผลสูง)", "sSET": "ดัชนี sSET (หุ้นขนาดเล็ก)", "MAI": "ดัชนี mai"}
IDX_SFX = ".INDEX"     # ชื่อในระบบ เช่น SET.INDEX (Yahoo ไม่มีประวัติดัชนีไทย จึงดึงจาก set.or.th)

NAMES.update({k + IDX_SFX: v for k, v in TH_INDEXES.items()})


def _th_index():
    for k, v in TH_INDEXES.items():
        NAMES[k + IDX_SFX] = v
    return [k + IDX_SFX for k in TH_INDEXES]


def _set_session(referer):
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                       "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
                         "Accept": "application/json, text/plain, */*", "Referer": referer})
    sess.get(referer, timeout=30)
    return sess


def fetch_th_indexes(tickers):
    """ประวัติดัชนีไทยรายวัน 5 ปี จาก set.or.th (มีแต่ราคาปิด) + ค่าวันนี้แบบ OHLC จากหน้าข้อมูลดัชนี"""
    out = {}
    try:
        sess = _set_session("https://www.set.or.th/en/market/index/set/overview")
        live = {}
        try:
            j = sess.get("https://www.set.or.th/api/set/index/info/list?type=INDEX", timeout=30).json()
            if isinstance(j, dict):     # บางครั้งห่อมาเป็น {"indexIndustrySectors": [...]} 
                j = j.get("indexIndustrySectors") or next((v for v in j.values() if isinstance(v, list)), [])
            for x in j:
                if isinstance(x, dict) and x.get("symbol"):
                    live[x["symbol"].upper()] = x
        except Exception as e:
            print(f"ดัชนีไทย: ไม่ได้ค่าวันนี้ ({e})")
        for t in tickers:
            sym = t[:-len(IDX_SFX)]
            try:
                r = sess.get(f"https://www.set.or.th/api/set/index/{sym}/chart-quotation?period=5Y&accumulated=false",
                             timeout=30)
                r.raise_for_status()
                q = r.json().get("quotations") or []
                rows = [(pd.Timestamp(x["localDatetime"][:10]), x["price"], x.get("volume") or 0)
                        for x in q if x.get("price")]
                if len(rows) < 100:
                    raise ValueError(f"ข้อมูลน้อย ({len(rows)})")
                df = pd.DataFrame(rows, columns=["Date", "Close", "Volume"]).drop_duplicates("Date").set_index("Date")
                df["Open"] = df["High"] = df["Low"] = df["Close"]
                lv = live.get(sym.upper())
                if lv and lv.get("last"):
                    d = pd.Timestamp(lv["marketDateTime"][:10])
                    df.loc[d] = {"Close": lv["last"], "Volume": lv.get("volume") or 0, "Open": lv.get("open") or lv["last"],
                                 "High": lv.get("high") or lv["last"], "Low": lv.get("low") or lv["last"]}
                out[t] = df.sort_index()[["Open", "High", "Low", "Close", "Volume"]]
            except Exception as e:
                print(f"ดัชนีไทย {sym}: ดึงไม่ได้ ({e})")
    except Exception as e:
        print(f"ดัชนีไทย: เชื่อม set.or.th ไม่ได้ ({e})")
    return out


UNIVERSES = {
    "th_index": ("ดัชนีตลาดหุ้นไทย (SET, SET50, SET100, SETHD, sSET, mai)", _th_index),
    "set100":   ("SET100 (หุ้นไทย 100 ตัว)", _set100),
    "theme":    ("หุ้นตามธีมที่คุยกัน (~30)", lambda: THEME_LIST),
    "thai":     ("หุ้นไทยตัวใหญ่ (~50 ชุดเดิม)", lambda: [s + ".BK" for s in THAI_LARGE]),
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
    idx = [t for t in tickers if t.endswith(IDX_SFX)]
    if idx:
        out.update(fetch_th_indexes(idx))
        tickers = [t for t in tickers if not t.endswith(IDX_SFX)]
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


