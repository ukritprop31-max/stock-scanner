# -*- coding: utf-8 -*-
"""
สแกนหุ้นทุกตลาด แล้วเขียนผลเป็นไฟล์ JSON ให้หน้าเว็บ (site/) อ่าน
รันอัตโนมัติบน GitHub Actions วันละ 2 รอบ (ดู .github/workflows/scan.yml)

หน้าเว็บจะกรองตามเงื่อนไขในเบราว์เซอร์เอง ไฟล์นี้จึงเก็บค่าของทุกหุ้นไว้
"""
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np

import scanner_core as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site", "data")
CHART_MIN_RSIMA = 55      # เก็บไฟล์กราฟเฉพาะหุ้นที่ RSI-MA ถึงระดับนี้ (+ หุ้นตามธีม/หุ้นไทยทุกตัว)
CHART_BARS = 600          # จำนวนแท่งที่เก็บไว้แสดงในกราฟ (~2.4 ปี)
STREAK_TH = 70
R30_LEVEL = 30


def months(today, d):
    return round((today - d).days / 30.44, 1) if d is not None else None


def main():
    os.makedirs(OUT, exist_ok=True)
    cdir = os.path.join(OUT, "c")
    shutil.rmtree(cdir, ignore_errors=True)
    os.makedirs(cdir)

    markets, errors = {}, []
    for key, (label, _) in C.UNIVERSES.items():
        try:
            for t in C.get_universe(key):
                markets.setdefault(t, []).append(key)
            print(f"{label}: ok")
        except Exception as e:
            errors.append(f"{label}: โหลดรายชื่อไม่ได้ ({e})")
            print(errors[-1])
    tickers = list(markets)
    print(f"รวม {len(tickers)} ตัว — กำลังดึงราคา…")

    data = C.download_prices(tickers, lambda d, t: print(f"  {d}/{t}", flush=True))
    always_chart = {"theme", "thai", "thai_dr"}
    dr_map = {x["y"]: x for x in C.load_dr()}     # หุ้นแม่ -> รายชื่อ DR ในไทย
    rows = []
    for t, df in data.items():
        if len(df) < 60:
            continue
        try:
            ind = C.indicators(df)
            rsima = C._f(ind["rsima"].iloc[-1])
            if rsima is None:
                continue
            close = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            today = df.index[-1]
            e = {n: C._f(ind[f"ema{n}"].iloc[-1], 4) for n in C.EMA_LENS}
            ribbon = all(e[a] is not None and e[b] is not None and e[a] > e[b]
                         for a, b in zip(C.EMA_LENS[:5], C.EMA_LENS[1:5]))
            nl, nh = C.lows_highs(df)
            t30 = df.index[(ind["rsima"] <= R30_LEVEL).fillna(False).values]
            r30 = t30[-1] if len(t30) else None
            streak = 0
            for v in ind["rsima"].iloc[::-1]:
                if not np.isnan(v) and v > STREAK_TH:
                    streak += 1
                else:
                    break
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in t)
            has_chart = rsima >= CHART_MIN_RSIMA or any(m in always_chart for m in markets[t])
            rows.append({
                "t": t, "m": markets[t], "c": C._f(close, 4), "chg": C._f((close / prev - 1) * 100),
                "rsi": C._f(ind["rsi"].iloc[-1]), "rsima": rsima, "streak": streak,
                "d800": C._f((close / e[800] - 1) * 100) if e[800] else None,
                "d200": C._f((close / e[200] - 1) * 100) if e[200] else None,
                "ribbon": ribbon, "a800": bool(e[800] and close > e[800]),
                "val": C._f(float(df["Volume"].tail(20).mean() or 0) * close / 1e6, 1),
                "nl_m": months(today, nl), "nh_d": (today - nh).days if nh is not None else None,
                "r30_m": months(today, r30), "date": today.strftime("%Y-%m-%d"),
                "f": safe if has_chart else None,
            })
            if t in dr_map:
                rows[-1]["dr"] = dr_map[t]["drs"]
                rows[-1]["nm"] = dr_map[t]["name"].title()
            if has_chart:
                write_chart(os.path.join(cdir, safe + ".json"), t, df, ind, nl, nh, r30)
        except Exception as ex:
            print(f"ข้าม {t}: {ex}")

    missing = len(tickers) - len(data)
    if missing:
        errors.append(f"ดึงข้อมูลไม่ได้ {missing} ตัว (ชื่อย่อผิดหรือเลิกซื้อขาย)")
    out = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "markets": {k: v[0] for k, v in C.UNIVERSES.items()},
        "count": len(rows), "errors": errors, "rows": rows,
    }
    with open(os.path.join(OUT, "scan.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"เสร็จ: {len(rows)} ตัว, กราฟ {sum(1 for r in rows if r['f'])} ไฟล์")
    if not rows:
        sys.exit("ไม่มีข้อมูลเลย — Yahoo อาจปิดกั้นชั่วคราว")


def write_chart(path, t, df, ind, nl, nh, r30):
    tail = df.tail(CHART_BARS)
    idx = tail.index
    times = [d.strftime("%Y-%m-%d") for d in idx]

    def ser(s, nd):
        s = s.reindex(idx)
        return [None if np.isnan(v) else round(float(v), nd) for v in s]

    obj = {
        "t": t, "time": times,
        "o": ser(tail["Open"], 4), "h": ser(tail["High"], 4), "l": ser(tail["Low"], 4),
        "c": ser(tail["Close"], 4), "v": [int(v) if not np.isnan(v) else 0 for v in tail["Volume"]],
        "ema": {str(n): ser(ind[f"ema{n}"], 4) for n in C.EMA_LENS},
        "rsi": ser(ind["rsi"], 2), "rsima": ser(ind["rsima"], 2),
        "nl": nl.strftime("%Y-%m-%d") if nl is not None and nl >= idx[0] else None,
        "nh": nh.strftime("%Y-%m-%d") if nh is not None and nh >= idx[0] else None,
        "r30": r30.strftime("%Y-%m-%d") if r30 is not None and r30 >= idx[0] else None,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
