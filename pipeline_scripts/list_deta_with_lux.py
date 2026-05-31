#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import pandas as pd

# "lux" 列名の候補（deta.csv 側で表記揺れがある可能性に備える）
LUX_ALIASES = [
    "lux", "lx", "illuminance", "illuminance_lux", "lux_value", "lux_lx"
]

def find_lux_col(columns):
    cols = list(columns)
    lower_map = {c.lower(): c for c in cols}
    for a in LUX_ALIASES:
        if a in lower_map:
            return lower_map[a]
    # 部分一致も少し許す（例: "hioki_lux" など）
    for c in cols:
        cl = c.lower()
        if "lux" in cl or "illuminance" in cl:
            return c
    return None

def analyze_one(csv_path, min_valid_ratio=0.8, min_valid_count=10):
    try:
        # まずヘッダだけ読む
        header = pd.read_csv(csv_path, nrows=0)
        lux_col = find_lux_col(header.columns)
        if lux_col is None:
            return None

        # lux列だけ読む（軽量）
        df = pd.read_csv(csv_path, usecols=[lux_col])
        s = pd.to_numeric(df[lux_col], errors="coerce")
        total = len(s)
        valid = s.notna().sum()

        if total == 0:
            return None

        valid_ratio = float(valid) / float(total)

        # “ちゃんと入っている”判定
        if valid < min_valid_count or valid_ratio < min_valid_ratio:
            return {
                "path": csv_path,
                "lux_col": lux_col,
                "total": total,
                "valid": int(valid),
                "valid_ratio": valid_ratio,
                "mean": None,
                "min": None,
                "max": None,
                "ok": False,
            }

        s2 = s.dropna()
        return {
            "path": csv_path,
            "lux_col": lux_col,
            "total": total,
            "valid": int(valid),
            "valid_ratio": valid_ratio,
            "mean": float(s2.mean()),
            "min": float(s2.min()),
            "max": float(s2.max()),
            "ok": True,
        }
    except Exception as e:
        return {"path": csv_path, "error": str(e), "ok": False}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.expanduser("~/challenge_ws/src/nav_cloning/data"))
    ap.add_argument("--min_valid_ratio", type=float, default=0.8)
    ap.add_argument("--min_valid_count", type=int, default=10)
    ap.add_argument("--show_ng", action="store_true", help="lux列はあるが条件を満たさない(NG)ものも表示")
    args = ap.parse_args()

    pattern = os.path.join(args.data_dir, "*", "dataset", "vel", "deta.csv")
    files = sorted(glob.glob(pattern))

    ok_list = []
    ng_list = []
    err_list = []

    for f in files:
        r = analyze_one(f, args.min_valid_ratio, args.min_valid_count)
        if r is None:
            continue
        if r.get("error"):
            err_list.append(r)
        elif r["ok"]:
            ok_list.append(r)
        else:
            ng_list.append(r)

    def time_from_path(p):
        # .../data/<TIME>/dataset/vel/deta.csv -> <TIME>
        parts = p.split(os.sep)
        try:
            idx = parts.index("data")
            return parts[idx+1]
        except Exception:
            return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))

    print("=== deta.csv に lux が“ちゃんと”入っている（OK） ===")
    for r in ok_list:
        t = time_from_path(r["path"])
        print(f"{t}\tcol={r['lux_col']}\tvalid={r['valid']}/{r['total']} ({r['valid_ratio']:.3f})"
              f"\tmean={r['mean']:.2f}\tmin={r['min']:.2f}\tmax={r['max']:.2f}")

    if args.show_ng:
        print("\n=== lux列はあるが条件未満（NG） ===")
        for r in ng_list:
            t = time_from_path(r["path"])
            print(f"{t}\tcol={r['lux_col']}\tvalid={r['valid']}/{r['total']} ({r['valid_ratio']:.3f})")

    if err_list:
        print("\n=== 読み込みエラー ===")
        for r in err_list:
            print(r["path"], "->", r["error"])

    print(f"\n[SUMMARY] deta.csv found={len(files)} / ok={len(ok_list)} / ng={len(ng_list)} / err={len(err_list)}")

if __name__ == "__main__":
    main()

