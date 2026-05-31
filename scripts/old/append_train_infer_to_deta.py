#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
append_train_infer_to_deta.py  (test.py と同じ config 読み込み規約)
- __file__/../config/config.yaml を既定で読み込む
- config の pc_user_name, ws_name, time, epoch, load_model から
  dataset_dir と weights の既定パスを自動生成
- 必要なら --dataset_dir / --weights で上書き可能
- 画像は <dataset_dir>/<img_dir>/<episode>_<cam>.npy
- CSVは <dataset_dir>/<vel_dir>/data.csv を読み、<vel_dir>/deta.csv に pred_<tag> を追記
"""

import os
import argparse
import numpy as np
import pandas as pd

# ====== test.py と同じ流儀で config を読む ======
def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config", filename)
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def default_dataset_dir(cfg):
    """ /home/<pc_user_name>/<ws_name>/src/nav_cloning/data/<time>/dataset """
    user = cfg.get("pc_user_name")
    ws   = cfg.get("ws_name")
    tm   = cfg.get("time")
    if not (user and ws and tm):
        return None
    return os.path.join("/home", user, ws, "src", "nav_cloning", "data", str(tm), "dataset")

def default_weights_path(cfg):
    """ /home/<pc_user_name>/<ws_name>/src/nav_cloning/data/<time>/model/<epoch>/<load_model> """
    user = cfg.get("pc_user_name")
    ws   = cfg.get("ws_name")
    tm   = cfg.get("time")
    ep   = cfg.get("epoch")
    lm   = cfg.get("load_model")
    if not (user and ws and tm and (ep is not None) and lm):
        return None
    return os.path.join("/home", user, ws, "src", "nav_cloning", "data",
                        str(tm), "model", str(ep), str(lm))

def _resize48x64(img):
    H, W = 48, 64
    try:
        import cv2
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    except Exception:
        try:
            from skimage.transform import resize
            return resize(img, (H, W), preserve_range=True).astype(np.float32)
        except Exception:
            y_idx = (np.linspace(0, img.shape[0]-1, H)).astype(int)
            x_idx = (np.linspace(0, img.shape[1]-1, W)).astype(int)
            return img[y_idx][:, x_idx].astype(np.float32)

def _load_model(weight_path):
    # test.py と同じ deep_learning を想定（.load() を使う）
    from net import deep_learning
    dl = deep_learning(n_action=1)
    dl.load(weight_path)  # ← test.py と同じ呼び方
    return dl

def main():
    cfg = load_config()  # ← test.py と同じ相対パス解決
    # 画像/速度ディレクトリ名（configにあれば使う／無ければ既定）
    img_dir = cfg.get("load_dataset_img", "img")
    vel_dir = cfg.get("load_dataset_vel", "vel")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=None,
                        help="dataset 直下（img/ と vel/ を含む）。未指定なら config から推定")
    parser.add_argument("--weights", default=None,
                        help="学習済み .pth 。未指定なら config から推定")
    parser.add_argument("--tag", required=True,
                        help="追記する列名の接尾辞（pred_<tag>）例: m12_base")
    parser.add_argument("--cam", default="center", choices=["center", "left", "right"])
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or default_dataset_dir(cfg)
    if not dataset_dir or not os.path.isdir(dataset_dir):
        raise SystemExit(f"[ERR] dataset_dir が見つかりません: {dataset_dir}\n"
                         f"例) /home/<user>/<ws_name>/src/nav_cloning/data/<time>/dataset")

    weights_path = args.weights or default_weights_path(cfg)
    if not weights_path or not os.path.exists(weights_path):
        raise SystemExit(f"[ERR] weights が見つかりません: {weights_path}\n"
                         f"config の epoch/load_model を見直すか --weights で指定してください")

    # 入出力CSV
    in_csv  = os.path.join(dataset_dir, vel_dir, "data.csv")
    out_csv = os.path.join(dataset_dir, vel_dir, "deta.csv")
    if not os.path.exists(in_csv):
        raise SystemExit(f"[ERR] 入力CSVが見つかりません: {in_csv}")

    df_in = pd.read_csv(in_csv)
    if "episode" not in df_in.columns:
        raise SystemExit("[ERR] CSV に 'episode' 列が必要です")

    print("=== data.csv preview ===")
    print(df_in.head(10).to_string(index=False))

    # 推論器ロード（test.py と同じ deep_learning.load() を使用）
    dl = _load_model(weights_path)

    pred_col = f"pred_{args.tag}"
    preds = []
    for ep in df_in["episode"]:
        npy = os.path.join(dataset_dir, img_dir, f"{ep}_{args.cam}.npy")
        if not os.path.exists(npy):
            preds.append(np.nan); continue
        img = np.load(npy).astype(np.float32)
        if img.ndim != 3 or img.shape[2] not in (1, 3):
            preds.append(np.nan); continue
        if img.shape[:2] != (48, 64):
            img = _resize48x64(img)
        wz = float(dl.act(img))
        preds.append(wz)

    # deta.csv を作成/更新（episode でマージして安全に上書き）
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if os.path.exists(out_csv):
        base = pd.read_csv(out_csv)
        if "episode" in base.columns:
            add = pd.DataFrame({"episode": df_in["episode"], pred_col: preds})
            out = pd.merge(base.drop(columns=[pred_col], errors="ignore"),
                           add, on="episode", how="outer")
            out.sort_values(by="episode", inplace=True, ignore_index=True)
        else:
            # 行対応しかできないCSVの場合は列を追加（長さチェック）
            if len(base) == len(df_in):
                base[pred_col] = preds
                out = base
            else:
                out = df_in.copy()
                out[pred_col] = preds
    else:
        out = df_in.copy()
        out[pred_col] = preds

    out.to_csv(out_csv, index=False)
    print(f"[done] 追記完了: {out_csv}  列名: {pred_col}")
    print(f"[info] dataset_dir={dataset_dir}")
    print(f"[info] weights    ={weights_path}")

if __name__ == "__main__":
    main()

