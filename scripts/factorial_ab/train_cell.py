#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A案: 因子計画の1セル {encoder × aug × rows × seed} を学習する。

例:
  python train_cell.py \
    --data_root ~/challenge_ws/nav_cloning_data \
    --train_times 20260308_100554 20260308_120428 \
    --rows 2000 --encoder dinov2_vits14 --aug taw3op --aug-k 3 --seed 1 \
    --out_model /path/to/models/dinov2__taw3op__r2000__s1.pt

encoder:
  scratch        既存 net.py と同一CNNを画素から学習 (48x64)
  resnet18       ImageNet 事前学習 ResNet18 を frozen、MLPヘッドのみ学習
  dinov2_vits14  DINOv2 ViT-S/14 を frozen、MLPヘッドのみ学習 (初回はDL必要)
aug:
  none | taw3op (Equalize,Brightness,AutoContrast の TrivialAugment 風、K個の拡張コピーを追加)
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (DEFAULT_OPS, Head, PhotoTAW, PixelDataset, ScratchNet,
                    build_backbone, build_samples, extract_features,
                    load_rows, pick_device, save_model, set_all_seeds)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--train_times", nargs="+", required=True)
    ap.add_argument("--rows", type=int, default=0, help="各 train_time から先頭 N 行 (0=全部)")
    ap.add_argument("--encoder", default="scratch",
                    choices=["scratch", "resnet18", "vit_b_16",
                             "dinov2_vits14", "dinov2_vitb14", "clip_vitb16"])
    ap.add_argument("--aug", default="none", choices=["none", "taw3op"])
    ap.add_argument("--aug-k", type=int, default=3, dest="aug_k")
    ap.add_argument("--allowed-ops", default=",".join(DEFAULT_OPS), dest="allowed_ops")
    ap.add_argument("--magnitude-bins", type=int, default=31, dest="magnitude_bins")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--feature_cache", default=None, help="frozen特徴のキャッシュdir")
    ap.add_argument("--out_model", required=True)
    return ap.parse_args()


def build_train_sample_groups(args, aug_k: int):
    """train_time ごとに (全行サンプル列, セルで使う先頭サンプル数) を返す。

    サンプル列は行順に生成されるので、先頭 rows 行ぶんは全行リストの接頭辞になる。
    これにより特徴キャッシュを rows 条件間で共有できる。
    """
    groups = []
    total_missing = 0
    for t in args.train_times:
        df_full = load_rows(args.data_root, t)
        s_full, miss = build_samples(args.data_root, t, df_full, aug_k=aug_k)
        total_missing += miss
        if args.rows and args.rows > 0:
            s_sub, _ = build_samples(args.data_root, t, df_full.head(args.rows),
                                     aug_k=aug_k)
            n_sub = len(s_sub)
        else:
            n_sub = len(s_full)
        groups.append((s_full, n_sub))
    if total_missing:
        print("[WARN] 画像欠損 %d 件をスキップ" % total_missing)
    if not any(n for _, n in groups):
        raise SystemExit("[ERR] 学習サンプルが0件です")
    return groups


def train_scratch(args, groups, aug, device):
    samples = []
    for s_full, n_sub in groups:
        samples.extend(s_full[:n_sub])
    set_all_seeds(args.seed)
    net = ScratchNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.MSELoss()
    loader = DataLoader(PixelDataset(samples, aug, args.seed), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers, drop_last=False)
    log = []
    for ep in range(args.epochs):
        net.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * xb.shape[0]
            n += xb.shape[0]
        log.append((ep, tot / max(1, n)))
        if ep % 10 == 0 or ep == args.epochs - 1:
            print("[scratch] epoch %d loss %.6f" % (ep, tot / max(1, n)), flush=True)
    return net.state_dict(), log


def train_head_on_features(args, groups, aug, device):
    set_all_seeds(args.seed)
    # 拡張なしなら特徴はシードに依存しないので seed=0 でキャッシュを共有する
    feat_seed = args.seed if aug is not None else 0
    feat_list, label_list = [], []
    backbone = None
    for s_full, n_sub in groups:
        if backbone is None:
            backbone = build_backbone(args.encoder, device)
        f, l = extract_features(
            s_full, args.encoder, device, aug, feat_seed,
            cache_dir=args.feature_cache, batch_size=args.batch_size,
            num_workers=args.num_workers, backbone=backbone)
        feat_list.append(f[:n_sub])
        label_list.append(l[:n_sub])
    X = torch.from_numpy(np.concatenate(feat_list))
    Y = torch.from_numpy(np.concatenate(label_list)).view(-1, 1)
    dim = X.shape[1]
    head = Head(dim).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=max(args.lr, 1e-3),
                           weight_decay=args.weight_decay)
    crit = nn.MSELoss()
    n = X.shape[0]
    log = []
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = X[idx].to(device), Y[idx].to(device)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * xb.shape[0]
        log.append((ep, tot / n))
        if ep % 20 == 0 or ep == args.epochs - 1:
            print("[head:%s] epoch %d loss %.6f" % (args.encoder, ep, tot / n), flush=True)
    return head.state_dict(), log


def main():
    args = parse_args()
    device = pick_device(args.device)
    print("[INFO] device=%s encoder=%s aug=%s rows=%s seed=%d" %
          (device, args.encoder, args.aug, args.rows or "all", args.seed))

    aug = None
    aug_k = 0
    if args.aug == "taw3op":
        aug = PhotoTAW(args.allowed_ops.split(","), args.magnitude_bins)
        aug_k = args.aug_k

    groups = build_train_sample_groups(args, aug_k)
    n_used = sum(n for _, n in groups)
    print("[INFO] train samples: %d (clean+aug)" % n_used)

    t0 = time.time()
    if args.encoder == "scratch":
        state, log = train_scratch(args, groups, aug, device)
    else:
        state, log = train_head_on_features(args, groups, aug, device)
    dt = time.time() - t0

    meta = {
        "encoder": args.encoder,
        "aug": args.aug,
        "aug_k": aug_k,
        "allowed_ops": args.allowed_ops if aug else "",
        "rows": args.rows,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_times": args.train_times,
        "n_train_samples": n_used,
        "final_loss": log[-1][1] if log else None,
        "train_seconds": round(dt, 1),
    }
    save_model(args.out_model, state, meta)
    loss_csv = args.out_model + ".loss.csv"
    with open(loss_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss"])
        w.writerows(log)
    print("[DONE] model: %s (%.1fs, final_loss=%.6f)" % (args.out_model, dt, log[-1][1]))


if __name__ == "__main__":
    main()
