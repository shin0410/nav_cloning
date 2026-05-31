#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

VIEWS = ["center","left","right"]

class Net(nn.Module):
    def __init__(self, n_channel=3, n_out=1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.cnn_layer = nn.Sequential(self.conv1, self.relu, self.conv2, self.relu, self.conv3, self.relu, self.flatten)
        self.fc_layer  = nn.Sequential(self.fc4, self.relu, self.fc5)

    def forward(self, x):
        return self.fc_layer(self.cnn_layer(x))

def load_img(img_path: str) -> torch.Tensor:
    arr = np.load(img_path, mmap_mode="r")  # HWC float32 0..1
    t = torch.from_numpy(arr).permute(2,0,1).contiguous()  # CHW
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--csv_in", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pred_col", required=True)
    ap.add_argument("--csv_out", required=True)
    ap.add_argument("--view", default="center", choices=VIEWS)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_in)
    if "episode" not in df.columns:
        raise ValueError("csv_inにepisode列がありません")
    # episode文字列化（"123.0"→"123"）
    ep = df["episode"].astype(str).str.replace(r"\.0$", "", regex=True).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    sd = torch.load(args.model, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    preds = []
    missing = 0

    with torch.no_grad():
        buf_x = []
        buf_idx = []

        def flush():
            nonlocal preds
            if not buf_x:
                return
            x = torch.stack(buf_x, dim=0).to(device, dtype=torch.float32)
            y = model(x).squeeze(1).detach().cpu().numpy().tolist()
            for i, p in zip(buf_idx, y):
                preds[i] = float(p)
            buf_x.clear()
            buf_idx.clear()

        preds = [float("nan")] * len(ep)

        for i, e in enumerate(ep):
            img_path = os.path.join(args.img_dir, f"{e}_{args.view}.npy")
            if not os.path.exists(img_path):
                missing += 1
                continue
            buf_x.append(load_img(img_path))
            buf_idx.append(i)
            if len(buf_x) >= args.batch:
                flush()
        flush()

    df[args.pred_col] = preds
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    df.to_csv(args.csv_out, index=False)
    print(f"[DONE] infer: {args.csv_out}  missing_images={missing}/{len(ep)}")

if __name__ == "__main__":
    main()

