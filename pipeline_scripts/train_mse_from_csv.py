#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


VIEWS = ["center", "left", "right"]


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

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)

        self.cnn_layer = nn.Sequential(
            self.conv1, self.relu,
            self.conv2, self.relu,
            self.conv3, self.relu,
            self.flatten
        )
        self.fc_layer = nn.Sequential(
            self.fc4, self.relu,
            self.fc5
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = self.fc_layer(x)
        return x


class ImgCsvDataset(Dataset):
    def __init__(self, img_dir: str, csv_path: str, views=VIEWS):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
        self.views = views
        self.pairs = []
        for _, row in self.df.iterrows():
            ep = str(row["episode"]).split(".")[0]
            for v in self.views:
                self.pairs.append((ep, v, float(row[v])))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ep, v, angle = self.pairs[idx]
        p = os.path.join(self.img_dir, f"{ep}_{v}.npy")

        # mmapのwarning/不定動作を避けるためcopy
        arr = np.load(p, mmap_mode="r")
        arr = np.array(arr, copy=True)  # writable
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()  # CHW

        y = torch.tensor([angle], dtype=torch.float32)
        return t, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--eps", type=float, default=1e-2)          # default側に合わせる
    ap.add_argument("--weight_decay", type=float, default=5e-4) # default側に合わせる
    ap.add_argument("--save", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    ds = ImgCsvDataset(args.img_dir, args.csv)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(3, 1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), eps=args.eps, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    print(f"[TRAIN] samples={len(ds)} batch={args.batch} epochs={args.epochs}")
    for e in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.squeeze(1), y.squeeze(1))
            loss.backward()
            optimizer.step()
            running += loss.item()

        scheduler.step()
        if (e + 1) % 10 == 0 or e == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {e+1}/{args.epochs} loss={running/len(dl):.6f} lr={lr:.6g}")

    torch.save(model.state_dict(), args.save)
    print(f"[DONE] saved: {args.save}")


if __name__ == "__main__":
    main()

