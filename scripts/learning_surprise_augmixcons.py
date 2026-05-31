#!/usr/bin/env python3
import os, sys, random
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Make sure we can import augmix package under scripts/edit_dataset
NAV_DIR = Path(__file__).resolve().parents[1]  # .../nav_cloning
EDIT_DIR = NAV_DIR / "scripts" / "edit_dataset"
if str(EDIT_DIR) not in sys.path:
    sys.path.insert(0, str(EDIT_DIR))

try:
    from augmix.augment_and_mix import augment_and_mix, MEAN, STD
except Exception as e:
    raise SystemExit(
        "Failed to import AugMix. Expected: nav_cloning/scripts/edit_dataset/augmix/augment_and_mix.py "
        f"(error={e})"
    )

def denormalize_rgb01(x: np.ndarray) -> np.ndarray:
    # AugMix implementation typically returns normalized (x-MEAN)/STD.
    # Convert back to [0,1] RGB.
    x = x * STD + MEAN
    return np.clip(x, 0.0, 1.0)

class Net(nn.Module):
    # Must match scripts/learning_surprise.py
    def __init__(self, n_channel=3, n_out=1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)

        self.flatten = nn.Flatten()
        self.cnn_layer = nn.Sequential(
            self.conv1, self.relu,
            self.conv2, self.relu,
            self.conv3, self.relu,
            self.flatten,
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
    def __init__(self, img_dir: Path, csv_path: Path, views=("center","left","right")):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
        self.views = list(views)
        self.items = []
        for _, r in self.df.iterrows():
            ep = str(r["episode"])
            if ep.endswith(".0"):
                ep = ep[:-2]
            for v in self.views:
                p = img_dir / f"{ep}_{v}.npy"
                if p.is_file():
                    self.items.append((p, float(r[v])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        x = np.load(p, mmap_mode="r")  # HWC BGR float32 [0,1]
        x = np.asarray(x, dtype=np.float32)
        x = torch.from_numpy(x).permute(2,0,1).contiguous()  # CHW
        return x, torch.tensor([y], dtype=torch.float32), idx

def compute_weights(labels: np.ndarray, bin_n: int) -> np.ndarray:
    # Inverse-frequency weights over |label| bins (normalized to mean=1).
    if bin_n <= 1:
        return np.ones_like(labels, dtype=np.float32)
    ya = np.abs(labels.astype(np.float32))
    mx = float(np.max(ya)) + 1e-6
    edges = np.linspace(0.0, mx, bin_n+1)
    b = np.digitize(ya, edges[1:], right=False)  # 0..bin_n
    counts = np.bincount(b, minlength=bin_n+1).astype(np.float32)
    counts[counts == 0] = 1.0
    w = (len(labels) / counts[b]).astype(np.float32)
    w /= (np.mean(w) + 1e-8)
    return w

def augmix_two_views(batch_bgr: torch.Tensor, alpha: float):
    """
    batch_bgr: Bx3xHxW float, BGR [0,1]
    returns: (aug1_bgr, aug2_bgr) Bx3xHxW
    """
    b, _, _, _ = batch_bgr.shape
    a1 = torch.empty_like(batch_bgr)
    a2 = torch.empty_like(batch_bgr)

    batch_np = batch_bgr.permute(0,2,3,1).detach().cpu().numpy()  # B H W C (BGR)
    for i in range(b):
        bgr = batch_np[i]
        rgb = bgr[..., ::-1]  # to RGB [0,1]
        try:
            m1 = augment_and_mix(rgb.astype(np.float32), alpha=alpha)
        except TypeError:
            m1 = augment_and_mix(rgb.astype(np.float32))
        try:
            m2 = augment_and_mix(rgb.astype(np.float32), alpha=alpha)
        except TypeError:
            m2 = augment_and_mix(rgb.astype(np.float32))

        m1 = denormalize_rgb01(np.asarray(m1, dtype=np.float32))[..., ::-1]  # back to BGR
        m2 = denormalize_rgb01(np.asarray(m2, dtype=np.float32))[..., ::-1]

        a1[i] = torch.from_numpy(m1).permute(2,0,1)
        a2[i] = torch.from_numpy(m2).permute(2,0,1)

    return a1, a2

def main():
    cfg_path = NAV_DIR / "config" / "config.yaml"
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    pc = cfg.get("pc_user_name")
    ws = cfg.get("ws_name")
    time_id = cfg.get("time")
    if not (pc and ws and time_id):
        raise SystemExit("config.yaml must contain pc_user_name, ws_name, time")

    public_path = Path(f"/home/{pc}/{ws}/src/nav_cloning/data")
    img_dir = public_path / time_id / "dataset" / str(cfg.get("load_dataset_img","img"))
    vel_csv = public_path / time_id / "dataset" / str(cfg.get("load_dataset_vel","vel")) / "data.csv"

    batch_size = int(cfg.get("batch_size", 8))
    epochs = int(cfg.get("epoch", 100))
    lr = float(cfg.get("learning_rate", 1e-4))
    bin_n = int(cfg.get("bin", 10))
    save_model = str(cfg.get("save_model", "model_gpu.pt"))
    out_model_dir = public_path / time_id / "model" / str(cfg.get("epoch"))  # same convention

    lam_cons = float(os.environ.get("AUGCONS_LAMBDA", "0.5"))
    alpha = float(os.environ.get("AUGCONS_ALPHA", "1.0"))
    seed = int(os.environ.get("SEED", "0"))

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ds = ImgCsvDataset(img_dir, vel_csv)
    if len(ds) == 0:
        raise SystemExit(f"dataset empty: {img_dir} / {vel_csv}")

    labels = np.array([y for _, y in ds.items], dtype=np.float32)
    weights = torch.from_numpy(compute_weights(labels, bin_n)).float()

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction="none")
    huber = nn.SmoothL1Loss(reduction="mean")

    out_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[AugMix-Cons] time={time_id} epochs={epochs} bs={batch_size} lr={lr} bin={bin_n} lam_cons={lam_cons} alpha={alpha}")
    print(f"[DATA] n={len(ds)} img_dir={img_dir}")
    print(f"[SAVE] {out_model_dir / save_model}")

    for ep in range(1, epochs+1):
        model.train()
        loss_sum = 0.0
        n_sum = 0

        for xb, yb, idxb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            a1, a2 = augmix_two_views(xb, alpha=alpha)
            a1 = a1.to(device, non_blocking=True)
            a2 = a2.to(device, non_blocking=True)

            pred0 = model(xb)
            pred1 = model(a1)
            pred2 = model(a2)

            wi = weights[idxb].to(device, non_blocking=True).view(-1,1)

            sup0 = (mse(pred0, yb) * wi).mean()
            sup1 = (mse(pred1, yb) * wi).mean()
            sup2 = (mse(pred2, yb) * wi).mean()
            sup = (sup0 + sup1 + sup2) / 3.0

            cons = huber(pred0, pred1) + huber(pred0, pred2)
            loss = sup + lam_cons * cons

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * xb.size(0)
            n_sum += xb.size(0)

        avg = loss_sum / max(1, n_sum)
        if ep == 1 or ep % max(1, epochs//10) == 0 or ep == epochs:
            print(f"epoch {ep}/{epochs} loss={avg:.6f}")

    torch.save(model.state_dict(), out_model_dir / save_model)
    print("[DONE] saved:", out_model_dir / save_model)

if __name__ == "__main__":
    main()
