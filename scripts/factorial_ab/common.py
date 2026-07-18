#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A案/B案 共通ユーティリティ。

データ規約 (既存パイプラインと同一):
  <data_root>/<time_id>/dataset/img/<episode>_<view>.npy  (48x64x3 float32 [0,1])
  <data_root>/<time_id>/dataset/vel/data.csv              (episode,center,left,right[,lux,...])
学習は center/left/right の3視点(各視点の列がラベル)、評価は center のみ。
"""

import hashlib
import json
import os
import zlib
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

VIEWS = ("center", "left", "right")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PRETRAINED_INPUT = 224

# encoder名 → (系統, 入力正規化)。系統は結果の解釈用メモ。
ENCODER_INFO = {
    "scratch":       "事前学習なし (net.py と同一CNN)",
    "resnet18":      "教師あり ImageNet / CNN",
    "vit_b_16":      "教師あり ImageNet / ViT",
    "dinov2_vits14": "自己教師 DINOv2 / ViT-S",
    "dinov2_vitb14": "自己教師 DINOv2 / ViT-B",
    "clip_vitb16":   "言語対比 CLIP(openai) / ViT-B",
}


class Sample(NamedTuple):
    npy_path: str
    label: float
    key: str      # "<time_id>/<episode>_<view>"
    variant: int  # 0 = clean, 1..K = 拡張コピー


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------

def ep_str(v) -> str:
    s = str(v)
    if s.replace(".", "", 1).isdigit():
        return s.split(".")[0]
    return s


def load_rows(data_root: str, time_id: str, vel_dir: str = "vel",
              first_n: Optional[int] = None) -> pd.DataFrame:
    csv_path = Path(data_root) / time_id / "dataset" / vel_dir / "data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))
    df = pd.read_csv(csv_path)
    missing = {"episode", "center", "left", "right"} - set(df.columns)
    if missing:
        raise ValueError("data.csv に必要な列がありません: %s (%s)" % (sorted(missing), csv_path))
    if first_n is not None and first_n > 0:
        df = df.head(first_n)
    return df


def build_samples(data_root: str, time_id: str, df: pd.DataFrame,
                  img_dir: str = "img", views: Sequence[str] = VIEWS,
                  aug_k: int = 0) -> Tuple[List[Sample], int]:
    """1行×視点×(clean+K拡張) のサンプル列を作る。返り値: (samples, missing数)"""
    base = Path(data_root) / time_id / "dataset" / img_dir
    samples: List[Sample] = []
    missing = 0
    for _, r in df.iterrows():
        ep = ep_str(r["episode"])
        for v in views:
            label = float(r[v])
            if not np.isfinite(label):
                continue
            p = base / ("%s_%s.npy" % (ep, v))
            if not p.is_file():
                missing += 1
                continue
            key = "%s/%s_%s" % (time_id, ep, v)
            for variant in range(aug_k + 1):
                samples.append(Sample(str(p), label, key, variant))
    return samples, missing


# ---------------------------------------------------------------------------
# 光度系拡張 (TrivialAugment 風・許可 op 制限つき、サンプル毎に決定的)
# ---------------------------------------------------------------------------

DEFAULT_OPS = ("Equalize", "Brightness", "AutoContrast")  # 屋外実験で有効だった3op


def _enhance(cls):
    def fn(img: Image.Image, mag: float, sign: int) -> Image.Image:
        return cls(img).enhance(1.0 + sign * 0.99 * mag)
    return fn


_OP_TABLE = {
    "Identity": lambda img, mag, sign: img,
    "Equalize": lambda img, mag, sign: ImageOps.equalize(img),
    "AutoContrast": lambda img, mag, sign: ImageOps.autocontrast(img),
    "Brightness": _enhance(ImageEnhance.Brightness),
    "Color": _enhance(ImageEnhance.Color),
    "Contrast": _enhance(ImageEnhance.Contrast),
    "Sharpness": _enhance(ImageEnhance.Sharpness),
    "Posterize": lambda img, mag, sign: ImageOps.posterize(img, max(1, 8 - int(mag * 6))),
    "Solarize": lambda img, mag, sign: ImageOps.solarize(img, int(255 * (1.0 - mag))),
}


class PhotoTAW:
    def __init__(self, allowed_ops: Sequence[str] = DEFAULT_OPS, num_magnitude_bins: int = 31):
        bad = [o for o in allowed_ops if o not in _OP_TABLE]
        if bad:
            raise ValueError("未知の op: %s (使用可能: %s)" % (bad, sorted(_OP_TABLE)))
        self.ops = list(allowed_ops)
        self.bins = int(num_magnitude_bins)

    def signature(self) -> str:
        return "taw(%s,b%d)" % ("+".join(self.ops), self.bins)

    def apply(self, img01: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        pil = Image.fromarray(np.clip(img01 * 255.0, 0, 255).astype(np.uint8))
        op = self.ops[rng.randint(len(self.ops))]
        mag = rng.randint(self.bins) / max(1, self.bins - 1)
        sign = 1 if rng.rand() < 0.5 else -1
        out = _OP_TABLE[op](pil, mag, sign)
        return np.asarray(out, dtype=np.float32) / 255.0


def sample_rng(base_seed: int, key: str, variant: int) -> np.random.RandomState:
    h = zlib.crc32(("%d|%s|%d" % (base_seed, key, variant)).encode("utf-8")) & 0xFFFFFFFF
    return np.random.RandomState(h)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PixelDataset(Dataset):
    """npy を遅延読み込みし、variant>0 なら決定的に拡張を当てる。出力 CHW float32。"""

    def __init__(self, samples: List[Sample], aug: Optional[PhotoTAW], base_seed: int):
        self.samples = samples
        self.aug = aug
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = np.load(s.npy_path).astype(np.float32)
        if s.variant > 0:
            if self.aug is None:
                raise RuntimeError("variant>0 なのに aug が None です")
            img = self.aug.apply(img, sample_rng(self.base_seed, s.key, s.variant))
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        y = torch.tensor([s.label], dtype=torch.float32)
        return x, y


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

class ScratchNet(nn.Module):
    """scripts/net.py の Net と同一構造 (48x64 入力前提)。"""

    def __init__(self, n_channel: int = 3, n_out: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)
        for m in (self.conv1, self.conv2, self.conv3, self.fc4, self.fc5):
            nn.init.kaiming_normal_(m.weight)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc4(x))
        return self.fc5(x)


class Head(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(hidden, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone(name: str, device: torch.device) -> Tuple[nn.Module, int]:
    """frozen 特徴抽出器を返す。(module, feature_dim)"""
    if name == "resnet18":
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        dim = 512
    elif name == "vit_b_16":
        import torchvision
        m = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity()
        dim = 768
    elif name.startswith("dinov2"):
        # torch.hub の facebookresearch/dinov2 は Python 3.10+ 前提のため timm 経由で読む
        import timm
        timm_name = {
            "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
            "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
        }.get(name)
        if timm_name is None:
            raise ValueError("未知の dinov2 encoder: %s" % name)
        try:
            m = timm.create_model(timm_name, pretrained=True, num_classes=0,
                                  img_size=PRETRAINED_INPUT)
        except Exception as e:
            raise RuntimeError(
                "dinov2 のロードに失敗しました (初回はネット接続が必要です): %s" % e)
        dim = int(m.num_features)
    elif name.startswith("clip_"):
        try:
            import open_clip
        except ImportError:
            raise RuntimeError(
                "open_clip が未インストールです: pip install open_clip_torch")
        arch = {"clip_vitb16": "ViT-B-16-quickgelu"}.get(name)
        if arch is None:
            raise ValueError("未知の CLIP encoder: %s" % name)
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained="openai")
        m = model.visual
        dim = int(getattr(m, "output_dim", 512))
    else:
        raise ValueError("未知の encoder: %s (使用可能: %s)" % (name, sorted(ENCODER_INFO)))
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, dim


def preprocess_for_backbone(batch: torch.Tensor, encoder: str = "") -> torch.Tensor:
    x = F.interpolate(batch, size=(PRETRAINED_INPUT, PRETRAINED_INPUT),
                      mode="bilinear", align_corners=False)
    mean_v, std_v = (CLIP_MEAN, CLIP_STD) if encoder.startswith("clip") \
        else (IMAGENET_MEAN, IMAGENET_STD)
    mean = torch.tensor(mean_v, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std_v, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# 特徴キャッシュ
# ---------------------------------------------------------------------------

def _feature_cache_path(cache_dir: str, encoder: str, samples: List[Sample],
                        aug_sig: str, base_seed: int) -> Path:
    h = hashlib.sha1()
    h.update(("%s|%s|%d" % (encoder, aug_sig, base_seed)).encode())
    for s in samples:
        h.update(("%s|%d|%.6f" % (s.key, s.variant, s.label)).encode())
    return Path(cache_dir) / ("feat_%s_%s.npz" % (encoder, h.hexdigest()[:16]))


@torch.no_grad()
def extract_features(samples: List[Sample], encoder: str, device: torch.device,
                     aug: Optional[PhotoTAW], base_seed: int,
                     cache_dir: Optional[str] = None, batch_size: int = 64,
                     num_workers: int = 2,
                     backbone: Optional[Tuple[nn.Module, int]] = None,
                     verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """frozen encoder で全サンプルの特徴を作る。(features [N,D], labels [N])"""
    aug_sig = aug.signature() if aug is not None else "none"
    cache = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache = _feature_cache_path(cache_dir, encoder, samples, aug_sig, base_seed)
        if cache.is_file():
            z = np.load(str(cache))
            return z["features"], z["labels"]

    if backbone is None:
        backbone = build_backbone(encoder, device)
    model, dim = backbone
    loader = DataLoader(PixelDataset(samples, aug, base_seed), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    feats = np.empty((len(samples), dim), dtype=np.float32)
    labels = np.empty(len(samples), dtype=np.float32)
    i = 0
    for xb, yb in loader:
        n = xb.shape[0]
        out = model(preprocess_for_backbone(xb.to(device), encoder))
        feats[i:i + n] = out.cpu().numpy()
        labels[i:i + n] = yb.view(-1).numpy()
        i += n
        if verbose and (i // batch_size) % 50 == 0:
            print("  [feat] %d/%d" % (i, len(samples)), flush=True)
    if cache is not None:
        np.savez_compressed(str(cache), features=feats, labels=labels)
    return feats, labels


# ---------------------------------------------------------------------------
# モデル保存 / 読み込み / 評価
# ---------------------------------------------------------------------------

def save_model(path: str, state_dict: dict, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_meta(model_path: str) -> dict:
    mp = model_path + ".meta.json"
    if os.path.isfile(mp):
        with open(mp) as f:
            return json.load(f)
    # meta 無し = 既存パイプラインの scratch Net (state_dict) とみなす
    return {"encoder": "scratch"}


def load_model_for_eval(model_path: str, device: torch.device,
                        backbone_cache: Optional[dict] = None):
    """(forward_fn(batch_pixels)->preds, meta) を返す。"""
    meta = load_meta(model_path)
    encoder = meta.get("encoder", "scratch")
    state = torch.load(model_path, map_location=device)
    if encoder == "scratch":
        net = ScratchNet().to(device)
        # 既存 net.py は conv1 等と cnn_layer.0 等の二重名で保存するため後者を落とす
        state = {k: v for k, v in state.items()
                 if not (k.startswith("cnn_layer.") or k.startswith("fc_layer."))}
        net.load_state_dict(state)
        net.eval()

        def forward(xb: torch.Tensor) -> torch.Tensor:
            return net(xb.to(device))
        return forward, meta

    if backbone_cache is not None and encoder in backbone_cache:
        bb, dim = backbone_cache[encoder]
    else:
        bb, dim = build_backbone(encoder, device)
        if backbone_cache is not None:
            backbone_cache[encoder] = (bb, dim)
    head = Head(dim).to(device)
    head.load_state_dict(state)
    head.eval()

    def forward(xb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = bb(preprocess_for_backbone(xb.to(device), encoder))
        return head(f)
    return forward, meta


@torch.no_grad()
def evaluate_model_on_time(model_path: str, data_root: str, time_id: str,
                           device: torch.device, view: str = "center",
                           first_n: Optional[int] = 6100, batch_size: int = 128,
                           num_workers: int = 2,
                           backbone_cache: Optional[dict] = None,
                           feature_cache_dir: Optional[str] = None) -> Dict[str, float]:
    df = load_rows(data_root, time_id, first_n=first_n)
    samples, missing = build_samples(data_root, time_id, df, views=(view,), aug_k=0)
    if not samples:
        raise RuntimeError("評価サンプルが0件: %s %s" % (time_id, view))
    meta = load_meta(model_path)
    encoder = meta.get("encoder", "scratch")
    if encoder != "scratch" and feature_cache_dir:
        # テスト特徴をキャッシュ: 同じテストセットを多数のモデルで評価するとき backbone を1回しか通さない
        if backbone_cache is not None and encoder in backbone_cache:
            bb = backbone_cache[encoder]
        else:
            bb = build_backbone(encoder, device)
            if backbone_cache is not None:
                backbone_cache[encoder] = bb
        feats, labels = extract_features(
            samples, encoder, device, None, 0, cache_dir=feature_cache_dir,
            batch_size=batch_size, num_workers=num_workers, backbone=bb,
            verbose=False)
        head = Head(bb[1]).to(device)
        head.load_state_dict(torch.load(model_path, map_location=device))
        head.eval()
        X = torch.from_numpy(feats)
        preds = []
        for i in range(0, X.shape[0], batch_size):
            preds.append(head(X[i:i + batch_size].to(device)).cpu().view(-1).numpy())
        e = np.concatenate(preds) - labels
    else:
        forward, _ = load_model_for_eval(model_path, device, backbone_cache)
        loader = DataLoader(PixelDataset(samples, None, 0), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
        errs: List[np.ndarray] = []
        for xb, yb in loader:
            pred = forward(xb).cpu().view(-1).numpy()
            errs.append(pred - yb.view(-1).numpy())
        e = np.concatenate(errs)
    ae = np.abs(e)
    return {
        "count": int(e.size),
        "missing_images": int(missing),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "median_abs_error": float(np.median(ae)),
        "p95_abs_error": float(np.percentile(ae, 95)),
        "bias_mean_error": float(e.mean()),
    }


# ---------------------------------------------------------------------------
# lux 統計 (B案)
# ---------------------------------------------------------------------------

def dataset_lux_stats(data_root: str, time_id: str, first_n: Optional[int] = None,
                      luma_samples: int = 200) -> Dict[str, float]:
    """CSVの lux 列があればそれを、無ければ center 画像の平均輝度(0-255)を代用。"""
    df = load_rows(data_root, time_id, first_n=first_n)
    out: Dict[str, float] = {"time_id": time_id, "rows": float(len(df))}
    lux = pd.to_numeric(df.get("lux"), errors="coerce").dropna() if "lux" in df.columns else pd.Series(dtype=float)
    if len(lux) > 0:
        out.update(mean_lux=float(lux.mean()), median_lux=float(lux.median()),
                   std_lux=float(lux.std()), lux_source="csv")
        return out
    img_dir = Path(data_root) / time_id / "dataset" / "img"
    eps = [ep_str(v) for v in df["episode"].tolist()]
    stride = max(1, len(eps) // luma_samples)
    vals = []
    for ep in eps[::stride]:
        p = img_dir / ("%s_center.npy" % ep)
        if p.is_file():
            a = np.load(str(p))
            vals.append(float((a * 255.0 if a.max() <= 1.5 else a).mean()))
    if not vals:
        out.update(mean_lux=float("nan"), median_lux=float("nan"),
                   std_lux=float("nan"), lux_source="none")
        return out
    v = np.asarray(vals)
    out.update(mean_lux=float(v.mean()), median_lux=float(np.median(v)),
               std_lux=float(v.std()), lux_source="image_luma")
    return out


def pick_device(arg: Optional[str] = None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
