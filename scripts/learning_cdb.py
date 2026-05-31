import os
import sys
import argparse
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

# ============================================================
# 目的:
# - offline_collect で保存した npy + vel/data.csv を使ってオフライン学習
# - オンライン(change_dataset_balance)の「distance に応じて学習を濃くする」
#   (distance<0.05: 1回, 0.05-0.1: 2回, >=0.1: 3回 相当) をオフライン側でも再現
# - |center| < 0.1 のときだけ left/right も学習（オンラインの条件に合わせる）
# ============================================================

DEFAULT_BATCH_SIZE = 8

class Net(nn.Module):
    def __init__(self, n_channel: int, n_out: int):
        super().__init__()
        # <Network CNN 3 + FC 2>
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)  # 48x64 入力を想定
        self.fc5 = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)

        # <Weight init>
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
            self.flatten
        )
        self.fc_layer = nn.Sequential(
            self.fc4, self.relu,
            self.fc5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layer(x)
        x = self.fc_layer(x)
        return x


def load_config(filename: str = "config.yaml") -> dict:
    """../config/config.yaml を読む（元の learning_*.py と同じ配置想定）"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config", filename)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_public_path(config: dict, data_root_override: Optional[str], speed: Optional[str]) -> str:
    """
    data ルートを決める。
    互換のため、以下を順に探索:
      1) --data_root
      2) /home/<user>/ws/<ws>/src/nav_cloning/data
      3) /home/<user>/<ws>/src/nav_cloning/data
      4) このスクリプト相対 ../data
    """
    if data_root_override:
        p = os.path.abspath(os.path.expanduser(data_root_override))
        if os.path.isdir(p):
            return p
        raise FileNotFoundError(f"--data_root が存在しません: {p}")

    pc_user_name = str(config.get("pc_user_name", "")).strip()
    ws_name = str(config.get("ws_name", "")).strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_data = os.path.abspath(os.path.join(script_dir, "..", "data"))

    candidates: List[str] = []
    if pc_user_name and ws_name:
        candidates.append(f"/home/{pc_user_name}/ws/{ws_name}/src/nav_cloning/data")
        candidates.append(f"/home/{pc_user_name}/{ws_name}/src/nav_cloning/data")
    candidates.append(local_data)

    if speed:
        for c in candidates:
            if os.path.isdir(os.path.join(c, speed)):
                return c

    for c in candidates:
        if os.path.isdir(c):
            return c

    raise FileNotFoundError(
        "data ルートが見つかりません。--data_root で明示してください。\n"
        f"  tried: {candidates}"
    )


def resolve_mode(
    public_path: str,
    speed: str,
    mode_from_config: Optional[str],
    mode_override: Optional[str],
    allow_create: bool = False,
) -> str:
    """
    mode を解決する。
    優先順:
      1) --mode
      2) config['mode']
      3) <public_path>/<speed> 配下で dataset を持つ mode が 1 つなら自動選択
    """
    mode = (mode_override if mode_override else mode_from_config)
    if mode is not None:
        mode = str(mode).strip()
    if mode:
        mode_dir = os.path.join(public_path, speed, mode)
        if not os.path.isdir(mode_dir):
            if allow_create:
                os.makedirs(mode_dir, exist_ok=True)
                print(f"[INFO] created mode directory: {mode_dir}")
            else:
                raise FileNotFoundError(f"mode ディレクトリが見つかりません: {mode_dir}")
        return mode

    speed_dir = os.path.join(public_path, speed)
    if not os.path.isdir(speed_dir):
        raise FileNotFoundError(f"speed ディレクトリが見つかりません: {speed_dir}")

    modes = []
    for d in sorted(os.listdir(speed_dir)):
        mode_dir = os.path.join(speed_dir, d)
        if not os.path.isdir(mode_dir):
            continue
        if os.path.isdir(os.path.join(mode_dir, "dataset")):
            modes.append(d)

    if len(modes) == 1:
        print(f"[INFO] mode is not set. Auto-selected mode='{modes[0]}'")
        return modes[0]

    if not modes:
        raise ValueError(
            f"mode 候補が見つかりませんでした: speed_dir={speed_dir}\n"
            "config に mode を追加するか、--mode を指定してください。"
        )

    raise ValueError(
        "mode が特定できません（候補が複数あります）。\n"
        f"  speed_dir={speed_dir}\n"
        f"  candidates={modes}\n"
        "config に mode を追加するか、--mode を指定してください。"
    )


def resolve_flat_dataset_root(data_root: str, try_num: str) -> Optional[str]:
    """
    旧/別環境の flat 配置:
      <data_root>/<try_num>/dataset/{img,vel}
    """
    candidate = os.path.join(data_root, try_num, "dataset")
    if os.path.isfile(os.path.join(candidate, "vel", "data.csv")):
        return candidate
    return None


@dataclass(frozen=True)
class Sample:
    npy_path: str
    target: float
    distance: float  # NaN あり得る


class OfflineCDBDataset(Dataset):
    """
    offline_collect の出力:
      <dataset_root>/img/{episode}_{center|left|right}.npy
      <dataset_root>/vel/data.csv

    オンライン(change_dataset_balance)の挙動をオフラインに反映:
      - distance < 0.05: サンプルを 1 回ぶん
      - 0.05 <= distance < 0.1: 2 回ぶん
      - distance >= 0.1: 3 回ぶん
    ※ オンラインでは make_dataset の回数 + その場学習(act_and_trains) を足した感じ。
       オフラインでは「同じサンプルを複製した dataset を作って shuffle 学習」すると等価に近い。
    """
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        label_source: str = "expert",     # center の教師値に使う列: expert/center/executed/policy
        use_lr: bool = True,
        lr_gate_abs: float = 0.1,         # |center| < lr_gate_abs のときだけ left/right も学習
        w_th1: float = 0.05,
        w_th2: float = 0.10,
        w_small: int = 1,
        w_mid: int = 2,
        w_large: int = 3,
        require_phase_training: bool = True,
    ):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"data.csv が見つかりません: {csv_path}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"img ディレクトリが見つかりません: {img_dir}")

        df = pd.read_csv(csv_path)

        # phase 列があるなら training のみに絞る（テストデータ混入防止）
        if require_phase_training and "phase" in df.columns:
            df = df[df["phase"] == "training"].reset_index(drop=True)

        # 必要列チェック
        required_cols = {"episode", "distance", "center", "left", "right"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"data.csv に必要列がありません: missing={sorted(list(missing))}")

        # center の教師値列（expert/center/executed/policy）
        if label_source not in df.columns:
            # expert が無い場合は center を使う等のフォールバック
            if label_source != "center" and "center" in df.columns:
                print(f"[WARN] label_source='{label_source}' が無いので 'center' を使います。")
                label_source = "center"
            else:
                raise ValueError(f"label_source='{label_source}' 列が data.csv に存在しません。")

        base_samples: List[Sample] = []
        n_center = 0
        n_lr = 0

        for _, r in df.iterrows():
            ep = int(r["episode"])
            dist = float(r["distance"]) if not pd.isna(r["distance"]) else float("nan")

            # center 教師値
            center_label = float(r[label_source])
            center_path = os.path.join(img_dir, f"{ep}_center.npy")
            base_samples.append(Sample(center_path, center_label, dist))
            n_center += 1

            # left/right 追加条件（オンラインに合わせる）
            if use_lr and abs(float(r["center"])) < lr_gate_abs:
                left_label = float(r["left"])
                right_label = float(r["right"])
                left_path = os.path.join(img_dir, f"{ep}_left.npy")
                right_path = os.path.join(img_dir, f"{ep}_right.npy")
                base_samples.append(Sample(left_path, left_label, dist))
                base_samples.append(Sample(right_path, right_label, dist))
                n_lr += 2

        # distance に応じた「複製インデックス」を作る（shuffle=True で学習できる）
        expanded_indices: List[int] = []
        for i, s in enumerate(base_samples):
            d = s.distance
            if np.isnan(d):
                w = w_small
            elif d < w_th1:
                w = w_small
            elif d < w_th2:
                w = w_mid
            else:
                w = w_large
            expanded_indices.extend([i] * int(w))

        self.base_samples = base_samples
        self.expanded_indices = expanded_indices

        print(f"[INFO] Loaded offline dataset:")
        print(f"  csv: {csv_path}")
        print(f"  img: {img_dir}")
        print(f"  base samples  : {len(base_samples)} (center={n_center}, left/right={n_lr})")
        print(f"  expanded len  : {len(expanded_indices)} (avg multiplier={len(expanded_indices)/max(1,len(base_samples)):.2f}x)")
        print(f"  label_source  : {label_source}")
        print(f"  lr_gate_abs   : {lr_gate_abs}")

    def __len__(self) -> int:
        return len(self.expanded_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = self.expanded_indices[idx]
        s = self.base_samples[base_idx]

        if not os.path.isfile(s.npy_path):
            raise FileNotFoundError(f"npy が見つかりません: {s.npy_path}")

        img = np.load(s.npy_path, mmap_mode="r")  # 想定: (H,W,3)
        img = np.array(img)

        # 型とスケール調整
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        # 0-255 っぽいなら 0-1 に正規化
        if img.max() > 1.5:
            img = img / 255.0

        # 形状を CHW に
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"想定外の画像形状: {s.npy_path} shape={img.shape} (H,W,3 を想定)")
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3,H,W)

        y = torch.tensor([s.target], dtype=torch.float32)
        return x, y


def maybe_extract_zip(zip_path: str, out_dir: str) -> None:
    """zip を out_dir に展開（既に中身があれば何もしない）"""
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Extracting zip: {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def resolve_dataset_root(default_base_dir: str, try_num: str, zip_path: Optional[str]) -> str:
    """
    dataset_root (= img/ と vel/ が入っているディレクトリ) を見つける。
    想定:
      <base_dir>/<try_num>/dataset/img
      <base_dir>/<try_num>/dataset/vel/data.csv
    """
    # 1) 標準配置
    candidate = os.path.join(default_base_dir, try_num, "dataset")
    if os.path.isfile(os.path.join(candidate, "vel", "data.csv")):
        return candidate

    # 1.5) 既に展開済み zip で <try_num>/<try_num>/dataset になっている配置
    candidate_nested = os.path.join(default_base_dir, try_num, try_num, "dataset")
    if os.path.isfile(os.path.join(candidate_nested, "vel", "data.csv")):
        return candidate_nested

    # 2) zip 指定があるなら展開して探す
    if zip_path is not None:
        extract_parent = os.path.join(default_base_dir, try_num)
        maybe_extract_zip(zip_path, extract_parent)

        candidate2 = os.path.join(extract_parent, try_num, "dataset")  # zip内に try_num/ がある場合
        if os.path.isfile(os.path.join(candidate2, "vel", "data.csv")):
            return candidate2

        candidate3 = os.path.join(extract_parent, "dataset")  # zip内が dataset/ 直下の場合
        if os.path.isfile(os.path.join(candidate3, "vel", "data.csv")):
            return candidate3

    # 3) カレント付近も探す
    cwd = os.getcwd()
    candidate4 = os.path.join(cwd, try_num, "dataset")
    if os.path.isfile(os.path.join(candidate4, "vel", "data.csv")):
        return candidate4

    raise FileNotFoundError(
        "dataset_root を見つけられませんでした。\n"
        f"  tried: {os.path.join(default_base_dir, try_num, 'dataset')}\n"
        "  ヒント: offline_collect の出力を\n"
        "    .../data/<speed>/<mode>/dataset/<try_num>/dataset/{img,vel}\n"
        "  に置くか、--zip で zip を指定してください。"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("try_num", help="データセットID（例: 20260209_201921）")
    parser.add_argument("epochs", type=int, help="学習エポック数（例: 50）")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--label_source", type=str, default="expert",
                        help="center の教師値に使う列（expert/center/executed/policy）。オンライン模倣なら expert 推奨。")
    parser.add_argument("--use_lr", action="store_true", help="left/right も学習に含める（デフォルトON）")
    parser.add_argument("--no_lr", action="store_true", help="left/right を使わない")
    parser.add_argument("--lr_gate_abs", type=float, default=0.1, help="|center| < この値のときだけ left/right を追加")
    parser.add_argument("--zip", type=str, default=None, help="offline_collect zip を直接指定（未展開なら自動展開）")
    parser.add_argument("--mode", type=str, default=None, help="mode を明示（例: teleop_normal）")
    parser.add_argument("--speed", type=str, default=None, help="speed を明示（未指定なら config の speed）")
    parser.add_argument("--data_root", type=str, default=None, help="data ルートを明示（例: /home/.../nav_cloning/data）")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")

    args = parser.parse_args()

    use_lr = True
    if args.no_lr:
        use_lr = False
    if args.use_lr:
        use_lr = True

    config = load_config()
    speed_src = args.speed if args.speed is not None else config.get("speed")
    speed = str(speed_src).strip() if speed_src is not None else ""

    public_path = resolve_public_path(config, args.data_root, speed if speed else None)

    # まず flat 配置 (<data_root>/<try_num>/dataset) を優先して見る
    dataset_root = resolve_flat_dataset_root(public_path, args.try_num)
    use_flat_layout = dataset_root is not None
    mode = None

    if not use_flat_layout:
        if not speed:
            raise ValueError(
                "speed が未設定です。flat 配置も見つかりませんでした。\n"
                "config に speed を追加するか --speed を指定してください。"
            )
        mode = resolve_mode(
            public_path,
            speed,
            config.get("mode"),
            args.mode,
            allow_create=(args.zip is not None),
        )
        dataset_base_dir = os.path.join(public_path, speed, mode, "dataset")
        dataset_root = resolve_dataset_root(dataset_base_dir, args.try_num, args.zip)

    csv_path = os.path.join(dataset_root, "vel", "data.csv")
    img_dir = os.path.join(dataset_root, "img")

    # 出力先（cdb で分ける）
    if use_flat_layout:
        output_model_path = os.path.join(public_path, "model", "cdb", args.try_num, str(args.epochs))
        result_path = os.path.join(public_path, "result", "cdb", args.try_num, str(args.epochs))
    else:
        output_model_path = os.path.join(public_path, speed, mode, "model", "cdb", args.try_num, str(args.epochs))
        result_path = os.path.join(public_path, speed, mode, "result", "cdb", args.try_num, str(args.epochs))
    os.makedirs(output_model_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(result_path, "run"))

    dataset = OfflineCDBDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        label_source=args.label_source,
        use_lr=use_lr,
        lr_gate_abs=args.lr_gate_abs,
        require_phase_training=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(n_channel=3, n_out=1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), eps=1e-2, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"[INFO] data_root={public_path}")
    if use_flat_layout:
        print("[INFO] dataset_layout=flat (<data_root>/<try_num>/dataset)")
    else:
        print(f"[INFO] dataset_layout=hierarchical (speed/mode), speed={speed}, mode={mode}")
    print(f"[INFO] device={device}, batch_size={args.batch_size}, epochs={args.epochs}")
    print(f"[INFO] log_dir={os.path.join(result_path, 'run')}")
    print(f"[INFO] model_out={os.path.join(output_model_path, 'model_gpu.pt')}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, angles in loader:
            images = images.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)  # (B,1)
            loss = criterion(outputs.squeeze(), angles.squeeze())
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        avg_loss = running_loss / max(1, len(loader))
        writer.add_scalar("loss", avg_loss, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        print(f"epoch [{epoch+1}/{args.epochs}] loss: {avg_loss:.6f} lr: {current_lr:.6f}")

    torch.save(model.state_dict(), os.path.join(output_model_path, "model_gpu.pt"))
    writer.close()
    print("[INFO] training finished.")


if __name__ == "__main__":
    main()
