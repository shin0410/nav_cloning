import os
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml


class Net(nn.Module):
    def __init__(self, n_channel: int, n_out: int):
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
            self.fc5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layer(x)
        x = self.fc_layer(x)
        return x


def load_config(filename: str = "config.yaml") -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_cfg = os.environ.get("NAV_CONFIG")
    if env_cfg:
        config_path = os.path.abspath(os.path.expanduser(env_cfg))
    else:
        config_path = os.path.join(script_dir, "..", "config", filename)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_public_path(config: dict, data_root_override: Optional[str], speed: Optional[str]) -> str:
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


def resolve_dataset_root(
    data_root: str,
    time_id: str,
    speed: Optional[str],
    mode: Optional[str],
    img_dir_name: str,
    vel_dir_name: str,
) -> Tuple[str, str]:
    """
    Returns:
      (dataset_root, layout)
      layout in {"flat", "hierarchical"}
    """
    tried: List[str] = []

    # flat layout: <data_root>/<time>/dataset/{img,vel}
    flat = os.path.join(data_root, time_id, "dataset")
    tried.append(flat)
    if os.path.isfile(os.path.join(flat, vel_dir_name, "data.csv")):
        return flat, "flat"

    if speed and mode:
        # hierarchical: <data_root>/<speed>/<mode>/dataset/<time>/dataset/{img,vel}
        h1 = os.path.join(data_root, speed, mode, "dataset", time_id, "dataset")
        tried.append(h1)
        if os.path.isfile(os.path.join(h1, vel_dir_name, "data.csv")):
            return h1, "hierarchical"

        # nested zip layout: .../<time>/<time>/dataset/{img,vel}
        h2 = os.path.join(data_root, speed, mode, "dataset", time_id, time_id, "dataset")
        tried.append(h2)
        if os.path.isfile(os.path.join(h2, vel_dir_name, "data.csv")):
            return h2, "hierarchical"

    raise FileNotFoundError(
        "dataset_root を見つけられませんでした。\n"
        f"  time={time_id}\n"
        f"  tried={tried}"
    )


@dataclass(frozen=True)
class Sample:
    npy_path: str
    target: float


class NpyCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        label_source_center: str = "center",
        max_steps: int = 0,
        require_phase_training: bool = True,
        aug_csv_path: Optional[str] = None,
        aug_img_dir: Optional[str] = None,
        aug_steps: int = 0,
        aug_seed: int = 42,
    ):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"data.csv が見つかりません: {csv_path}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"img ディレクトリが見つかりません: {img_dir}")

        base_df = pd.read_csv(csv_path)
        base_df = self._filter_rows(base_df, require_phase_training, max_steps)

        required = {"episode", "center", "left", "right"}
        missing = required - set(base_df.columns)
        if missing:
            raise ValueError(f"data.csv に必要列がありません: missing={sorted(list(missing))}")

        if label_source_center not in base_df.columns:
            if "center" in base_df.columns:
                print(f"[WARN] label_source_center='{label_source_center}' が無いので center を使います。")
                label_source_center = "center"
            else:
                raise ValueError("center 列が data.csv に存在しません。")

        samples: List[Sample] = self._rows_to_samples(base_df, img_dir, label_source_center)

        # 追加データ（例: TAW）を比率ぶん混ぜる
        aug_added_rows = 0
        if aug_csv_path and aug_img_dir and aug_steps > 0:
            if not os.path.isfile(aug_csv_path):
                raise FileNotFoundError(f"aug csv が見つかりません: {aug_csv_path}")
            if not os.path.isdir(aug_img_dir):
                raise FileNotFoundError(f"aug img ディレクトリが見つかりません: {aug_img_dir}")
            aug_df = pd.read_csv(aug_csv_path)
            aug_df = self._filter_rows(aug_df, require_phase_training, 0)
            if len(aug_df) > 0:
                k = min(int(aug_steps), len(aug_df))
                aug_df = aug_df.sample(n=k, random_state=int(aug_seed)).reset_index(drop=True)
                aug_samples = self._rows_to_samples(aug_df, aug_img_dir, label_source_center)
                aug_added_rows = len(aug_df)
                samples.extend(aug_samples)

        if not samples:
            raise RuntimeError("有効なサンプルがありません。CSV列やnpy配置を確認してください。")

        self.samples = samples
        print(f"[INFO] Loaded default dataset: {len(self.samples)} samples")
        print(f"  csv: {csv_path}")
        print(f"  img: {img_dir}")
        if max_steps and max_steps > 0:
            print(f"  max_steps: {max_steps} (expected ~= {max_steps * 3})")
        if aug_added_rows > 0:
            print(f"  aug_rows: {aug_added_rows} (added ~= {aug_added_rows * 3} samples)")
            print(f"  aug_csv: {aug_csv_path}")
            print(f"  aug_img: {aug_img_dir}")

    @staticmethod
    def _filter_rows(df: pd.DataFrame, require_phase_training: bool, max_steps: int) -> pd.DataFrame:
        if require_phase_training and "phase" in df.columns:
            df = df[df["phase"] == "training"].reset_index(drop=True)
        if max_steps and max_steps > 0:
            df = df.head(int(max_steps)).reset_index(drop=True)
        return df

    @staticmethod
    def _rows_to_samples(df: pd.DataFrame, img_dir: str, label_source_center: str) -> List[Sample]:
        samples: List[Sample] = []
        views = ("center", "left", "right")
        for _, r in df.iterrows():
            ep = str(r["episode"]).split(".")[0]
            for v in views:
                label_col = label_source_center if v == "center" else v
                if label_col not in r.index:
                    continue
                label = float(r[label_col])
                if not np.isfinite(label):
                    continue
                npy_path = os.path.join(img_dir, f"{ep}_{v}.npy")
                if not os.path.isfile(npy_path):
                    continue
                samples.append(Sample(npy_path=npy_path, target=label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        img = np.load(s.npy_path, mmap_mode="r")
        img = np.array(img)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"想定外の画像形状: {s.npy_path} shape={img.shape} (H,W,3 を想定)")

        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        y = torch.tensor([s.target], dtype=torch.float32)
        return x, y


def normalize_time_list(raw_times) -> List[str]:
    if raw_times is None:
        return []
    if isinstance(raw_times, str):
        raw_times = [raw_times]
    if not isinstance(raw_times, (list, tuple)):
        return []
    out: List[str] = []
    seen = set()
    for t in raw_times:
        s = str(t).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("time", nargs="?", default=None, help="データセットID（例: 20260209_201921）")
    parser.add_argument("epochs", nargs="?", type=int, default=None, help="学習エポック数")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--speed", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--load_dataset_img", type=str, default=None)
    parser.add_argument("--load_dataset_vel", type=str, default=None)
    parser.add_argument("--label_source_center", type=str, default="center")
    parser.add_argument("--max_steps", type=int, default=None, help="先頭N stepのみ使う（例: 4000 -> 約12000サンプル）")
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--mix_taw_ratio", type=float, default=None, help="元step数に対するTAW追加比率（0.5 => 4000stepなら2000step追加）")
    parser.add_argument("--mix_taw_img_dir", type=str, default=None, help="TAW画像ディレクトリ名（dataset直下）")
    parser.add_argument("--mix_taw_vel_dir", type=str, default=None, help="TAWラベルディレクトリ名（dataset直下）")
    parser.add_argument("--mix_taw_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--no_phase_training", action="store_true")
    args = parser.parse_args()

    cfg = load_config()

    time_id = str(args.time if args.time else os.environ.get("NAV_TIME", cfg.get("time", ""))).strip()
    if not time_id:
        raise ValueError("time が未設定です。引数または NAV_TIME か config['time'] を指定してください。")

    epochs = int(args.epochs if args.epochs is not None else os.environ.get("NAV_EPOCH", cfg.get("epoch", 100)))
    batch_size = int(args.batch_size if args.batch_size is not None else os.environ.get("NAV_BATCH_SIZE", cfg.get("batch_size", 8)))
    speed = str(args.speed if args.speed is not None else os.environ.get("NAV_SPEED", cfg.get("speed", ""))).strip()
    mode = str(args.mode if args.mode is not None else os.environ.get("NAV_MODE", cfg.get("mode", ""))).strip()
    load_dataset_img = str(args.load_dataset_img if args.load_dataset_img is not None else os.environ.get("NAV_LOAD_DATASET_IMG", cfg.get("load_dataset_img", "img")))
    load_dataset_vel = str(args.load_dataset_vel if args.load_dataset_vel is not None else os.environ.get("NAV_LOAD_DATASET_VEL", cfg.get("load_dataset_vel", "vel")))
    max_steps = int(args.max_steps if args.max_steps is not None else os.environ.get("NAV_MAX_STEPS", cfg.get("max_steps", 0)))
    save_model = str(args.save_model if args.save_model is not None else os.environ.get("NAV_SAVE_MODEL", "model_gpu.pt"))
    mix_taw_ratio = float(
        args.mix_taw_ratio
        if args.mix_taw_ratio is not None
        else os.environ.get("NAV_MIX_TAW_RATIO", cfg.get("mix_taw_ratio", 0.0))
    )
    mix_taw_img_dir = str(
        args.mix_taw_img_dir
        if args.mix_taw_img_dir is not None
        else os.environ.get("NAV_MIX_TAW_IMG_DIR", cfg.get("output_aug_dataset_img", ""))
    ).strip()
    mix_taw_vel_dir = str(
        args.mix_taw_vel_dir
        if args.mix_taw_vel_dir is not None
        else os.environ.get("NAV_MIX_TAW_VEL_DIR", cfg.get("output_aug_dataset_vel", ""))
    ).strip()
    train_times_env = os.environ.get("NAV_TRAIN_TIMES")
    if train_times_env:
        train_times = normalize_time_list(train_times_env.split(","))
    else:
        train_times = normalize_time_list(cfg.get("train_times", []))
    if not train_times:
        train_times = [time_id]

    data_root = resolve_public_path(cfg, args.data_root if args.data_root else os.environ.get("NAV_DATA_ROOT"), speed if speed else None)
    primary_root, layout = resolve_dataset_root(data_root, time_id, speed if speed else None, mode if mode else None, load_dataset_img, load_dataset_vel)

    if layout == "flat":
        output_model_path = os.path.join(data_root, time_id, "model", "default", str(epochs))
        result_path = os.path.join(data_root, time_id, "result", "default", str(epochs))
    else:
        output_model_path = os.path.join(data_root, speed, mode, "model", "default", time_id, str(epochs))
        result_path = os.path.join(data_root, speed, mode, "result", "default", time_id, str(epochs))

    os.makedirs(output_model_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(result_path, "run"))

    datasets: List[Dataset] = []
    for t in train_times:
        dataset_root_i, _layout_i = resolve_dataset_root(
            data_root, t, speed if speed else None, mode if mode else None, load_dataset_img, load_dataset_vel
        )
        csv_path_i = os.path.join(dataset_root_i, load_dataset_vel, "data.csv")
        img_dir_i = os.path.join(dataset_root_i, load_dataset_img)

        aug_csv_path_i: Optional[str] = None
        aug_img_dir_i: Optional[str] = None
        aug_steps_i = 0
        if mix_taw_ratio > 0 and max_steps > 0 and mix_taw_img_dir and mix_taw_vel_dir:
            cand_aug_csv = os.path.join(dataset_root_i, mix_taw_vel_dir, "data.csv")
            cand_aug_img = os.path.join(dataset_root_i, mix_taw_img_dir)
            if os.path.isfile(cand_aug_csv) and os.path.isdir(cand_aug_img):
                aug_csv_path_i = cand_aug_csv
                aug_img_dir_i = cand_aug_img
                aug_steps_i = max(1, int(round(max_steps * mix_taw_ratio)))
            else:
                print("[WARN] mix_taw_ratio > 0 ですが TAWディレクトリが見つからないため無視します。")
                print(f"       time={t}")
                print(f"       expected img: {cand_aug_img}")
                print(f"       expected csv: {cand_aug_csv}")

        ds_i = NpyCsvDataset(
            csv_path=csv_path_i,
            img_dir=img_dir_i,
            label_source_center=args.label_source_center,
            max_steps=max_steps,
            require_phase_training=(not args.no_phase_training),
            aug_csv_path=aug_csv_path_i,
            aug_img_dir=aug_img_dir_i,
            aug_steps=aug_steps_i,
            aug_seed=args.mix_taw_seed,
        )
        datasets.append(ds_i)

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(n_channel=3, n_out=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), eps=1e-2, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] layout={layout} time={time_id}")
    if len(train_times) > 1:
        print(f"[INFO] train_times={','.join(train_times)}")
    print(f"[INFO] device={device}, batch_size={batch_size}, epochs={epochs}")
    if mix_taw_ratio > 0 and max_steps > 0:
        print(f"[INFO] mix_taw_ratio={mix_taw_ratio} (each dataset adds round(max_steps*ratio) steps when TAW exists)")
    print(f"[INFO] log_dir={os.path.join(result_path, 'run')}")
    print(f"[INFO] model_out={os.path.join(output_model_path, save_model)}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, angles in loader:
            images = images.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), angles.squeeze())
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        avg_loss = running_loss / max(1, len(loader))
        writer.add_scalar("loss", avg_loss, epoch)
        writer.add_scalar("lr", current_lr, epoch)
        print(f"epoch [{epoch + 1}/{epochs}] loss: {avg_loss:.6f} lr: {current_lr:.6f}")

    torch.save(model.state_dict(), os.path.join(output_model_path, save_model))
    writer.close()
    print("[INFO] training finished.")


if __name__ == "__main__":
    main()
