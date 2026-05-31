import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

import numpy as np

BATCH_SIZE = 8
BIN = 5

# ---------------------------
# AugMix (regression) settings
# ---------------------------
USE_AUGMIX_CONSISTENCY = True   # デバッグで切りたい時は False
AUGMIX_SEVERITY = 3
AUGMIX_WIDTH = 3
AUGMIX_DEPTH = -1
AUGMIX_ALPHA = 1.0

# 整合性（本来JSD）→ 回帰用 SmoothL1 の重み
LAMBDA_CONS = 1.0

# 入力が BGR で保存されている場合だけ True（色が変ならここを疑う）
INPUT_IS_BGR = False


# ---------------------------
# Try to import AugMix
# ---------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_ROOTS = [
    os.path.abspath(os.path.join(SCRIPT_DIR, "..")),
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")),
]
for p in CANDIDATE_ROOTS:
    if p not in sys.path:
        sys.path.insert(0, p)

augment_and_mix = None
MEAN = None
STD = None

try:
    # あなたの aug_mix.py と同じ import 形式
    from augmix.augment_and_mix import augment_and_mix, MEAN, STD
except Exception:
    try:
        # AugMix repo をそのまま置いた場合の fallback
        from augment_and_mix import augment_and_mix, MEAN, STD
    except Exception as e:
        raise ImportError(
            "AugMix の import に失敗しました。\n"
            "想定: nav_cloning 配下に augmix/augment_and_mix.py があり、augmix/ に __init__.py がある構成。\n"
            "または augment_and_mix.py が PYTHONPATH に入っている構成。\n"
            "エラー: %s" % str(e)
        )


# denormalize は aug_mix.py と同じ発想（AugMix出力(正規化空間) → [0,1]RGB に戻す）
_mean = np.array(MEAN, dtype=np.float32)
_std = np.array(STD, dtype=np.float32)

def denormalize_rgb(img_norm):
    img = img_norm * _std[None, None, :] + _mean[None, None, :]
    return np.clip(img, 0.0, 1.0).astype(np.float32)

def chw_torch_to_rgb01_hwc_numpy(img_chw):
    """
    img_chw: torch.Tensor (C,H,W)
    return: np.float32 (H,W,C) in RGB, range [0,1]
    """
    x = img_chw.detach().cpu()
    if x.dtype != torch.float32:
        x = x.float()

    # スケール推定（0..1 or 0..255 を想定）
    mx = float(x.max().item()) if x.numel() > 0 else 0.0
    if mx > 1.5:
        x = x / 255.0

    x = torch.clamp(x, 0.0, 1.0)

    # CHW -> HWC
    x = x.permute(1, 2, 0).contiguous().numpy()  # HWC

    # BGR -> RGB の可能性
    if INPUT_IS_BGR:
        x = x[..., ::-1]

    return x.astype(np.float32)

def rgb01_hwc_numpy_to_chw_torch(img_rgb01, like_tensor):
    """
    img_rgb01: np.float32 (H,W,C) in RGB, range [0,1]
    like_tensor: 元の torch.Tensor (C,H,W) のスケールに戻すため参照
    return: torch.float32 (C,H,W)
    """
    x = np.clip(img_rgb01, 0.0, 1.0).astype(np.float32)

    # RGB -> BGR の可能性
    if INPUT_IS_BGR:
        x = x[..., ::-1]

    # HWC -> CHW
    t = torch.from_numpy(x).permute(2, 0, 1).contiguous().float()

    # 元が 0..255 系だったら戻す
    mx = float(like_tensor.detach().max().item()) if like_tensor.numel() > 0 else 0.0
    if mx > 1.5:
        t = t * 255.0

    return t

def augmix_one_torch(img_chw, severity, width, depth, alpha):
    """
    img_chw: torch.Tensor (C,H,W)
    return: torch.Tensor (C,H,W) (入力スケールに合わせて返す)
    """
    img_rgb01 = chw_torch_to_rgb01_hwc_numpy(img_chw)  # HWC RGB [0,1]
    mixed_norm = augment_and_mix(
        img_rgb01.astype(np.float32),
        severity=severity,
        width=width,
        depth=depth,
        alpha=alpha
    )  # 返り値は「正規化空間」

    mixed_rgb01 = denormalize_rgb(mixed_norm)  # RGB [0,1]
    out = rgb01_hwc_numpy_to_chw_torch(mixed_rgb01, like_tensor=img_chw)
    return out

def augmix_batch(images_bchw, severity, width, depth, alpha):
    """
    images_bchw: torch.Tensor (B,C,H,W)
    return: torch.Tensor (B,C,H,W)
    """
    outs = []
    # CPUで作ってGPUへ（AugMixはnumpy/PILなので基本CPU）
    imgs_cpu = images_bchw.detach().cpu()
    for i in range(imgs_cpu.size(0)):
        outs.append(augmix_one_torch(imgs_cpu[i], severity, width, depth, alpha))
    return torch.stack(outs, dim=0).to(images_bchw.device)


class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
        # <Network CNN 3 + FC 2>
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)

        # <Weight set>
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)

        self.flatten = nn.Flatten()

        # <CNN layer>
        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            self.flatten
        )

        # <FC layer (output)>
        self.fc_layer = nn.Sequential(
            self.fc4,
            self.relu,
            self.fc5,
        )

    def forward(self, x):
        x1 = self.cnn_layer(x)
        x2 = self.fc_layer(x1)
        return x2


def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config", filename)  # configディレクトリ内を想定
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


config = load_config()

pc_user_name = config["pc_user_name"]
ws_name = config["ws_name"]
speed = config["speed"]
mode = config["mode"]

try_num = sys.argv[1]
ep = sys.argv[2]

public_path = "/home/" + pc_user_name + "/ws/" + ws_name + "/src/nav_cloning/data"
input_dataset_path = os.path.join(public_path, speed, mode, "dataset", try_num, "tensor")
output_model_path = os.path.join(public_path, speed, mode, "model", "surprise", try_num, ep)
try:
    os.makedirs(output_model_path)
except FileExistsError:
    pass

# tensor_data = torch.load(input_dataset_path + "/" + "dataset.pt")
tensor_data = torch.load(input_dataset_path + "/" + "dataset_fixed.pt")
# tensor_data = torch.load(input_dataset_path + "/" + "dataset_augmix.pt")
print(f"The dataset contains {len(tensor_data)} samples.")

result_path = os.path.join(public_path, speed, mode, "result", "surprise", try_num, ep)
try:
    os.makedirs(result_path)
except FileExistsError:
    pass

writer = SummaryWriter(log_dir=result_path + "/" + "run")
tensor_dataloader = DataLoader(tensor_data, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net(n_channel=3, n_out=1).to(device)

# 教師あり（元のまま：MSE）
criterion_sup = nn.MSELoss(reduction='none')
# 整合性（本来JSDの置き場）：回帰なのでSmoothL1
criterion_cons = nn.SmoothL1Loss(reduction='none')

optimizer = optim.Adam(model.parameters(), eps=1e-2, weight_decay=5e-4)  # 5e-4(1e-4)
num_epoch = int(sys.argv[2])
scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-6)  # 1e-6

# 事前に全データの分布を計算
all_angles = torch.cat([a for _, a in tensor_dataloader], dim=0).to(device)
all_angles = all_angles.squeeze(-1)

global_min_val = all_angles.min()
global_max_val = all_angles.max()
num_bins = BIN
global_bins = torch.linspace(
    global_min_val.item(), global_max_val.item(),
    num_bins + 1, device=device, dtype=all_angles.dtype
)

all_angles_binned = torch.bucketize(all_angles, global_bins) - 1
all_angles_binned = all_angles_binned.clamp(min=0, max=num_bins - 1)
global_bin_counts = torch.bincount(all_angles_binned, minlength=num_bins).float()
global_bin_probs = global_bin_counts / (global_bin_counts.sum() + 1e-6)


def calculate_weighted_shannon_surprise(angles, bins=global_bins, bin_probs=global_bin_probs, eps=1e-6):
    if angles.device != bins.device:
        bins = bins.to(angles.device)
        bin_probs = bin_probs.to(angles.device)
    angles_binned = torch.bucketize(angles, bins) - 1
    angles_binned = angles_binned.clamp(min=0, max=len(bin_probs) - 1)
    return -torch.log(bin_probs[angles_binned] + eps)


for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    running_sup = 0.0
    running_cons = 0.0

    for images, angles in tensor_dataloader:
        images, angles = images.to(device), angles.to(device)
        angles = angles.squeeze(1)

        shannon_weights = calculate_weighted_shannon_surprise(angles)

        optimizer.zero_grad()

        if USE_AUGMIX_CONSISTENCY:
            # 2つのAugMix viewを作る（分類のAugMixと同じ構成）
            images_aug1 = augmix_batch(images, AUGMIX_SEVERITY, AUGMIX_WIDTH, AUGMIX_DEPTH, AUGMIX_ALPHA)
            images_aug2 = augmix_batch(images, AUGMIX_SEVERITY, AUGMIX_WIDTH, AUGMIX_DEPTH, AUGMIX_ALPHA)

            # forwardはまとめて（速度のため）
            x_cat = torch.cat([images, images_aug1, images_aug2], dim=0)
            y_cat = model(x_cat).squeeze(1)

            b = images.size(0)
            y_clean = y_cat[:b]
            y_a1 = y_cat[b:2*b]
            y_a2 = y_cat[2*b:3*b]

            # 教師あり
            loss_sup = criterion_sup(y_clean, angles)

            # 整合性（JSDの代わり）：clean予測（detach）を teacher にして SmoothL1
            with torch.no_grad():
                y_t = y_clean.detach()
            loss_cons = 0.5 * (criterion_cons(y_a1, y_t) + criterion_cons(y_a2, y_t))

            # 合成して、あなたのコード通り Shannon 重みを掛ける
            loss_total = loss_sup + (LAMBDA_CONS * loss_cons)
            weighted_loss = (loss_total * shannon_weights).sum()

            # ログ用
            running_sup += float((loss_sup * shannon_weights).sum().item())
            running_cons += float((loss_cons * shannon_weights).sum().item())

        else:
            # 元の挙動（教師ありのみ）
            outputs = model(images).squeeze(1)
            loss_sup = criterion_sup(outputs, angles)
            weighted_loss = (loss_sup * shannon_weights).sum()
            running_sup += float(weighted_loss.item())

        weighted_loss.backward()
        optimizer.step()
        running_loss += float(weighted_loss.item())

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    denom = max(1, len(tensor_dataloader))
    writer.add_scalar('loss_total', running_loss / denom, epoch)
    writer.add_scalar('loss_sup_weighted', running_sup / denom, epoch)
    writer.add_scalar('loss_cons_weighted', running_cons / denom, epoch)
    print(
        f'epoch [{epoch+1}/{num_epoch}], '
        f'loss_total: {running_loss/denom:.4f}, '
        f'loss_sup(w): {running_sup/denom:.4f}, '
        f'loss_cons(w): {running_cons/denom:.4f}, '
        f'lr: {current_lr:.6f}'
    )

torch.save(model.state_dict(), output_model_path + '/model_gpu.pt')

