import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import pandas as pd
from tqdm import tqdm  # ★変更: tqdm追加
# import cv2  # unused
import warnings
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")



def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_cfg = os.environ.get("NAV_CONFIG")
    if env_cfg:
        config_path = os.path.abspath(os.path.expanduser(env_cfg))
    else:
        config_path = os.path.join(script_dir, "..", "config", filename)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
config = load_config()

PC_USER_NAME = str(config["pc_user_name"])
WS_NAME = str(config["ws_name"])
BATCH_SIZE = int(os.environ.get("NAV_BATCH_SIZE", config["batch_size"]))
BIN = int(os.environ.get("NAV_BIN", config["bin"]))
TIME = str(os.environ.get("NAV_TIME", config["time"]))
EPOCH = int(os.environ.get("NAV_EPOCH", config["epoch"]))
LOAD_DATASET_IMG = str(os.environ.get("NAV_LOAD_DATASET_IMG", config["load_dataset_img"]))
LOAD_DATASET_VEL = str(os.environ.get("NAV_LOAD_DATASET_VEL", config["load_dataset_vel"]))
SAVE_MODEL = str(os.environ.get("NAV_SAVE_MODEL", config["save_model"]))
MAX_STEPS = int(os.environ.get("NAV_MAX_STEPS", config.get("max_steps", 0)))
OUTPUT_TIME = str(os.environ.get("NAV_OUTPUT_TIME", "")).strip()

train_times_env = os.environ.get("NAV_TRAIN_TIMES")
if train_times_env:
    TRAIN_TIMES = [s.strip() for s in train_times_env.split(",") if s.strip()]
else:
    TRAIN_TIMES = config.get("train_times", [])


VIEWS = ["center", "left", "right"]


def normalize_train_times(raw_times):
    if raw_times is None:
        return []
    if isinstance(raw_times, str):
        raw_times = [raw_times]
    if not isinstance(raw_times, (list, tuple)):
        return []
    normalized = []
    seen = set()
    for t in raw_times:
        t_str = str(t).strip()
        if t_str and t_str not in seen:
            seen.add(t_str)
            normalized.append(t_str)
    return normalized


TRAIN_TIMES = normalize_train_times(TRAIN_TIMES)



class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    #<Network CNN 3 + FC 2> 
        self.conv1 = nn.Conv2d(n_channel, 32,kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64,64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512,n_out)
        self.relu = nn.ReLU(inplace=True)
    #<Weight set>
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        #self.maxpool = nn.MaxPool2d(2,2)
        #self.batch = nn.BatchNorm2d(0.2)
        self.flatten = nn.Flatten()
    #<CNN layer>   
        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            #self.maxpool,
            self.flatten
        )
    #<FC layer (output)>
        self.fc_layer = nn.Sequential(
            self.fc4,
            self.relu,
            self.fc5,
        )

    #<forward layer>
    def forward(self,x):
        x1 = self.cnn_layer(x)
        x2 = self.fc_layer(x1)
        return x2
    


class ImgCsvDataset(Dataset):
    def __init__(self, img_dir, csv_path, views=VIEWS, max_steps=0):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
        if max_steps and max_steps > 0:
            self.df = self.df.head(int(max_steps)).reset_index(drop=True)
        self.views = views
        self.data_pairs = []
        for _, row in self.df.iterrows():
            for view in self.views:
                episode = str(row['episode']).split('.')[0]
                img_path = os.path.join(img_dir, f"{episode}_{view}.npy")
                angle = row[view]
                self.data_pairs.append((img_path, angle))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, angle = self.data_pairs[idx]
        img = np.load(img_path, mmap_mode='r')  # float32（0-1）のまま読み込み
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC → CHW
        return img, torch.tensor(angle, dtype=torch.float32).unsqueeze(0)



def calculate_weighted_shannon_surprise(angles, bins, bin_probs, eps=1e-6):
    angles_binned = torch.bucketize(angles, bins) - 1
    angles_binned = angles_binned.clamp(min=0)
    return -torch.log(bin_probs[angles_binned] + eps)


def resolve_dataset_paths(public_path, time_id):
    dataset_base = os.path.join(public_path, time_id, "dataset")
    img_dir = os.path.join(dataset_base, LOAD_DATASET_IMG)
    csv_path = os.path.join(dataset_base, LOAD_DATASET_VEL, "data.csv")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"img dir not found: {img_dir}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"csv not found: {csv_path}")
    return img_dir, csv_path


def build_train_dataset(public_path):
    if TRAIN_TIMES:
        datasets = []
        csv_paths = []
        used_times = []
        for time_id in TRAIN_TIMES:
            try:
                img_dir, csv_path = resolve_dataset_paths(public_path, time_id)
                ds = ImgCsvDataset(img_dir, csv_path, max_steps=MAX_STEPS)
                if len(ds) == 0:
                    print(f"[WARN] skip empty dataset: {time_id}")
                    continue
                datasets.append(ds)
                csv_paths.append(csv_path)
                used_times.append(time_id)
            except Exception as e:
                print(f"[WARN] skip {time_id}: {e}")
        if not datasets:
            raise RuntimeError("train_times was set but no valid dataset was found.")
        if len(datasets) == 1:
            return datasets[0], csv_paths, used_times[0], used_times
        return ConcatDataset(datasets), csv_paths, "multi", used_times

    img_dir, csv_path = resolve_dataset_paths(public_path, TIME)
    dataset = ImgCsvDataset(img_dir, csv_path, max_steps=MAX_STEPS)
    if len(dataset) == 0:
        raise RuntimeError(f"dataset is empty: {TIME}")
    return dataset, [csv_path], TIME, [TIME]


def compute_global_bins(csv_paths, device):
    all_angles_np = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if MAX_STEPS and MAX_STEPS > 0:
            df = df.head(int(MAX_STEPS)).reset_index(drop=True)
        use_cols = [c for c in VIEWS if c in df.columns]
        if not use_cols:
            continue
        arr = df[use_cols].apply(pd.to_numeric, errors="coerce").values.flatten()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            all_angles_np.append(arr.astype(np.float32))

    if not all_angles_np:
        raise RuntimeError("No valid angle values were found in CSV files.")

    all_angles = torch.tensor(np.concatenate(all_angles_np), dtype=torch.float32, device=device)
    global_min_val, global_max_val = all_angles.min(), all_angles.max()
    if global_min_val == global_max_val:
        global_min_val = global_min_val - 1e-6
        global_max_val = global_max_val + 1e-6

    global_bins = torch.linspace(global_min_val, global_max_val, BIN + 1, device=device)
    all_angles_binned = torch.bucketize(all_angles, global_bins) - 1
    all_angles_binned = all_angles_binned.clamp(min=0, max=BIN - 1)
    global_bin_counts = torch.bincount(all_angles_binned, minlength=BIN).float()
    global_bin_probs = global_bin_counts / (global_bin_counts.sum() + 1e-6)
    return global_bins, global_bin_probs

def main():
    #public_path = f"/home/{PC_USER_NAME}/ws/{WS_NAME}/src/nav_cloning/data"
    public_path = os.environ.get(
        "NAV_DATA_DIR",
        f"/home/{PC_USER_NAME}/{WS_NAME}/src/nav_cloning/data",
    )
    dataset, csv_paths, out_time, used_times = build_train_dataset(public_path)
    if OUTPUT_TIME:
        out_time = OUTPUT_TIME
    output_model_path = os.path.join(public_path, out_time, "model", str(EPOCH))
    result_path = os.path.join(public_path, out_time, "result", str(EPOCH))

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(output_model_path, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(result_path, "run"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(used_times) > 1:
        print(f"Use train_times ({len(used_times)}): {', '.join(used_times)}")
    else:
        print(f"Use train time: {used_times[0]}")
    if OUTPUT_TIME:
        print(f"Save output time: {out_time}")
    if MAX_STEPS and MAX_STEPS > 0:
        print(f"Use max_steps: {MAX_STEPS} (expected samples ~= {MAX_STEPS * len(VIEWS)})")
    print(f"The dataset contains {len(dataset)} samples.")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    sample_img, _ = dataset[0]
    model = Net(n_channel=int(sample_img.shape[0]), n_out=1).to(device)
    # criterion = nn.MSELoss()
    criterion = nn.MSELoss(reduction='none')  # ★変更: MSELossのreductionを'none'に設定
    optimizer = optim.Adam(model.parameters(), eps=1e-8, weight_decay=5e-4) # ★変更: epsを1e-8に設定 #1e-2から 1e-8に変更
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)



    # ★変更: 角速度一括取得して bin 確率を計算
    global_bins, global_bin_probs = compute_global_bins(csv_paths, device)



    for epoch in range(EPOCH):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(dataset)) as pbar:
            for images, angles in dataloader:
                images, angles = images.to(device), angles.to(device)
                angles = angles.squeeze(1)
                shannon_weights = calculate_weighted_shannon_surprise(angles, global_bins, global_bin_probs)
                optimizer.zero_grad()
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, angles)
                weighted_loss = (loss * shannon_weights).sum()
                weighted_loss.backward()
                optimizer.step()
                running_loss += weighted_loss.item()
        
                pbar.update(len(images))# バッチサイズ分進める
                
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        avg_loss = running_loss / max(1, len(dataloader))
        writer.add_scalar('loss', avg_loss, epoch)
        writer.add_scalar('lr', current_lr, epoch)
        
        print(f"epoch [{epoch+1}/{EPOCH}], loss: {avg_loss:.4f}, lr: {current_lr:.6f}")

    torch.save(model.state_dict(), os.path.join(output_model_path, SAVE_MODEL))

if __name__ == "__main__":
    main()
