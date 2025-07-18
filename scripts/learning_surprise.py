import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import pandas as pd
from tqdm import tqdm  # ★変更: tqdm追加
import cv2
import warnings
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")



def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config", filename)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
config = load_config()

PC_USER_NAME = config["pc_user_name"]
WS_NAME = config["ws_name"]
BATCH_SIZE = int(config["batch_size"])
BIN = int(config["bin"])
TIME = config["time"]
EPOCH = int(config["epoch"])
LOAD_DATASET_IMG = config["load_dataset_img"]
LOAD_DATASET_VEL = config["load_dataset_vel"]
SAVE_MODEL = config["save_model"]


VIEWS = ["center", "left", "right"]



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
    def __init__(self, img_dir, csv_path, views=VIEWS):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
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

def main():
    public_path = f"/home/{PC_USER_NAME}/ws/{WS_NAME}/src/nav_cloning/data"
    dataset_base = os.path.join(public_path, TIME, "dataset")
    img_dir = os.path.join(dataset_base, LOAD_DATASET_IMG)
    csv_path = os.path.join(dataset_base, LOAD_DATASET_VEL, "data.csv")
    output_model_path = os.path.join(public_path, TIME, "model", str(EPOCH))
    result_path = os.path.join(public_path, TIME, "result", str(EPOCH))

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(output_model_path, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(result_path, "run"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ImgCsvDataset(img_dir, csv_path)
    print(f"The dataset contains {len(dataset)} samples.")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Net(n_channel=3, n_out=1).to(device)
    # criterion = nn.MSELoss()
    criterion = nn.MSELoss(reduction='none')  # ★変更: MSELossのreductionを'none'に設定
    optimizer = optim.Adam(model.parameters(), eps=1e-8) # ★変更: epsを1e-8に設定 #1e-2から 1e-8に変更
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)



    # ★変更: 角速度一括取得して bin 確率を計算
    df = pd.read_csv(csv_path)
    all_angles = torch.tensor(df[VIEWS].values.flatten(), dtype=torch.float32).to(device)
    global_min_val, global_max_val = all_angles.min(), all_angles.max()
    global_bins = torch.linspace(global_min_val, global_max_val, BIN + 1, device=device)
    all_angles_binned = torch.bucketize(all_angles, global_bins) - 1
    all_angles_binned = all_angles_binned.clamp(min=0)
    global_bin_counts = torch.bincount(all_angles_binned, minlength=BIN).float()
    global_bin_probs = global_bin_counts / (global_bin_counts.sum() + 1e-6)



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
        avg_loss = running_loss / len(dataloader)
        writer.add_scalar('loss', avg_loss, epoch)
        
        print(f"epoch [{epoch+1}/{EPOCH}], loss: {avg_loss:.4f}, lr: {current_lr:.6f}")

    torch.save(model.state_dict(), os.path.join(output_model_path, SAVE_MODEL))

if __name__ == "__main__":
    main()