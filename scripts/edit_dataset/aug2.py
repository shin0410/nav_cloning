import os
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import cv2

def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", filename)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

PC_USER_NAME = config["pc_user_name"]
WS_NAME = config["ws_name"]
TIME = config["time"]
INPUT_AUG_DATASET_IMG = config["input_aug_dataset_img"]
INPUT_AUG_DATASET_VEL = config["input_aug_dataset_vel"]
OUTPUT_AUG_DATASET_IMG = config["output_aug_dataset_img"]
OUTPUT_AUG_DATASET_VEL = config["output_aug_dataset_vel"]

def adjust_gamma(image, gamma=1.0):
    """float32 (0〜1) 前提のガンマ補正"""
    return np.clip(image ** (1.0 / gamma), 0.0, 1.0)

def add_random_shadow(image):
    """影追加（float32 BGR）"""
    h, w = image.shape[:2]
    top_x, bot_x = np.random.randint(0, w, 2)
    mask = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        x = int((bot_x - top_x) * (i / h) + top_x)
        mask[i, :x] = 1.0

    shadow_strength = np.random.uniform(0.3, 0.7)
    shadow_mask = np.stack([mask]*3, axis=-1)
    return np.clip(image * (1 - shadow_mask + shadow_mask * shadow_strength), 0.0, 1.0)

def apply_local_brightness(image):
    """局所的な明るさ変更（float32）"""
    h, w = image.shape[:2]
    center = (np.random.randint(w), np.random.randint(h))
    radius = np.random.randint(min(w, h) // 4, min(w, h) // 2)
    brightness = np.random.uniform(0.5, 1.5)

    Y, X = np.ogrid[:h, :w]
    dist = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    mask = np.exp(-dist / (2 * radius ** 2)).astype(np.float32)
    mask = np.expand_dims(mask, axis=2)  # チャンネル対応

    return np.clip(image * (1 + (brightness - 1) * mask), 0.0, 1.0)

def color_jitter_bgr(image, brightness, contrast, saturation, hue):
    """float32 BGR画像に対する色変化（OpenCV使用）"""
    image = image.copy()

    # Brightness
    if brightness > 0:
        factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
        image = np.clip(image * factor, 0.0, 1.0)

    # Contrast
    if contrast > 0:
        factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mean = np.mean(gray)
        image = np.clip((image - mean) * factor + mean, 0.0, 1.0)

    # Saturation & Hue in HSV
    if saturation > 0 or hue > 0:
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        if saturation > 0:
            s_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)
        if hue > 0:
            h_shift = np.random.uniform(-hue, hue) * 180
            hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    return np.clip(image, 0.0, 1.0)


def augment_gamma_color_shadow_variants(image):
    """ガンマ＋色調補正＋影＋局所照明の複合変換（float32）"""
    variants = []
    base_images = [image]

    for _ in range(4):
        img = image.copy()
        gamma_val = np.random.uniform(0.5, 1.5)
        img = adjust_gamma(img, gamma_val)
        img = color_jitter_bgr(img, brightness=0.4, contrast=0.4, saturation=0.1, hue=0.01)
        base_images.append(img)

    for base in base_images:
        variants.append(base)
        variants.append(add_random_shadow(base))
        variants.append(apply_local_brightness(base))

    return variants


def main():
    #base_dir = f"/home/{PC_USER_NAME}/ws/{WS_NAME}/src/nav_cloning/data/{TIME}/dataset"
    base_dir = f"/home/{PC_USER_NAME}/{WS_NAME}/src/nav_cloning/data/{TIME}/dataset"
    img_dir = os.path.join(base_dir, INPUT_AUG_DATASET_IMG)
    vel_csv_path = os.path.join(base_dir, INPUT_AUG_DATASET_VEL, "data.csv")

    aug_img_dir = os.path.join(base_dir, OUTPUT_AUG_DATASET_IMG)
    aug_vel_dir = os.path.join(base_dir, OUTPUT_AUG_DATASET_VEL)

    # 修正: 不要なtry-except削除
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_vel_dir, exist_ok=True)

    # CSVファイルの存在確認
    if not os.path.exists(vel_csv_path):
        raise FileNotFoundError(f"Velocity CSV file not found: {vel_csv_path}")

    df = pd.read_csv(vel_csv_path)
    augmented_records = []
    episodes = df['episode'].unique()

    for episode in tqdm(episodes, desc="Processing episodes"):
        row = df[df['episode'] == episode].iloc[0]
        all_variants = {}

        # 各ポジションの画像を処理
        for pos in ['center', 'left', 'right']:
            filename = f"{episode}_{pos}.npy"
            filepath = os.path.join(img_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: Image file not found: {filepath}")
                continue
            
            try:
                img_bgr = np.load(filepath, mmap_mode='r')
                # img_bgr = np.array(img_bgr).astype(np.float32)
                if img_bgr.max() > 1.0:
                    img_bgr /= 255.0


                combined_variants = []
                combined_variants = augment_gamma_color_shadow_variants(img_bgr)
                all_variants[pos] = combined_variants

                
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
                continue

        # 全ポジションのデータが揃っているかチェック
        if len(all_variants) != 3:
            print(f"Warning: Missing position data for episode {episode}")
            continue

        # 各ポジションのバリアント数が一致しているかチェック
        variant_counts = [len(all_variants[pos]) for pos in ['center', 'left', 'right']]
        if len(set(variant_counts)) > 1:
            print(f"Warning: Inconsistent variant counts for episode {episode}: {variant_counts}")
            continue

        # 画像枚数はposのどれかのバリエーション数（基本一致している想定）
        num_variants = len(all_variants['center'])
        for i in range(num_variants):
            for pos in ['center', 'left', 'right']:
                out_img_bgr = all_variants[pos][i]
                out_fname = f"{episode}_{i}_{pos}.npy"
                out_path = os.path.join(aug_img_dir, out_fname)
                np.save(out_path, out_img_bgr)

            csv_episode = f"{episode}_{i}"
            record = row.to_dict()
            record['episode'] = csv_episode
            augmented_records.append(record)

    # 結果をCSVに保存
    df_aug = pd.DataFrame(augmented_records)
    out_csv_path = os.path.join(aug_vel_dir, "data.csv")
    df_aug.to_csv(out_csv_path, index=False)

    print(f"Original episodes: {len(episodes)}")
    print(f"Augmented images saved: {len(os.listdir(aug_img_dir))}")
    print(f"Augmented CSV records: {len(augmented_records)}")

if __name__ == "__main__":
    main()
