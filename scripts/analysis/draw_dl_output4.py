#!/usr/bin/env python3
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def plot_multiple_trajectories(directories, image_path, output_dir):
    """
    複数ディレクトリ(走行結果)から経路データを読み込み、
    1枚の地図画像の上に全ての走行軌跡を重ねてプロットして保存する。
    """
    # 地図画像の読み込みと表示用オブジェクトの作成
    try:
        image = Image.open(image_path).convert("L")
        arr = np.asarray(image)
        fig, ax = plt.subplots()
        # extent=[minX, maxX, minY, maxY] を自分の地図に合わせて設定
        ax.imshow(arr, cmap='gray', extent=[-30, 20, -4.5, 14])
    except FileNotFoundError:
        print(f"Map image not found at {image_path}. Proceeding without background image.")
        fig, ax = plt.subplots()
    
    # 各走行ディレクトリごとにCSVを読み込み、軌跡をプロットする
    for directory in directories:
        csv_path = os.path.join(directory, 'training_all.csv')
        #csv_path = os.path.join(directory, 'training.csv')
        if not os.path.isfile(csv_path):
            print(f"CSV file not found at {csv_path}. Skipping.")
            continue
        
        x_list = []
        y_list = []

        # CSVファイルを開いてデータを読み込み
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f, delimiter=',')  # 区切り文字をカンマに設定
                header = next(reader, None)  # ヘッダーを取得（存在しない場合はNone）
                if header:
                    print(f"[{os.path.basename(directory)}] Header: {header}")

                # 29999行スキップ（ヘッダー含む）
                lines_skipped = 0
                skip_lines = 0
                for _ in range(skip_lines):
                    try:
                        next(reader)
                        lines_skipped += 1
                    except StopIteration:
                        break  # ファイルの終端に達した場合

                if lines_skipped < skip_lines:
                    print(f"Warning [{os.path.basename(directory)}]: Only {lines_skipped} lines skipped. Not enough data to skip {skip_lines} lines.")

                # データの読み込み
                for row in reader:
                    # 必要な列数が足りなければスキップ
                    if len(row) < 5:
                        print(f"Skipping row in [{os.path.basename(directory)}] due to insufficient columns: {row}")
                        continue
                    
                    # row[5], row[6] に x, y が格納されている想定
                    str_x, str_y = row[3], row[4]
                    try:
                        x = float(str_x)
                        y = float(str_y)
                        x_list.append(x)
                        y_list.append(y)
                    except ValueError:
                        print(f"Skipping row in [{os.path.basename(directory)}] due to invalid data: {row}")
        except Exception as e:
            print(f"Error reading file {csv_path}: {e}")
            continue

        # 軌跡が取得できていれば地図上に重ねて描画
        if x_list and y_list:
            # ディレクトリ名(または任意のラベル)を凡例として利用label=os.path.basename(directory)
            ax.plot(x_list, y_list, linewidth=1.0, color='red')

        else:
            print(f"No valid trajectory data in [{os.path.basename(directory)}].")

    # 凡例の表示
    ax.legend()
    
    # 軸ラベルやタイトルなどの設定
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Robot Trajectories on a Single Map')
    plt.grid(True)

    # 出力ファイル名やパスを指定して保存
    # ディレクトリ名ではなくまとめ図用の名前を付ける
    output_file = os.path.join(output_dir, 'all_trajectories.png')
    plt.savefig(output_file)
    plt.close()
    print(f"All trajectories plot saved to {output_file}")

if __name__ == '__main__':
    # パッケージフォルダから各パスを生成
    base_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_night_change_dataset_balance'
    #base_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_with_dir_$(arg mode)'
    #image_path = roslib.packages.get_pkg_dir('nav_cloning') + '/maps/Experimental Environment-EDIT (1).jpg'
    image_path = roslib.packages.get_pkg_dir('nav_cloning') + '/maps/cit_3f_map (4).png'
    output_dir = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/plots'

    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    """
    # 例: RGB
    specified_directories = [
        '20241112_17_43_37', '20241112_18_15_16', '20241112_18_39_35',
        '20241112_19_15_39', '20241121_20_52_38', '20241128_20_30_20',
        '20241128_20_59_13', '20241129_19_09_01', '20241129_19_16_43',
        '20241128_20_59_13', '20241129_19_09_01', '20241129_19_30_42',
        '20241130_19_16_43', 
    ]
        
    """
    """
        # 例: RGB day
    specified_directories = [
        '20241214_13_04_15', '20241214_13_05_57', '20241214_13_12_39',
        '20241214_13_16_35', '20241214_13_20_32', '20241214_13_24_55',
        '20241214_13_28_53', '20241214_13_33_41', '20241214_13_36_51',
        '20241214_13_45_15', '20241214_13_50_20', '20241214_13_53_42'
    ]
    
    """
    
        # 例: augday
    specified_directories = [
        '20250120_09_37_52', '20250120_09_45_07', '20250120_09_50_19',
        '20250120_09_53_19', '20250120_09_57_29', '20250120_09_59_13',
        '20250120_10_00_54', '20250120_10_03_20', '20250120_10_05_14',
        '20250120_10_07_51', '20250120_10_27_15', '20250120_09_47_34'
    ]
    
    """
   
        # 例: aug
    specified_directories = [
        '20250120_19_22_12', '20250120_19_23_59', '20250120_19_26_17',
        '20250120_19_31_23', '20250120_19_34_27', '20250120_19_36_46',
        '20250120_19_38_32', '20250120_19_42_35', '20250120_19_46_24',
        '20250120_19_49_50', '20250120_19_52_58', '20250120_19_48_00',
        '20250120_19_54_58' 
    ]
   """
    # ディレクトリが実際に存在するものだけを抽出
    directories = [
        os.path.join(base_path, d)
        for d in specified_directories
        if os.path.isdir(os.path.join(base_path, d))
    ]

    # 1枚の図に全ての軌跡を重ねてプロット
    plot_multiple_trajectories(directories, image_path, output_dir)


