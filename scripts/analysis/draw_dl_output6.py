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
    1枚の地図画像の上に全ての走行軌跡を、制御源(dlなら赤、navなら青)で重ねてプロットし保存する。
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
    
    # 各ディレクトリごとにCSVを読み込み、軌跡をプロットする
    for directory in directories:
        csv_path = os.path.join(directory, 'training_all.csv')
        if not os.path.isfile(csv_path):
            print(f"CSV file not found at {csv_path}. Skipping.")
            continue
        
        # dl 用と nav 用の軌跡リストを初期化
        x_dl, y_dl = [], []
        x_nav, y_nav = [], []
        
        # CSVファイルを開いてデータを読み込み
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                header = next(reader, None)
                if header:
                    print(f"[{os.path.basename(directory)}] Header: {header}")

                for row in reader:
                    # 必要な列数が足りなければスキップ
                    if len(row) < 9:
                        print(f"Skipping row in [{os.path.basename(directory)}] due to insufficient columns: {row}")
                        continue
                    
                    try:
                        # x, yはそれぞれインデックス 5,6、directionはインデックス 8
                        x = float(row[5])
                        y = float(row[6])
                        direction = row[8].strip().lower()
                    except ValueError:
                        print(f"Skipping row in [{os.path.basename(directory)}] due to invalid numeric data: {row}")
                        continue
                    
                    # 制御源に応じて軌跡を分ける
                    if direction == "dl":
                        x_dl.append(x)
                        y_dl.append(y)
                    elif direction == "nav":
                        x_nav.append(x)
                        y_nav.append(y)
                    else:
                        # それ以外の場合は、任意のデフォルト色（例: 緑）にするかスキップ
                        x_nav.append(x)
                        y_nav.append(y)
        except Exception as e:
            print(f"Error reading file {csv_path}: {e}")
            continue

        # 軌跡が取得できていれば描画
        if x_dl and y_dl:
            ax.plot(x_dl, y_dl, linewidth=1.0, color='red', label='dl')
        if x_nav and y_nav:
            ax.plot(x_nav, y_nav, linewidth=1.0, color='blue', label='nav')

    # 凡例の設定（重複を避けるために一度だけ設定）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # 軸ラベル、タイトル、グリッドの設定
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Robot Trajectories on a Single Map')
    plt.grid(True)

    # 結果画像の出力パスと保存
    output_file = os.path.join(output_dir, 'all_trajectories.png')
    plt.savefig(output_file)
    plt.close()
    print(f"All trajectories plot saved to {output_file}")

if __name__ == '__main__':
    # パッケージフォルダから各パスを生成
    base_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_with_dir_use_dl_output'
    image_path = roslib.packages.get_pkg_dir('nav_cloning') + '/maps/cit_3f_map (4).png'
    output_dir = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/plots'

    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 指定ディレクトリの例（必要に応じて変更）
    specified_directories = [
        '20250120_19_22_12', '20250120_19_23_59', '20250120_19_26_17',
        '20250120_19_31_23', '20250120_19_34_27', '20250120_19_36_46',
        '20250120_19_38_32', '20250120_19_42_35', '20250120_19_46_24',
        '20250120_19_49_50', '20250120_19_52_58', '20250120_19_48_00',
        '20250120_19_54_58' 
    ]
"""    
        # 指定ディレクトリの例（必要に応じて変更）
    specified_directories = [
        '20250120_19_22_12', '20250120_19_23_59', '20250120_19_26_17',
        '20250120_19_31_23', '20250120_19_34_27', '20250120_19_36_46',
        '20250120_19_38_32', '20250120_19_42_35', '20250120_19_46_24',
        '20250120_19_49_50', '20250120_19_52_58', '20250120_19_48_00',
        '20250120_19_54_58' 
    ]
"""    
    # 存在するディレクトリだけを抽出
    directories = [os.path.join(base_path, d) for d in specified_directories if os.path.isdir(os.path.join(base_path, d))]
    
    # 全軌跡を重ねてプロット
    plot_multiple_trajectories(directories, image_path, output_dir)

