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

def plot_trajectory(directory, image_path, output_dir):
    # 地図画像の読み込みと表示
    try:
        image = Image.open(image_path).convert("L")
        arr = np.asarray(image)
        fig, ax = plt.subplots()
        ax.imshow(arr, cmap='gray', extent=[-10, 50, -10, 50])
    except FileNotFoundError:
        print(f"Map image not found at {image_path}. Proceeding without background image.")
        fig, ax = plt.subplots()

    x_list = []
    y_list = []

    # CSVファイルからデータを読み込む
    try:
        with open(os.path.join(directory, 'training.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')  # 区切り文字をカンマに設定
            header = next(reader, None)  # ヘッダーを取得（存在しない場合はNone）
            if header:
                print(f"Header: {header}")
            
            # 6000行目までスキップ（ヘッダーを含む）
            lines_skipped = 0
            for _ in range(29999):
                try:
                    next(reader)
                    lines_skipped += 1
                except StopIteration:
                    break  # ファイルの終端に達した場合
            if lines_skipped < 29999:
                print(f"Warning: Only {lines_skipped + 1} lines in the file including header. Not enough data to skip 6000 lines.")
            
            # データの読み込み
            for row in reader:
                if len(row) < 8:
                    print(f"Skipping row due to insufficient columns: {row}")
                    continue  # 列数が足りない場合はスキップ
                
                # 必要な列を取得（8列の場合）
                str_x, str_y = row[5], row[6]
                try:
                    x = float(str_x)
                    y = float(str_y)
                    x_list.append(x)
                    y_list.append(y)
                except ValueError:
                    print(f"Skipping row due to invalid data: {row}")
    except FileNotFoundError:
        print(f"CSV file not found at {os.path.join(directory, 'training.csv')}")
        return
    
    # 経路の描画
    if x_list and y_list:
        ax.plot(x_list, y_list, color='red', linewidth=2, label='Robot Trajectory')
        ax.legend()
    else:
        print("No valid trajectory data to plot.")
    
    # プロットの保存
    output_file = os.path.join(output_dir, os.path.basename(directory) + '.png')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Robot Trajectory on Map')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Trajectory plot saved to {output_file}")

if __name__ == '__main__':
    base_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_with_dir_$(arg mode)'
    image_path = roslib.packages.get_pkg_dir('nav_cloning') + '/maps/map.png'
    output_dir = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/plots'

    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 時間帯をスクリプトで選択できるように設定aug30000
    #specified_directories = [
    #    '20241027_22:32:32', '20241028_00:21:12', '20241028_02:11:55',
    #    '20241028_04:02:27', '20241028_05:52:50', '20241028_07:43:22',
    #    '20241028_09:34:08', '20241028_11:24:25', '20241028_13:14:47',
    #    '20241028_15:05:04'
    #]

    # 時間帯をスクリプトで選択できるように設定aug10000severity2
    #specified_directories = [
    #    '20241103_23:40:00', '20241103_22:17:14', '20241103_22:58:07',
    #    '20241104_00:22:04', '20241104_01:04:12', '20241104_01:46:11',
    #    '20241104_02:28:17', '20241104_03:09:42', '20241104_03:51:58',
    #    '20241104_04:34:06'
    #]
    
    # 時間帯をスクリプトで選択できるように設定aug10000severity1
    #specified_directories = [
    #    '20241102_15:48:17', '20241102_16:29:17', '20241102_17:10:16',
    #    '20241102_17:51:13', '20241102_18:32:15', '20241102_19:13:19',
    #    '20241102_19:54:16', '20241102_20:36:03', '20241102_21:18:26',
    #    '20241102_21:59:50'
    #]
    
        # 時間帯をスクリプトで選択できるように設定aug30000severity2
    specified_directories = [
        '20241104_07:27:15', '20241104_09:15:06', '20241104_11:05:21',
        '20241104_12:55:17', '20241104_05:52:50', '20241104_07:43:22',
        '20241104_09:34:08', '20241104_11:24:25', '20241104_13:14:47',
        '20241104_15:05:04'
    ]
        
     # 時間帯をスクリプトで選択できるように設定aug回転なし10000
    #specified_directories = [
    #    '20241030_21:56:45', '20241030_22:39:34', '20241030_23:21:22',
    #    '20241031_00:03:20', '20241031_00:45:10', '20241031_01:27:12',
    #    '20241031_02:09:12', '20241031_02:51:15', '20241031_03:33:16',
    #    '20241031_04:15:18'
    #]

    directories = [os.path.join(base_path, d) for d in specified_directories if os.path.isdir(os.path.join(base_path, d))]
    for directory in directories:
        plot_trajectory(directory, image_path, output_dir)

