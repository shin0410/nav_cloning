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
    複数のディレクトリ(走行結果)から各種CSVを読み込み、
    1枚の地図画像上に軌跡を重ねて描画する。
    """

    #--- 地図画像を読み込み、描画環境を準備する ---#
    try:
        image = Image.open(image_path).convert("L")
        arr = np.asarray(image)
        fig, ax = plt.subplots()
        # 地図の表示範囲を適宜調整してください
        ax.imshow(arr, cmap='gray', extent=[-10, 50, -10, 50])
    except FileNotFoundError:
        print(f"Map image not found at {image_path}. Proceeding without background image.")
        fig, ax = plt.subplots()

    #--- 各ディレクトリのCSVから軌跡を読み込み、同じFigureに描画する ---#
    for directory in directories:
        csv_file = os.path.join(directory, 'training.csv')
        if not os.path.isfile(csv_file):
            print(f"CSV file not found at {csv_file}. Skipping.")
            continue

        x_list = []
        y_list = []

        # CSVからデータを読み込み
        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                header = next(reader, None)  # ヘッダー行(存在しなければNone)
                if header:
                    print(f"[{os.path.basename(directory)}] Header: {header}")

                # 29999行スキップ（必要に応じて変更）
                lines_skipped = 0
                skip_count = 5999
                for _ in range(skip_count):
                    try:
                        next(reader)
                        lines_skipped += 1
                    except StopIteration:
                        break
                if lines_skipped < skip_count:
                    print(f"Warning [{os.path.basename(directory)}]: "
                          f"Only {lines_skipped} lines skipped. "
                          f"Not enough data to skip {skip_count} lines.")

                # 残りの行からX,Y座標を取得
                for row in reader:
                    if len(row) < 8:
                        print(f"Skipping row in [{os.path.basename(directory)}] due to insufficient columns: {row}")
                        continue
                    str_x, str_y = row[5], row[6]
                    try:
                        x = float(str_x)
                        y = float(str_y)
                        x_list.append(x)
                        y_list.append(y)
                    except ValueError:
                        print(f"Skipping row in [{os.path.basename(directory)}] due to invalid data: {row}")

        except Exception as e:
            print(f"Error reading CSV from [{directory}]: {e}")
            continue

        # 軌跡を地図上に描画 (ディレクトリ名を凡例として利用)
        if x_list and y_list:
            ax.plot(x_list, y_list, linewidth=1, color='red')
        else:
            print(f"No valid trajectory data in [{os.path.basename(directory)}].")

    #--- 軸や凡例、タイトルなどを設定して可視化を整える ---#
    ax.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Robot Trajectories')
    plt.grid(True)

    #--- 画像ファイルとして出力 ---#
    output_file = os.path.join(output_dir, 'all_trajectories.png')
    plt.savefig(output_file)
    plt.close()
    print(f"All trajectories plot saved to {output_file}")

#---------------------------------------------#
if __name__ == '__main__':
    base_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_with_dir_$(arg mode)'
    image_path = roslib.packages.get_pkg_dir('nav_cloning') + '/maps/map.png'
    output_dir = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/plots'

    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
        # 時間帯をスクリプトで選択できるように設定(ふつう6,000)
    specified_directories = [
        '20241022_22:40:11', '20241022_23:22:04', '20241023_00:03:57',
        '20241023_00:45:57', '20241023_01:27:47', '20241023_02:09:36',
        '20241023_02:51:38', '20241023_03:33:30', '20241023_04:15:16',
        '20241023_04:57:15'
    ]
        # 時間帯をスクリプトで選択できるように設定(ふつう6,000)
    """
    specified_directories = [
         '20240929_15:23:48', '20240929_14:55:41', '20240929_14:27:36',
         '20240929_13:59:27', '20240929_13:31:26', '20240929_13:03:20',
         '20240929_12:35:15', '20240929_12:07:15', '20240929_11:39:07',
         '20240929_11:10:54'
     ]
     """
    
    # 時間帯をスクリプトで選択できるように設定(明るいふつう6,000)
    """
    specified_directories = [
        '20240930_02:09:57', '20240930_01:41:49', '20240930_01:13:52',
        '20240930_00:45:45', '20240930_00:17:31', '20240929_23:49:23',
        '20240929_23:21:13', '20240929_22:53:09', '20240929_22:24:59',
        '20240929_21:57:29'
    ]
    """
    
    # 時間帯をスクリプトで選択できるように設定(aug6,000)
    """
    specified_directories = [
        '20241022_20:00:48', '20241022_19:32:32', '20241022_18:36:10',
        '20241022_18:08:22', '20241022_04:43:23', '20241022_04:15:07',
        '20241022_03:46:32', '20241022_03:18:09', '20241022_02:49:45',
        '20241022_02:18:52'
    ]
    """
    # 実在するディレクトリだけリスト化
    directories = [
        os.path.join(base_path, d)
        for d in specified_directories
        if os.path.isdir(os.path.join(base_path, d))
    ]

    # まとめて1枚の図に軌跡を描画
    plot_multiple_trajectories(directories, image_path, output_dir)


