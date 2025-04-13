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

def plot_trajectory():
    # 必要に応じてコメントアウトしてください
    # rospy.init_node('trajectory_node', anonymous=True)
    path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/'
    image_path = roslib.packages.get_pkg_dir('nav_cloning')+'/maps/map.png'
    
    # 地図画像の読み込みと表示
    try:
        image = Image.open(image_path).convert("L")
        arr = np.asarray(image)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(arr, cmap='gray', extent=[-10,50,-10,50])
    except FileNotFoundError:
        print(f"Map image not found at {image_path}. Proceeding without background image.")
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    x_list = []
    y_list = []

    try:
        with open(path + 'training.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')  # 区切り文字をカンマに設定
            header = next(reader)  # ヘッダーを取得
            print(f"Header: {header}")
            # 6000行目までスキップ（ヘッダーを含む）
            lines_skipped = 0
            for _ in range(5999):
                try:
                    next(reader)
                    lines_skipped += 1
                except StopIteration:
                    break  # ファイルの終端に達した場合
            if lines_skipped < 5999:
                print(f"Warning: Only {lines_skipped+1} lines in the file including header. Not enough data to skip 6000 lines.")
            # データの読み込み
            data_points_read = 0
            for row in reader:
                if len(row) < 8:
                    print(f"Skipping row due to insufficient columns: {row}")
                    continue  # 列数が足りない場合はスキップ
                # 必要な列を取得（8列の場合）
                str_step, str_mode, str_loss, str_angle_error, str_distance, str_x, str_y, str_the = row
                try:
                    x = float(str_x)
                    y = float(str_y)
                except ValueError:
                    print(f"Skipping row due to invalid data: {row}")
                    continue  # 数値に変換できない場合はスキップ
                x_list.append(x)
                y_list.append(y)
                data_points_read += 1

            print(f"Data points read: {data_points_read}")
            if data_points_read == 0:
                print("No data points were read after line 6000.")
                return
    except FileNotFoundError:
        print(f"CSV file not found at {path + 'training.csv'}.")
        return

    # データポイントを表示（デバッグ用）
    print("First few data points:")
    for i in range(min(5, len(x_list))):
        print(f"x: {x_list[i]}, y: {y_list[i]}")

    # 軌跡のプロット
    ax.plot(x_list, y_list, marker='o', color='blue')
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_title('Robot Trajectory (from line 6000 onwards)')
    ax.set_xlim([-5, 30])
    ax.set_ylim([-5, 15])
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plot_trajectory()

