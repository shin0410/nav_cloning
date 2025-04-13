#!/usr/bin/env python3
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_loss(directories, output_dir):
    plt.figure(figsize=(10, 6))
    combined_losses = []

    for idx, directory in enumerate(directories):
        x_list = []
        y_list = []

        # CSVファイルからデータを読み込む
        try:
            with open(os.path.join(directory, 'training.csv'), 'r') as f:
                reader = csv.reader(f, delimiter=',')  # 区切り文字をカンマに設定
                header = next(reader, None)  # ヘッダーを取得（存在しない場合はNone）
                if header:
                    print(f"Header: {header}")

                # データの読み込み（3列目のデータを取得、最大30000ステップまで）
                for i, row in enumerate(reader):
                    if i >= 6000:
                        break
                    if len(row) < 3:
                        print(f"Skipping row due to insufficient columns: {row}")
                        continue  # 列数が足りない場合はスキップ
                    try:
                        loss_value = float(row[2])
                        x_list.append(len(x_list) + 1)  # インデックスをステップとして使用
                        y_list.append(loss_value)
                    except ValueError:
                        print(f"Skipping row due to invalid data: {row}")
        except FileNotFoundError:
            print(f"CSV file not found at {os.path.join(directory, 'training.csv')}")
            continue

        # 各CSVファイルのデータを1つのグラフにプロット
        if x_list and y_list:
            plt.plot(x_list, y_list, label=f'Loss {idx + 1}')
            combined_losses.append(y_list)
        else:
            print(f"No valid loss data to plot for directory: {directory}")

    # 各CSVファイルの平均をプロット
    if combined_losses:
        max_length = max(map(len, combined_losses))
        padded_losses = [np.pad(loss, (0, max_length - len(loss)), 'constant', constant_values=np.nan) for loss in combined_losses]
        avg_losses = np.nanmean(padded_losses, axis=0)
        plt.plot(range(1, len(avg_losses) + 1), avg_losses, label='Average Loss', color='black', linewidth=2)

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.ylim(0, 0.05)
    plt.title('Loss vs Step for Multiple CSV Files')
    plt.legend()
    plt.grid(True)

    # プロットの保存
    output_file = os.path.join(output_dir, 'combined_loss_plot.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Combined loss plot saved to {output_file}")

if __name__ == '__main__':
    #base_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_with_dir_$(arg mode)'
    base_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_with_dir_use_dl_output'
    output_dir = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/plots'

    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 時間帯をスクリプトで選択できるように設定(aug10,000)
    #specified_directories = [
    #    '20240930_02:09:57', '20240930_01:41:49', '20240930_01:13:57',
    #    '20241023_00:45:57', '20241023_01:27:47', '20241023_02:09:36',
    #    '20241023_02:51:38', '20241023_03:33:30', '20241023_04:15:16',
    #    '20241023_04:57:15'
    #]
    
    # 時間帯をスクリプトで選択できるように設定(ふつう6,000)
    #specified_directories = [
    #    '20241022_22:40:11', '20241022_23:22:04', '20241023_00:03:57',
    #    '20241023_00:45:57', '20241023_01:27:47', '20241023_02:09:36',
    #    '20241023_02:51:38', '20241023_03:33:30', '20241023_04:15:16',
    #    '20241023_04:57:15'
    #]
        # 時間帯をスクリプトで選択できるように設定(ふつう6,000)
    #specified_directories = [
    #     '20240929_15:23:48', '20240929_14:55:41', '20240929_14:27:36',
    #     '20240929_13:59:27', '20240929_13:31:26', '20240929_13:03:20',
    #     '20240929_12:35:15', '20240929_12:07:15', '20240929_11:39:07',
    #     '20240929_11:10:54'
    # ]
    
    # 時間帯をスクリプトで選択できるように設定(明るいふつう6,000)
    specified_directories = [
        '20240930_02:09:57', '20240930_01:41:49', '20240930_01:13:52',
        '20240930_00:45:45', '20240930_00:17:31', '20240929_23:49:23',
        '20240929_23:21:13', '20240929_22:53:09', '20240929_22:24:59',
        '20240929_21:57:29'
    ]
    
    # 時間帯をスクリプトで選択できるように設定(aug6,000)
    #specified_directories = [
    #    '20241022_20:00:48', '20241022_19:32:32', '20241022_18:36:10',
    #    '20241022_18:08:22', '20241022_04:43:23', '20241022_04:15:07',
    #    '20241022_03:46:32', '20241022_03:18:09', '20241022_02:49:45',
    #    '20241022_02:18:52'
    #]
    
    # 時間帯をスクリプトで選択できるように設定(aug30,000)
    #specified_directories = [
    #    '20241027_22:32:32', '20241028_00:21:12', '20241028_02:11:55',
    #    '20241028_04:02:27', '20241028_05:52:50', '20241028_07:43:22',
    #    '20241028_09:34:08', '20241028_11:24:25', '20241028_13:14:47',
    #    '20241028_15:05:04',
    #]
    
    # 時間帯をスクリプトで選択できるように設定(aug回転なし10000)
    #specified_directories = [
    #    '20241030_21:56:45', '20241030_22:39:34', '20241030_23:21:22',
    #    '20241031_00:03:20', '20241031_00:45:10', '20241031_01:27:12',
    #    '20241031_02:09:12', '20241031_02:51:15', '20241031_03:33:16',
    #    '20241031_04:15:18'
    #]

    directories = [os.path.join(base_path, d) for d in specified_directories if os.path.isdir(os.path.join(base_path, d))]
    plot_loss(directories, output_dir)

