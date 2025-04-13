#!/usr/bin/env python3
from __future__ import print_function

import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_net import *  # deep_learning クラスなどが含まれている前提
from skimage.transform import resize
from geometry_msgs.msg import Twist, PoseArray, PoseWithCovarianceStamped
from std_msgs.msg import Int8, Int8MultiArray
from std_srvs.srv import Trigger, Empty, SetBool, SetBoolResponse
from nav_msgs.msg import Path, Odometry
import csv
import os
import time
import copy
import sys
import tf
import numpy as np

class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        
        # パラメータ取得
        self.mode = rospy.get_param("/nav_cloning_node/mode", "use_dl_output")
        self.action_num = 1
        
        # deep learning オブジェクトの生成（モジュール nav_cloning_net 内の deep_learning クラスを利用）
        self.dl = deep_learning(n_action=self.action_num)
        self.bridge = CvBridge()
        
        # 各種サブスクライバの設定
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
        self.tracker_sub = rospy.Subscriber("/tracker", Odometry, self.callback_tracker)
        
        # パブリッシャとサービスの設定
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.mode_save_srv = rospy.Service('/model_save', Trigger, self.callback_model_save)
        
        # 状態変数の初期化
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()
        self.cv_image = np.zeros((480, 640, 3), np.uint8)
        self.cv_left_image = np.zeros((480, 640, 3), np.uint8)
        self.cv_right_image = np.zeros((480, 640, 3), np.uint8)
        self.learning = True
        self.select_dl = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        package_dir = roslib.packages.get_pkg_dir('nav_cloning')
        self.path = os.path.join(package_dir, 'data', 'result_with_dir_' + str(self.mode))
        self.save_path = os.path.join(package_dir, 'data', 'model_with_dir_' + str(self.mode))
        self.previous_reset_time = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        self.start_time_s = rospy.get_time()
        
        # ログ保存用ディレクトリの作成
        os.makedirs(os.path.join(self.path, self.start_time), exist_ok=True)
        csv_path = os.path.join(self.path, self.start_time, 'training.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)', 'x(m)', 'y(m)', 'the(rad)', 'direction'])

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print("Error converting main image: ", e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print("Error converting left image: ", e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print("Error converting right image: ", e)

    def callback_tracker(self, data):
        self.pos_x = data.pose.pose.position.x
        self.pos_y = data.pose.pose.position.y
        rot = data.pose.pose.orientation
        angle = tf.transformations.euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
        self.pos_the = angle[2]

    def callback_path(self, data):
        self.path_pose = data

    def callback_pose(self, data):
        distance_list = []
        pos = data.pose.pose.position
        for pose in self.path_pose.poses:
            path_pos = pose.pose.position
            distance = np.sqrt((pos.x - path_pos.x)**2 + (pos.y - path_pos.y)**2)
            distance_list.append(distance)
        if distance_list:
            self.min_distance = min(distance_list)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        self.learning = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp

    def callback_model_save(self, data):
        model_res = SetBoolResponse()
        self.dl.save(self.save_path)
        model_res.message = "model_save"
        model_res.success = True
        return model_res

    def loop(self):
        # 入力画像のサイズが期待通りでない場合は処理をスキップ
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return

        # ロボットが動き出しているかで開始判定
        if self.vel.linear.x != 0:
            self.is_started = True
        if not self.is_started:
            return

        # 画像をリサイズ（DL入力用）
        img = resize(self.cv_image, (48, 64), mode='constant')
        img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        img_right = resize(self.cv_right_image, (48, 64), mode='constant')

        # エピソード数に応じた処理
        if self.episode == 10000:
            self.learning = False
            self.dl.save(self.save_path)
        if self.episode == 12000:
            os.system('killall roslaunch')
            sys.exit()

        print("Starting loop iteration")

        if self.learning:
            target_action = self.action
            distance = self.min_distance

            if self.mode == "use_dl_output":
                # DL出力を取得し学習（左右画像も補助学習）
                action, loss = self.dl.act_and_trains(img, target_action)
                if abs(target_action) < 0.1:
                    action_left, loss_left = self.dl.act_and_trains(img_left, target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right, target_action + 0.2)
                angle_error = abs(action - target_action)

                # 距離に基づいて制御信号の採用方法を決定
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True

                if self.select_dl and self.episode >= 0:
                    target_action = action
                    control_source = "dl"  # Deep Learning 出力を採用
                else:
                    control_source = "nav"  # ナビゲーション（教師信号）を採用

                print("{}, training, loss: {}, angle_error: {}, distance: {}, control_source: {}".format(
                    self.episode, loss, angle_error, distance, control_source))

                self.episode += 1
                csv_path = os.path.join(self.path, self.start_time, 'training.csv')
                line = [str(self.episode), "training", str(loss), str(angle_error), str(distance),
                        str(self.pos_x), str(self.pos_y), str(self.pos_the), control_source]
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)

                self.vel.linear.x = 0.2
                self.vel.angular.z = target_action
                self.nav_pub.publish(self.vel)
            else:
                # self.mode が "use_dl_output" 以外のときの処理（必要なら追加）
                pass
        else:
            # テストモード時の処理
            print("Calling act")
            target_action = self.dl.act(img)
            print("act completed")
            distance = self.min_distance
            print("{}, test, angular: {}, distance: {}".format(self.episode, target_action, distance))
            self.episode += 1
            angle_error = abs(self.action - target_action)
            csv_path = os.path.join(self.path, self.start_time, 'training.csv')
            line = [str(self.episode), "test", "0", str(angle_error), str(distance),
                    str(self.pos_x), str(self.pos_y), str(self.pos_the)]
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        # 画像表示（デバッグ用）
        cv2.imshow("Resized Image", img)
        cv2.imshow("Resized Left Image", img_left)
        cv2.imshow("Resized Right Image", img_right)
        cv2.waitKey(1)

if __name__ == '__main__':
    node = nav_cloning_node()
    DURATION = 0.2
    rate = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        node.loop()
        rate.sleep()

