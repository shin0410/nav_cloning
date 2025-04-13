#!/usr/bin/env python3
from __future__ import print_function

from numpy import dtype
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_net import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from std_msgs.msg import Int8MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
import sys
import tf
from nav_msgs.msg import Odometry
import numpy as np
from PIL import Image as PILImage

# AugMix 
current_dir = os.path.dirname(os.path.abspath(__file__))
augmix_dir = os.path.join(current_dir, '..', 'augmix')
augmix_dir = os.path.abspath(augmix_dir)
sys.path.append(augmix_dir)
from augment_and_mix import augment_and_mix

def process_image_from_array(image_array, target_size=(64, 48)):
    """
    「小さい解像度」でAugMixを行い、そのまま返す関数。
    例として (64,48) -> (縦48, 横64) の順序の場合は
    target_size=(64,48) に注意(幅×高さなのか、高さ×幅なのか)。

    この関数が返す画像は、AugMix後の「小さい解像度」かつ[0.0~1.0]のfloat配列。
    """
    try:
        # 1. OpenCVのBGR -> RGB
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # 2. PILで読み込み、指定された小さい解像度にリサイズ
        image_pil = PILImage.fromarray(image_rgb)
        if target_size is not None:
            image_pil = image_pil.resize(target_size, PILImage.LANCZOS)

        # 3. NumPy配列に変換 (0.0 ~ 1.0 の float)
        image_np = np.array(image_pil, dtype=np.float32) / 255.0

        # 4. AugMixを適用 (結果も [0.0~1.0] で返る想定)
        augmented_image = augment_and_mix(image_np)

        # 5. 念のため0~1にクリップ
        min_val, max_val = np.min(augmented_image), np.max(augmented_image)
        if max_val > min_val:  
            augmented_image = (augmented_image - min_val) / (max_val - min_val)
        augmented_image = np.clip(augmented_image, 0.0, 1.0)

        # AugMix後の小さい画像 ([0.0~1.0] float32) をそのまま return
        return augmented_image

    except Exception as e:
        print(f"Error processing image array: {str(e)}")
        return None

class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.mode = rospy.get_param("/nav_cloning_node/mode", "use_dl_output")
        self.action_num = 1
        self.dl = deep_learning(n_action=self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.mode_save_srv = rospy.Service('/model_save', Trigger, self.callback_model_save)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
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
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_with_dir_' + str(self.mode) + '/'
        self.save_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_with_dir_' + str(self.mode) + '/'
        self.previous_reset_time = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.start_time)

        with open(self.path + self.start_time + '/' + 'training.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)', 'x(m)', 'y(m)', 'the(rad)', 'direction'])
        self.tracker_sub = rospy.Subscriber("/tracker", Odometry, self.callback_tracker)

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

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
            path = pose.pose.position
            distance = np.sqrt(abs((pos.x - path.x)**2 + (pos.y - path.y)**2))
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
        # カメラ画像が正しく取得できていない場合はスキップ
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return

        # ロボットが動き始めるまで学習・推論を停止
        if self.vel.linear.x != 0:
            self.is_started = True
        if self.is_started == False:
            return

        #-----------------------------
        # 1. 学習フェーズ: AugMixを小さい解像度で実施
        #-----------------------------
        if self.episode < 7000:
            # 例: (64,48)の解像度でAugMixを行う
            img_augmented       = process_image_from_array(self.cv_image, target_size=(64,48))
            img_left_augmented  = process_image_from_array(self.cv_left_image, target_size=(64,48))
            img_right_augmented = process_image_from_array(self.cv_right_image, target_size=(64,48))

            if img_augmented is None or img_left_augmented is None or img_right_augmented is None:
                print("Augmentation failed for one or more images.")
                return

            # ここではすでに(64,48)のサイズになっている(※float32, 0~1)
            # 学習に使うなら(48,64)にしたい場合は転置か別途指定するなど、
            # 必要に応じてアスペクトを合わせてください
            # 例: skimage.transform.resizeで(48,64)にする
            img       = resize(img_augmented,       (48,64), mode='constant')
            img_left  = resize(img_left_augmented,  (48,64), mode='constant')
            img_right = resize(img_right_augmented, (48,64), mode='constant')

        else:
            # テストフェーズ: 小さい解像度のAugMixなし → 直接 (48,64) へリサイズ
            img       = resize(self.cv_image,       (48,64), mode='constant')
            img_left  = resize(self.cv_left_image,  (48,64), mode='constant')
            img_right = resize(self.cv_right_image, (48,64), mode='constant')

        # エピソード区切りで学習->テストへ
        if self.episode == 5000:
            self.learning = False
            self.dl.save(self.save_path)

        if self.episode == 7000:
            os.system('killall roslaunch')
            sys.exit()

        print("Starting loop iteration")

        #-----------------------------
        # 2. 学習 or テスト
        #-----------------------------
        if self.learning:
            target_action = self.action
            distance = self.min_distance

            # メインカメラで学習
            action, loss = self.dl.act_and_trains(img, target_action)

            # 左右カメラは補足学習用
            if abs(target_action) < 0.1:
                action_left, loss_left   = self.dl.act_and_trains(img_left,  target_action - 0.2)
                action_right, loss_right = self.dl.act_and_trains(img_right, target_action + 0.2)

            angle_error = abs(action - target_action)
            print(f"{self.episode}, training, loss: {loss}, angle_error: {angle_error}, distance: {distance}")

            self.episode += 1
            line = [
                str(self.episode),
                "training",
                str(loss),
                str(angle_error),
                str(distance),
                str(self.pos_x),
                str(self.pos_y),
                str(self.pos_the)
            ]
            with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)

            # 実際のロボット動作
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        else:
            # テストフェーズ
            print("Calling act")
            target_action = self.dl.act(img)
            print("act completed")

            distance = self.min_distance
            print(f"{self.episode}, test, angular:{target_action}, distance: {distance}")

            self.episode += 1
            angle_error = abs(self.action - target_action)
            line = [
                str(self.episode),
                "test",
                "0",
                str(angle_error),
                str(distance),
                str(self.pos_x),
                str(self.pos_y),
                str(self.pos_the)
            ]
            with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)

            # テスト動作
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        #-----------------------------
        # 3. 表示
        #-----------------------------
        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        temp = copy.deepcopy(img_left)
        cv2.imshow("Resized Left Image", temp)
        temp = copy.deepcopy(img_right)
        cv2.imshow("Resized Right Image", temp)
        cv2.waitKey(1)

if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()

