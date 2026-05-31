#!/usr/bin/env python3

import numpy as np
import roslib
roslib.load_manifest('nav_cloning')
import rospy
from net import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skimage.transform import resize
import os
import sys
from geometry_msgs.msg import Twist
import yaml
import csv
import time
from sensor_msgs.msg import Joy


def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config", filename)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

SPEED = config["speed"]
DURATION = float(config["duration"])


class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)

        self.speed = float(SPEED)
        self.bridge = CvBridge()

        # 3 cameras
        self.image_center_sub = rospy.Subscriber(
            "/camera_center/usb_cam/image_raw",
            Image,
            self.callback_center,
            queue_size=1
        )
        self.image_left_sub = rospy.Subscriber(
            "/camera_left/usb_cam/image_raw",
            Image,
            self.callback_left,
            queue_size=1
        )
        self.image_right_sub = rospy.Subscriber(
            "/camera_right/usb_cam/image_raw",
            Image,
            self.callback_right,
            queue_size=1
        )

        self.vel = Twist()
        self.vel_sub = rospy.Subscriber("/teleop_vel", Twist, self.callback_vel, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback, queue_size=1)

        self.action_num = 1
        self.dl = deep_learning(n_action=self.action_num)

        self.is_started = False
        self.action = 0.0
        self.episode = 0

        # latest images from each camera
        self.cv_center_image = None
        self.cv_left_image = None
        self.cv_right_image = None

        self.count = 0
        self.joy_flg = False

        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path_dataset = roslib.packages.get_pkg_dir('nav_cloning') + '/data/' + self.start_time + "/dataset/"
        self.img_store_path = self.path_dataset + "/img/"
        self.vel_store_path = self.path_dataset + "/vel/"
        self.vel_csv_file = self.vel_store_path + "data.csv"

        os.makedirs(self.path_dataset, exist_ok=True)
        os.makedirs(self.img_store_path, exist_ok=True)
        os.makedirs(self.vel_store_path, exist_ok=True)

    def callback_center(self, data):
        try:
            self.cv_center_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print("center:", e)

    def callback_left(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print("left:", e)

    def callback_right(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print("right:", e)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def joy_callback(self, data):
        if len(data.buttons) > 0 and data.buttons[0] == 1:
            self.joy_flg = True

    def images_ready(self):
        if self.cv_center_image is None:
            return False
        if self.cv_left_image is None:
            return False
        if self.cv_right_image is None:
            return False
        return True

    def preprocess(self, image):
        return resize(image, (48, 64), mode='constant').astype(np.float32)

    def loop(self):
        if not self.images_ready():
            return

        if self.count == 0:
            time.sleep(10)
            self.vel.linear.x = self.speed
            self.vel.angular.z = 0.0
            self.nav_pub.publish(self.vel)
            self.count = 1

        if self.count == 1:
            if self.joy_flg:
                os.system('killall roslaunch')
                sys.exit()

            # resize each real camera image
            img_center = self.preprocess(self.cv_center_image)
            img_left = self.preprocess(self.cv_left_image)
            img_right = self.preprocess(self.cv_right_image)

            # save images
            np.save(self.img_store_path + f"{self.episode}_center.npy", img_center)
            np.save(self.img_store_path + f"{self.episode}_left.npy", img_left)
            np.save(self.img_store_path + f"{self.episode}_right.npy", img_right)

            target_action = self.action

            # save csv (keep current label policy)
            file_exists = os.path.isfile(self.vel_csv_file)
            with open(self.vel_csv_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(['episode', 'center', 'left', 'right'])
                writer.writerow([
                    self.episode,
                    round(target_action, 4),
                    round(target_action - 0.2, 4),
                    round(target_action + 0.2, 4)
                ])

            self.vel.linear.x = self.speed
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

            self.episode += 1
            print(self.episode)


if __name__ == '__main__':
    rg = nav_cloning_node()
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
