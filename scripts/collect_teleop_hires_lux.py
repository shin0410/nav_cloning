#!/usr/bin/env python3
# collect_teleop+lux.py の高解像度保存版。
# 従来の 48x64 npy / data.csv / trajectory はそのまま出力し、互換性を保ったまま
# 追加で img_hires/ にカメラ元解像度(640x480)の JPEG を保存する。
# 容量目安: JPEG 約30-80KB/枚 x 3視点 x 10000行 ≈ 1-2.5GB (float32 npy 換算の約1/60)
# 48x64 は後からでも作れないため、事前学習エンコーダ(224x224入力)用の資産として残す。

import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image, Imu
from sensor_msgs.msg import Illuminance
from cv_bridge import CvBridge, CvBridgeError
from skimage.transform import resize
import os
import sys
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import yaml
import csv
import time
from sensor_msgs.msg import Joy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from trajectory_logger import TrajectoryLogger


PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)


def load_config(filename="config.yaml"):
    config_path = os.path.join(PACKAGE_DIR, "config", filename)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_data_root():
    configured_root = os.environ.get("NAV_CLONING_DATA_ROOT") or config.get("dataset_root")
    if configured_root:
        return os.path.abspath(os.path.expanduser(configured_root))
    return os.path.join(PACKAGE_DIR, "data")


config = load_config()

SPEED = config["speed"]
DURATION = float(config["duration"])
JPEG_QUALITY = int(os.environ.get("NAV_CLONING_JPEG_QUALITY", 92))


def simulate_yaw_rotation(image, yaw_angle_degrees, camera_fov):
    height, width = image.shape[:2]
    f = (width / 2) / np.tan(np.radians(camera_fov / 2))
    yaw_rad = np.radians(yaw_angle_degrees)
    horizontal_shift = f * np.tan(yaw_rad)
    y, x = np.indices((height, width))
    map_x = x - horizontal_shift
    map_y = y.astype(np.float32)
    warped_image = cv2.remap(image, map_x.astype(np.float32), map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
    return warped_image


class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.speed = float(SPEED)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera_center/usb_cam/image_raw", Image, self.callback, queue_size=1)
        self.vel = Twist()
        self.vel_sub = rospy.Subscriber("/teleop_vel", Twist, self.callback_vel, queue_size=1)
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.nav_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback, queue_size=1)
        self.imu_topic = rospy.get_param("~imu_topic", "/spresense/imu/data_raw")
        self.imu_queue_size = int(rospy.get_param("~imu_queue_size", 1000))
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.odom_queue_size = int(rospy.get_param("~odom_queue_size", 1000))
        self.trajectory_flush_every = int(rospy.get_param("~trajectory_flush_every", 120))
        self.latest_imu_angular_z = float('nan')
        self.imu_sub = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=self.imu_queue_size)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=self.odom_queue_size)
        self.is_started = False
        self.action = 0.0
        self.episode = 0
        self.cv_image = np.zeros((480, 640, 3), np.uint8)
        self.camera_angle = 24.61
        self.camera_fov = 150
        self.count = 0
        self.joy_flg = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.data_root = get_data_root()
        self.path_dataset = os.path.join(self.data_root, self.start_time, "dataset")
        self.img_store_path = os.path.join(self.path_dataset, "img")
        self.img_hires_store_path = os.path.join(self.path_dataset, "img_hires")
        self.vel_store_path = os.path.join(self.path_dataset, "vel")
        self.trajectory_store_path = os.path.join(self.path_dataset, "trajectory")
        self.vel_csv_file = os.path.join(self.vel_store_path, "data.csv")

        self.latest_lux = float('nan')
        self.lux_sub = rospy.Subscriber("/lux", Illuminance, self.lux_callback, queue_size=1)

        os.makedirs(self.path_dataset, exist_ok=True)
        os.makedirs(self.img_store_path, exist_ok=True)
        os.makedirs(self.img_hires_store_path, exist_ok=True)
        os.makedirs(self.vel_store_path, exist_ok=True)
        os.makedirs(self.trajectory_store_path, exist_ok=True)
        self.vel_csv_handle = open(self.vel_csv_file, 'a', newline='')
        self.vel_writer = csv.writer(self.vel_csv_handle)
        if self.vel_csv_handle.tell() == 0:
            self.vel_writer.writerow(['episode', 'center', 'left', 'right', 'lux'])
            self.vel_csv_handle.flush()
        self.trajectory_logger = TrajectoryLogger(
            self.trajectory_store_path,
            "collect_teleop_hires_lux",
            prefer_imu_yaw=rospy.get_param("~trajectory_prefer_imu_yaw", True),
            flush_every=self.trajectory_flush_every
        )
        rospy.on_shutdown(self.close_files)

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def joy_callback(self, data):
        if len(data.buttons) > 0 and data.buttons[0] == 1:
            self.joy_flg = True

    def lux_callback(self, msg):
        self.latest_lux = msg.illuminance

    def imu_callback(self, msg):
        self.latest_imu_angular_z = msg.angular_velocity.z
        if hasattr(self, "trajectory_logger"):
            self.trajectory_logger.log_imu(msg, rospy.Time.now().to_sec())

    def odom_callback(self, msg):
        if hasattr(self, "trajectory_logger"):
            self.trajectory_logger.log_odom(msg, rospy.Time.now().to_sec())

    def close_files(self):
        if hasattr(self, "vel_csv_handle") and not self.vel_csv_handle.closed:
            self.vel_csv_handle.close()
        if hasattr(self, "trajectory_logger"):
            self.trajectory_logger.close()

    def save_hires(self, name, image_bgr):
        path = os.path.join(self.img_hires_store_path, name)
        ok = cv2.imwrite(path, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            rospy.logwarn_throttle(10.0, "hires JPEG の保存に失敗: %s" % path)

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.count == 0:
            time.sleep(10)
            self.vel.linear.x = self.speed
            self.vel.angular.z = 0.0
            self.nav_pub.publish(self.vel)
            self.count = 1

        if self.count == 1:
            sample_time = rospy.Time.now().to_sec()
            if self.joy_flg:
                self.close_files()
                os.system('killall roslaunch')
                sys.exit()

            cv_left_image = simulate_yaw_rotation(self.cv_image, self.camera_angle, self.camera_fov)
            cv_right_image = simulate_yaw_rotation(self.cv_image, -self.camera_angle, self.camera_fov)

            self.save_hires(f"{self.episode}_center.jpg", self.cv_image)
            self.save_hires(f"{self.episode}_left.jpg", cv_left_image)
            self.save_hires(f"{self.episode}_right.jpg", cv_right_image)

            img = resize(self.cv_image, (48, 64), mode='constant').astype(np.float32)
            img_left = resize(cv_left_image, (48, 64), mode='constant').astype(np.float32)
            img_right = resize(cv_right_image, (48, 64), mode='constant').astype(np.float32)

            np.save(os.path.join(self.img_store_path, f"{self.episode}_center.npy"), img)
            np.save(os.path.join(self.img_store_path, f"{self.episode}_left.npy"), img_left)
            np.save(os.path.join(self.img_store_path, f"{self.episode}_right.npy"), img_right)

            target_action = self.action

            self.vel_writer.writerow([
                self.episode,
                round(target_action, 4),
                round(target_action - 0.2, 4),
                round(target_action + 0.2, 4),
                None if np.isnan(self.latest_lux) else round(float(self.latest_lux), 3)
            ])
            self.vel_csv_handle.flush()

            self.vel.linear.x = self.speed
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)
            self.trajectory_logger.log(
                self.episode,
                sample_time,
                self.vel.linear.x,
                self.vel.angular.z,
                self.latest_imu_angular_z
            )
            self.episode += 1
            print(self.episode)


if __name__ == '__main__':
    rg = nav_cloning_node()
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
