#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
from geometry_msgs.msg import Twist
import yaml
import csv
import time
from sensor_msgs.msg import Joy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
SMALL_IMAGE_SIZE = (64, 48)  # OpenCV order: width, height
VIDEO_SIZE = (640, 480)  # OpenCV order: width, height
VIDEO_FOURCC = "mp4v"


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

        self.is_started = False
        self.action = 0.0
        self.episode = 0

        # latest images from each camera
        self.cv_center_image = None
        self.cv_left_image = None
        self.cv_right_image = None

        self.count = 0
        self.joy_flg = False
        self.last_slow_warning_time = 0.0

        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.data_root = get_data_root()
        self.path_dataset = os.path.join(self.data_root, self.start_time, "dataset")
        self.img_store_path = os.path.join(self.path_dataset, "img")
        self.vel_store_path = os.path.join(self.path_dataset, "vel")
        self.video_store_path = os.path.join(self.path_dataset, "video")
        self.vel_csv_file = os.path.join(self.vel_store_path, "data.csv")
        self.video_index_file = os.path.join(self.video_store_path, "index.csv")

        os.makedirs(self.path_dataset, exist_ok=True)
        os.makedirs(self.img_store_path, exist_ok=True)
        os.makedirs(self.vel_store_path, exist_ok=True)
        os.makedirs(self.video_store_path, exist_ok=True)

        self.vel_csv_handle = open(self.vel_csv_file, 'a', newline='')
        self.vel_writer = csv.writer(self.vel_csv_handle)
        if self.vel_csv_handle.tell() == 0:
            self.vel_writer.writerow(['episode', 'center', 'left', 'right'])
            self.vel_csv_handle.flush()

        self.video_fps = 1.0 / DURATION
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
        self.video_writers = {
            "center": cv2.VideoWriter(
                os.path.join(self.video_store_path, "center.mp4"),
                fourcc,
                self.video_fps,
                VIDEO_SIZE
            ),
            "left": cv2.VideoWriter(
                os.path.join(self.video_store_path, "left.mp4"),
                fourcc,
                self.video_fps,
                VIDEO_SIZE
            ),
            "right": cv2.VideoWriter(
                os.path.join(self.video_store_path, "right.mp4"),
                fourcc,
                self.video_fps,
                VIDEO_SIZE
            ),
        }
        for name, writer in self.video_writers.items():
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {name}")

        self.video_index_handle = open(self.video_index_file, 'a', newline='')
        self.video_index_writer = csv.writer(self.video_index_handle)
        if self.video_index_handle.tell() == 0:
            self.video_index_writer.writerow([
                'episode',
                'video_frame',
                'timestamp',
                'center_video',
                'left_video',
                'right_video',
                'action'
            ])
            self.video_index_handle.flush()

        self.metadata_file = os.path.join(self.video_store_path, "metadata.yaml")
        with open(self.metadata_file, 'w') as metadata:
            yaml.safe_dump({
                "video_width": VIDEO_SIZE[0],
                "video_height": VIDEO_SIZE[1],
                "video_fps": self.video_fps,
                "video_fourcc": VIDEO_FOURCC,
                "npy_width": SMALL_IMAGE_SIZE[0],
                "npy_height": SMALL_IMAGE_SIZE[1],
                "duration": DURATION,
                "speed": self.speed,
            }, metadata, sort_keys=False)
        rospy.on_shutdown(self.close_files)

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
        resized = cv2.resize(image, SMALL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def make_video_frame(self, image):
        if image.shape[1] != VIDEO_SIZE[0] or image.shape[0] != VIDEO_SIZE[1]:
            image = cv2.resize(image, VIDEO_SIZE, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image)

    def close_files(self):
        if hasattr(self, "video_writers"):
            for writer in self.video_writers.values():
                writer.release()
        if hasattr(self, "vel_csv_handle") and not self.vel_csv_handle.closed:
            self.vel_csv_handle.close()
        if hasattr(self, "video_index_handle") and not self.video_index_handle.closed:
            self.video_index_handle.close()

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
            sample_start = time.time()
            if self.joy_flg:
                self.close_files()
                os.system('killall roslaunch')
                sys.exit()

            center_frame = self.cv_center_image.copy()
            left_frame = self.cv_left_image.copy()
            right_frame = self.cv_right_image.copy()

            # save small training images
            img_center = self.preprocess(center_frame)
            img_left = self.preprocess(left_frame)
            img_right = self.preprocess(right_frame)

            np.save(os.path.join(self.img_store_path, f"{self.episode}_center.npy"), img_center)
            np.save(os.path.join(self.img_store_path, f"{self.episode}_left.npy"), img_left)
            np.save(os.path.join(self.img_store_path, f"{self.episode}_right.npy"), img_right)

            target_action = self.action

            # save 640x480 video frames at the same sample timing
            self.video_writers["center"].write(self.make_video_frame(center_frame))
            self.video_writers["left"].write(self.make_video_frame(left_frame))
            self.video_writers["right"].write(self.make_video_frame(right_frame))
            self.video_index_writer.writerow([
                self.episode,
                self.episode,
                f"{time.time():.6f}",
                "center.mp4",
                "left.mp4",
                "right.mp4",
                round(target_action, 4)
            ])
            self.video_index_handle.flush()

            # save csv (keep current label policy)
            self.vel_writer.writerow([
                self.episode,
                round(target_action, 4),
                round(target_action - 0.2, 4),
                round(target_action + 0.2, 4)
            ])
            self.vel_csv_handle.flush()

            self.vel.linear.x = self.speed
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

            elapsed = time.time() - sample_start
            if elapsed > DURATION:
                now = time.time()
                if now - self.last_slow_warning_time > 2.0:
                    rospy.logwarn(
                        "collect_teleop loop took %.3fs > duration %.3fs",
                        elapsed,
                        DURATION
                    )
                    self.last_slow_warning_time = now

            self.episode += 1
            print(self.episode)


if __name__ == '__main__':
    rg = nav_cloning_node()
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
