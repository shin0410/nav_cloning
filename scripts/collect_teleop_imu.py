#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
import threading
from geometry_msgs.msg import Twist
import yaml
import csv
import time
from sensor_msgs.msg import Image, Imu, Joy
from std_msgs.msg import Float32


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

        self.imu_topic = rospy.get_param("~imu_topic", "/spresense/imu/data_raw")
        self.temperature_topic = rospy.get_param("~temperature_topic", "/spresense/imu/temperature")
        self.imu_flush_every = int(rospy.get_param("~imu_flush_every", 120))
        self.imu_lock = threading.Lock()
        self.latest_imu = None
        self.latest_temperature = float("nan")
        self.imu_raw_count = 0
        self.temperature_raw_count = 0
        self.closing = False
        self.last_imu_warning_time = 0.0
        self.imu_sub = rospy.Subscriber(
            self.imu_topic,
            Imu,
            self.callback_imu,
            queue_size=200
        )
        self.temperature_sub = rospy.Subscriber(
            self.temperature_topic,
            Float32,
            self.callback_temperature,
            queue_size=50
        )

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
        self.imu_store_path = os.path.join(self.path_dataset, "imu")
        self.vel_csv_file = os.path.join(self.vel_store_path, "data.csv")
        self.video_index_file = os.path.join(self.video_store_path, "index.csv")
        self.imu_raw_file = os.path.join(self.imu_store_path, "raw.csv")
        self.imu_sample_index_file = os.path.join(self.imu_store_path, "sample_index.csv")
        self.temperature_raw_file = os.path.join(self.imu_store_path, "temperature.csv")

        os.makedirs(self.path_dataset, exist_ok=True)
        os.makedirs(self.img_store_path, exist_ok=True)
        os.makedirs(self.vel_store_path, exist_ok=True)
        os.makedirs(self.video_store_path, exist_ok=True)
        os.makedirs(self.imu_store_path, exist_ok=True)

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

        self.imu_raw_handle = open(self.imu_raw_file, 'a', newline='')
        self.imu_raw_writer = csv.writer(self.imu_raw_handle)
        if self.imu_raw_handle.tell() == 0:
            self.imu_raw_writer.writerow(
                [
                    'ros_time',
                    'header_stamp',
                    'seq',
                    'frame_id',
                    'orientation_x',
                    'orientation_y',
                    'orientation_z',
                    'orientation_w',
                    'angular_velocity_x',
                    'angular_velocity_y',
                    'angular_velocity_z',
                    'linear_acceleration_x',
                    'linear_acceleration_y',
                    'linear_acceleration_z',
                ]
                + [f'orientation_covariance_{i}' for i in range(9)]
                + [f'angular_velocity_covariance_{i}' for i in range(9)]
                + [f'linear_acceleration_covariance_{i}' for i in range(9)]
            )
            self.imu_raw_handle.flush()

        self.imu_sample_index_handle = open(self.imu_sample_index_file, 'a', newline='')
        self.imu_sample_index_writer = csv.writer(self.imu_sample_index_handle)
        if self.imu_sample_index_handle.tell() == 0:
            self.imu_sample_index_writer.writerow([
                'episode',
                'sample_time',
                'imu_header_stamp',
                'imu_ros_time',
                'imu_age_sec',
                'frame_id',
                'angular_velocity_x',
                'angular_velocity_y',
                'angular_velocity_z',
                'linear_acceleration_x',
                'linear_acceleration_y',
                'linear_acceleration_z',
                'orientation_x',
                'orientation_y',
                'orientation_z',
                'orientation_w',
                'temperature',
            ])
            self.imu_sample_index_handle.flush()

        self.temperature_raw_handle = open(self.temperature_raw_file, 'a', newline='')
        self.temperature_raw_writer = csv.writer(self.temperature_raw_handle)
        if self.temperature_raw_handle.tell() == 0:
            self.temperature_raw_writer.writerow(['ros_time', 'temperature'])
            self.temperature_raw_handle.flush()

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

        self.imu_metadata_file = os.path.join(self.imu_store_path, "metadata.yaml")
        with open(self.imu_metadata_file, 'w') as metadata:
            yaml.safe_dump({
                "imu_topic": self.imu_topic,
                "temperature_topic": self.temperature_topic,
                "raw_csv": "raw.csv",
                "sample_index_csv": "sample_index.csv",
                "temperature_csv": "temperature.csv",
                "imu_flush_every": self.imu_flush_every,
                "note": "raw.csv stores every incoming sensor_msgs/Imu message; sample_index.csv stores the latest IMU value aligned to each image/video sample.",
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

    def callback_imu(self, msg):
        ros_time = rospy.Time.now().to_sec()
        header_stamp = msg.header.stamp.to_sec()
        row = [
            f"{ros_time:.9f}",
            f"{header_stamp:.9f}",
            msg.header.seq,
            msg.header.frame_id,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ] + list(msg.orientation_covariance) \
          + list(msg.angular_velocity_covariance) \
          + list(msg.linear_acceleration_covariance)

        latest = {
            "ros_time": ros_time,
            "header_stamp": header_stamp,
            "frame_id": msg.header.frame_id,
            "orientation_x": msg.orientation.x,
            "orientation_y": msg.orientation.y,
            "orientation_z": msg.orientation.z,
            "orientation_w": msg.orientation.w,
            "angular_velocity_x": msg.angular_velocity.x,
            "angular_velocity_y": msg.angular_velocity.y,
            "angular_velocity_z": msg.angular_velocity.z,
            "linear_acceleration_x": msg.linear_acceleration.x,
            "linear_acceleration_y": msg.linear_acceleration.y,
            "linear_acceleration_z": msg.linear_acceleration.z,
        }

        with self.imu_lock:
            if self.closing:
                return
            self.latest_imu = latest
            self.imu_raw_writer.writerow(row)
            self.imu_raw_count += 1
            if self.imu_flush_every > 0 and self.imu_raw_count % self.imu_flush_every == 0:
                self.imu_raw_handle.flush()

    def callback_temperature(self, msg):
        ros_time = rospy.Time.now().to_sec()
        with self.imu_lock:
            if self.closing:
                return
            self.latest_temperature = float(msg.data)
            self.temperature_raw_writer.writerow([f"{ros_time:.9f}", self.latest_temperature])
            self.temperature_raw_count += 1
            if self.imu_flush_every > 0 and self.temperature_raw_count % self.imu_flush_every == 0:
                self.temperature_raw_handle.flush()

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
        with self.imu_lock:
            self.closing = True
        if hasattr(self, "video_writers"):
            for writer in self.video_writers.values():
                writer.release()
        if hasattr(self, "vel_csv_handle") and not self.vel_csv_handle.closed:
            self.vel_csv_handle.close()
        if hasattr(self, "video_index_handle") and not self.video_index_handle.closed:
            self.video_index_handle.close()
        if hasattr(self, "imu_raw_handle") and not self.imu_raw_handle.closed:
            self.imu_raw_handle.close()
        if hasattr(self, "imu_sample_index_handle") and not self.imu_sample_index_handle.closed:
            self.imu_sample_index_handle.close()
        if hasattr(self, "temperature_raw_handle") and not self.temperature_raw_handle.closed:
            self.temperature_raw_handle.close()

    def write_imu_sample_index(self, episode, sample_time):
        with self.imu_lock:
            latest_imu = None if self.latest_imu is None else dict(self.latest_imu)
            latest_temperature = self.latest_temperature

        if latest_imu is None:
            now = time.time()
            if now - self.last_imu_warning_time > 2.0:
                rospy.logwarn("No IMU message received yet on %s", self.imu_topic)
                self.last_imu_warning_time = now
            nan = float("nan")
            self.imu_sample_index_writer.writerow([
                episode,
                f"{sample_time:.9f}",
                nan,
                nan,
                nan,
                "",
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                latest_temperature,
            ])
            self.imu_sample_index_handle.flush()
            return

        stamp_for_age = latest_imu["header_stamp"]
        if stamp_for_age <= 0.0:
            stamp_for_age = latest_imu["ros_time"]
        imu_age = sample_time - stamp_for_age
        self.imu_sample_index_writer.writerow([
            episode,
            f"{sample_time:.9f}",
            f"{latest_imu['header_stamp']:.9f}",
            f"{latest_imu['ros_time']:.9f}",
            f"{imu_age:.9f}",
            latest_imu["frame_id"],
            latest_imu["angular_velocity_x"],
            latest_imu["angular_velocity_y"],
            latest_imu["angular_velocity_z"],
            latest_imu["linear_acceleration_x"],
            latest_imu["linear_acceleration_y"],
            latest_imu["linear_acceleration_z"],
            latest_imu["orientation_x"],
            latest_imu["orientation_y"],
            latest_imu["orientation_z"],
            latest_imu["orientation_w"],
            latest_temperature,
        ])
        self.imu_sample_index_handle.flush()

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
            sample_time = rospy.Time.now().to_sec()

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
                f"{sample_time:.6f}",
                "center.mp4",
                "left.mp4",
                "right.mp4",
                round(target_action, 4)
            ])
            self.video_index_handle.flush()
            self.write_imu_sample_index(self.episode, sample_time)

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
