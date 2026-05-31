#!/usr/bin/env python3

import numpy as np
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from net import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
import os
import yaml

def load_config(filename="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_cfg = os.environ.get("NAV_CONFIG")
    if env_cfg:
        config_path = os.path.abspath(os.path.expanduser(env_cfg))
    else:
        config_path = os.path.join(script_dir, "..", "config", filename)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
config = load_config()

PC_USER_NAME = str(config.get("pc_user_name", "shin"))
WS_NAME = str(config.get("ws_name", "challenge_ws"))
SPEED = str(config.get("speed", "0.8"))
DURATION = float(config.get("duration", 0.1))
TIME = config.get("time")
EPOCH = int(config.get("epoch", 100))
LOAD_MODEL = config.get("load_model")
LOAD_MODEL_PATH_CFG = config.get("load_model_path")




class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.action_num = 1
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera_center/usb_cam/image_raw", Image, self.callback)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.episode = 0
        self.vel = Twist()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.speed = float(SPEED)
        self.dir_name = TIME
        self.epoch = EPOCH
        self.load_model = LOAD_MODEL
        self.load_model_path = self.resolve_model_path()

    def resolve_model_path(self):
        # 1) 最優先: 環境変数で絶対パス指定
        env_model = os.environ.get("NAV_MODEL_PATH")
        if env_model:
            p = os.path.abspath(os.path.expanduser(env_model))
            if os.path.isfile(p):
                return p
            raise FileNotFoundError(f"NAV_MODEL_PATH not found: {p}")

        # 2) config に load_model_path があれば優先
        if LOAD_MODEL_PATH_CFG:
            p = os.path.abspath(os.path.expanduser(str(LOAD_MODEL_PATH_CFG)))
            if os.path.isfile(p):
                return p
            raise FileNotFoundError(f"load_model_path not found: {p}")

        # 3) 従来互換: time/epoch/load_model から組み立て
        if self.dir_name and self.load_model:
            p = f"/home/{PC_USER_NAME}/{WS_NAME}/src/nav_cloning/data/{self.dir_name}/model/{self.epoch}/{self.load_model}"
            if os.path.isfile(p):
                return p
            raise FileNotFoundError(
                f"model not found: {p}\n"
                "Set NAV_MODEL_PATH or load_model_path in config to use an arbitrary model file."
            )

        raise KeyError(
            "config keys are insufficient. Need one of:\n"
            "  - NAV_MODEL_PATH env var\n"
            "  - load_model_path in config\n"
            "  - (time + epoch + load_model) in config"
        )


    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
       
        img = resize(self.cv_image, (48, 64), mode='constant')

        if self.episode == 0:
            print(f"[INFO] load model: {self.load_model_path}")
            self.dl.load(self.load_model_path)

        target_action = self.dl.act(img)
        self.vel.linear.x = self.speed
        self.vel.angular.z = target_action
        self.nav_pub.publish(self.vel)
        self.episode += 1
        print(str(self.episode) + ", test, angle:" + str(target_action))

        # cv2.imshow("Image", self.cv_image)
        # cv2.imshow("Resized Image", img)
        # cv2.waitKey(1)


if __name__ == '__main__':
    rg = nav_cloning_node()
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
