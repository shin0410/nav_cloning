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
    config_path = os.path.join(script_dir, "..", "config", filename)  # configディレクトリ内を想定
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
config = load_config()

PC_USER_NAME = config["pc_user_name"]
WS_NAME = config["ws_name"]
SPEED = config["speed"]
DURATION = float(config["duration"])
TIME = config["time"]
EPOCH = int(config["epoch"])
LOAD_MODEL = config["load_model"]




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
        self.load_model_path = "/home/" + PC_USER_NAME + "/ws/" + WS_NAME + "/src/nav_cloning/data/" + self.dir_name + "/model/" + str(self.epoch) + "/" + self.load_model


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