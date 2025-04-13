#!/bin/bash

# パラメータの設定
SCRIPT="nav_cloning_node_augmix.py"
MODE="use_dl_output"
WORLD_NAME="Tsudanuma_2-3.world"
MAP_FILE="cit_3f_map"  
WAYPOINTS_FILE="cit_3f_loop.yaml" 
DIST_ERR="0.8"
INITIAL_POSE_X="-5.0"
INITIAL_POSE_Y="7.7"
INITIAL_POSE_A="3.14"
USE_WAYPOINT_NAV="true"
ROBOT_X="-5.0"
ROBOT_Y="7.7"
ROBOT_YAW="3.14"

# 繰り返し回数（デフォルトは1）
REPEAT_COUNT=${1:-1}

# 指定した回数だけroslaunchを実行
for i in $(seq $REPEAT_COUNT)
do
  roslaunch nav_cloning nav_cloning_all_augmix.launch \
    script:=$SCRIPT \
    mode:=$MODE \
    world_name:=$WORLD_NAME \
    map_file:=$MAP_FILE \
    waypoints_file:=$WAYPOINTS_FILE \
    dist_err:=$DIST_ERR \
    initial_pose_x:=$INITIAL_POSE_X \
    initial_pose_y:=$INITIAL_POSE_Y \
    initial_pose_a:=$INITIAL_POSE_A \
    use_waypoint_nav:=$USE_WAYPOINT_NAV \
    robot_x:=$ROBOT_X \
    robot_y:=$ROBOT_Y \
    robot_yaw:=$ROBOT_YAW

  # 各実行の後に10秒間待機
  sleep 10
done

