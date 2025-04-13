for i in `seq 1`
do
  roslaunch nav_cloning nav_cloning_all.launch script:=nav_cloning_node.py mode:=selected_training world_name:=Tsudanuma_2-3.world map_file:=cit_3f_map waypoints_file:=cit3f_way_fix.yaml dist_err:=0.8 initial_pose_x:=-5.0 initial_pose_y:=7.7 initial_pose_a:=3.14 use_waypoint_nav:=false robot_x:=-5.0 robot_y:=7.7 robot_Y:=3.14
  sleep 10
done

