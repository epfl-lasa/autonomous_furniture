#!/bin/bash

# it: Do iterative or non-iterative terminal
docker run \
		-it \
		-e DISPLAY=$DISPLAY \
		-h $HOSTNAME \
		--net host \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $HOME/.Xauthority:/home/ros/.Xauthority \
		-v "$(pwd)"/../../autonomous_furniture/:/home/ros/ros2_ws/src/autonomous_furniture:rw\
		-v /home/lukas/Code/various_tools/vartools:/home/ros/python/various_tools/vartools:rw\
		-v /home/lukas/Code/dynamic_obstacle_avoidance/dynamic_obstacle_avoidance:/home/ros/python/dynamic_obstacle_avoidance/dynamic_obstacle_avoidance:rw\
		-v /home/lukas/Code/nonlinear_obstacle_avoidance/nonlinear_avoidance:/home/ros/python/nonlinear_obstacle_avoidance/nonlinear_avoidance:rw\
		-v /home/lukas/Code/nonlinear_obstacle_avoidance/media:/home/ros/ros2_ws/src/autonomous_furniture/scripts/qolo_visualizations/media:rw\
		ros2_autonomous_furniture

# Temporary store media for nice plots...

# Local libraries
# -v ~/Code/dynamic_obstacle_avoidance/dynamic_obstacle_avoidance:/home/ros/python/dynamic_obstacle_avoidance/dynamic_obstacle_avoidance\
	# -v ~/Code/various_tools/vartools:/home/ros/python/various_tools/vartools\
	
# Alternative mounting?!
# --mount type=bind,source="$(pwd)"/visualization,target=/home/ros/rviz \

# Change user to root
# -u root \

# Copy specify file
# -v "$(pwd)"/docker-rviz/qolo_env.sh:/home/ros/qolo_env.sh\


# Run with specific user
# -u root \


