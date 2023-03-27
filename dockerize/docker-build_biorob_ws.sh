#!/bin/bash
cd .. && sudo docker build -t ros2_autonomous_furniture -f dockerize/Dockerfile . && cd dockerize

# install ros package

# TO run:
# 
# $ docker run -it --rm ros_with_rviz
 
