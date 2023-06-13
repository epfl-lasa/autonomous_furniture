#!/bin/bash
cd ../.. && docker build -t ros2_autonomous_furniture -f autonomous_furniture/dockerize/Dockerfile . && cd autonomous_furniture/dockerize

# DOCKER_BUILDKIT=1 docker build --build-arg HOST_GID=$(id -g) --build-arg HOST_UID=$(id -u)
# install ros package

# TO run:
# 
# $ docker run -it --rm ros_with_rviz
