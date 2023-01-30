ARG ROS_DISTRO=humble
FROM osrf/ros:humble-desktop
# FROM osrf/ros:${ROS_DISTRO}-desktop
# FROM osrf/ros:${ROS_DISTRO}-ros-base

ARG HOME=/home/ros

# INSTALL NECESSARY PACKAGES
RUN apt update \
	&& apt install -y \
	tmux \
    nano \
	vim-python-jedi

# ROS dependent environment
RUN apt install \
	ros-${ROS_DISTRO}-joint-state-publisher-gui \
	ros-${ROS_DISTRO}-robot-state-publisher \
	ros-${ROS_DISTRO}-rviz2

# RUN apt-get install -y python3.10
RUN apt-get install -y python3-pip

# RUN	apt clean
# Make python 3.9 the default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create a user called ROS
RUN groupadd -g 1000 ros
RUN useradd -d /home/ros -s /bin/bash -m ros -u 1000 -g 1000

# Install Python-Libraries
USER ros
RUN mkdir -p ${HOME}/python
WORKDIR ${HOME}/python

RUN git clone -b main --single-branch https://github.com/hubernikus/various_tools.git
RUN git clone -b main --single-branch https://github.com/epfl-lasa/dynamic_obstacle_avoidance

# Enforce partial rebuild (temp fix -> remove in the future)
RUN echo 1

# USER root
WORKDIR ${HOME}/python/various_tools
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install --editable .

# Dynamic Obstacle Avoidance Library [Only minor changes]
WORKDIR ${HOME}/python/dynamic_obstacle_avoidance
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install --editable .

# ROS-ENV setup
USER ros
ENV HOME /home/ros
# USER ros

RUN mkdir -p ${HOME}/ros2_ws/src
RUN rosdep update

WORKDIR ${HOME}/ros2_ws/src

COPY autonomous_furniture autonomous_furniture
COPY objects_descriptions objects_descriptions

# Setup Colcon
# RUN mkdir -p ${COLCON_HOME}
# COPY --chown=${USER}:${USER} ./config/colcon ${COLCON_HOME}
# RUN /bin/bash ${COLCON_HOME}/setup.sh

# Setup ROS2 workspace
USER ros
WORKDIR ${HOME}/ros2_ws

RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; colcon build --symlink-install"
# RUN /bin/bash -c '/opt/ros/humble/setup.bash; colcon build --symlink-install'

# Update bash file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash; \
	source ~/ros2_ws/install/setup.bash" | cat - ${HOME}/.bashrc > tmp && mv tmp ${HOME}/.bashrc
# RUN echo "source /opt/ros/humble/setup.bash" >> ${HOME}/.bashrc


# Matplotlib setup
USER root
RUN apt-get install python3-tk
RUN python3 -m pip install pyqt5

USER root
WORKDIR ${HOME}/ros2_ws/src/autonomous_furniture

# COPY setup_ros_env.sh ${HOME}/setup_ros_env.sh
# RUN echo "bash ~/setup_ros_env.sh" >> ~/.bashrc

ENTRYPOINT tmux
