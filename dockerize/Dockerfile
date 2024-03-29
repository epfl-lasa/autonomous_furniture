ARG ROS_DISTRO=humble
FROM osrf/ros:${ROS_DISTRO}-desktop
# FROM osrf/ros:${ROS_DISTRO}-desktop
# FROM osrf/ros:${ROS_DISTRO}-ros-base

ARG USER=ros
ARG HOME=/home/${USER}
# ENV HOME /home/ros

ARG HOST_GID=1000
ARG HOST_UID=1000

# INSTALL NECESSARY PACKAGES
RUN apt update \
    && apt install -y \
    tmux \
    nano \
    vim-python-jedi

# ROS dependent environment
RUN apt update \
    && apt install -y \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-xacro

# RUN apt-get install -y python3.10
RUN apt-get install -y python3-pip

RUN apt-get install python3-tk
RUN python3 -m pip install pyqt5

# Make python 3.10 the default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN addgroup --gid ${HOST_GID} ${USER}
RUN adduser --gecos "ROS User" --uid ${HOST_UID} --gid ${HOST_GID} ${USER} && yes | passwd ${USER}
RUN usermod -a -G dialout ${USER}
RUN echo "${USER} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/99_aptget
RUN chmod 0440 /etc/sudoers.d/99_aptget && chown root:root /etc/sudoers.d/99_aptget

# RUN groupadd -g  ${HOST_GID} ros
# RUN useradd -d ${HOME} -s /bin/bash -m ros -u ${HOST_UID} -g ${HOST_GID}

# Install Python-Libraries
USER ros
RUN mkdir -p ${HOME}/python
WORKDIR ${HOME}/python

# Enforce partial rebuild (temp fix -> remove in the future)
RUN git clone -b main --single-branch https://github.com/hubernikus/various_tools.git
RUN git clone -b main --single-branch https://github.com/epfl-lasa/dynamic_obstacle_avoidance

# USER root
WORKDIR ${HOME}/python/various_tools
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install --editable .

# Dynamic Obstacle Avoidance Library [Only minor changes]
WORKDIR ${HOME}/python/dynamic_obstacle_avoidance
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install --editable .

# Install Nonlinear Dynamics
WORKDIR ${HOME}/python
RUN git clone -b main https://github.com/hubernikus/nonlinear_obstacle_avoidance.git
WORKDIR ${HOME}/python/nonlinear_obstacle_avoidance
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install --editable .
# pip install git+git://github.com/hubernikus/nonlinear_obstacle_avoidance.git

# ROS-ENV setup
USER ros
RUN mkdir -p ${HOME}/ros2_ws/src/autonomous_furniture
RUN rosdep update

# # Download dependencies
WORKDIR ${HOME}/ros2_ws/src
RUN git clone -b main https://github.com/epfl-lasa/furniture_descriptions.git
# # RUN git clone -b main https://github.com/epfl-lasa/autonomous_furniture

# Copy local directory Set up work directory
COPY autonomous_furniture autonomous_furniture

# WORKDIR ${HOME}/ros2_ws/src/autonomous_furniture
WORKDIR ${HOME}/ros2_ws/src/autonomous_furniture
RUN python3 -m pip install -r requirements.txt
# # TODO: remove requirements.txt files

# Setup Colcon
# RUN mkdir -p ${COLCON_HOME}
# COPY --chown=${USER}:${USER} ./config/colcon ${COLCON_HOME}
# RUN /bin/bash ${COLCON_HOME}/setup.sh

# Setup ROS2 workspace
USER ros
WORKDIR ${HOME}/ros2_ws

RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; colcon build --symlink-install"

# # Update bash file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash; \
	source ~/ros2_ws/install/setup.bash" | cat - ${HOME}/.bashrc > tmp && mv tmp ${HOME}/.bashrc
# RUN echo "source /opt/ros/humble/setup.bash" >> ${HOME}/.bashrc

# Set root password [does not work for now ?!]
# USER root
# RUN echo 'root:root' | chpasswd
# RUN echo 'Docker!' | passwd --stdin root 

# For development / testing purposes
RUN python3 -m pip install -U pytest
RUN sudo apt install python-is-python3

# USER ros
# USER root

# WORKDIR ${HOME}/ros2_ws/src/autonomous_furniture
WORKDIR ${HOME}/ros2_ws/src/autonomous_furniture/scripts/qolo_visualizations
# COPY setup_ros_env.sh ${HOME}/setup_ros_env.sh
# RUN echo "bash ~/setup_ros_env.sh" >> ~/.bashrc

ENTRYPOINT tmux
