# Autonomous Furniture

---
This repository contains an implementation of dynamic obstacle avoidance algorithm for autonomous furniture as developped in [1], [2], and [3]
---

## Setup using Docker
It is advised to use ubuntu, as the docker virtual images have the best compatibility with it.
For this check
https://docs.docker.com/engine/install/ubuntu/

Once installed, build the docker:

``` shell
bash docker-build.sh
```

And finally run the docker in a command line:
``` shell
bash docker-run.sh
```

Now, check out files in the `scripts` folder, which can be run. For example
``` shell
python3 corner_case.py
```

Make sure, the visualization is enabled (check issues below).

The docker-run file is setup to automatically update local changes.

### Test / Develop
If you want to make sure that everything runs correctly, execute following command in the autonomous_furniture folder:
``` shell
pytest
```

## Setup without ROS
Following libraries are needed and need to be installed:

``` shell
https://github.com/epfl-lasa/dynamic_obstacle_avoidance
```

## Run Rviz-Environment
To run the example `assistive_environment.launch.py`
``` shell
ros2 launch autonomous_furniture assistive_environment.launch.py
```

ros2 launch autonomous_furniture dense_environmen.launch.py

Sometimes, the rviz does not open the correct configuration, in that case open it:
`CTRL+o` -> go to `/home/ros/ros2_ws/autonomous_furniture/rviz/assistive_environment.rviz`


Launch the obstacle avoidance simulation (from the autonomous_furniture folder):
``` shell
python3 furniture_publishers/assistive_environment.py
```

Run the indoor, dense environment:

``` shell
python3 furniture_publishers/dense_environment.py
ros2 launch autonomous_furniture dense_environment.launch.py
```

### Build and Run
If you need to build AND run
``` shell
colcon build --symlink-install && ros2 launch autonomous_furniture assistive_environment.launch.py
```

## Setup on Host / Main computer
Requirements: python3.10, ROS2 (prefereably Humble)

To setup the repository create "workspace/src" directory then clone the repository into the "src" directory:
```shell
cd workspace/src
```

```shell
git clone --recurse-submodules git@github.com:epfl-lasa/autonomous_furniture.git
git clone git@github.com:epfl-lasa/furniture_descriptions.git
```
(Make sure that the submodule dynamic_obstacle_avoidance is in autonomous_furniture/libraries, and various_tools is in autonomous_furniture/libraries/dynamic_obstacle_avoidance/libraries)

if you forgot to add the "--recurse-submodules" flag, you can fetch the submodules with:
```
git submodule update --init --recursive
```
Be sure your directory structure looks like this:
```bash
.
└── workspace
    └──src  
        └── furniture_descriptions
        └── autonomous_furniture
             └── libraries
                └── dynamic_obstacle_avoidance
                    └── libraries
                        └── various_tools
```
if not ROS2 will fail to build and through mising package errors.

### ROS2 Setup
For the 3D visualization to work properly, a working version of ROS2 is needed. It has been tested on foxy, galactic, and humble. The required python version is 3.10, so in case you decide not to use Humble, you will need to install Foxy or Galactic from source defining a python version >=3.10.
With the basic version of ROS2 there are a few missing packages needed to properly run the code.
First install rosdep:
```shell
sudo apt install python3-rosdep2
rosdep update
```
Install colcon tu build the environment:
https://colcon.readthedocs.io/en/released/user/installation.html

Add the remaining missing packages.
For Foxy:
```shell
sudo apt install ros-foxy-joint-state-publisher-gui
sudo apt install ros-foxy-joint-state-publisher
sudo apt install ros-foxy-xacro
sudo apt install ros-foxy-tf-transformations
```
For Humble:
```shell
sudo apt install ros-humble-joint-state-publisher-gui
sudo apt install ros-humble-joint-state-publisher
sudo apt install ros-humble-xacro
sudo apt install ros-humble-tf-transformations
```
### Python Install
Go to file directory, create a virtual environment containing a python version >= 3.10 and activate it:
```shell
cd autonomous_furniture
virtualenv --python=python3.10 venv
source venv/bin/activate
```
Setup the obstacle avoidance and various tools libraries:
```shell
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
cd libraries/dynamic_obstacle_avoidance/
pip install -r requirements.txt
pip install -e .
cd libraries/various_tools/
pip install -r requirements.txt
pip install -e .
```

### ROS2 install
Build the ROS2 packages outside of the virtual environment:
```shell
deactivate
cd workspace
colcon build --symlink-install
```
Don't forget to source the package:
```shell
. install/setup.bash
```

Build and run
<!-- colcon build --symlink-install && ros2 launch autonomous_furniture example_launch.launch.py -->
``` shell
colcon build --symlink-install && ros2 launch autonomous_furniture assistive_environment.launch.py
```


## Running 3D Environment
To run the 3D env you have to open 2 terminal, 1 will run the launch file for RViz and the subscriber and 1 will run the publisher.
In the first terminal install and source de ROS2 packages, then run:
```shell
ros2 launch autonomous_furniture <env of choice>.launch.py
```
Replace "env of choice" with any of the available environment in the launch directory.
In the second terminal go to the autonomous_furniture directory, start the virtual environment, and run the corresponding python publisher:
```shell
cd workspace/src/autonomous_furniture/
source venv/bin/activate
python furniture_publisher/<env of choice>_state_publisher.py
```

### Publishers
assistive_environment_state_publisher.py
hospital_environment_state_publisher.py

The publishers are compatible with the current launch files

### 2D Examples and Recorder
assistive_environment.py
hospital_environment.py

record_hospital_environment.py

## Known Errors
You may encounter as few errors at first a few of those are the following:
* If the simulation crashed after a while with "Exception: Normal direciton is not pointing along reference.-> change the sign of the inequality", that means something got run over, might need to adjust the margins
* ROS2 doesn't build --> check file struct mentioned in Setup
* Missing packages when launching ROS2 launch files --> check ROS2 Setup
* Missing packages when launching python files --> this may be due to detached submodules
  * Reattach submodules:
```shell
cd workspace/src/autonomous_furniture/autonomous_furniture/libraries/dynamic_obstacle_avoidance/
git checkout feat/mobilerobot
cd lib/various_tools/
git pull origin main
```
### Errors when using ROS Foxy and Ubuntu 20.04
* If the command for upgrading pip results in an error, try to upgrade pip with the following command:
```shell
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
```
* Cannot import rclpy: this errors come from the fact that all debian packages in ROS Foxy rely on python3.8. You can solve it by building ROS Foxy from source and define its python version as 3.10.


### Docker - Cannot connect to host
Add the xhost set to your `.profile` setup file by running:
``` shell
echo "xhost + local:" >> ~/.profile
```


**References**     
> [1] Lukas Huber, Aude Billard, and Jean-Jacques E. Slotine. "Avoidance of Convex and Concave Obstacles with Convergence ensured through Contraction." IEEE Robotics and Automation Letters (2019).
> 
> [2] Lukas Huber, Jean-Jacques E. Slotine, and Aude Billard. "Avoiding Dense and Dynamic Obstacles in Enclosed Spaces: Application to Moving in a Simulated Crowd." arXiv preprint arXiv:2105.11743 (2021).
> 
> [3] Federico M. Conzelmann, Lukas Huber, Diego Paez-Granados, Anastasia Bolotnikova, Auke Ijspeert, and Aude Billard "A Dynamical System Approach to Decentralized Collision-free Autonomous Coordination of a Mobile Assistive Furniture Swarm"
