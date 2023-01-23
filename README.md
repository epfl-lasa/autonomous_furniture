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

The docker is setup, such that it automatically shares updates local changes.

## Setup without ROS
Following libraries are needed and need to be installed:

``` shell
https://github.com/epfl-lasa/dynamic_obstacle_avoidance
```




## Setup on Host / Main computer
Requirements: python, ROS2

To setup the repository create "workspace/src" directory then clone the repository into the "src" directory:
```shell
cd workspace 
```

```shell
git clone --recurse-submodules git@github.com:epfl-lasa/autonomous_furniture.git ./src
```
(Make sure that the submodule dynamic_obstacle_avoidance is in autonomous_furniture/libraries, and various_tools is in autonomous_furniture/libraries/dynamic_obstacle_avoidance/lib)

if you forgot to add the "--recurse-submodules" flag, you can fetch the submodules with:
```
git submodule update --init --recursive
```
Be sure your directory structure looks like this:
```bash
.
└── workspace
    └──src
        ├── autonomous_furniture
        └── objects_descriptions
```
if not ROS2 will fail to build and through mising package errors.
### ROS2 Setup
For the 3D visualization to work properly, a working version of ROS2 is needed. It has been tested on foxy and galactic.
With the basic version of ROS2 there are a few missing packages needed to properly run the code.
First install rosdep:
```shell
sudo apt install python3-rosdep2
rosdep update
```
Install colcon:
https://colcon.readthedocs.io/en/released/user/installation.html

Add the remaining missing packages:
```shell
sudo apt install ros-<ros2-distro>-joint-state-publisher-gui
sudo apt install ros-<ros2-distro>-joint-state-publisher
sudo apt install ros-<ros2-distro>-xacro
```
### Python Install
Go to file directory:
```shell
cd src/autonomous_furniture
```

## 
Create a virtual environment and install the dependencies from the Pipfile:
```shell
pipenv install
```
This will create a Pipfile.lock and will take some time, so be patient.
Once the virtual environment is setup start it with:
```shell
pipenv shell
```

Setup the obstacle avoidance and various tools libraries:
```shell
cd libraries/dynamic_obstacle_avoidance/
pip install -r requirements.txt
pip install -e .
cd libraries/various_tools/
python setup.py develop
pip install -e .
```

### ROS2 install
Build the ROS2 packages outside of the "pipenv shell":
```shell
cd workspace
colcon build --symlink-install
```
Don't forget to source the package:
```shell
. install/setup.bash
```

## Running 3D Environment
To run the 3D env you have to open 2 terminal, 1 will run the launch file for RViz and the subscriber and 1 will run the publisher.
In the first terminal install and source de ROS2 packages, then run:
```shell
ros2 launch autonomous_furniture <env of choice>.launch.py
```
Replace "env of choice" with any of the available environment in the launch directory.
In the second terminal go to the autonomous_furniture directory and start the pipenv shell:
```shell
cd workspace/src/autonomous_furniture/
pipenv shell
```
The run the corresponding python publisher:
```shell
python furniture_publisher/<env of choice>_state_publisher.py
```

## Running 2D Environment
You can run any of the record files in Scenarios and set the "rec" flag to "False", should be False by default:
```shell
cd workspace/src/autonomous_furniture/
pipenv shell
python Scenarios/rec_anim_<script of choice>.py --rec=<False or True>
```

## Updated Code
I added a furniture class which is more elegant implementation of the smart furniture.
New examples are provided as well as the corresponding ROS2 publishers.

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
cd workspace/src/autonomous_furniture/libraries/dynamic_obstacle_avoidance/
git checkout feat/mobilerobot
cd lib/various_tools/
git pull origin main
```

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
