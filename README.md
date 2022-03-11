# Autonomous Furniture

---
This repository contains an implementation of dynamic obstacle avoidance algorithm for autonomous furniture as developped in [1], [2], and [3]
---
Requirements: python, pipenv, ROS2

## Setup
To setup the repository create "workspace/src" directory then clone the repository into the "src" directory:
```shell
cd workspace 
```
```
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
### Python Install
Go to file directory:
```shell
cd src/autonomous_furniture
```
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
python setup.py develop
cd lib/various_tools/
python setup.py develop
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


**References**     
> [1] Huber, Lukas, Aude Billard, and Jean-Jacques E. Slotine. "Avoidance of Convex and Concave Obstacles with Convergence ensured through Contraction." IEEE Robotics and Automation Letters (2019).
> 
> [2] Huber, Lukas, and Slotine Aude Billard. "Avoiding Dense and Dynamic Obstacles in Enclosed Spaces: Application to Moving in a Simulated Crowd." arXiv preprint arXiv:2105.11743 (2021).
> 
> [3] Add reference if my paper gets published
