from test.test_orientation_ctrl import DynamicalSystemAnimation
from autonomous_furniture.agent import Furniture, Person
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from vartools.states import ObjectPose
from dynamic_obstacle_avoidance.containers import ObstacleContainer

import numpy as np

class ScenarioLauncher:
    def __init_(self, nb_sim = 5, nb_furniture = 5, record = False):
        self._nb_furniture = nb_furniture 
        self._nb_sim = nb_sim

    def run(self):
        pass    

def main():
    axis = [2, 1]
    max_ax_len = max(axis)
    min_ax_len = min(axis)

    # List of environment shared by all the furniture/agent
    obstacle_environment = ObstacleContainer()

    # control_points for the cuboid
    control_points = np.array([[0.3, 0], [-0.3, 0]])

    # , orientation = 1.6) Goal of the CuboidXd
    goal = ObjectPose(position=np.array([7, 1]), orientation=np.pi/2)

    table_shape = CuboidXd(axes_length=[max_ax_len, min_ax_len],
                           center_position=np.array([-2, 1]),
                           margin_absolut=0.6,
                           orientation=np.pi/2,
                           tail_effect=False,)

    goal2 = ObjectPose(position=np.array([-2, 0.5]), orientation=np.pi/2)
    table_shape2 = CuboidXd(axes_length=[max_ax_len, min_ax_len],
                            center_position=np.array([7, 0.5]),
                            margin_absolut=0.6,
                            orientation=0,
                            tail_effect=False,)

    my_furniture = [Furniture(shape=table_shape, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal, priority_value=1),
                    Furniture(shape=table_shape2, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal2, priority_value=1)]  # ,    Furniture(shape=table_shape2, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal2)]
if __name__ == "__main__":
    main()
    
