from autonomous_furniture.agent import Furniture, Person
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
import numpy as np


furniture_dic = {
    "2x1": CuboidXd(
        axes_length=[2, 1],
        center_position=np.array([-2, 1]),
        margin_absolut=0.6,
        orientation=np.pi / 2,
        tail_effect=False,
    ),
    # obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal, priority_value=1)
}
