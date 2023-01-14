import numpy as np

import matplotlib.pyplot as plt
from scipy import rand
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization import plot_obstacles
from autonomous_furniture.dynamical_system_animation import DynamicalSystemAnimation


from autonomous_furniture.agent import BaseAgent, Furniture, Person

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--rec", action="store", default=False, help="Record flag")
parser.add_argument(
    "--name", action="store", default="recording", help="Name of the simulation"
)
args = parser.parse_args()


def corner_case():
    axis = [2.4, 1.1]
    max_ax_len = max(axis)
    min_ax_len = min(axis)

    # List of environment shared by all the furniture/agent
    obstacle_environment = ObstacleContainer()

    # control_points for the cuboid
    control_points = np.array([[0.6, 0], [-0.6, 0]])

    # , orientation = 1.6) Goal of the CuboidXd
    # , orientation = 1.6) Goal of the CuboidXd
    goal = ObjectPose(position=np.array([6, 2]), orientation=np.pi / 2)

    table_shape = CuboidXd(
        axes_length=[max_ax_len, min_ax_len],
        center_position=np.array([-2, 2]),
        margin_absolut=1,
        orientation=np.pi / 2,
        tail_effect=False,
    )

    my_furniture = [
        Person(
            center_position=[2, 2.2],
            radius=0.8,
            obstacle_environment=obstacle_environment,
            goal_pose=goal,
            priority_value=1,
            margin=1,
            static=True,
            name="elder",
        ),
        Furniture(
            shape=table_shape,
            obstacle_environment=obstacle_environment,
            control_points=control_points,
            goal_pose=goal,
            priority_value=1,
            name="elder",
        ),
    ]
    my_animation = DynamicalSystemAnimation(
        it_max=450,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )

    my_animation.setup(
        obstacle_environment, agent=my_furniture, x_lim=[-3, 8], y_lim=[-2, 7]
    )

    version = "v2"
    do_drag = "nodrag"

    my_animation.run(save_animation=args.rec, mini_drag=do_drag, version=version)
    my_animation.logs(len(my_furniture), do_drag, version=version)


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    corner_case()
