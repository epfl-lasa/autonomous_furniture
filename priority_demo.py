import numpy as np

import matplotlib.pyplot as plt
from scipy import rand
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from autonomous_furniture.dynamical_system_animation import DynamicalSystemAnimation

from autonomous_furniture.agent import Furniture, Person

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--rec", action="store", default=False, help="Record flag")
parser.add_argument(
    "--name", action="store", default="recording", help="Name of the simulation"
)
args = parser.parse_args()


def priority_demo():
    axis = [2.4, 1.1]
    max_ax_len = max(axis)
    min_ax_len = min(axis)

    # List of environment shared by all the furniture/agent
    obstacle_environment = ObstacleContainer()

    # control_points for the cuboid
    control_points = np.array([[0.6, 0], [-0.6, 0]])

    # , orientation = 1.6) Goal of the CuboidXd
    # , orientation = 1.6) Goal of the CuboidXd
    goal = ObjectPose(position=np.array([6, 2.5]), orientation=np.pi / 2)

    table_shape = CuboidXd(
        axes_length=[max_ax_len, min_ax_len],
        center_position=np.array([-1, 2.5]),
        margin_absolut=1,
        orientation=np.pi / 2,
        tail_effect=False,
    )

    goal2 = ObjectPose(position=np.array([2.5, 5.5]), orientation=np.pi / 2)

    table_shape2 = CuboidXd(
        axes_length=[1, 1],
        center_position=np.array([2, 0]),
        margin_absolut=1,
        orientation=goal.orientation,
        tail_effect=False,
    )

    my_furniture = [
        Person(
            center_position=[2, 5.5],
            radius=0.8,
            obstacle_environment=obstacle_environment,
            goal_pose=ObjectPose(position=np.array([2, 5.5]), orientation=np.pi / 2),
            priority_value=1,
            margin=1,
            static=True,
            name="pers",
        ),
        Furniture(
            shape=table_shape,
            obstacle_environment=obstacle_environment,
            control_points=control_points,
            goal_pose=goal,
            priority_value=1,
            name="fur",
        ),
        Furniture(
            shape=table_shape2,
            obstacle_environment=obstacle_environment,
            control_points=np.array([[0, 0], [0, 0]]),
            goal_pose=ObjectPose(position=np.array([2, 0]), orientation=np.pi / 2),
            priority_value=1,
            static=True,
            name="static",
        ),
    ]

    my_animation = DynamicalSystemAnimation(
        it_max=200,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )

    my_animation.setup(
        obstacle_environment,
        agent=my_furniture,
        x_lim=[-3, 8],
        y_lim=[-2, 7],
        version="v2",
        mini_drag="dragdist",
    )

    my_animation.run(save_animation=args.rec)
    my_animation.logs(len(my_furniture))


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    priority_demo()
