import argparse
import numpy as np

import matplotlib.pyplot as plt
from scipy import rand

from vartools.states import Pose
from vartools.states import ObjectPose
from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from autonomous_furniture.agent3D import Furniture3D
from autonomous_furniture.dynamical_system_animation3D import DynamicalSystemAnimation3D
from autonomous_furniture.furniture_creators import assign_agent_virtual_drag

parser = argparse.ArgumentParser()

parser.add_argument("--rec", action="store", default=False, help="Record flag")
parser.add_argument(
    "--name", action="store", default="recording", help="Name of the simulation"
)
args = parser.parse_args()


def test(visualize=False):
    axis = [2.4, 1.1]
    max_ax_len = max(axis)
    min_ax_len = min(axis)

    # List of environment shared by all the furniture/agent
    obstacle_environment_with_decoupling = ObstacleContainer()
    obstacle_environment_no_decoupling = ObstacleContainer()

    # control_points for the cuboid
    control_points = np.array([[0.6, 0], [-0.6, 0]])

    # , orientation = 1.6) Goal of the CuboidXd
    # , orientation = 1.6) Goal of the CuboidXd
    goal = ObjectPose(position=np.array([6, 3]), orientation=0)

    table_shape_with_decoupling = CuboidXd(
        axes_length=[max_ax_len, min_ax_len],
        center_position=np.array([1, 3]),
        margin_absolut=1,
        orientation=np.pi / 2,
        tail_effect=False,
    )

    table_shape_no_decoupling = CuboidXd(
        axes_length=[max_ax_len, min_ax_len],
        center_position=np.array([1, 3]),
        margin_absolut=1,
        orientation=np.pi / 2,
        tail_effect=False,
    )

    my_furniture_with_decoupling = assign_agent_virtual_drag([
        Furniture3D(
            shape_list=[table_shape_with_decoupling],
            obstacle_environment=obstacle_environment_with_decoupling,
            control_points=control_points,
            goal_pose=goal,
            name="fur",
        ),
    ])

    my_furniture_no_decoupling = assign_agent_virtual_drag([
        Furniture3D(
            shape_list=[table_shape_no_decoupling],
            obstacle_environment=obstacle_environment_no_decoupling,
            control_points=control_points,
            goal_pose=goal,
            name="fur",
        ),
    ])

    # Furniture(shape=table_shape, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal, priority_value=1, name="fur")]

    my_animation_with_decoupling = DynamicalSystemAnimation3D(
        it_max=1000,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )
    my_animation_with_decoupling.setup(
        layer_list=[my_furniture_with_decoupling],
        x_lim=[-3, 8],
        y_lim=[-2, 8],
        mini_drag="nodrag",
        version="v2",
        safety_module=False,
        emergency_stop=False,
    )

    my_animation_no_decoupling = DynamicalSystemAnimation3D(
        it_max=1000,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )
    my_animation_no_decoupling.setup(
        layer_list=[my_furniture_no_decoupling],
        x_lim=[-3, 8],
        y_lim=[-2, 8],
        mini_drag="nodrag",
        version="v1",
        safety_module=False,
        emergency_stop=False,
    )

    if visualize:
        my_animation_with_decoupling.run(save_animation=args.rec)
        my_animation_no_decoupling.run(save_animation=args.rec)

    # Check Dynamic Agent

    my_furniture_with_decoupling[0].update_velocity(
        mini_drag=my_animation_with_decoupling.mini_drag,
        version=my_animation_with_decoupling.version,
        emergency_stop=my_animation_with_decoupling.emergency_stop,
        safety_module=my_animation_with_decoupling.safety_module,
        time_step=my_animation_with_decoupling.dt_simulation,
    )

    my_furniture_no_decoupling[0].update_velocity(
        mini_drag=my_animation_no_decoupling.mini_drag,
        version=my_animation_no_decoupling.version,
        emergency_stop=my_animation_no_decoupling.emergency_stop,
        safety_module=my_animation_no_decoupling.safety_module,
        time_step=my_animation_no_decoupling.dt_simulation,
    )

    assert (
        my_furniture_with_decoupling[0].angular_velocity < 0
    ), "Expected rotate negatively"
    assert np.linalg.norm(
        my_furniture_with_decoupling[0].angular_velocity
    ) > np.linalg.norm(
        my_furniture_no_decoupling[0].angular_velocity
    ), "Expected soft decoupling rotates faster than no drag"


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    test(visualize=False)
