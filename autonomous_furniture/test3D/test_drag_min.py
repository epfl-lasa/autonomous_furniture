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
    obstacle_environment_with_drag = ObstacleContainer()
    obstacle_environment_nodrag = ObstacleContainer()

    # control_points for the cuboid
    control_points = np.array([[0.6, 0], [-0.6, 0]])

    # , orientation = 1.6) Goal of the CuboidXd
    # , orientation = 1.6) Goal of the CuboidXd
    start1 = ObjectPose(position=np.array([1, 3]), orientation=np.pi / 2)
    goal1 = ObjectPose(position=np.array([6, 3]), orientation=np.pi / 2)

    table_shape_with_drag = CuboidXd(
        axes_length=[max_ax_len, min_ax_len],
        center_position=start1.position,
        margin_absolut=1,
        orientation=start1.orientation,
        tail_effect=False,
    )

    goal2 = ObjectPose(position=np.array([-1, 3]), orientation=np.pi / 2)
    table_shape2_with_drag = CuboidXd(
        axes_length=[max_ax_len, min_ax_len],
        center_position=goal2.position,
        margin_absolut=1,
        orientation=goal2.orientation,
        tail_effect=False,
    )

    table_shape_nodrag = CuboidXd(
        axes_length=[max_ax_len, min_ax_len],
        center_position=start1.position,
        margin_absolut=1,
        orientation=start1.orientation,
        tail_effect=False,
    )

    table_shape2_nodrag = CuboidXd(
        axes_length=[max_ax_len, min_ax_len],
        center_position=goal2.position,
        margin_absolut=1,
        orientation=goal2.orientation,
        tail_effect=False,
    )

    my_furniture_with_drag = [
        Furniture3D(
            shape_list=[table_shape_with_drag],
            obstacle_environment=obstacle_environment_with_drag,
            control_points=control_points,
            goal_pose=goal1,
            name="fur",
        ),
        Furniture3D(
            shape_list=[table_shape2_with_drag],
            obstacle_environment=obstacle_environment_with_drag,
            control_points=control_points,
            goal_pose=goal2,
            name="fur",
            static=True,
        ),
    ]

    my_furniture_no_drag = [
        Furniture3D(
            shape_list=[table_shape_nodrag],
            obstacle_environment=obstacle_environment_nodrag,
            control_points=control_points,
            goal_pose=goal1,
            name="fur",
        ),
        Furniture3D(
            shape_list=[table_shape2_nodrag],
            obstacle_environment=obstacle_environment_nodrag,
            control_points=control_points,
            goal_pose=goal2,
            name="fur",
            static=True,
        ),
    ]

    # Furniture(shape=table_shape, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal, priority_value=1, name="fur")]

    my_animation_with_drag = DynamicalSystemAnimation3D(
        it_max=200,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )
    my_animation_with_drag.setup(
        layer_list=[my_furniture_with_drag],
        x_lim=[-3, 8],
        y_lim=[-2, 8],
        mini_drag="dragdist",
        version="v2",
        safety_module=True,
        emergency_stop=True,
    )

    my_animation_nodrag = DynamicalSystemAnimation3D(
        it_max=200,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )
    my_animation_nodrag.setup(
        layer_list=[my_furniture_no_drag],
        x_lim=[-3, 8],
        y_lim=[-2, 8],
        mini_drag="nodrag",
        version="v2",
        safety_module=True,
        emergency_stop=True,
    )

    if visualize:
        my_animation_with_drag.run(save_animation=args.rec)
        my_animation_nodrag.run(save_animation=args.rec)

    # Check Dynamic Agent

    my_furniture_with_drag[0].update_velocity(
        mini_drag=my_animation_with_drag.mini_drag,
        version=my_animation_with_drag.version,
        emergency_stop=my_animation_with_drag.emergency_stop,
        safety_module=my_animation_with_drag.safety_module,
        time_step=my_animation_with_drag.dt_simulation,
    )

    my_furniture_no_drag[0].update_velocity(
        mini_drag=my_animation_nodrag.mini_drag,
        version=my_animation_nodrag.version,
        emergency_stop=my_animation_nodrag.emergency_stop,
        safety_module=my_animation_nodrag.safety_module,
        time_step=my_animation_nodrag.dt_simulation,
    )

    assert my_furniture_with_drag[0].angular_velocity < 0, "Expected rotate negatively"
    assert np.linalg.norm(my_furniture_with_drag[0].angular_velocity) > np.linalg.norm(
        my_furniture_no_drag[0].angular_velocity
    ), "Expected min drag rotates faster than no drag"


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    test(visualize=False)
