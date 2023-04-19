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


def test_uneven_priority(visualize=False):
    axis = [2.4, 1.1]
    max_ax_len = max(axis)
    min_ax_len = min(axis)

    # List of environment shared by all the furniture/agent
    obstacle_environment = ObstacleContainer()

    # control_points for the cuboid
    control_points = np.array([[0.6, 0], [-0.6, 0]])

    # , orientation = 1.6) Goal of the CuboidXd
    # , orientation = 1.6) Goal of the CuboidXd
    start1 = ObjectPose(position=np.array([-1, 3]), orientation=np.pi / 2)
    goal1 = ObjectPose(position=np.array([6, 3]), orientation=np.pi / 2)
    table_shape = CuboidXd(
        axes_length=[max_ax_len, min_ax_len],
        center_position=start1.position,
        margin_absolut=1,
        orientation=start1.orientation,
        tail_effect=False,
    )

    goal2 = ObjectPose(position=np.array([2, 6]), orientation=np.pi / 2)
    table_shape2 = CuboidXd(
        axes_length=[1, 1],
        center_position=goal2.position,
        margin_absolut=1,
        orientation=goal2.orientation,
        tail_effect=False,
    )

    goal3 = ObjectPose(position=np.array([2, 0]), orientation=np.pi / 2)
    table_shape3 = CuboidXd(
        axes_length=[1, 1],
        center_position=goal3.position,
        margin_absolut=1,
        orientation=goal3.orientation,
        tail_effect=False,
    )

    my_furniture = [
        Furniture3D(
            shape_list=[table_shape2],
            obstacle_environment=obstacle_environment,
            control_points=np.array([[0, 0], [0, 0]]),
            goal_pose=goal2,
            priority_value=1e3,
            static=True,
            name="static",
        ),
        Furniture3D(
            shape_list=[table_shape3],
            obstacle_environment=obstacle_environment,
            control_points=np.array([[0, 0], [0, 0]]),
            goal_pose=goal3,
            priority_value=1e-3,
            static=True,
            name="static",
        ),
        Furniture3D(
            shape_list=[table_shape],
            obstacle_environment=obstacle_environment,
            control_points=control_points,
            goal_pose=goal1,
            priority_value=1,
            name="fur",
        ),
    ]

    # Furniture(shape=table_shape, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal, priority_value=1, name="fur")]
    my_animation = DynamicalSystemAnimation3D(
        it_max=200,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )
    my_animation.setup(
        obstacle_environment,
        agent_list=my_furniture,
        x_lim=[-3, 8],
        y_lim=[-2, 8],
        mini_drag="dragdist",
        version="v1",
        safety_module=False,
        emergency_stop=True
    )

    if visualize:
        my_animation.run(save_animation=args.rec)
        # my_animation.logs(len(my_furniture))

    # Check Dynamic Agent
    my_furniture[2].update_velocity(
        mini_drag=my_animation.mini_drag,
        version=my_animation.version,
        emergency_stop=my_animation.emergency_stop,
        safety_module=my_animation.safety_module,
        time_step=my_animation.dt_simulation,
    )

    assert my_furniture[2].linear_velocity[0] > 0, "Expected to move towards attractor"
    assert (
        my_furniture[2].linear_velocity[1] < 0
    ), "Expected to move away from high priority"

    # Check that agent is really statig
    my_furniture[1].update_velocity(
        mini_drag=my_animation.mini_drag,
        version=my_animation.version,
        emergency_stop=my_animation.emergency_stop,
        safety_module=my_animation.safety_module,
        time_step=my_animation.dt_simulation,
    )

    assert np.allclose(
        my_furniture[1].linear_velocity, np.zeros(2)
    ), "agent[1] should be static."


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    test_uneven_priority(visualize=True)