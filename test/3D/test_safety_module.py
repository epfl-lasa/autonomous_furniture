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

import pathlib

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

    parameter_file = (
        str(pathlib.Path(__file__).parent.resolve()) + "/parameters/test.json"
    )

    # List of environment shared by all the furniture/agent
    obstacle_environment = ObstacleContainer()

    # control_points for the cuboid
    control_points = np.array([[0.6, 0], [-0.6, 0]])

    # , orientation = 1.6) Goal of the CuboidXd
    # , orientation = 1.6) Goal of the CuboidXd
    start1 = ObjectPose(position=np.array([1, 3]), orientation=np.pi / 2)
    goal = ObjectPose(position=np.array([6, 3]), orientation=np.pi / 2)
    table_shape = CuboidXd(
        axes_length=[max_ax_len, min_ax_len],
        center_position=start1.position,
        margin_absolut=1,
        orientation=start1.orientation,
        tail_effect=False,
    )

    start2 = ObjectPose(position=np.array([3, 3]), orientation=np.pi / 2)
    goal2 = ObjectPose(position=np.array([-1, 3]), orientation=np.pi / 2)
    table_shape2 = CuboidXd(
        axes_length=[max_ax_len, min_ax_len],
        center_position=start2.position,
        margin_absolut=1,
        orientation=start2.orientation,
        tail_effect=False,
    )

    my_furniture = [
        Furniture3D(
            shape_list=[table_shape],
            shape_positions=np.array([[0.0, 0.0]]),
            starting_pose=table_shape.pose,
            obstacle_environment=obstacle_environment,
            control_points=control_points,
            goal_pose=goal,
            name="fur",
            safety_damping=1,
            gamma_critic_max=2,
            parameter_file=parameter_file,
        ),
        Furniture3D(
            shape_list=[table_shape2],
            shape_positions=np.array([[0.0, 0.0]]),
            starting_pose=table_shape2.pose,
            obstacle_environment=obstacle_environment,
            control_points=control_points,
            goal_pose=goal2,
            name="fur",
            safety_damping=1,
            gamma_critic_max=2,
            parameter_file=parameter_file,
        ),
    ]

    for i in range(len(my_furniture)):
        agent = assign_agent_virtual_drag([my_furniture[i]])
        my_furniture[i] = agent[0]

    # Furniture(shape=table_shape, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal, priority_value=1, name="fur")]
    my_animation = DynamicalSystemAnimation3D(
        it_max=200,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )
    my_animation.setup(
        layer_list=[my_furniture],
        x_lim=[-3, 8],
        y_lim=[-2, 8],
        mini_drag="dragdist",
        version="v2",
        safety_module=True,
        emergency_stop=True,
    )

    if visualize:
        my_animation.run(save_animation=args.rec)
        # my_animation.logs(len(my_furniture))

    # Check Dynamic Agent
    my_furniture[0].update_velocity(
        mini_drag=my_animation.mini_drag,
        version=my_animation.version,
        emergency_stop=my_animation.emergency_stop,
        safety_module=my_animation.safety_module,
        time_step=my_animation.dt_simulation,
    )
    my_furniture[1].update_velocity(
        mini_drag=my_animation.mini_drag,
        version=my_animation.version,
        emergency_stop=my_animation.emergency_stop,
        safety_module=my_animation.safety_module,
        time_step=my_animation.dt_simulation,
    )

    assert my_furniture[0].linear_velocity[0] < 0, "Expected to move towards attractor"
    assert (
        my_furniture[1].linear_velocity[0] > 0
    ), "Expected to move away from high priority"


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    test(visualize=False)
