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

from autonomous_furniture.furniture_creators import (
    create_standard_table_3D_surface_legs,
)

parser = argparse.ArgumentParser()

parser.add_argument("--rec", action="store", default=False, help="Record flag")
parser.add_argument(
    "--name", action="store", default="recording", help="Name of the simulation"
)
args = parser.parse_args()


def test_straight(visualize=False):
    # List of environment shared by all the furniture/agent in the same layer
    obstacle_environment_lower = ObstacleContainer()
    obstacle_environment_upper = ObstacleContainer()

    ### CREATE TABLE SECTIONS FOR ALL THE LAYERS
    table_reference_start = ObjectPose(
        position=np.array([-1, 1]), orientation=0
    )
    table_reference_goal = ObjectPose(position=np.array([1, 1]), orientation=0)

    table_legs_agent, table_surface_agent = create_standard_table_3D_surface_legs(
        obstacle_environment_lower,
        obstacle_environment_upper,
        table_reference_start,
        table_reference_goal,
        margins=0.1,
        static=False,
    )
    
    obstacle_shape = CuboidXd(
        center_position = table_reference_goal.position,
        orientation = table_reference_goal.orientation,
        axes_length=[1,1]
    )
    obstacle = Furniture3D(
        shape_list=[obstacle_shape],
        starting_pose=ObjectPose(position=table_reference_goal.position, orientation=table_reference_goal.orientation),
        goal_pose=ObjectPose(position=table_reference_goal.position, orientation=table_reference_goal.orientation),
        static=True,
        obstacle_environment=obstacle_environment_upper,
        control_points=np.array([[0.0, 0.0]])
    )

    # Furniture(shape=table_shape, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal, priority_value=1, name="fur")]
    my_animation = DynamicalSystemAnimation3D(
        it_max=200,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )
    my_animation.setup(
        layer_list=[[table_surface_agent, obstacle]],
        x_lim=[-3, 8],
        y_lim=[-2, 8],
        mini_drag="nodrag",
        version="v1",
        safety_module=False,
        emergency_stop=False,
    )

    if visualize:
        my_animation.run(save_animation=args.rec)
        # my_animation.logs(len(my_furniture))

    # Check Dynamic Agent
    table_surface_agent.update_velocity(
        mini_drag=my_animation.mini_drag,
        version=my_animation.version,
        emergency_stop=my_animation.emergency_stop,
        safety_module=my_animation.safety_module,
        time_step=my_animation.dt_simulation,
    )

    assert table_surface_agent.linear_velocity[0] > 0, "Expected to move towards attractor"
    assert (
        np.linalg.norm(table_surface_agent.linear_velocity[1]) < 1e-6
    ), "Expected to move perfectly vertical"


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    test_straight(visualize=False)
