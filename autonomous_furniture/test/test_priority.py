import argparse
import numpy as np

import matplotlib.pyplot as plt
from scipy import rand

from vartools.states import Pose
from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from autonomous_furniture.agent import Furniture, Person
from autonomous_furniture.dynamical_system_animation import DynamicalSystemAnimation

parser = argparse.ArgumentParser()

parser.add_argument("--rec", action="store", default=False, help="Record flag")
parser.add_argument(
    "--name", action="store", default="recording", help="Name of the simulation"
)
args = parser.parse_args()


def test_uneven_priority(visualize=False):
    # List of environment shared by all the furniture/agent
    obstacle_environment = ObstacleContainer()
    position1 = np.array([-2.0, 0.0])
    position2 = np.array([0.0, 2.0])
    position3 = np.array([0.0, -2.0])

    goal = Pose(position=np.array([2.0, 0]), orientation=0.0)
    margin = 0.4

    my_furniture = [
        Person(
            center_position=position1,
            radius=0.8,
            obstacle_environment=obstacle_environment,
            goal_pose=goal,
            priority_value=1,
            margin=margin,
            static=False,
        ),
        Person(
            center_position=position2,
            radius=0.8,
            obstacle_environment=obstacle_environment,
            goal_pose=Pose(position2),
            priority_value=1,
            margin=margin,
            static=True,
        ),
        Person(
            center_position=position3,
            radius=0.8,
            obstacle_environment=obstacle_environment,
            goal_pose=Pose(position3),
            priority_value=4,
            margin=margin,
            static=True,
        ),
    ]

    # Furniture(shape=table_shape, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal, priority_value=1, name="fur")]
    my_animation = DynamicalSystemAnimation(
        it_max=200,
        dt_simulation=0.05,
        dt_sleep=0.05,
        animation_name=args.name,
    )
    my_animation.setup(
        obstacle_environment,
        agent=my_furniture,
        x_lim=[-4, 4],
        y_lim=[-4, 4],
        mini_drag="dragdist",
        version="v2",
    )

    if visualize:
        my_animation.run(save_animation=args.rec)
        my_animation.logs(len(my_furniture))

    # Check Dynamic Agent
    my_furniture[0].update_velocity(
        mini_drag=my_animation.mini_drag,
        version=my_animation.version,
        emergency_stop=my_animation.emergency_stop,
        safety_module=my_animation.safety_module,
        time_step=my_animation.dt_simulation,
    )

    assert my_furniture[0].linear_velocity[0] > 0, "Expected to move towards attractor"
    assert (
        my_furniture[0].linear_velocity[1] > 0
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

    test_uneven_priority(visualize=False)
