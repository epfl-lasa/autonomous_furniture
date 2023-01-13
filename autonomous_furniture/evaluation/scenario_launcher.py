import random
from turtle import position
from autonomous_furniture.agent import Furniture, Person
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from dynamic_obstacle_avoidance.obstacles import Obstacle
from vartools.states import ObjectPose
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from evaluation.grid import Grid
import numpy as np


class ScenarioLauncher:
    def __init__(self, nb_furniture=5, record=False):

        x_lim = [-3, 8]
        y_lim = [-2, 7]
        self._resolution = [40, 40]

        self._nb_furniture = nb_furniture

        self._init_index_free_space = [
            (i, j)
            for i in range(self._resolution[0])
            for j in range(self._resolution[1])
        ]
        self._init_index_occupied_space = []

        self._goal_index_free_space = [
            (i, j)
            for i in range(self._resolution[0])
            for j in range(self._resolution[1])
        ]
        self._goal_index_occupied_space = []

        self._init_setup = []  # Stores all the starting position of the agents
        self._goal_setup = []  # Stores all the goal pose of the agents

        self._fur_shape = CuboidXd(
            axes_length=[2, 1],  # TODO TEMPORARY Remove from being it hardcoded
            center_position=[0, 0],
            margin_absolut=0.9,
            orientation=0,
            tail_effect=False,
        )
        # TODO modifiy to remove the useless "[]"
        self.grid = Grid(x_lim, y_lim, [], self._resolution)

    def creation(
        self,
    ):  # Peut être mettre ça dans l'initialisation plutôt que dans une autre méthode
        # Reset the list othe tuple representing the coordianates of free and occupied space
        self.reset_index()
        self._init_setup = []  # Stores all the starting position of the agents
        self._goal_setup = []  # Stores all the goal pose of the agents

        for ii in range(self._nb_furniture):
            # Finding the initial pose of the  furniture
            init_pose = self.place_agent_randomly(
                self._init_index_free_space, self._init_index_occupied_space
            )
            # Finding the goal pose of the furniture
            goal_pose = self.place_agent_randomly(
                self._goal_index_free_space, self._goal_index_occupied_space
            )

            # TODO This is hardcoded and has to be changed for instance the goal location has to be randomly posed as well
            self._init_setup.append(init_pose)
            self._goal_setup.append(goal_pose)
            print(f"Furniture #{ii} successfuly placed")

    def place_agent_randomly(
        self, free_space: list = None, occupied_space: list = None
    ) -> ObjectPose:
        # Pose of the furniture that will be randomly placed in the arena
        new_pose = ObjectPose()

        is_placed = False
        nb_tries = 1

        # Only use for iteration during the filling of cells_new_fur
        grid_coord = free_space.copy()

        if True:  # while is_placed is not True:
            # Chosing from the occupied list of cell a potential candidate/challenger
            index_pos = random.choice(free_space)
            # index_pos is the tuple of the grid coordinate
            new_pose.position = self.grid._grid[index_pos]
            # Randomly choosing a pose between [-pi, pi]
            new_pose.orientation = (
                random.randint(np.floor(-np.pi * 100), np.ceil(np.pi * 100)) / 100
            )

            self._fur_shape = CuboidXd(
                axes_length=[2, 1],  # TODO Remove from being it hardcoded
                center_position=new_pose.position,
                margin_absolut=0.9,
                orientation=new_pose.orientation,
                tail_effect=False,
            )

            # reseted- List of tuple representing the coordinate of the cells occupied by the new furniture
            cells_new_fur = []

            # Filling cells_new_fur with the tuple coordinate of the cells occupied by the candidate furniture in the grid
            for idx in grid_coord:
                position = self.grid._grid[idx]

                # Checking if the tuple position is inside the new furniture
                if self._fur_shape.is_inside(position, margin=1.5):
                    cells_new_fur.append(idx)

            if any(cell in cells_new_fur for cell in occupied_space):
                nb_tries += 1
                print(
                    f"Failed to place the furniture (overlaping). Trying again(#{nb_tries}"
                )
            else:
                for cells in cells_new_fur:
                    # A bug could appear if pos was not in _init_index_free_space but should not happen by construction
                    free_space.remove(cells)
                    occupied_space.append(cells)

                is_placed = True  # Ending condition to place a furniture

        return new_pose

    def setup(self):
        self.agents = []
        self.obstacle_environment = ObstacleContainer()

        for ii in range(self._nb_furniture):
            self._fur_shape = CuboidXd(
                axes_length=[2, 1],  # TODO TEMPORARY Remove from being it hardcoded
                center_position=[0, 0],
                margin_absolut=0.9,
                orientation=0,
                tail_effect=False,
            )
            self._fur_shape.pose.position = self._init_setup[ii].position
            self._fur_shape.pose.orientation = self._init_setup[ii].orientation
            self.agents.append(
                Furniture(
                    shape=self._fur_shape,
                    obstacle_environment=self.obstacle_environment,
                    control_points=np.array([[0.4, 0], [-0.4, 0]]),
                    goal_pose=self._goal_setup[ii],
                )
            )

    def reset_index(self):
        self._init_index_free_space = [
            (i, j)
            for i in range(self._resolution[0])
            for j in range(self._resolution[1])
        ]
        self._init_index_occupied_space = []

        self._goal_index_free_space = [
            (i, j)
            for i in range(self._resolution[0])
            for j in range(self._resolution[1])
        ]
        self._goal_index_occupied_space = []

    def check_overlaping(self):
        pass

    def update_freespace(self):
        pass

    def run(self):
        pass
