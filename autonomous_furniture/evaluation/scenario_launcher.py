from random import random
from turtle import position
from test.test_orientation_ctrl import DynamicalSystemAnimation
from autonomous_furniture.agent import Furniture, Person
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd
from dynamic_obstacle_avoidance.obstacles import Obstacle
from vartools.states import ObjectPose
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from evaluation.grid import Grid
import numpy as np

class ScenarioLauncher:
    def __init_(self, nb_sim = 5, nb_furniture = 5, record = False):
        x_lim=[-3, 8]
        y_lim=[-2, 7]
        resolution = [20, 20]

        self._nb_furniture = nb_furniture 
        self._nb_sim = nb_sim

        self._init_index_free_space = [ (i, j) for i in range(resolution[0]) for j in range(resolution[1])]
        self._init_index_occupied_space =[]
        self.goal_pos_free_space = []

        self.obstacle_environment = ObstacleContainer()
        agents = []

        self.grid = Grid(x_lim, y_lim, agents,resolution)


    
    def creation_scenario(self):
        new_pose = ObjectPose() # Pose of the furniture that will be randomly placed in the arena 
        
        for ii in range(self._nb_furniture):
            is_placed = False
            nb_tries = 1
            while is_placed is not True:
                index_pos = random.choice(self._init_index_free_space) # Chosing from the occupied list of cell a potential candidate/challenger
                new_pose.position = self.grid._grid[index_pos] # index_pos is the tuple of the grid coordinate
                new_pose.orientation = random.randint(np.floor(-np.pi*100), np.ceil(np.pi*100))/100 # Randomly choosing a pose between [-pi, pi]

                fur_shape = CuboidXd(axes_length=[2, 1],   # TODO Remove from being it hardcoded
                            center_position=new_pose.position,
                            margin_absolut=0.6,
                            orientation=new_pose.orientation,
                            tail_effect=False,)
                
                cells_new_fur =[] # List of tuple representing the coordinate of the cells occupied by the new furniture

                for idx in self._init_index_free_space:
                    position = self.grid._grid[idx]
                    
                    if fur_shape.is_inside(position): # Checking if the tuple position is inside the new furniture
                        cells_new_fur.append(position) 

                if any(cell in cells_new_fur for cell in self._init_index_occupied_space):
                    nb_tries += 1
                    print(f"Failed to place one furniture (overlaping). Trying again(#{nb_tries}")
                else:
                    self._init_index_occupied_space.append(cells_new_fur)
                    for pos in cells_new_fur:
                        self._init_index_free_space.remove(pos) # A bug could appear if pos was not in _init_index_free_space but should not happen by construction
                        self._init_index_occupied_space.append(pos)
                        
                        is_placed = True # Ending condition to place a furniture
                    #TODO This is hardcoded and has to be changed for instance the goal location has to be randomly posed as well
                    Furniture(shape=fur_shape, obstacle_environment=self.obstacle_environment, control_points=np.array([[0.4, 0], [-0.4, 0]]), goal_pose=fur_shape.pose.position)

                

    def place_agent_randomly(self, free_space):
        
        pass

    def check_overlaping(self):
        pass

    def update_freespace(self):
        pass
    


    def run(self):
        pass    

def main():
    axis = [2, 1]
    max_ax_len = max(axis)
    min_ax_len = min(axis)

    # List of environment shared by all the furniture/agent
    obstacle_environment = ObstacleContainer()

    # control_points for the cuboid
    control_points = np.array([[0.3, 0], [-0.3, 0]])

    # , orientation = 1.6) Goal of the CuboidXd
    goal = ObjectPose(position=np.array([7, 1]), orientation=np.pi/2)

    table_shape = CuboidXd(axes_length=[max_ax_len, min_ax_len],
                           center_position=np.array([-2, 1]),
                           margin_absolut=0.6,
                           orientation=np.pi/2,
                           tail_effect=False,)

    goal2 = ObjectPose(position=np.array([-2, 0.5]), orientation=np.pi/2)
    table_shape2 = CuboidXd(axes_length=[max_ax_len, min_ax_len],
                            center_position=np.array([7, 0.5]),
                            margin_absolut=0.6,
                            orientation=0,
                            tail_effect=False,)

    my_furniture = [Furniture(shape=table_shape, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal, priority_value=1),
                    Furniture(shape=table_shape2, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal2, priority_value=1)]  # ,    Furniture(shape=table_shape2, obstacle_environment=obstacle_environment, control_points=control_points, goal_pose=goal2)]
if __name__ == "__main__":
    main()
    
