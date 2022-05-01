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
    def __init__(self, nb_sim = 5, nb_furniture = 5, furnitures :list =[] , obs_environment : ObstacleContainer = None, record = False):
        x_lim=[-3, 8]
        y_lim=[-2, 7]
        resolution = [40, 40]

        self._nb_furniture = nb_furniture 
        self._nb_sim = nb_sim

        self._init_index_free_space = [ (i, j) for i in range(resolution[0]) for j in range(resolution[1])]
        self._init_index_occupied_space =[]
        
        self._goal_index_free_space = [ (i, j) for i in range(resolution[0]) for j in range(resolution[1])]
        self._goal_index_occupied_space =[]

        self.obstacle_environment = obs_environment
        self.agents = furnitures

        self._fur_shape = CuboidXd(axes_length=[2, 1],   # TODO TEMPORARY Remove from being it hardcoded
                        center_position=[0,0],
                        margin_absolut=0.6,
                        orientation=0,
                        tail_effect=False,)
        self.grid = Grid(x_lim, y_lim, self.agents,resolution)


    
    def creation_scenario(self):
        new_pose = ObjectPose() # Pose of the furniture that will be randomly placed in the arena 
        #TODO change the fact that we have to copy to fill the "cells_fur_new"
        copy_indx = self._init_index_free_space.copy() # If we don't copy the list is copied by reference

        for ii in range(self._nb_furniture):
            # Finding the initial pose of the  furniture
            init_pose = self.place_agent_randomly(self._init_index_free_space, self._init_index_occupied_space)
            # Finding the goal pose of the furniture
            goal_pose = self.place_agent_randomly(self._goal_index_free_space, self._goal_index_occupied_space)

            furniture = CuboidXd(axes_length=[2, 1],   # TODO TEMPORARY Remove from being it hardcoded
                        center_position=init_pose.position,
                        margin_absolut=0.6,
                        orientation=init_pose.orientation,
                        tail_effect=False,)
            #TODO This is hardcoded and has to be changed for instance the goal location has to be randomly posed as well
            self.agents.append(Furniture(shape=furniture, obstacle_environment=self.obstacle_environment, control_points=np.array([[0.4, 0], [-0.4, 0]]), goal_pose=goal_pose))
            print(f"Furniture #{ii} successfuly placed")
                                                          
    def place_agent_randomly(self, free_space : list = None, occupied_space : list = None)->ObjectPose:
        new_pose = ObjectPose() # Pose of the furniture that will be randomly placed in the arena 
        is_placed = False
        nb_tries = 1

        grid_coord = free_space.copy() # Only use for iteration during the filling of cells_new_fur

        while is_placed is not True:
            index_pos = random.choice(free_space) # Chosing from the occupied list of cell a potential candidate/challenger
            new_pose.position = self.grid._grid[index_pos] # index_pos is the tuple of the grid coordinate
            new_pose.orientation = random.randint(np.floor(-np.pi*100), np.ceil(np.pi*100))/100 # Randomly choosing a pose between [-pi, pi]

            self._fur_shape = CuboidXd(axes_length=[2, 1],   # TODO Remove from being it hardcoded
                        center_position=new_pose.position,
                        margin_absolut=0.6,
                        orientation=new_pose.orientation,
                        tail_effect=False,)
            
            cells_new_fur =[] # reseted- List of tuple representing the coordinate of the cells occupied by the new furniture 

            # Filling cells_new_fur with the tuple coordinate of the cells occupied by the candidate furniture in the grid
            for idx in grid_coord:
                position = self.grid._grid[idx]
                
                if self._fur_shape.is_inside(position, margin = 1): # Checking if the tuple position is inside the new furniture
                    cells_new_fur.append(idx)

            if any(cell in cells_new_fur for cell in occupied_space):
                nb_tries += 1
                print(f"Failed to place the furniture (overlaping). Trying again(#{nb_tries}")
            else:
                for cells in cells_new_fur:
                    free_space.remove(cells) # A bug could appear if pos was not in _init_index_free_space but should not happen by construction
                    occupied_space.append(cells)
                    
                is_placed = True # Ending condition to place a furniture 
        
        return new_pose

    def check_overlaping(self):
        pass

    def update_freespace(self):
        pass
    


    def run(self):
        pass    
