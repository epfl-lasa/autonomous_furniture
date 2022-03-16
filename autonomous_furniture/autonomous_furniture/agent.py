from abc import ABC, abstractmethod
from asyncio import get_running_loop
import warnings 
import numpy as np
from dynamic_obstacle_avoidance.containers.obstacle_container import ObstacleContainer
from vartools.states import ObjectPose, ObjectTwist
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import Ellipse
from vartools.dynamical_systems import LinearSystem
from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving

# from vartools.states

class BaseAgent(ABC):
    def __init__(self, shape : Obstacle = None, priority_value: float = 1, control_points : np.array = None, parking_pose : ObjectPose = None,
                goal_pose :ObjectPose = None, obstacle_environment : ObstacleContainer = None) -> None:
        super().__init__()
        self._shape = shape
        self._priority_value = priority_value
        self._obstacle_environment = obstacle_environment # TODO maybe append the shape directly in bos env, and then do a destructor to remove it from the list
        self._control_points = control_points
        self._parking_pose = parking_pose
        self._goal_pose = goal_pose

    @property
    def position(self):
        return self._shape.pose.position
    @property
    def dimension(self):
        return self._shape.pose.dimension
    @property
    def linear_velocity(self):
        return self._shape.twist.linear
    @linear_velocity.setter
    def linear_velocity(self, value):
        self._shape.twist.linear = value
    @property 
    def angular_velocity(self):
        return self._shape.twist.angular
    @angular_velocity.setter
    def angular_velocity(self, value):
        self._shape.twist.angular = value

    def do_velocity_step(self, dt):
        return self._shape.do_velocity_step(dt)
    
    def get_global_control_points(self):
        return np.array([self._shape.pose.transform_position_from_local_to_reference(ctp) for ctp in self._control_points]).T
    
    def get_goal_control_points(self):
        return np.array([self._goal_pose.transform_position_from_local_to_reference(ctp) for ctp in self._control_points]).T

    @staticmethod
    def get_weight_from_gamma(gammas, cutoff_gamma, n_points, gamma0=1.0, frac_gamma_nth=0.5):
        weights = (gammas - gamma0) / (cutoff_gamma - gamma0)
        weights = weights / frac_gamma_nth
        weights = 1.0 / weights
        weights = (weights - frac_gamma_nth) / (1 - frac_gamma_nth)
        weights = weights / n_points
        return weights
    
    @staticmethod
    def get_gamma_product_crowd(position, env):
        if not len(env):
            # Very large number
            return 1e20

        gamma_list = np.zeros(len(env))
        for ii, obs in enumerate(env):
            # gamma_type needs to be implemented for all obstacles
            gamma_list[ii] = obs.get_gamma(
                position, in_global_frame=True
                # , gamma_type=gamma_type
            )

        n_obs = len(gamma_list)
        # Total gamma [1, infinity]
        # Take root of order 'n_obs' to make up for the obstacle multiple
        if any(gamma_list < 1):
            warnings.warn("Collision detected.")
            # breakpoint()
            return 0

        # gamma = np.prod(gamma_list-1)**(1.0/n_obs) + 1
        gamma = np.min(gamma_list)
        if np.isnan(gamma):
            breakpoint()
        return gamma

    def get_obstacles_without_me(self):
        return [obs for obs in self._obstacle_environment if not obs == self._shape]
    
    def get_weight_of_control_points(self, control_points, environment_without_me):
        cutoff_gamma = 1e-4
        # gamma_values = self.get_gamma_at_control_point(control_points[self.obs_multi_agent[obs]], obs, temp_env)
        gamma_values = np.zeros(control_points.shape[1])
        
        for ii in range(control_points.shape[1]):
            gamma_values[ii] = self.get_gamma_product_crowd(control_points[ii, :], environment_without_me)    
        
        ctl_point_weight = np.zeros(gamma_values.shape)
        ind_nonzero = gamma_values < cutoff_gamma
        if not any(ind_nonzero):
            # ctl_point_weight[-1] = 1
            ctl_point_weight = np.full(gamma_values.shape, 1/control_points.shape[1])
        # for index in range(len(gamma_values)):
        ctl_point_weight[ind_nonzero] = self.get_weight_from_gamma(
            gamma_values[ind_nonzero],
            cutoff_gamma=cutoff_gamma,
            n_points=control_points.shape[1]
        )

        ctl_point_weight_sum = np.sum(ctl_point_weight)
        if ctl_point_weight_sum > 1:
            ctl_point_weight = ctl_point_weight / ctl_point_weight_sum
        else:
            ctl_point_weight[-1] += 1 - ctl_point_weight_sum
        
        return ctl_point_weight

class Furniture(BaseAgent):
    def __init__(self,
                **kwargs) -> None:
        super().__init__(**kwargs)
        self._dynamics = LinearSystem(attractor_position=self.position,
                                    maximum_velocity=1)
        # self._dynamic_avoider = DynamicCrowdAvoider(initial_dynamics=self._dynamics, environment=self._obstacle_environment)

    def update_velocity(self):
        environment_without_me = self.get_obstacles_without_me()
        # TODO : Make it a method to be called outside the class
        
        global_control_points = self.get_global_control_points()
        global_goal_control_points = self.get_goal_control_points()

        weights = self.get_weight_of_control_points(global_control_points, environment_without_me)
        
        velocities = np.zeros((self.dimension, self._control_points.shape[1]))
        angular_vel = np.zeros((1,self._control_points.shape[1])) #TODO : Do we want to enable rotation along other axis in the futur ?

        for ii in range(self._control_points.shape[1]) :
            ctp = global_control_points[:, ii]
            self._dynamics.attractor_position = global_goal_control_points[:, ii]
            initial_velocity = self._dynamics.evaluate(ctp) 
            velocities[:, ii] = obs_avoidance_interpolation_moving(position=ctp, initial_velocity=initial_velocity, obs=environment_without_me)

            
        self.linear_velocity = np.sum(velocities*np.tile(weights, (self.dimension,1)), axis=1)

        for ii in range(self._control_points.shape[1]):
             angular_vel[0,ii] = weights[ii]*np.cross(self._shape.center_position - self._control_points[ii], 
                                                    velocities[:,ii]-self.linear_velocity)
        self.angular_velocity = -2*np.sum(angular_vel) #TODO : Remove the hard coded 2

        # for agent in self.obs_w_multi_agent[obs]:
        #         angular_vel[agent - (obs * 2)] = weights[obs][agent - (obs * 2)] * np.cross(
        #             (self.obstacle_environment[obs].center_position - self.position_list[agent, :, ii]),
        #             (self.velocity[agent, :] - obs_vel))
        
            # angular_vel_obs = angular_vel.sum()
            # self.obstacle_environment[obs].linear_velocity = obs_vel
            # self.obstacle_environment[obs].angular_velocity = -2 * angular_vel_obs
            # self.obstacle_environment[obs].do_velocity_step(self.dt_simulation)
            # for agent in self.obs_w_multi_agent[obs]:
            #     self.position_list[agent, :, ii + 1] = self.obstacle_environment[obs].transform_relative2global(
            #         self.relative_agent_pos[agent, :])
        # TODO : Make it for the angular velocity

class Person(BaseAgent):
    def __init__(self, priority : int = 1 , person_radius : float = 0.6,  **kwargs) -> None:
        super().__init__(**kwargs)
        self._shape = Ellipse(axes_length=[person_radius, person_radius])
        breakpoint()
        self._priority_value = priority

        self._control_points = np.array([0, 0]) # Only one control point at the center when dealing with a Person
    
    def update_velocity(self):
        environment_without_me = self.get_obstacles_without_me()
        
        global_control_points = self.get_global_control_points()
        global_goal_control_points = self.get_goal_control_points()

        velocities = np.zeros((self.dimension, self._control_points.shape[1]))
        angular_vel = np.zeros((1,self._control_points.shape[1])) #TODO : Do we want to enable rotation along other axis in the futur ?

        self.linear_velocity = np.array([0, 0])
        self.angular_velocity = 0.1

   


