from abc import ABC, abstractmethod
from asyncio import get_running_loop
import warnings
import numpy as np
from numpy import linalg as LA
from dynamic_obstacle_avoidance.containers.obstacle_container import ObstacleContainer
from vartools.dynamical_systems.linear import ConstantValue
from vartools.states import ObjectPose, ObjectTwist
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.obstacles.ellipse_xd import EllipseWithAxes

from vartools.dynamical_systems import LinearSystem
from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving

# from vartools.states


class BaseAgent(ABC):
    def __init__(self, shape: Obstacle, obstacle_environment: ObstacleContainer, priority_value: float = 1., control_points: np.array = None, parking_pose: ObjectPose = None,
                 goal_pose: ObjectPose = None) -> None:
        super().__init__()
        self._shape = shape
        self.priority = priority_value
        self.virtual_drag = max(self._shape.axes_length)/min(self._shape.axes_length)
        # TODO maybe append the shape directly in bos env, and then do a destructor to remove it from the list
        self._obstacle_environment = obstacle_environment
        self._control_points = control_points
        self._parking_pose = parking_pose
        self._goal_pose = goal_pose
        # Adding the current shape of the agent to the list of obstacle_env so as to be visible to other agents
        self._obstacle_environment.append(self._shape)

    @property
    def position(self):
        return self._shape.pose.position
    @property
    def orientation(self):
        return self._shape.pose.orientation

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

    @property
    def priority(self):
        return self._shape.reactivity

    @priority.setter
    def priority(self,value):
        self._shape.reactivity = value

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
            gamma_values[ii] = self.get_gamma_product_crowd(
                control_points[ii, :], environment_without_me)

        ctl_point_weight = np.zeros(gamma_values.shape)
        ind_nonzero = gamma_values < cutoff_gamma
        if not any(ind_nonzero): # TODO Case he there is ind_nonzero
            # ctl_point_weight[-1] = 1
            ctl_point_weight = np.full(
                gamma_values.shape, 1/control_points.shape[1])
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
        self.minimize_drag : bool = False

    def update_velocity(self):
        initial_velocity = np.zeros(2)
        environment_without_me = self.get_obstacles_without_me()
        # TODO : Make it a method to be called outside the class
        global_control_points = self.get_global_control_points()
        global_goal_control_points = self.get_goal_control_points()

        weights = self.get_weight_of_control_points(
            global_control_points, environment_without_me)

        velocities = np.zeros((self.dimension, self._control_points.shape[1]))
        init_velocities = np.zeros((self.dimension, self._control_points.shape[1]))
        # TODO : Do we want to enable rotation along other axis in the futur ?
        angular_vel = np.zeros((1, self._control_points.shape[1]))

        self._dynamics.attractor_position = self._goal_pose.position
        initial_velocity = self._dynamics.evaluate(self.position) # First we compute the initial velocity at the "center", ugly
        
        # Computing the weights of the angle to reach
        w1_hat = self.virtual_drag
        w2_hat = self._dynamics.maximum_velocity/LA.norm(initial_velocity)-1
        w1 = w1_hat/(w1_hat + w2_hat)
        w2 = 1 - w1

        print(w1)
        lin_vel_dir = np.arctan2(initial_velocity[1], initial_velocity[0]) # Direction (angle), of the linear_velocity in the global frame

        if np.abs(lin_vel_dir-self.orientation) < np.abs(lin_vel_dir-(self.orientation -np.pi)): # Make the smallest rotation- the furniture has to pi symetric
            drag_angle = lin_vel_dir-self.orientation
        else:
            drag_angle =  lin_vel_dir-(self.orientation -np.pi)
        
        goal_angle = self._goal_pose.orientation- self.orientation

        # TODO Very clunky : Rather make a function out of it
        K = 1 # K proportionnal parameter for the speed
        initial_angular_vel = K*(w1*drag_angle + w2*goal_angle) # Initial angular_velocity is computed

        for ii in range(self._control_points.shape[1]):
            tang_vel = [-initial_angular_vel*initial_velocity[1], initial_angular_vel*initial_velocity[0]] # doing the cross product formula by "hand" than using the funct
            init_velocities[:,ii] = initial_velocity/2 + tang_vel

            ctp = global_control_points[:, ii]
            # self._dynamics.attractor_position = global_goal_control_points[:, ii]
            # initial_velocity = self._dynamics.evaluate(ctp)
            velocities[:, ii] = obs_avoidance_interpolation_moving(
                position=ctp, initial_velocity=init_velocities[:,ii], obs=environment_without_me, self_priority=self.priority)

        self.linear_velocity = np.sum(
            velocities*np.tile(weights, (self.dimension, 1)), axis=1)

        for ii in range(self._control_points.shape[1]):
            angular_vel[0, ii] = weights[ii]*np.cross(self._shape.center_position - self._control_points[ii],
                                                      velocities[:, ii]-self.linear_velocity)
        # TODO : Remove the hard coded 2
        self.angular_velocity = -2*np.sum(angular_vel)
    
    def controller(self, initial_velocity):
        distance_to_goal = LA.norm(self._goal_pose.position-self.position)
        min_dist_to_minimize_drag = 4
        dist_to_goal_thr = 2

        goal_angle = self._goal_pose.orientation- self.orientation
        opti_drag_angle = np.arctan2(initial_velocity[1], initial_velocity[0])- self.orientation

        w1 = self.angular_velocity^2/self.linear_velocity^2
        w2 = 1
        weighted_angle = w1*goal_angle + w2*opti_drag_angle
        
        return weighted_angle
        # if distance_to_goal < dist_to_goal_thr:
        #     self.minimize_drag = False
        #     return self._goal_pose.orientation- self.orientation

        # elif distance_to_goal > min_dist_to_minimize_drag or self.minimize_drag is True:
        #     self.minimize_drag = True
        #     goal_dir = np.arctan2(initial_velocity[1], initial_velocity[0])
            
        #     if np.abs(goal_dir-self.orientation) < np.abs(goal_dir-(self.orientation -np.pi)): # Make the smallest rotation- the furniture has to pi symetric
        #         return goal_dir-self.orientation
        #     else:
        #         return goal_dir-(self.orientation -np.pi)

        # else:            
        #     return self._goal_pose.orientation- self.orientation


class Person(BaseAgent):
    def __init__(self, priority_value: float = 1, center_position=None, radius=0.5, **kwargs) -> None:
        _shape = EllipseWithAxes(center_position=np.array(center_position),
                                 margin_absolut=1,
                                 orientation=0,
                                 tail_effect=False,
                                 axes_length=np.array([radius, radius]))

        super().__init__(shape=_shape, priority_value=priority_value,
                         control_points=np.array([[0, 0]]), **kwargs) # ALWAYS USE np.array([[0,0]]) and not np.array([0,0])

        self._dynamics = LinearSystem(attractor_position=self.position,
                                      maximum_velocity=1)

    def update_velocity(self):
        environment_without_me = self.get_obstacles_without_me()
        global_control_points = self.get_global_control_points()
        global_goal_control_points = self.get_goal_control_points()
        # Person only holds one control point, thus modulation is simplified
        ctp = global_control_points[:,0] # 0 hardcoded because we assume in person, we will have only one control point
        self._dynamics.attractor_position = global_goal_control_points[:,0] # 0 hardcoded because we assume in person, we will have only one control point
        initial_velocity = self._dynamics.evaluate(ctp)
        velocity = obs_avoidance_interpolation_moving(
              position=ctp, initial_velocity=initial_velocity, obs=environment_without_me,self_priority=self.priority)

        self.linear_velocity = velocity