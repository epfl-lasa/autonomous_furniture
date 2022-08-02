from abc import ABC, abstractmethod
from asyncio import get_running_loop
import warnings
import numpy as np
import matplotlib.pyplot as plt
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
    # Static variable, to trak the number of collisions within a scenario
    number_collisions = 0
    number_serious_collisions = 0

    def __init__(
        self,
        shape: Obstacle,
        obstacle_environment: ObstacleContainer,
        priority_value: float = 1.0,
        control_points: np.array = None,
        parking_pose: ObjectPose = None,
        goal_pose: ObjectPose = None,
        name: str = "no_name",
        static: bool = False,
    ) -> None:
        super().__init__()
        self._shape = shape
        self.priority = priority_value
        self.virtual_drag = max(self._shape.axes_length) / min(self._shape.axes_length)
        # TODO maybe append the shape directly in bos env, and then do a destructor to remove it from the list
        self._obstacle_environment = obstacle_environment
        self._control_points = control_points
        self._parking_pose = parking_pose
        self._goal_pose = goal_pose
        # Adding the current shape of the agent to the list of obstacle_env so as to be visible to other agents
        self._obstacle_environment.append(self._shape)
        
        self._static = static
        # name of the furniture, useful for debugging stuff
        self.name = name

        self.converged: bool = False
        # metrics
        self.direct_distance = LA.norm(goal_pose.position - self.position)
        self.total_distance = 0
        self.time_conv = 0
        self.time_conv_direct = 0
        self._proximity = 0
        self._list_prox = [] # Tempory attribute only use for the qualitative example of the report. To be deleted after
        #plot
        self._list_center_pos = [self.position]
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
    def priority(self, value):
        self._shape.reactivity = value
    
    @property
    def name(self):
        return self._shape.name
    
    @name.setter
    def name(self, name):
        self._shape.name = name

    def do_velocity_step(self, dt):
        return self._shape.do_velocity_step(dt)

    def get_global_control_points(self):
        return np.array(
            [
                self._shape.pose.transform_position_from_local_to_reference(ctp)
                for ctp in self._control_points
            ]
        ).T

    def get_goal_control_points(self):
        return np.array(
            [
                self._goal_pose.transform_position_from_local_to_reference(ctp)
                for ctp in self._control_points
            ]
        ).T

    def get_veloctity_in_global_frame(self, velocity):
        return self._shape.pose.transform_direction_from_local_to_reference(velocity)
    
    def get_velocity_in_local_frame(self, velocity):
        return self._shape.pose.transform_direction_from_reference_to_local(velocity)

    @staticmethod
    def get_weight_from_gamma(
        gammas, cutoff_gamma, n_points, gamma0=1.0, frac_gamma_nth=0.5
    ):
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
                position,
                in_global_frame=True
                # , gamma_type=gamma_type
            )

        n_obs = len(gamma_list)
        # Total gamma [1, infinity]
        # Take root of order 'n_obs' to make up for the obstacle multiple
        if any(gamma_list < 1):
            BaseAgent.number_collisions += 1
            warnings.warn("Collision detected.")
            return None, 0

        # gamma = np.prod(gamma_list-1)**(1.0/n_obs) + 1
        gamma = np.min(gamma_list)
        index = np.argmin(gamma_list)

        if np.isnan(gamma):
            breakpoint()
        return index, gamma

    def get_obstacles_without_me(self):
        return [obs for obs in self._obstacle_environment if not obs == self._shape]

    def get_weight_of_control_points(self, control_points, environment_without_me):
        cutoff_gamma = 1e-4  # TODO : This value has to be big and not small
        # gamma_values = self.get_gamma_at_control_point(control_points[self.obs_multi_agent[obs]], obs, temp_env)
        gamma_values = np.zeros(control_points.shape[1])
        obs_idx = np.zeros(control_points.shape[1])
        for ii in range(control_points.shape[1]):
           obs_idx[ii] ,gamma_values[ii] = self.get_gamma_product_crowd(
                control_points[:, ii], environment_without_me
            )

        ctl_point_weight = np.zeros(gamma_values.shape)
        ind_nonzero = gamma_values < cutoff_gamma
        if not any(ind_nonzero):  # TODO Case he there is ind_nonzero
            # ctl_point_weight[-1] = 1
            ctl_point_weight = np.full(gamma_values.shape, 1 / control_points.shape[1])
        # for index in range(len(gamma_values)):
        ctl_point_weight[ind_nonzero] = self.get_weight_from_gamma(
            gamma_values[ind_nonzero],
            cutoff_gamma=cutoff_gamma,
            n_points=control_points.shape[1],
        )

        ctl_point_weight_sum = np.sum(ctl_point_weight)
        if ctl_point_weight_sum > 1:
            ctl_point_weight = ctl_point_weight / ctl_point_weight_sum
        else:
            ctl_point_weight[-1] += 1 - ctl_point_weight_sum

        return ctl_point_weight


class Furniture(BaseAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._dynamics = LinearSystem(
            attractor_position=self._goal_pose.position, maximum_velocity=1
        )

        # self._dynamic_avoider = DynamicCrowdAvoider(initial_dynamics=self._dynamics, environment=self._obstacle_environment)
        self.minimize_drag: bool = False # Seems to be used nowhere, mb to be removed

        # Metrics
        self.time_conv_direct = self.direct_distance / self._dynamics.maximum_velocity

    @property
    def margin_absolut(self):
        return self._shape._margin_absolut

    def update_velocity(self, mini_drag: str = "nodrag", version: str = "v1"):
        #self.corner_case(mini_drag, version)

        initial_velocity = np.zeros(2)
        environment_without_me = self.get_obstacles_without_me()
        # TODO : Make it a method to be called outside the class
        global_control_points = self.get_global_control_points()
        global_goal_control_points = self.get_goal_control_points()

        weights = self.get_weight_of_control_points(
            global_control_points, environment_without_me
        )

        velocities = np.zeros((self.dimension, self._control_points.shape[1]))
        init_velocities = np.zeros((self.dimension, self._control_points.shape[1]))

        if not self._static:
            # TODO : Do we want to enable rotation along other axis in the futur ?
            angular_vel = np.zeros((1, self._control_points.shape[1]))

            # First we compute the initial velocity at the "center", ugly
            initial_velocity = self._dynamics.evaluate(self.position)

            plt.arrow(self.position[0], self.position[1], initial_velocity[0],
                  initial_velocity[1], head_width=0.1, head_length=0.2, color='g')

            if version == "v2":
                initial_velocity = obs_avoidance_interpolation_moving(
                    position=self.position,
                    initial_velocity=initial_velocity,
                    obs=environment_without_me,
                    self_priority=self.priority,
                )
            plt.arrow(self.position[0], self.position[1], initial_velocity[0], initial_velocity[1], head_width=0.1, head_length=0.2, color='m')
            
            initial_magnitude = LA.norm(initial_velocity)

            # Computing the weights of the angle to reach
            if mini_drag == "dragvel":
                
                w1_hat = self.virtual_drag
                w2_hat_max = 1000
                if LA.norm(initial_velocity) != 0:
                    w2_hat = self._dynamics.maximum_velocity / LA.norm(initial_velocity) - 1
                    if w2_hat > w2_hat_max:
                        w2_hat = w2_hat_max
                else:
                    w2_hat = w2_hat_max

                w1 = w1_hat / (w1_hat + w2_hat)
                w2 = 1 - w1
            elif mini_drag == "dragdist":
                
                d = LA.norm(self.position-self._goal_pose.position)
                kappa = self.virtual_drag
                w1 = 1/2*(1+np.tanh((d*kappa-1.5*kappa)/2))
                w2 = 1- w1

            elif mini_drag == "nodrag":
                w1 = 0
                w2 = 1 
            else :
                print("Error in the name of the type of drag to use")
                w1 = 0
                w2 = 1 

            # Direction (angle), of the linear_velocity in the global frame
            lin_vel_dir = np.arctan2(initial_velocity[1], initial_velocity[0])

            # Make the smallest rotation- the furniture has to pi symetric
            drag_angle = lin_vel_dir - self.orientation
            # Case where there is no symetry in the furniture
            if np.abs(drag_angle) > np.pi:
                drag_angle = -1 * (2 * np.pi - drag_angle)

            # Case where we consider for instance PI-symetry for the furniture
            if (
                np.abs(drag_angle) > np.pi / 2
            ):  # np.pi/2 is the value hard coded in case for PI symetry of the furniture, if we want to introduce PI/4 symetry for instance we ahve to change this value
                if self.orientation > 0:
                    orientation_sym = self.orientation - np.pi
                else:
                    orientation_sym = self.orientation + np.pi

                drag_angle = lin_vel_dir - orientation_sym
                if drag_angle > np.pi / 2:
                    drag_angle = -1 * (2 * np.pi - drag_angle)

            goal_angle = self._goal_pose.orientation - self.orientation
            if np.abs(goal_angle) > np.pi:
                goal_angle = -1 * (2 * np.pi - goal_angle)

            if (
                np.abs(goal_angle) > np.pi / 2
            ):  # np.pi/2 is the value hard coded in case for PI symetry of the furniture, if we want to introduce PI/4 symetry for instance we ahve to change this value
                if self.orientation > 0:
                    orientation_sym = self.orientation - np.pi
                else:
                    orientation_sym = self.orientation + np.pi

                goal_angle = self._goal_pose.orientation - orientation_sym
                if goal_angle > np.pi / 2:
                    goal_angle = -1 * (2 * np.pi - goal_angle)

            # TODO Very clunky : Rather make a function out of it
            K = 3  # K proportionnal parameter for the speed
            # Initial angular_velocity is computed
            initial_angular_vel = K * (w1 * drag_angle + w2 * goal_angle)

            for ii in range(self._control_points.shape[1]):
                # doing the cross product formula by "hand" than using the funct
                tang_vel = [
                    -initial_angular_vel * self._control_points[ii, 1],
                    initial_angular_vel * self._control_points[ii, 0],
                ]
                tang_vel = self.get_veloctity_in_global_frame(tang_vel)
                init_velocities[:, ii] = initial_velocity + tang_vel

                ctp = global_control_points[:, ii]
                velocities[:, ii] = obs_avoidance_interpolation_moving(
                    position=ctp,
                    initial_velocity=init_velocities[:, ii],
                    obs=environment_without_me,
                    self_priority=self.priority,
                )
                # plt.arrow(ctp[0], ctp[1], init_velocities[0, ii],
                #           init_velocities[1, ii], head_width=0.1, head_length=0.2, color='g')
                # plt.arrow(ctp[0], ctp[1], velocities[0, ii], velocities[1,
                #           ii], head_width=0.1, head_length=0.2, color='m')

            self.linear_velocity = np.sum(
                velocities * np.tile(weights, (self.dimension, 1)), axis=1
            )

            # normalization to the initial velocity
            self.linear_velocity = (
                initial_magnitude * self.linear_velocity / LA.norm(self.linear_velocity)
            )
            # plt.arrow(self.position[0], self.position[1], self.linear_velocity[0],
            #           self.linear_velocity[1], head_width=0.1, head_length=0.2, color='b')

            for ii in range(self._control_points.shape[1]):
                angular_vel[0, ii] = weights[ii] * np.cross(
                    global_control_points[:, ii] - self._shape.center_position,
                    velocities[:, ii] - self.linear_velocity,
                )

            self.angular_velocity = np.sum(angular_vel)
        
        else:
            self.linear_velocity = [0,0]
            self.angular_velocity = 0


    def compute_metrics(self, dt):
        # Compute distance
        if not self.converged:
            self.total_distance += LA.norm(self.linear_velocity) * dt
            self.time_conv += dt

        # compute proximity 
        R = 3 # radius
        distance = []

        for obs in self.get_obstacles_without_me():
            distance.append(obs.get_distance_to_surface(self.position, in_obstacle_frame=False, margin_absolut=self.margin_absolut))
        
        dmin  = min(distance)
        dmin = dmin if dmin < R else R
        dmin = dmin if dmin >0 else 0

        self._proximity += dmin/R
        self._list_prox.append(self._proximity) # Temporary metric used for the prox graph of the report, can be deleted

    def corner_case(self, mini_drag: str = "nodrag", version: str = "v1"):
        
        initial_velocity = np.zeros(2)
        environment_without_me = self.get_obstacles_without_me()
        # TODO : Make it a method to be called outside the class
        global_control_points = self.get_global_control_points()

        weights = self.get_weight_of_control_points(
            global_control_points, environment_without_me
        )
        
        gamma_values = np.zeros(global_control_points.shape[1])
        for ii in range(global_control_points.shape[1]):
            gamma_values[ii] = self.get_gamma_product_crowd(
                global_control_points[:, ii], environment_without_me
            )

        velocities = np.zeros((self.dimension, self._control_points.shape[1]))
        init_velocities = np.zeros((self.dimension, self._control_points.shape[1]))

        if not self._static:
            # TODO : Do we want to enable rotation along other axis in the futur ?
            angular_vel = np.zeros((1, self._control_points.shape[1]))

            # First we compute the initial velocity at the "center", ugly
            initial_velocity = self._dynamics.evaluate(self.position)

            
            initial_velocity = obs_avoidance_interpolation_moving(
                                position=self.position,
                                initial_velocity=initial_velocity,
                                obs=environment_without_me,
                                self_priority=self.priority,)
        
            
            initial_magnitude = LA.norm(initial_velocity)

            # Computing the weights of the angle to reach
            if mini_drag == "dragvel":
                
                w1_hat = self.virtual_drag
                w2_hat_max = 1000
                if LA.norm(initial_velocity) != 0:
                    w2_hat = self._dynamics.maximum_velocity / LA.norm(initial_velocity) - 1
                    if w2_hat > w2_hat_max:
                        w2_hat = w2_hat_max
                else:
                    w2_hat = w2_hat_max

                w1 = w1_hat / (w1_hat + w2_hat)
                w2 = 1 - w1
            elif mini_drag == "dragdist":
                
                d = LA.norm(self.position-self._goal_pose.position)
                kappa = self.virtual_drag
                w1 = 1/2*(1+np.tanh((d*kappa-1.5*kappa)/2))
                w2 = 1- w1

            elif mini_drag == "nodrag":
                w1 = 0
                w2 = 1 
            else :
                print("Error in the name of the type of drag to use")
                w1 = 0
                w2 = 1 
                
            # Direction (angle), of the linear_velocity in the global frame
            lin_vel_dir = np.arctan2(initial_velocity[1], initial_velocity[0])

            # Make the smallest rotation- the furniture has to pi symetric
            drag_angle = lin_vel_dir - self.orientation
            # Case where there is no symetry in the furniture
            if np.abs(drag_angle) > np.pi:
                drag_angle = -1 * (2 * np.pi - drag_angle)

            # Case where we consider for instance PI-symetry for the furniture
            if (
                np.abs(drag_angle) > np.pi / 2
            ):  # np.pi/2 is the value hard coded in case for PI symetry of the furniture, if we want to introduce PI/4 symetry for instance we ahve to change this value
                if self.orientation > 0:
                    orientation_sym = self.orientation - np.pi
                else:
                    orientation_sym = self.orientation + np.pi

                drag_angle = lin_vel_dir - orientation_sym
                if drag_angle > np.pi / 2:
                    drag_angle = -1 * (2 * np.pi - drag_angle)

            goal_angle = self._goal_pose.orientation - self.orientation
            if np.abs(goal_angle) > np.pi:
                goal_angle = -1 * (2 * np.pi - goal_angle)

            if (
                np.abs(goal_angle) > np.pi / 2
            ):  # np.pi/2 is the value hard coded in case for PI symetry of the furniture, if we want to introduce PI/4 symetry for instance we ahve to change this value
                if self.orientation > 0:
                    orientation_sym = self.orientation - np.pi
                else:
                    orientation_sym = self.orientation + np.pi

                goal_angle = self._goal_pose.orientation - orientation_sym
                if goal_angle > np.pi / 2:
                    goal_angle = -1 * (2 * np.pi - goal_angle)

            # TODO Very clunky : Rather make a function out of it
            K = 3  # K proportionnal parameter for the speed
            # Initial angular_velocity is computed
            initial_angular_vel = K * (w1 * drag_angle + w2 * goal_angle)

            sign_project =np.zeros(self._control_points.shape[1])
            for ii in range(self._control_points.shape[1]):
                # doing the cross product formula by "hand" than using the funct
                tang_vel = [
                    -initial_angular_vel * self._control_points[ii, 1],
                    initial_angular_vel * self._control_points[ii, 0],
                ]
                tang_vel = self.get_veloctity_in_global_frame(tang_vel)
                init_velocities[:, ii] = initial_velocity + tang_vel

                ctp = global_control_points[:, ii]
                velocities[:, ii] = obs_avoidance_interpolation_moving(
                    position=ctp,
                    initial_velocity=init_velocities[:, ii],
                    obs=environment_without_me,
                    self_priority=self.priority,
                )

                normal = np.array([1,0])
                velocities_temp = self.get_velocity_in_local_frame(velocities[:,ii])
                sign_project[ii] = np.sign(np.dot(normal, velocities_temp))

                # plt.arrow(ctp[0], ctp[1], init_velocities[0, ii],
                #           init_velocities[1, ii], head_width=0.1, head_length=0.2, color='g')
                # plt.arrow(ctp[0], ctp[1], velocities[0, ii], velocities[1,
                #           ii], head_width=0.1, head_length=0.2, color='m')
            
            gamma_critic = 2
            magnitude =1
            if np.sign(np.prod(sign_project)) < 0:
                if all(gamma_values < gamma_critic):
                    print("ANTI COLLSION")
                    magnitude = 1/(1-gamma_critic)*(min(gamma_values)-gamma_critic)
                else: 
                    magnitude = 1

            self.linear_velocity = np.sum(
                velocities * np.tile(weights, (self.dimension, 1)), axis=1
            )

            # normalization to the initial velocity
            self.linear_velocity = (
                initial_magnitude * self.linear_velocity / LA.norm(self.linear_velocity)
            )*magnitude
            # plt.arrow(self.position[0], self.position[1], self.linear_velocity[0],
            #           self.linear_velocity[1], head_width=0.1, head_length=0.2, color='b')

            for ii in range(self._control_points.shape[1]):
                angular_vel[0, ii] = weights[ii] * np.cross(
                    global_control_points[:, ii] - self._shape.center_position,
                    velocities[:, ii] - self.linear_velocity,
                )

            self.angular_velocity = np.sum(angular_vel)*magnitude
        
        else:
            self.linear_velocity = [0,0]
            self.angular_velocity = 0

    def emergency_stop(self, mini_drag: str = "nodrag", version: str = "v2"):
        
        initial_velocity = np.zeros(2)
        environment_without_me = self.get_obstacles_without_me()
        # TODO : Make it a method to be called outside the class
        global_control_points = self.get_global_control_points()

        weights = self.get_weight_of_control_points(
            global_control_points, environment_without_me
        )
        
        gamma_values = np.zeros(global_control_points.shape[1])
        obs_idx = np.zeros(global_control_points.shape[1]) # Idx of the obstacle in the environment where the Gamma is calculated from
        for ii in range(global_control_points.shape[1]):
            obs_idx[ii], gamma_values[ii] = self.get_gamma_product_crowd(
                global_control_points[:, ii], environment_without_me
            )
        
        # Check if the gamma function is below a critical value
        gamma_critic = 2
        list_critic_gammas = []
        for ii in range(global_control_points.shape[1]):
            if gamma_values[ii] < gamma_critic:
                list_critic_gammas.append(ii)
        

        # If so check if the final velocity is going toward the normal to the close obstacle
        # If yes decrease the velocity regarding the angle between the final velocity and the normal to the obstacle
        
        velocities = np.zeros((self.dimension, self._control_points.shape[1]))
        init_velocities = np.zeros((self.dimension, self._control_points.shape[1]))

        if not self._static:
            # TODO : Do we want to enable rotation along other axis in the futur ?
            angular_vel = np.zeros((1, self._control_points.shape[1]))

            # First we compute the initial velocity at the "center", ugly
            initial_velocity = self._dynamics.evaluate(self.position)

            
            initial_velocity = obs_avoidance_interpolation_moving(
                                position=self.position,
                                initial_velocity=initial_velocity,
                                obs=environment_without_me,
                                self_priority=self.priority,)
        
            
            initial_magnitude = LA.norm(initial_velocity)

            # Computing the weights of the angle to reach
            if mini_drag == "dragvel":
                
                w1_hat = self.virtual_drag
                w2_hat_max = 1000
                if LA.norm(initial_velocity) != 0:
                    w2_hat = self._dynamics.maximum_velocity / LA.norm(initial_velocity) - 1
                    if w2_hat > w2_hat_max:
                        w2_hat = w2_hat_max
                else:
                    w2_hat = w2_hat_max

                w1 = w1_hat / (w1_hat + w2_hat)
                w2 = 1 - w1
            elif mini_drag == "dragdist":
                
                d = LA.norm(self.position-self._goal_pose.position)
                kappa = self.virtual_drag
                w1 = 1/2*(1+np.tanh((d*kappa-1.5*kappa)/2))
                w2 = 1- w1

            elif mini_drag == "nodrag":
                w1 = 0
                w2 = 1 
            else :
                print("Error in the name of the type of drag to use")
                w1 = 0
                w2 = 1 
                
            # Direction (angle), of the linear_velocity in the global frame
            lin_vel_dir = np.arctan2(initial_velocity[1], initial_velocity[0])

            # Make the smallest rotation- the furniture has to pi symetric
            drag_angle = lin_vel_dir - self.orientation
            # Case where there is no symetry in the furniture
            if np.abs(drag_angle) > np.pi:
                drag_angle = -1 * (2 * np.pi - drag_angle)

            # Case where we consider for instance PI-symetry for the furniture
            if (
                np.abs(drag_angle) > np.pi / 2
            ):  # np.pi/2 is the value hard coded in case for PI symetry of the furniture, if we want to introduce PI/4 symetry for instance we ahve to change this value
                if self.orientation > 0:
                    orientation_sym = self.orientation - np.pi
                else:
                    orientation_sym = self.orientation + np.pi

                drag_angle = lin_vel_dir - orientation_sym
                if drag_angle > np.pi / 2:
                    drag_angle = -1 * (2 * np.pi - drag_angle)

            goal_angle = self._goal_pose.orientation - self.orientation
            if np.abs(goal_angle) > np.pi:
                goal_angle = -1 * (2 * np.pi - goal_angle)

            if (
                np.abs(goal_angle) > np.pi / 2
            ):  # np.pi/2 is the value hard coded in case for PI symetry of the furniture, if we want to introduce PI/4 symetry for instance we ahve to change this value
                if self.orientation > 0:
                    orientation_sym = self.orientation - np.pi
                else:
                    orientation_sym = self.orientation + np.pi

                goal_angle = self._goal_pose.orientation - orientation_sym
                if goal_angle > np.pi / 2:
                    goal_angle = -1 * (2 * np.pi - goal_angle)

            # TODO Very clunky : Rather make a function out of it
            K = 3  # K proportionnal parameter for the speed
            # Initial angular_velocity is computed
            initial_angular_vel = K * (w1 * drag_angle + w2 * goal_angle)

            sign_project =np.zeros(self._control_points.shape[1])
            for ii in range(self._control_points.shape[1]):
                # doing the cross product formula by "hand" than using the funct
                tang_vel = [
                    -initial_angular_vel * self._control_points[ii, 1],
                    initial_angular_vel * self._control_points[ii, 0],
                ]
                tang_vel = self.get_veloctity_in_global_frame(tang_vel)
                init_velocities[:, ii] = initial_velocity + tang_vel

                ctp = global_control_points[:, ii]
                velocities[:, ii] = obs_avoidance_interpolation_moving(
                    position=ctp,
                    initial_velocity=init_velocities[:, ii],
                    obs=environment_without_me,
                    self_priority=self.priority,
                )

            self.linear_velocity = np.sum(
                velocities * np.tile(weights, (self.dimension, 1)), axis=1
            )

            for ii in range(self._control_points.shape[1]):
                angular_vel[0, ii] = weights[ii] * np.cross(
                    global_control_points[:, ii] - self._shape.center_position,
                    velocities[:, ii] - self.linear_velocity,
                )

            self.angular_velocity = np.sum(angular_vel)
            
            if list_critic_gammas is not None:
                for ii in list_critic_gammas:
                    
                    # This only works if control points are on the longest axis of the cuboid, calculation of Omega x R + linear_velocity
                    instant_velocity = [0, self.angular_velocity*self._control_points[0][ii]] + self.get_velocity_in_local_frame(self.linear_velocity) 

                    temp = [0, self.angular_velocity*self._control_points[ii][0]]

                    normal = environment_without_me[ii].get_normal_direction(self.get_global_control_points()[:,ii], in_obstacle_frame=False)
                    plt.arrow(self.get_global_control_points()[0][ii], self.get_global_control_points()[1][ii], temp[0],
                                temp[1], head_width=0.1, head_length=0.2, color='b')

                    plt.arrow(self.get_global_control_points()[0][ii], self.get_global_control_points()[1][ii], normal[0], normal[1],
                                head_width=0.1, head_length=0.2, color='r')
                    if np.dot(instant_velocity, normal) < 0:
                        print("Collision trajectory")
                    else:
                        print("Not in collision trajectory")
        
        else:
            self.linear_velocity = [0,0]
            self.angular_velocity = 0


class Person(BaseAgent):
    def __init__(
        self, priority_value: float = 1, center_position=None, radius=0.5, margin: float =1, **kwargs
    ) -> None:
        _shape = EllipseWithAxes(
            center_position=np.array(center_position),
            margin_absolut=margin,
            orientation=0,
            tail_effect=False,
            axes_length=np.array([radius, radius]),
        )

        super().__init__(
            shape=_shape,
            priority_value=priority_value,
            control_points=np.array([[0, 0]]),
            **kwargs
        )  # ALWAYS USE np.array([[0,0]]) and not np.array([0,0])

        self._dynamics = LinearSystem(
            attractor_position=self.position, maximum_velocity=1
        )
        
    @property
    def margin_absolut(self):
        return self._shape.margin_absolut


    def update_velocity(self, **kwargs):
        environment_without_me = self.get_obstacles_without_me()
        global_control_points = self.get_global_control_points()
        global_goal_control_points = self.get_goal_control_points()
        # Person only holds one control point, thus modulation is simplified
        # 0 hardcoded because we assume in person, we will have only one control point
        ctp = global_control_points[:, 0]
        # 0 hardcoded because we assume in person, we will have only one control point
        self._dynamics.attractor_position = global_goal_control_points[:, 0]
        
        if not self._static:
            initial_velocity = self._dynamics.evaluate(ctp)
            velocity = obs_avoidance_interpolation_moving(
                position=ctp,
                initial_velocity=initial_velocity,
                obs=environment_without_me,
                self_priority=self.priority,
            )

            self.linear_velocity = velocity
        else:
            self.linear_velocity = [0,0]
        

    def compute_metrics(self, delta_t):
        # compute the proximity
        R = 3 # radius
        distance = []

        for obs in self.get_obstacles_without_me():
            distance.append(obs.get_distance_to_surface(self.position, in_obstacle_frame=False, margin_absolut=self.margin_absolut))
        
        dmin  = min(distance)
        dmin = dmin if dmin < R else R
        dmin = dmin if dmin >0 else 0
        self._proximity += dmin/R
        self._list_prox.append(self._proximity) # Temporary metric used for the prox graph of the report, can be deleted
    
    def get_distance_to_surface(self, position, in_obstacle_frame: bool = True, margin_absolut: float = None):
        self.get_point_on_surface(position=position,
        in_obstacle_frame=in_obstacle_frame,
        margin_absolut=margin_absolut,)
         
