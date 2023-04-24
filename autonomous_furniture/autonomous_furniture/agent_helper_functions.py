import numpy as np
from numpy import linalg as LA
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving


def apply_linear_and_angular_acceleration_constraints(
    linear_velocity_old,
    angular_velocity_old,
    linear_velocity,
    angular_velocity,
    maximum_linear_acceleration,
    maximum_angular_acceleration,
    time_step,
):
    # This function checks whether the difference in new computed kinematics and old kinematics exceeds the acceleration limits
    linear_velocity_difference = linear_velocity - linear_velocity_old
    angular_velocity_difference = angular_velocity - angular_velocity_old
    linear_velocity_difference_allowed = maximum_linear_acceleration * time_step
    angular_velocity_difference_allowed = maximum_angular_acceleration * time_step

    if LA.norm(linear_velocity_difference) > linear_velocity_difference_allowed:
        vel_correction = (
            linear_velocity_difference
            / LA.norm(linear_velocity_difference)
            * linear_velocity_difference_allowed
        )
        # plt.clf()
        # plt.arrow(self.position[0], self.position[1], linear_velocity_old[0],
        #           linear_velocity_old[1], head_width=0.01, head_length=0.01, color='red')
        # plt.arrow(self.position[0], self.position[1], self.linear_velocity[0],
        #           self.linear_velocity[1], head_width=0.01, head_length=0.01, color='yellow')
        # plt.arrow(self.position[0]+linear_velocity_old[0], self.position[1]+linear_velocity_old[1], linear_velocity_difference[0],
        #           linear_velocity_difference[1], head_width=0.01, head_length=0.01, color='purple')

        linear_velocity = linear_velocity_old + vel_correction

        # plt.arrow(self.position[0], self.position[1], self.linear_velocity[0],
        #           self.linear_velocity[1], head_width=0.01, head_length=0.01, color='green')
        # plt.arrow(self.position[0]+linear_velocity_old[0], self.position[1]+linear_velocity_old[1], vel_correction[0],
        #           vel_correction[1], head_width=0.01, head_length=0.01, color='blue')

    if LA.norm(angular_velocity_difference) > angular_velocity_difference_allowed:
        angular_velocity = (
            angular_velocity_old
            + angular_velocity_difference
            / LA.norm(angular_velocity_difference)
            * angular_velocity_difference_allowed
        )

    return linear_velocity, angular_velocity


def compute_gamma_critic(d, d_critic, gamma_critic_max, gamma_critic_min):
    # compute gamma critic
    if d > d_critic:
        gamma_critic = gamma_critic_max
    else:
        gamma_critic = (
            gamma_critic_min + d * (gamma_critic_max - gamma_critic_min) / d_critic
        )

    return gamma_critic


def compute_ctr_point_vel_from_obs_avoidance(
    number_ctrpt, goal_pos_ctr_pts, actual_pos_ctr_pts, environment_without_me, priority
):
    velocities = np.zeros((2, number_ctrpt))
    for i in range(number_ctrpt):
        # define direction as initial velocities
        ctr_pt_i = np.array(
            [actual_pos_ctr_pts[0][i], actual_pos_ctr_pts[1][i]]
        )  # extract i-th actual control points position
        ctr_pt_i_goal = np.array(
            [goal_pos_ctr_pts[0][i], goal_pos_ctr_pts[1][i]]
        )  # extract i-th goal control points position
        initial_velocity = ctr_pt_i_goal - ctr_pt_i
        # initial_velocity /= LA.norm(initial_velocity) #normalize vector
        velocities[:, i] = obs_avoidance_interpolation_moving(
            position=ctr_pt_i,
            initial_velocity=initial_velocity,
            obs=environment_without_me,
            self_priority=priority,
        )
    return velocities


def compute_drag_angle(initial_velocity, actual_orientation):
    # Direction (angle), of the linear_velocity in the global frame
    lin_vel_dir = np.arctan2(initial_velocity[1], initial_velocity[0])

    # Make the smallest rotation- the furniture has to pi symetric
    drag_angle = lin_vel_dir - actual_orientation
    # Case where there is no symetry in the furniture
    if np.abs(drag_angle) > np.pi:
        drag_angle = -1 * (2 * np.pi - drag_angle)

    # Case where we consider for instance PI-symetry for the furniture
    if (
        np.abs(drag_angle) > np.pi / 2
    ):  # np.pi/2 is the value hard coded in case for PI symetry of the furniture, if we want to introduce PI/4 symetry for instance we have to change this value
        if actual_orientation > 0:
            orientation_sym = actual_orientation - np.pi
        else:
            orientation_sym = actual_orientation + np.pi

        drag_angle = lin_vel_dir - orientation_sym
        if drag_angle > np.pi / 2:
            drag_angle = -1 * (2 * np.pi - drag_angle)
    return drag_angle


def compute_goal_angle(goal_orientation, actual_orientation):
    # compute orientation difference to reach goal
    goal_angle = goal_orientation - actual_orientation
    if np.abs(goal_angle) > np.pi:
        goal_angle = -1 * (2 * np.pi - goal_angle)

    if (
        np.abs(goal_angle) > np.pi / 2
    ):  # np.pi/2 is the value hard coded in case for PI symetry of the furniture, if we want to introduce PI/4 symetry for instance we ahve to change this value
        if actual_orientation > 0:
            orientation_sym = actual_orientation - np.pi
        else:
            orientation_sym = actual_orientation + np.pi

        goal_angle = goal_orientation - orientation_sym
        if goal_angle > np.pi / 2:
            goal_angle = -1 * (2 * np.pi - goal_angle)
    return goal_angle


def ctr_point_vel_from_agent_kinematics(
    initial_angular_vel,
    initial_velocity,
    number_ctrpt,
    local_control_points,
    global_control_points,
    actual_orientation,
    environment_without_me,
    priority,
    DSM,
):
    ### CALCULATE THE VELOCITY OF THE CONTROL POINTS GIVEN THE INITIAL ANGULAR AND LINEAR VELOCITY OF THE AGENT ###
    velocities = np.zeros((2, number_ctrpt))

    for ii in range(number_ctrpt):
        # doing the cross product formula by "hand" than using the funct
        tang_vel = [
            -initial_angular_vel * local_control_points[ii, 1],
            initial_angular_vel * local_control_points[ii, 0],
        ]
        tang_vel = get_veloctity_in_global_frame(actual_orientation, tang_vel)
        velocities[:, ii] = initial_velocity + tang_vel

        ctp = global_control_points[:, ii]
        if DSM:
            velocities[:, ii] = obs_avoidance_interpolation_moving(
                position=ctp,
                initial_velocity=velocities[:, ii],
                obs=environment_without_me,
                self_priority=priority,
            )
    return velocities


def agent_kinematics_from_ctr_point_vel(
    velocities, weights, global_control_points, ctrpt_number, global_reference_position
):
    ### CALCULATE FINAL LINEAR AND ANGULAT VELOCITY OF AGENT GIVEN THE LINEAR VELOCITY OF EACH CONTROL POINT ###
    # linear velocity
    linear_velocity = np.sum(velocities * np.tile(weights, (2, 1)), axis=1)
    # if initial_magnitude == None:
    #     # check whether velocity is greater than maximum speed
    #     if LA.norm(self.linear_velocity) > self._dynamics.maximum_velocity:
    #         self.linear_velocity = (
    #             self._dynamics.maximum_velocity
    #             * self.linear_velocity
    #             / LA.norm(self.linear_velocity)
    #         )
    # else:
    #     # normalization to the initial linear velocity
    #     self.linear_velocity = (
    #         initial_magnitude * self.linear_velocity / LA.norm(self.linear_velocity)
    #     )
    # angular velocity
    angular_vel = np.zeros(ctrpt_number)
    for ii in range(ctrpt_number):
        angular_vel[ii] = weights[ii] * np.cross(
            global_control_points[:, ii] - global_reference_position,
            velocities[:, ii] - linear_velocity,
        )
    angular_velocity = np.sum(angular_vel)

    return linear_velocity, angular_velocity


def compute_ang_weights(mini_drag, d, virtual_drag):
    if mini_drag == "dragdist":  # a1 computed as in the paper depending on the distance
        kappa = virtual_drag
        k = 0.01
        r = d / (d + k)
        alpha = 1.5
        w1 = 1 / 2 * (1 + np.tanh(kappa * (d - alpha))) * r
        w2 = 1 - w1

    elif mini_drag == "nodrag":  # no virtual drag
        w1 = 0
        w2 = 1
    else:
        print("Error in the name of the type of drag to use")
        w1 = 0
        w2 = 1

    return w1, w2


def evaluate_safety_repulsion(
    list_critic_gammas_indx: list[int],
    environment_without_me,
    global_control_points: np.ndarray,
    obs_idx: list[int],
    gamma_values: np.ndarray,
    velocities,
    gamma_critic,
    local_control_points,
) -> None:
    (
        gamma_list_colliding,  # list with critical gamma value
        normal_list_tot,  # list with all normal directions of critical obstacles for each control point
        weight_list_tot,  # list with all weights of each normal direction for each critical obstacle
        normals_for_ang_vel,  # list with already weighted normal direction for each control point
        control_point_d_list,  # list with distance from each control point with critical obstacle to center of mass
    ) = collect_infos_for_crit_ctr_points(
        list_critic_gammas_indx,
        environment_without_me,
        local_control_points,
        global_control_points,
        obs_idx,
        gamma_values,
        gamma_critic,
    )
    # # CALCULATE THE OPTIMAL ESCAPE DIRECTION AND THE ANGULAR VELOCITY CORRECTION TERM
    # normal_list_tot_combined = []
    # weight_list_tot_combined = []
    # ang_vel_weights = []
    # ang_vel_corr = []
    # for i in range(len(normal_list_tot)):
    #     normal_list_tot_combined += normal_list_tot[i]
    #     weight_list_tot_combined += weight_list_tot[i]
    #     normal_in_local_frame = self.get_velocity_in_local_frame(
    #         normals_for_ang_vel[i]
    #     )
    #     ang_vel_corr.append(normal_in_local_frame[1] * control_point_d_list[i])
    #     ang_vel_weights.append(1 / gamma_list_colliding[i])

    # weight_list_tot_combined = weight_list_tot_combined / np.sum(
    #     weight_list_tot_combined
    # )  # normalize weights
    # normal_combined = np.sum(
    #     normal_list_tot_combined
    #     * np.tile(weight_list_tot_combined, (self.dimension, 1)).transpose(),
    #     axis=0,
    # )  # calculate the escape direction given all obstacles proximity

    # ang_vel_weights = ang_vel_weights / np.sum(ang_vel_weights)  # normalize weights
    # ang_vel_corr = np.sum(
    #     ang_vel_corr * np.tile(ang_vel_weights, (1, 1)).transpose(), axis=0
    # )

    # if np.dot(self.linear_velocity, normal_combined) < 0:
    #     # the is a colliding trajectory we need to correct!
    #     b = 1 / (
    #         (self.gamma_critic - 1) * (np.min(gamma_list_colliding) - 1)
    #     )  # compute the correction parameter
    #     self.linear_velocity += (
    #         b * normal_combined
    #     )  # correct linear velocity to deviate it away from collision trajectory

    #     self.angular_velocity += (
    #         ang_vel_corr * b
    #     )  # correct angular velocity to rotate in a safer position
    #     self.angular_velocity = self.angular_velocity[
    #         0
    #     ]  # make angular velocity a scalar instead of a 1x1 array

    for i in range(len(gamma_list_colliding)):
        ctrpt_indx = np.where(gamma_values == gamma_list_colliding[i])[0][0]
        if np.dot(velocities[:, ctrpt_indx], normals_for_ang_vel[i]) < 0:
            b = 1 / ((gamma_critic - 1) * (gamma_list_colliding[i] - 1))
            velocities[:, ctrpt_indx] += b * normals_for_ang_vel[i]

    return velocities


def apply_velocity_constraints(
    linear_velocity, angular_velocity, maximum_linear_velocity, maximum_angular_velocity
):
    if (
        LA.norm(linear_velocity) > maximum_linear_velocity
    ):  # resize speed if it passes maximum speed
        linear_velocity *= maximum_linear_velocity / LA.norm(linear_velocity)

    if (
        LA.norm(angular_velocity) > maximum_angular_velocity
    ):  # resize speed if it passes maximum speed
        angular_velocity = (
            angular_velocity / LA.norm(angular_velocity) * maximum_angular_velocity
        )

    return linear_velocity, angular_velocity


def collect_infos_for_crit_ctr_points(
    list_critic_gammas_indx,
    environment_without_me,
    local_control_points,
    global_control_points,
    obs_idx,
    gamma_values,
    gamma_critic,
):
    normal_list_tot = []
    weight_list_tot = []
    normals_for_ang_vel = []
    gamma_list_colliding = []
    control_point_d_list = []
    for ii in list_critic_gammas_indx:
        # get all the critical normal directions for the given control point
        normal_list = []
        gamma_list = []
        for j, obs in enumerate(environment_without_me):
            # gamma_type needs to be implemented for all obstacles
            gamma = obs.get_gamma(global_control_points[:, ii], in_global_frame=True)
            if gamma < gamma_critic:
                normal = environment_without_me[obs_idx[ii]].get_normal_direction(
                    global_control_points[:, ii],
                    in_obstacle_frame=False,
                )
                normal_list.append(normal)
                gamma_list.append(gamma)
        # weight the critical normal directions depending on its gamma value
        n_obs_critic = len(normal_list)
        weight_list = []
        for j in range(n_obs_critic):
            weight = 1 / (gamma_list[j] - 1)
            weight_list.append(weight)
        weight_list_prov = weight_list / np.sum(
            weight_list
        )  # normalize weights but only to calculate normal for this ctrpoint
        # calculate the escape direction to avoid collision
        normal = np.sum(
            normal_list * np.tile(weight_list_prov, (2, 1)).transpose(),
            axis=0,
        )
        normal = normal / LA.norm(normal)

        # plt.arrow(self.get_global_control_points()[0][ii], self.get_global_control_points()[1][ii], instant_velocity[0],
        #             instant_velocity[1], head_width=0.1, head_length=0.2, color='b')
        gamma_list_colliding.append(gamma_values[ii])

        normal_list_tot.append(normal_list)
        weight_list_tot.append(weight_list)
        normals_for_ang_vel.append(normal)
        control_point_d_list.append(local_control_points[ii][0])

    return (
        gamma_list_colliding,
        normal_list_tot,
        weight_list_tot,
        normals_for_ang_vel,
        control_point_d_list,
    )


def get_veloctity_in_global_frame(orientation, velocity) -> np.ndarray:
    """Returns the transform of the velocity from relative to global frame."""
    R = np.array(
        [
            [np.cos(orientation), (-1) * np.sin(orientation)],
            [np.sin(orientation), np.cos(orientation)],
        ]
    )
    velocity = np.dot(R, velocity)
    return velocity


def get_gamma_product_crowd(position, env):
    if not len(env):
        # Very large number
        return 1e20

    gamma_list = np.zeros(len(env))
    for ii, obs in enumerate(env):
        gamma_list[ii] = obs.get_gamma(position, in_global_frame=True)

    n_obs = len(gamma_list)

    # Total gamma [1, infinity]
    # Take root of order 'n_obs' to make up for the obstacle multiple
    if any(gamma_list < 1):
        # BaseAgent.number_collisions += 1
        # if show_collision_info:
        #     print("[INFO] Collision")
        return 0, 0

    gamma = np.min(gamma_list)
    index = int(np.argmin(gamma_list))

    if np.isnan(gamma):
        # Debugging
        breakpoint()
    return index, gamma


def get_weight_from_gamma(
    gammas, cutoff_gamma, n_points, gamma0=1.0, frac_gamma_nth=0.5
):
    weights = (gammas - gamma0) / (cutoff_gamma - gamma0)
    weights = weights / frac_gamma_nth
    weights = 1.0 / weights
    weights = (weights - frac_gamma_nth) / (1 - frac_gamma_nth)
    weights = weights / n_points
    # in case there are some critical gammas, ignore the rest
    # critic_indx = np.where(gammas<1.3)[0]
    # if np.any(critic_indx):
    #     for i in range(len(weights)):
    #         if not np.any(np.where(critic_indx==i)[0]):
    #             weights[i] = 1e-3
    return weights


def get_weight_of_control_points(control_points, environment_without_me):
    cutoff_gamma = 10  # TODO : This value has to be big and not small
    # gamma_values = self.get_gamma_at_control_point(control_points[self.obs_multi_agent[obs]], obs, temp_env)
    gamma_values = np.zeros(control_points.shape[1])
    obs_idx = np.zeros(control_points.shape[1])
    for ii in range(control_points.shape[1]):
        obs_idx[ii], gamma_values[ii] = get_gamma_product_crowd(
            control_points[:, ii], environment_without_me
        )

    ctl_point_weight = np.zeros(gamma_values.shape)
    ind_nonzero = gamma_values < cutoff_gamma
    if not any(ind_nonzero):  # TODO Case he there is ind_nonzero
        # ctl_point_weight[-1] = 1
        ctl_point_weight = np.full(gamma_values.shape, 1 / control_points.shape[1])
    # for index in range(len(gamma_values)):
    ctl_point_weight[ind_nonzero] = get_weight_from_gamma(
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


def update_multi_layer_simulation(
    number_layer,
    number_agent,
    layer_list,
    mini_drag,
    version,
    emergency_stop,
    safety_module,
    dt_simulation,
    agent_pos_saver,
):
    for k in range(number_layer):
        # calculate the agents velocity in each layer
        for jj in range(number_agent):
            if not layer_list[k][jj] == None:
                layer_list[k][jj].update_velocity(
                    mini_drag=mini_drag,
                    version=version,
                    emergency_stop=emergency_stop,
                    safety_module=safety_module,
                    time_step=dt_simulation,
                )
    for jj in range(number_agent):
        # in case the agent has an emergency stop triggered in one of the layers set kinematics to zero
        stop_triggerred = False
        for k in range(number_layer):
            if layer_list[k][jj].stop:
                linear_velocity = 0.0
                angular_velocity = 0.0
                for l in range(number_layer):
                    layer_list[l][jj].linear_velocity = linear_velocity
                    layer_list[l][jj].angular_velocity = angular_velocity
                stop_triggerred = True
                break
        if not stop_triggerred:
            # collect the velocities of each layer for each agent
            agent_linear_velocities = np.zeros((2, number_layer))
            agent_angular_velocities = np.zeros((number_layer))
            for k in range(number_layer):
                agent_linear_velocities[:, k] = np.copy(
                    layer_list[k][jj].linear_velocity
                )
                agent_angular_velocities[k] = np.copy(
                    layer_list[k][jj].angular_velocity
                )
            # weight each layer for this specific agent
            weights = compute_layer_weights(
                jj,
                number_layer,
                layer_list,
            )  ####     NEEDS TO BE CHANGED!!!!  ######
            # calculate the weighted linear and angular velocity for the agent and overwrite the kinematics of each layer
            linear_velocity = np.sum(
                agent_linear_velocities * np.tile(weights, (2, 1)), axis=1
            )
            angular_velocity = np.sum(
                agent_angular_velocities * np.tile(weights, (1, 1))
            )

            # update each layers positions and orientations
            for k in range(number_layer):
                layer_list[k][jj].linear_velocity = linear_velocity
                layer_list[k][jj].angular_velocity = angular_velocity
                layer_list[k][jj].apply_kinematic_constraints()
                layer_list[k][jj].do_velocity_step(dt_simulation)
                agent_pos_saver[jj].append(layer_list[k][jj]._reference_pose.position)

    return layer_list, agent_pos_saver


def compute_layer_weights(agent_number, number_layer, layer_list):
    gamma_list = np.zeros(number_layer)
    for k in range(number_layer):
        gamma_list[k] = layer_list[k][agent_number].min_gamma
    weights = get_weight_from_gamma(
        gammas=gamma_list, cutoff_gamma=10, n_points=number_layer
    )
    weights = weights / np.sum(weights)
    return weights
