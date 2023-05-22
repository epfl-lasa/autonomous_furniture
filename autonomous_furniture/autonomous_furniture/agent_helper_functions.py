import numpy as np
from numpy import linalg as LA
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving
import yaml
from dynamic_obstacle_avoidance.visualization import plot_obstacles
import matplotlib.pyplot as plt


def apply_linear_and_angular_acceleration_constraints(
    linear_velocity_old,
    angular_velocity_old,
    linear_velocity,
    angular_velocity,
    maximum_linear_acceleration,
    maximum_angular_acceleration,
    time_step,
):
    # This function checks whether the difference in new computed kinematics and old kinematics exceeds the acceleration limits and adapts the kinematics in case th elimits are exceeded
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
        linear_velocity = linear_velocity_old + vel_correction

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
    number_ctrpt,
    goal_pos_ctr_pts,
    actual_pos_ctr_pts,
    environment_without_me,
    priority,
    cutoff_gamma_obs,
):
    # This function calculates all the control point velocities using DSM
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
        environment_without_me_adapted = []
        for k in range(len(environment_without_me)):
            obs = environment_without_me[k]
            gamma = obs.get_gamma(ctr_pt_i, in_global_frame=True)
            if gamma < cutoff_gamma_obs:
                environment_without_me_adapted.append(obs)
        velocities[:, i] = obs_avoidance_interpolation_moving(
            position=ctr_pt_i,
            initial_velocity=initial_velocity,
            obs=environment_without_me_adapted,
            self_priority=priority,
        )

    return attenuate_DSM_velocities(number_ctrpt, velocities)
    # return velocities


def compute_drag_angle(initial_velocity, actual_orientation):
    # This function caomputed the drag angle which is the angle between the linear velocity of the reference point and the orientation with least virtual drag

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
    # if np.abs(goal_angle) > np.pi:
    #     goal_angle = -1 * (2 * np.pi - goal_angle)

    # if (
    #     np.abs(goal_angle) > np.pi / 2
    # ):  # np.pi/2 is the value hard coded in case for PI symetry of the furniture, if we want to introduce PI/4 symetry for instance we ahve to change this value
    #     if actual_orientation > 0:
    #         orientation_sym = actual_orientation - np.pi
    #     else:
    #         orientation_sym = actual_orientation + np.pi

    #     goal_angle = goal_orientation - orientation_sym
    #     if goal_angle > np.pi / 2:
    #         goal_angle = -1 * (2 * np.pi - goal_angle)
    return goal_angle


def ctr_point_vel_from_agent_kinematics(
    initial_angular_vel,
    initial_velocity,
    number_ctrpt,
    global_control_points,
    environment_without_me,
    priority,
    DSM,
    reference_position,
    cutoff_gamma_obs,
):
    ### CALCULATE THE VELOCITY OF THE CONTROL POINTS GIVEN THE INITIAL ANGULAR AND LINEAR VELOCITY OF THE AGENT ###
    velocities = np.zeros((2, number_ctrpt))

    for i in range(number_ctrpt):
        control_point_global_relative = global_control_points[:, i] - reference_position
        velocity_3D = np.append(initial_velocity, 0.0) + np.cross(
            np.array([0.0, 0.0, initial_angular_vel]),
            np.append(control_point_global_relative, 0.0),
        )
        velocities[:, i] = velocity_3D[0:2]

        ctp = global_control_points[:, i]
        if DSM:
            environment_without_me_adapted = []
            for k in range(len(environment_without_me)):
                obs = environment_without_me[k]
                ctr_pt_i = np.array(
                    [global_control_points[0][i], global_control_points[1][i]]
                )
                gamma = obs.get_gamma(ctr_pt_i, in_global_frame=True)
                if gamma < cutoff_gamma_obs:
                    environment_without_me_adapted.append(obs)
            velocities[:, i] = obs_avoidance_interpolation_moving(
                position=ctp,
                initial_velocity=velocities[:, i],
                obs=environment_without_me_adapted,
                self_priority=priority,
            )

    return velocities


def agent_kinematics_from_ctr_point_vel(
    velocities, weights, global_control_points, ctrpt_number, global_reference_position
):
    # CALCULATE FINAL LINEAR AND ANGULAT VELOCITY OF AGENT GIVEN THE LINEAR VELOCITY OF EACH CONTROL POINT ###
    cotrol_points_relative_global = []
    for i in range(ctrpt_number):
        cotrol_points_relative_global.append(
            global_control_points[:, i] - global_reference_position
        )

    N = 2 * ctrpt_number
    A = np.zeros((N, 3))
    b = np.zeros((N))
    w_diag = np.zeros((N))

    for i in range(ctrpt_number):
        A[2 * i : 2 * i + 2, 0:2] = np.eye(2)
        A[2 * i : 2 * i + 2, 2] = np.array(
            [-cotrol_points_relative_global[i][1], cotrol_points_relative_global[i][0]]
        )
        b[2 * i : 2 * i + 2] = velocities[:, i]
        w_diag[2 * i : 2 * i + 2] = np.array([weights[i], weights[i]])
    W = np.diag(w_diag)
    Aw = np.dot(W, A)
    bw = np.dot(b, W)
    x = np.linalg.lstsq(Aw, bw, rcond=None)

    linear_velocity = x[0][0:2]
    angular_velocity = x[0][2]

    return linear_velocity, angular_velocity


def compute_ang_weights(mini_drag, d, virtual_drag, k, alpha):
    # This function computes the amount of virtual drag that should be used (a1)

    if mini_drag:  # virtual drag
        r = d / (d + k)
        a1 = (
            1
            / 2
            * (1 + np.tanh(virtual_drag * (d - alpha)))
            * r
            * (virtual_drag - 1)
            / (virtual_drag - 1 + 1e-6)
        )
        a2 = 1 - a1

    else:  # no virtual drag
        a1 = 0
        a2 = 1

    return a1, a2


def evaluate_safety_repulsion(
    list_critic_gammas_indx: list[int],
    environment_without_me,
    global_control_points: np.ndarray,
    obs_idx: list[int],
    gamma_values: np.ndarray,
    velocities,
    gamma_critic,
    safety_gain,
) -> None:
    # This function takes the control point velocities and checkes wether any of the control point velocities need to be modulated using the safety module and modulates those

    (
        gamma_list_colliding,  # list with critical gamma value
        normals_collision,  # list with already weighted normal direction for each control point
    ) = collect_infos_for_crit_ctr_points(
        list_critic_gammas_indx,
        environment_without_me,
        global_control_points,
        obs_idx,
        gamma_values,
        gamma_critic,
    )

    for i in range(len(gamma_list_colliding)):
        ctrpt_indx = np.where(gamma_values == gamma_list_colliding[i])[0][0]
        if np.dot(velocities[:, ctrpt_indx], normals_collision[i]) < 0:
            b = 1 / ((gamma_critic - 1) * (gamma_list_colliding[i] - 1))
            velocities[:, ctrpt_indx] += safety_gain * b * normals_collision[i]

    return velocities


def apply_velocity_constraints(
    linear_velocity, angular_velocity, maximum_linear_velocity, maximum_angular_velocity
):
    # This function check wether the velocity constraints are resepcted and adapts the linear and angular velocity in case
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
    global_control_points,
    obs_idx,
    gamma_values,
    gamma_critic,
):
    # This function checks wether any control point is on a colliding trajectory with a neighbourg and give back the gamma values of those points and the average normal direction of collision

    normals_collision = []
    gamma_list_colliding = []

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

        gamma_list_colliding.append(gamma_values[ii])

        normals_collision.append(normal)

    return (
        gamma_list_colliding,
        normals_collision,
    )


def get_gamma_product_crowd(position, env):
    # This fuction gives back the smallest gamma value and its index for one control point

    if not len(env):
        # Very large number
        return 1e20

    gamma_list = np.zeros(len(env))
    for ii, obs in enumerate(env):
        gamma_list[ii] = obs.get_gamma(position, in_global_frame=True)

    if any(gamma_list < 1):
        return 0, 0

    gamma = np.min(gamma_list)
    index = int(np.argmin(gamma_list))

    return index, gamma


def get_weight_from_gamma(gammas, cutoff_gamma, n_points, gamma0, frac_gamma_nth):
    # This function calculates the weights of each control point regarding the given gamma list

    weights = (gammas - gamma0) / (cutoff_gamma - gamma0)
    weights = weights / frac_gamma_nth
    weights = 1.0 / weights
    weights = (weights - frac_gamma_nth) / (1 - frac_gamma_nth)
    weights = weights / n_points
    return weights


def get_weight_of_control_points(
    control_points, environment_without_me, cutoff_gamma, gamma0, frac_gamma_nth
):
    # This function calculates the weights of each control point regarding the smallest gamma value of each point
    gamma_values = np.zeros(control_points.shape[1])
    obs_idx = np.zeros(control_points.shape[1])
    for ii in range(control_points.shape[1]):
        obs_idx[ii], gamma_values[ii] = get_gamma_product_crowd(
            control_points[:, ii], environment_without_me
        )

    ctl_point_weight = np.zeros(gamma_values.shape)
    ind_nonzero = gamma_values < cutoff_gamma
    if not any(ind_nonzero):
        ctl_point_weight = np.full(gamma_values.shape, 1 / control_points.shape[1])
    # for index in range(len(gamma_values)):
    ctl_point_weight[ind_nonzero] = get_weight_from_gamma(
        gamma_values[ind_nonzero],
        cutoff_gamma=cutoff_gamma,
        n_points=control_points.shape[1],
        gamma0=gamma0,
        frac_gamma_nth=frac_gamma_nth,
    )

    ctl_point_weight_sum = np.sum(ctl_point_weight)
    ctl_point_weight = ctl_point_weight / ctl_point_weight_sum

    return ctl_point_weight


def update_multi_layer_simulation(
    number_layer,
    number_agent,
    layer_list,
    dt_simulation=None,
    agent_pos_saver=None,
):
    # This function calculates the agent velocities given the list of layers for one step

    for k in range(number_layer):
        # calculate the agents velocity in each layer
        for jj in range(number_agent):
            if not layer_list[k][jj] == None:
                layer_list[k][jj].update_velocity(
                    time_step=dt_simulation,
                )
    for jj in range(number_agent):
        # in case the agent has an emergency stop triggered in one of the layers set kinematics to zero
        stop_triggerred = False
        for k in range(number_layer):
            if not layer_list[k][jj] == None:
                if layer_list[k][jj].stop:
                    linear_velocity = [0.0, 0.0]
                    angular_velocity = 0.0
                    for l in range(number_layer):
                        if not layer_list[l][jj] == None:
                            layer_list[l][jj].linear_velocity = linear_velocity
                            layer_list[l][jj].angular_velocity = angular_velocity
                    stop_triggerred = True
                    break
        if not stop_triggerred:
            # collect the velocities of each layer for each agent
            agent_linear_velocities = []
            agent_angular_velocities = []
            for k in range(number_layer):
                if not layer_list[k][jj] == None:
                    agent_linear_velocities.append(
                        np.copy(layer_list[k][jj].linear_velocity)
                    )
                    agent_angular_velocities.append(
                        np.copy(layer_list[k][jj].angular_velocity)
                    )
            # weight each layer for this specific agent
            weights = compute_layer_weights(
                jj,
                number_layer,
                layer_list,
            )
            # calculate the weighted linear and angular velocity for the agent and overwrite the kinematics of each layer
            linear_velocity = np.zeros((2))
            for i in range(len(agent_linear_velocities)):
                linear_velocity[0] += agent_linear_velocities[i][0] * weights[i]
                linear_velocity[1] += agent_linear_velocities[i][1] * weights[i]

            angular_velocity = np.sum(
                agent_angular_velocities * np.tile(weights, (1, 1))
            )

            # update each layers positions and orientations
            for k in range(number_layer):
                if not layer_list[k][jj] == None:
                    layer_list[k][jj].linear_velocity = linear_velocity
                    layer_list[k][jj].angular_velocity = angular_velocity
                    layer_list[k][jj].apply_kinematic_constraints()
                    if not dt_simulation == None:
                        layer_list[k][jj].do_velocity_step(dt_simulation)
                    if not agent_pos_saver == None:
                        agent_pos_saver[jj].append(
                            layer_list[k][jj]._reference_pose.position
                        )

    if not agent_pos_saver == None:
        return layer_list, agent_pos_saver
    else:
        return layer_list


def compute_layer_weights(agent_number, number_layer, layer_list):
    # This function calculates the weight das each for a piece of furniture layer should have when calculating the final furniture velocitties
    gamma_list = []
    for k in range(number_layer):
        if not layer_list[k][agent_number] == None:
            gamma_list.append(layer_list[k][agent_number].min_gamma)
            gamma0 = layer_list[k][agent_number].gamma0
            frac_gamma_nth = layer_list[k][agent_number].frac_gamma_nth
    weights = get_weight_from_gamma(
        gammas=np.array(gamma_list),
        cutoff_gamma=10,
        n_points=len(gamma_list),
        gamma0=gamma0,
        frac_gamma_nth=frac_gamma_nth,
    )
    weights = weights / np.sum(weights)
    return weights


def attenuate_DSM_velocities(ctr_pt_number, velocities):
    # This function changes the control points linear velocities lenghts in order to have all the same length without influencing the weighting
    norms = LA.norm(velocities, axis=0)
    avg_norm = np.average(norms)
    for i in range(ctr_pt_number):
        velocities[:, i] = velocities[:, i] / LA.norm(velocities[:, i]) * avg_norm

    return velocities


def get_params_from_file(
    agent,
    parameter_file,
    min_drag,
    soft_decoupling,
    safety_module,
    emergency_stop,
    maximum_linear_velocity,
    maximum_angular_velocity,
    maximum_linear_acceleration,
    maximum_angular_acceleration,
    safety_gain,
    gamma_critic_max,
    gamma_critic_min,
    gamma_stop,
    d_critic,
    cutoff_gamma_weights,
    cutoff_gamma_obs,
    static,
    name,
    priority_value,
):
    """This function checks wether any variable already is assigned and if not assigns the value from the parameter file"""

    # check if any non-mandatory variable was defined, otherwise take the value from the json file
    with open(parameter_file, "r") as openfile:
        yaml_object = yaml.safe_load(openfile)

    # used algorithms for agent
    if min_drag == None:
        agent.min_drag = yaml_object["minimize virtual drag"]
    else:
        agent.min_drag = min_drag

    if soft_decoupling == None:
        agent.soft_decoupling = yaml_object["soft decoupling"]
    else:
        agent.soft_decoupling = soft_decoupling

    if safety_module == None:
        agent.safety_module = yaml_object["safety module"]
    else:
        agent.safety_module = safety_module

    if emergency_stop == None:
        agent.emergency_stop = yaml_object["emergency stop"]
    else:
        agent.emergency_stop = emergency_stop

    # kinematic constraints
    if maximum_linear_velocity == None:
        agent.maximum_linear_velocity = yaml_object["maximum linear velocity"]
    else:
        agent.maximum_linear_velocity = maximum_linear_velocity

    if maximum_angular_velocity == None:
        agent.maximum_angular_velocity = yaml_object["maximum angular velocity"]
    else:
        agent.maximum_angular_velocity = maximum_angular_velocity

    if maximum_linear_acceleration == None:
        agent.maximum_linear_acceleration = yaml_object["maximum linear acceleration"]
    else:
        agent.maximum_linear_acceleration = maximum_linear_acceleration

    if maximum_angular_acceleration == None:
        agent.maximum_angular_acceleration = yaml_object["maximum angular acceleration"]
    else:
        agent.maximum_angular_acceleration = maximum_angular_acceleration

    # safety module
    if safety_gain == None:
        agent.safety_gain = yaml_object["safety module gain"]
    else:
        agent.safety_gain = safety_gain

    if (
        gamma_critic_max == None
    ):  # value of gamma_critic before being closer than d_critic
        agent.gamma_critic_max = yaml_object["max gamma critic"]
    else:
        agent.gamma_critic_max = gamma_critic_max

    if (
        gamma_critic_min == None
    ):  # minimal value of gamma_critic as it should stay vigilant
        agent.gamma_critic_min = yaml_object["min gamma critic"]
    else:
        agent.gamma_critic_min = gamma_critic_min

    if (
        gamma_stop == None
    ):  # agent should stop when a ctrpoint reaches a gamma value under this threshold
        agent.gamma_stop = yaml_object["gamma stop"]
    else:
        agent.gamma_stop = gamma_stop

    if d_critic == None:  # distance from which gamma_critic starts shrinking
        agent.d_critic = yaml_object["critical distance"]
    else:
        agent.d_critic = d_critic

    # cutoff gammas
    if cutoff_gamma_weights == None:
        agent.cutoff_gamma_weights = yaml_object[
            "cutoff gamma for control point weights"
        ]
    else:
        agent.cutoff_gamma_weights = cutoff_gamma_weights

    if cutoff_gamma_obs == None:
        agent.cutoff_gamma_obs = yaml_object["cutoff gamma for obstacle environment"]
    else:
        agent.cutoff_gamma_obs = cutoff_gamma_obs

    # static or dynamic agent
    if static == None:
        agent.static = yaml_object["static"]
    else:
        agent.static = static

    # agent name
    if name == "no_name":
        agent.name = yaml_object["name"]
    else:
        agent.name = name

    if priority_value == None:
        agent.priority = yaml_object["priority"]
    else:
        agent.priority = priority_value

    # save variables only for the agent helper functions
    # compute_ang_weights
    agent.k = yaml_object[
        "k"
    ]  # parameter for term d/(d+k) which ensures the virtual drag weight w1 goes to zero when d=0
    agent.alpha = yaml_object[
        "angle switch distance"
    ]  # distance at which the soft decoupling becomes stronger than the virtual drag

    # get_weight_from_gamma
    agent.gamma0 = yaml_object["gamma surface"]  # gamma value on obstacle surface
    agent.frac_gamma_nth = yaml_object["frac_gamma_nth"]  # boh this I don't know

    return agent


def plot_animation(
    ax,
    layer_list,
    number_layer,
    number_agent,
    obstacle_colors,
    agent_pos_saver,
    x_lim,
    y_lim,
):
    ax.clear()
    for k in range(number_layer):
        if len(obstacle_colors) > k:
            color = obstacle_colors[k]
        else:
            color = "black"

        for jj in range(number_agent):
            if not layer_list[k][jj] == None:
                goal_control_points = layer_list[k][
                    jj
                ].get_goal_control_points()  ##plot agent center position

                global_control_points = layer_list[k][jj].get_global_control_points()
                ax.plot(
                    global_control_points[0, :],
                    global_control_points[1, :],
                    color="black",
                    marker=".",
                    linestyle="",
                )

                ax.plot(
                    goal_control_points[0, :],
                    goal_control_points[1, :],
                    color=color,
                    marker=".",
                    linestyle="",
                )

                ax.plot(
                    layer_list[k][jj]._goal_pose.position[0],
                    layer_list[k][jj]._goal_pose.position[1],
                    color=color,
                    marker="*",
                )

                ax.plot(
                    layer_list[k][jj]._reference_pose.position[0],
                    layer_list[k][jj]._reference_pose.position[1],
                    color="black",
                    marker="*",
                )

                x_values = np.zeros(len(agent_pos_saver[jj]))
                y_values = x_values.copy()
                for i in range(len(agent_pos_saver[jj])):
                    x_values[i] = agent_pos_saver[jj][i][0]
                    y_values[i] = agent_pos_saver[jj][i][1]

                ax.plot(
                    x_values,
                    y_values,
                    color=color,
                    linestyle="dashed",
                )

                for i in range(len(layer_list[k][jj]._shape_list)):
                    plot_obstacles(
                        ax=ax,
                        obstacle_container=[layer_list[k][jj]._shape_list[i]],
                        x_lim=x_lim,
                        y_lim=y_lim,
                        showLabel=False,
                        obstacle_color=color,
                        draw_reference=False,
                        set_axes=False,
                        drawVelArrow=True,
                    )

    ax.set_xlabel("x [m]", fontsize=9)
    ax.set_ylabel("y [m]", fontsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.tight_layout()
