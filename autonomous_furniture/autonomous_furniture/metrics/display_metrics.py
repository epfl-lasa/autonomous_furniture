import json
from unittest.util import _MIN_DIFF_LEN
import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())


# diff_dist = []
# for nb_fur in [3]:
#     dist_data = np.zeros((2, 100))

#     time_data = []

#     conv_data = []
#     kk = 0
#     for algo in ["drag", "nodrag"]:

#         data = json.load(
#             open(f"autonomous_furniture/metrics/newmetrics/distance_nb{nb_fur}_{algo}.json", "r")
#         )

#         nb_folds = len(data["agent_0"]["total_dist"])

#         dist = np.zeros(nb_folds)
#         sum_direct_dist = np.zeros(nb_folds)
#         times = np.zeros(nb_folds)

#         # for agent in data.keys():
#         for ii in range(nb_fur):
#             dist = dist + data[f"agent_{ii}"]["total_dist"]
#             sum_direct_dist += data[f"agent_{ii}"]["direct_dist"]
#             # times += data[f"agent_{ii}"]["time_conv"]
#         dist /= nb_fur
#         # times /= nb_fur

#         time_data.append(times)
#         dist_data[kk, :] = dist
#         kk += 1

#         conv_data.append(data["converged"].count(True))

#     temp = dist_data[0, :] - dist_data[1, :]
#     diff_dist.append(dist_data[0, :] - dist_data[1, :])

#     # Sort the list and keep their old index which corresponds to the number of the scenario
#     sort_dist = sorted(enumerate(temp), key=lambda i: i[1])


folder_new_safety = "/home/menichel/ros2_ws/src/autonomous_furniture/autonomous_furniture/metrics/new_safety_module"
folder_no_safety = "/home/menichel/ros2_ws/src/autonomous_furniture/autonomous_furniture/metrics/no_safety_module"
folders = [folder_no_safety,folder_new_safety]

fig_size=[4,3]
fig_dpi=120
tick_size=8.5
label_size=9
legend_size=9

def compare_convergence_rate():
    diff_dist = []
    list_algo = ["dragdist"]
    list_vers = ["v2"]
    list_fur = [3, 4, 5, 6, 7, 8, 9, 10]
    converg_data = np.zeros((len(folders), len(list_fur)))
    nb_folds = number_scen(3, list_algo[0], list_vers[0])

    # for ll, nb_fur in enumerate(list_fur):
    #     dist_data = np.zeros((2, nb_folds))
    #     for algo in list_algo:
    #         kk = 0

    #         for jj, folder in enumerate(folders):
    #             data = json.load(
    #                 open(
    #                     f"{folder}/distance_nb{nb_fur}_{algo}_{list_vers[0]}.json",
    #                     "r",
    #                 )
    #             )

    #             dist = np.zeros(nb_folds)

    #             # for agent in data.keys():
    #             for ii in range(nb_fur):
    #                 dist = dist + [
    #                     data[f"agent_{ii}"]["total_dist"][mm]
    #                     / data[f"agent_{ii}"]["direct_dist"][mm]
    #                     for mm in range(nb_folds)
    #                 ]
    #                 # sum_direct_dist += data[f"agent_{ii}"]["direct_dist"]
    #                 # times += data[f"agent_{ii}"]["time_conv"]

    #             dist /= nb_fur

    #             dist_data[kk, :] = dist
    #             converg_data[jj, ll] = data["converged"].count(True)

    #             kk += 1
    #             # conv_data.append(data["converged"].count(True))

    #         temp = dist_data[0, :] - dist_data[1, :]
    #         diff_dist.append(dist_data[0, :] - dist_data[1, :])
    
    converg_data = np.zeros([3, len(list_fur)])
    for ll, nb_fur in enumerate(list_fur):
        data_prev = json.load( #load data from previous work
            open(
                f"{folder_no_safety}/distance_nb{nb_fur}_nodrag_v1.json",
                "r",
            )
        )
        converg_data[0, ll] = data_prev["converged"].count(True)

        data_no_safety = json.load( #load data from new work without safety module
            open(
                f"{folder_no_safety}/distance_nb{nb_fur}_dragdist_v2.json",
                "r",
            )
        )
        converg_data[1, ll] = data_no_safety["converged"].count(True)

        data_yes_safety = json.load( #load data from new work without safety module
            open(
                f"{folder_new_safety}/distance_nb{nb_fur}_dragdist_v2.json",
                "r",
            )
        )
        converg_data[2, ll] = data_yes_safety["converged"].count(True)

    # Nb fur to compare :
    # fur_number = 6
    # idx = list_fur.index(fur_number)
    # extract_best_worst_scn(
    #     diff_dist[idx],
    #     fur_number,
    #     diff_between=list_vers,
    #     alg=list_algo,
    #     version=list_vers,
    # )

    # plot_box(diff_dist, list_fur)
    label = ["HDSM", "DDHDSM", "SDDHDSM"]
    barWidth = 0.25
    fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    # plt.legend(fontsize=legend_size)
    
    br1 = np.arange(len(converg_data[0,:]))
    br0 = br1 - barWidth
    br2 = br1 + barWidth
    
    plt.bar(br1, converg_data[0,:], color="blue", width=barWidth, edgecolor="grey", label=label[0])
    plt.bar(br2, converg_data[1,:], color="green", width=barWidth, edgecolor="grey", label=label[1])
    plt.bar(br0, converg_data[2,:], color="orange", width=barWidth, edgecolor="grey", label=label[2])

    plt.xlabel("Number of furnitures", fontsize=label_size)
    plt.ylabel("Converged scenario[%]", fontsize=label_size)
    plt.xticks(np.arange(len(converg_data[0,:])), [3, 4, 5, 6, 7, 8, 9, 10])
    plt.tight_layout()

    for bars in ax.containers:
        ax.bar_label(bars, fontsize=label_size)

    plt.legend(fontsize=legend_size)
    plt.show()

def compare_travelled_distance():

    delete_collisions_from_data = True  # Whether or not we take into consideration for the graph all the scenarios
    # or the ones where both alg did not register collisions

    list_algo = ["dragdist", "nodrag"]
    list_vers = ["v2"]
    list_fur = [3, 4, 5, 6, 7, 8, 9, 10]
    converg_data = np.zeros((len(list_algo), len(list_fur)))
    nb_folds = number_scen(3, list_algo[0], list_vers[0])

    data_drag_temp = []  # TODO temp to remove
    data_nodrag_temp = []  # TODO temp to remove

    for ll, nb_fur in enumerate(list_fur):
        dist_data = np.zeros((2, 100))

        for version in list_vers:
            kk = 0

            data = []
            for jj, algo in enumerate(list_algo):
                data.append(
                    json.load(
                        open(
                            f"{folder_new_safety}/distance_nb{nb_fur}_{algo}_{version}.json",
                            "r",
                        )
                    )
                )

            idx_with_coll = []
            for data_alg in data:
                idx_with_coll = idx_with_coll + [
                    idx for idx, i in enumerate(data_alg["collisions"]) if i > 0
                ]  # We check scenarios were collisions occurend to discard them

            idx_with_coll = list(
                dict.fromkeys(idx_with_coll)
            )  # Little trick to remove duplicants from a list as a dictionnary cannot have duplicate keys as the same index
            # index scenario with coll can appear 2 times (if collision in both algo)

        for algo, data_alg in enumerate(
            data
        ):  # Becareful here to the order alg = 0 is drag alg = 1 is nondrag

            dist = np.zeros(nb_folds)
            for ii in range(nb_fur):
                dist = dist + [
                    data_alg[f"agent_{ii}"]["total_dist"][mm]
                    / data_alg[f"agent_{ii}"]["direct_dist"][mm]
                    for mm in range(nb_folds)
                ]

            dist /= nb_fur

            kk += 1

            if delete_collisions_from_data == True:
                dist = np.delete(dist, idx_with_coll)
            if algo == 0:  # 0 is drag, verify the order in list_version
                data_drag_temp.append(list(dist))
            if algo == 1:  # 1 is nondrag, verify the order in list_version
                data_nodrag_temp.append(list(dist))

    ticks = list_fur

    def set_box_color(bp, color):
        plt.setp(bp["boxes"], facecolor=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color="black")

    plt.figure(figsize=fig_size, dpi=fig_dpi)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    bpr = plt.boxplot(
        data_drag_temp,
        positions=np.array(range(len(data_drag_temp))) * 2.0 + 0.4,
        sym="",
        widths=0.6,
        patch_artist=True,
    )
    bpl = plt.boxplot(
        data_nodrag_temp,
        positions=np.array(range(len(data_nodrag_temp))) * 2.0 - 0.4,
        sym="",
        widths=0.6,
        patch_artist=True,
    )
    set_box_color(bpl, "red")
    set_box_color(bpr, "orange")  # colors are from http://colorbrewer2.org/

    plt.plot([], c="red", label="DHDSM")
    plt.plot([], c="orange", label="DDHDSM")

    plt.xlabel("Number of agents", fontsize=label_size)
    plt.ylabel(
        "Mean relative distance travelled, $\overline{\mathcal{D}}$ [-]", fontsize=label_size
    )
    plt.legend(fontsize=legend_size)

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)

    plt.tight_layout()

    # conv_data.append(data["converged"].count(True))

    # diff_dist.append(dist_data[0, :] - dist_data[1, :])

    # plot_box(diff_dist, list_fur)
    # plot_bar(converg_data, list_algo)


def extract_best_worst_scn(diff_data, nb_fur, diff_between, alg, version):

    sort_dist = sorted(enumerate(diff_data), key=lambda i: i[1])

    print(
        f"Parameters : number_furniture: {nb_fur} | algorith: {alg} | version: {version}"
    )
    print(f"Looking difference between : {diff_between[0]} and {diff_between[1]}")
    print(f"Best performances during scenario :")
    print(sort_dist[:5])
    print(f"Worst performances during scenario ")
    print(sort_dist[-5:])


def number_scen(nb_fur, algo, vers):
    data = json.load(
        open(
            f"{folder_new_safety}/distance_nb{nb_fur}_{algo}_{vers}.json",
            "r",
        )
    )
    nb_folds = len(data["agent_0"]["total_dist"])
    return nb_folds


def plot_box(data, labels: list, save: bool = False):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xticklabels(labels)
    ax.set(
        ylabel="Distance[m]", xlabel="Number of furniture"
    )  # title=f"{nb_fur} Furnitures")
    # Creating plot
    bp = plt.boxplot(data)

    plt.axhline(y=0, color="r", linestyle="--", linewidth=0.5)
    plt.title("Comparison between v1 and v2 with drag")
    if save:
        plt.savefig(
            f"autonomous_furniture/metrics/distance_nb{nb_fur}.png", format="png"
        )

def plot_collisions():

    list_nb_fur = [3, 4, 5, 6, 7, 8, 9]
    list_version = ["v2"]
    list_algo = ["dragdist"]

    collisions_data = []
    nb_collisions_data = []
    for algo in list_algo:
        for vers in list_version:
            for folder in folders:
                collisions_data_temp = []
                nb_collisions = []

                for idx, nb_fur in enumerate(list_nb_fur):

                    data = json.load(
                        open(
                            f"{folder}/distance_nb{nb_fur}_{algo}_{vers}.json",
                            "r",
                        )
                    )
                    collisions_data_temp.append(data["collisions"])
                    has_collision = [1 for i in data["collisions"] if i > 0]
                    nb_collisions.append(len(has_collision))

                nb_collisions_data.append(nb_collisions)

    width = 0.25
    #Full ERM_temp= [4, 17, 38, 56, 74, 87, 97, 99]
    ERM_temp = [
        17,
        38,
        56,
        74,
        87,
        97,
        99
    ]  # , 97, 99] # TODO TO be removed after presentation, report
    pos = np.arange(len(list_nb_fur))

    # br1 = np.arange(len(drag))
    # br2 = [x + barWidth for x in br1]
    fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
    # plt.figure(figsize=(3.5, 3.5), dpi=120)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # plt.bar(
    #     pos + 0.5 * width, ERM_temp, color="blue", width=width, label="previous work"
    # )
    br1 = pos
    br0 = pos-width
    br2 = pos+width
    
    plt.bar(br0,ERM_temp,color="blue",width=width,label="HDSM",)
    plt.bar(br1, nb_collisions_data[0], color="green", width=width, label="DDHDSM")
    plt.bar(br2, nb_collisions_data[1], color="orange", width=width, label="SDDHDSM")
    
    plt.xticks(pos, list_nb_fur)

    for bars in ax.containers:
        ax.bar_label(bars, fontsize=label_size)

    plt.xlabel("Number of agents", fontsize=label_size)
    plt.ylabel("Scenarios with virtual collisions [%]", fontsize=label_size)
    plt.legend(fontsize=legend_size)
    plt.tight_layout()
    plt.show()


def plot_proximity():

    delete_collisions_from_data = True

    list_algo = ["dragdist"]
    list_vers = ["v2"]
    list_fur = [3, 4, 5, 6, 7, 8, 9, 10]
    nb_folds = number_scen(list_fur[0], list_algo[0], list_vers[0])

    data_drag_temp = []  # TODO temp to remove
    data_nodrag_temp = []  # TODO temp to remove

    for nb_fur in list_fur:

        for version in list_vers:
            kk = 0
            data = []

            for algo in list_algo:
                data.append(
                    json.load(
                        open(
                            f"{folder_new_safety}/distance_nb{nb_fur}_{algo}_{version}.json",
                            "r",
                        )
                    )
                )
                data.append(
                    json.load(
                        open(
                            f"{folder_no_safety}/distance_nb{nb_fur}_{algo}_{version}.json",
                            "r",
                        )
                    )
                )


            idx_with_coll = []

            for data_alg in data:
                idx_with_coll = idx_with_coll + [
                    idx for idx, i in enumerate(data_alg["collisions"]) if i > 0
                ]

            idx_with_coll = list(
                dict.fromkeys(idx_with_coll)
            )  # Little trick to remove duplicants from a list as a dictionnary cannot have duplicate keys

        for algo, data_alg in enumerate(
            data
        ):  # Be careful here to the order alg = 0 is drag alg = 1 is nondrag

            prox = np.zeros(nb_folds)
            for ii in range(nb_fur):
                prox = prox + [
                    data_alg[f"agent_{ii}"]["prox"][mm] / nb_fur
                    for mm in range(nb_folds)
                ]

            kk += 1
            if delete_collisions_from_data == True:
                prox = np.delete(prox, idx_with_coll)

            if algo == 0:  # 0 is drag, verify the order in list_version
                data_drag_temp.append(list(prox))
            if algo == 1:  # 1 is nondrag, verify the order in list_version
                data_nodrag_temp.append(list(prox))

    ticks = list_fur

    def set_box_color(bp, color):
        plt.setp(bp["boxes"], facecolor=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color="black")

    plt.figure()
    plt.figure(figsize=(3.5, 3.5), dpi=120)
    plt.xticks(fontsize=8.5)
    plt.yticks(fontsize=8.5)
    plt.legend(fontsize=1.5)

    bpr = plt.boxplot(
        data_drag_temp,
        positions=np.array(range(len(data_drag_temp))) * 2.0 + 0.4,
        sym="",
        widths=0.6,
        patch_artist=True,
    )
    set_box_color(bpr, "green")
    plt.plot([], c="green", label="yes_safety")

    bpl = plt.boxplot(
        data_nodrag_temp,
        positions=np.array(range(len(data_nodrag_temp))) * 2.0 - 0.4,
        sym="",
        widths=0.6,
        patch_artist=True,
    )
    set_box_color(bpl, "red")      # colors are from http://colorbrewer2.org/
    plt.plot([], c="red", label="no_safety")

    plt.xlabel("Number of agents", fontsize=9)
    plt.ylabel("Mean proximity, $\overline{\mathcal{P}}$ [-]", fontsize=9)
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.tight_layout()


def plot_prox_graph():
    nb_fur = 3
    algo = "dragdist"
    version = "v2"

    step = 200
    data_equal = json.load(
        open(
            f"/home/menichel/ros2_ws/src/autonomous_furniture/distance_nb{nb_fur}_{algo}_{version}_equal.json",
            "r",
        )
    )
    data_priority = json.load(
        open(
            f"/home/menichel/ros2_ws/src/autonomous_furniture/distance_nb{nb_fur}_{algo}_{version}_priority.json",
            "r",
        )
    )
    pers_list_equal = data_equal["agent_0"]["list_prox"]
    furn_list_equal = data_equal["agent_2"]["list_prox"]
    
    pers_list_priority = data_priority["agent_0"]["list_prox"]
    furn_list_priority = data_priority["agent_2"]["list_prox"]

    # pers_list = [1 - pers_list[ii] / (ii + 1) for ii in range(step)]
    # furn_list = [1 - furn_list[ii] / (ii + 1) for ii in range(step)]

    fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
    xmin = 0
    xmax = 200
    ymin = 2
    ymax = 5

    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)

    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel("Step", fontsize=label_size)
    
    plt.ylabel("Distance from mobile agent [m]", fontsize=label_size)
    plt.plot(range(step), pers_list_equal, color="orange", linestyle="dashed")
    plt.plot(range(step), furn_list_equal, color="red", linestyle="dashed")
    # plt.plot(
    #     step - 1, pers_list[step - 1], "o", color=(246 / 255, 178 / 255, 107 / 255)
    # )
    # plt.plot(60, pers_list[60], "o", color=(246 / 255, 178 / 255, 107 / 255))
    # plt.plot(0, pers_list[0], "o", color=(246 / 255, 178 / 255, 107 / 255))
    # plt.vlines(step-1, 0, pers_list[step-1], linestyle="dashed", color="black")
    # plt.hlines(pers_list[step-1], 0, step-1, linestyle="dashed")
    plt.plot(range(step), pers_list_priority, color="orange")
    plt.plot(range(step), furn_list_priority, color="red")
    # plt.plot(step - 1, furn_list[step - 1], "o", color=(221 / 255, 16 / 255, 16 / 255))
    # plt.plot(60, furn_list[60], "o", color=(221 / 255, 16 / 255, 16 / 255))
    # plt.plot(0, furn_list[0], "o", color=(221 / 255, 16 / 255, 16 / 255))
    # plt.vlines(step-1, 0, furn_list[step-1], linestyle="dashed", color="black")
    plt.plot(np.where(pers_list_equal==np.min(pers_list_equal))[0][0], np.min(pers_list_equal), "go")
    ax.text(np.where(pers_list_equal==np.min(pers_list_equal))[0][0]-5, np.min(pers_list_equal)+0.15, "%.2f" %np.round(np.min(pers_list_equal),2))
    
    plt.plot(np.where(pers_list_priority==np.min(pers_list_priority))[0][0], np.min(pers_list_priority), "go")
    ax.text(np.where(pers_list_priority==np.min(pers_list_priority))[0][0]-5, np.min(pers_list_priority)+0.15, "%.2f" %np.round(np.min(pers_list_priority),2))

    plt.legend(["$d_P^{equal}$", "$d_O^{equal}$", "$d_P^{priority}$", "$d_O^{priority}$"], fontsize=legend_size)
    plt.tight_layout()
    plt.show()

def plot_time():

    delete_collisions_from_data = True

    list_algo = ["nodrag"]
    list_vers = ["v1", "v2"]
    list_fur = [3, 4, 5, 6, 7]
    nb_folds = number_scen(list_fur[0], list_algo[0], list_vers[0])

    data_drag_temp = []  # TODO temp to remove
    data_nodrag_temp = []  # TODO temp to remove

    for nb_fur in list_fur:

        for algo in list_algo:
            data = []

            for version in list_vers:
                data.append(
                    json.load(
                        open(
                            f"{folder_no_safety}/distance_nb{nb_fur}_{algo}_{version}.json",
                            "r",
                        )
                    )
                )

            idx_conv = []

            for data_alg in data:
                idx_conv = idx_conv + [
                    idx for idx, i in enumerate(data_alg["converged"]) if i == False
                ]

            idx_conv = list(
                dict.fromkeys(idx_conv)
            )  # Little trick to remove duplicants from a list as a dictionnary cannot have duplicate keys

            for algo, data_alg in enumerate(
                data
            ):  # Becareful here to the order alg = 0 is drag alg = 1 is nondrag

                time_rel = np.zeros(nb_folds)
                for ii in range(nb_fur):
                    time_rel = time_rel + [
                        data_alg[f"agent_{ii}"]["time_conv"][mm]
                        / data_alg[f"agent_{ii}"]["time_direct"][mm]
                        for mm in range(nb_folds)
                    ]

                time_rel /= nb_fur

                if delete_collisions_from_data == True:
                    time_rel = np.delete(time_rel, idx_conv)
                if algo == 0:  # 0 is drag, verify the order in list_version
                    data_drag_temp.append(list(time_rel))
                if algo == 1:  # 1 is nondrag, verify the order in list_version
                    data_nodrag_temp.append(list(time_rel))

    ticks = list_fur

    def set_box_color(bp, color):
        plt.setp(bp["boxes"], facecolor=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color="black")

    plt.figure()
    plt.figure(figsize=fig_size, dpi=fig_dpi)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    bpr = plt.boxplot(
        data_drag_temp,
        positions=np.array(range(len(data_drag_temp))) * 2.0 + 0.4,
        sym="",
        widths=0.6,
        patch_artist=True,
    )
    bpl = plt.boxplot(
        data_nodrag_temp,
        positions=np.array(range(len(data_nodrag_temp))) * 2.0 - 0.4,
        sym="",
        widths=0.6,
        patch_artist=True,
    )
    set_box_color(bpl, "blue")
    set_box_color(bpr, "red")  # colors are from http://colorbrewer2.org/

    plt.plot([], c="blue", label="HDSM")
    plt.plot([], c="red", label="DHDSM")

    plt.xlabel("Number of agents", fontsize=label_size)
    plt.ylabel("Mean relative time to converge, $\overline{\mathcal{T}}$ [-]", fontsize=label_size)
    plt.legend(fontsize=label_size)

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.tight_layout()


if __name__ == "__main__":
    # compare_convergence_rate()
    # compare_travelled_distance()
    # plot_collisions()
    # plot_prox_graph()
    # plot_proximity()
    # plot_time()
    plt.legend()
    plt.show()