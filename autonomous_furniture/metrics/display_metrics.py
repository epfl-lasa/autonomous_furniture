import json
from unittest.util import _MIN_DIFF_LEN
import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())


diff_dist = []
for nb_fur in [3]:
    dist_data = np.zeros((2, 100))

    time_data = []

    conv_data = []
    kk = 0
    for algo in ["drag", "nodrag"]:

        data = json.load(
            open(f"metrics/newmetrics/distance_nb{nb_fur}_{algo}.json", "r")
        )

        nb_folds = len(data["agent_0"]["total_dist"])

        dist = np.zeros(nb_folds)
        sum_direct_dist = np.zeros(nb_folds)
        times = np.zeros(nb_folds)

        # for agent in data.keys():
        for ii in range(nb_fur):
            dist = dist + data[f"agent_{ii}"]["total_dist"]
            sum_direct_dist += data[f"agent_{ii}"]["direct_dist"]
            # times += data[f"agent_{ii}"]["time_conv"]
        dist /= nb_fur
        # times /= nb_fur

        time_data.append(times)
        dist_data[kk, :] = dist
        kk += 1

        conv_data.append(data["converged"].count(True))

    temp = dist_data[0, :] - dist_data[1, :]
    diff_dist.append(dist_data[0, :] - dist_data[1, :])

    # Sort the list and keep their old index which corresponds to the number of the scenario
    sort_dist = sorted(enumerate(temp), key=lambda i: i[1])


def compare_v2_vs_v1():
    diff_dist = []
    list_algo = ["nodrag"]
    list_vers = ["v2", "v1"]
    list_fur = [2, 3, 5, 6, 7, 8]
    converg_data = np.zeros((len(list_vers), len(list_fur)))
    nb_folds = number_scen(3, list_algo[0], list_vers[0])

    for ll, nb_fur in enumerate(list_fur):
        dist_data = np.zeros((2, 100))
        for algo in list_algo:
            kk = 0

            for jj, version in enumerate(list_vers):
                data = json.load(
                    open(f"new_seed_gen/distance_nb{nb_fur}_{algo}_{version}.json", "r")
                )

                dist = np.zeros(nb_folds)

                # for agent in data.keys():
                for ii in range(nb_fur):
                    dist = dist + data[f"agent_{ii}"]["total_dist"]
                    # sum_direct_dist += data[f"agent_{ii}"]["direct_dist"]
                    # times += data[f"agent_{ii}"]["time_conv"]

                dist /= nb_fur

                dist_data[kk, :] = dist
                converg_data[jj, ll] = data["converged"].count(True)

                kk += 1
                conv_data.append(data["converged"].count(True))

            temp = dist_data[0, :] - dist_data[1, :]
            diff_dist.append(dist_data[0, :] - dist_data[1, :])

    # Nb fur to compare :
    fur_number = 6
    idx = list_fur.index(fur_number)
    extract_best_worst_scn(
        diff_dist[idx],
        fur_number,
        diff_between=list_vers,
        alg=list_algo,
        version=list_vers,
    )

    plot_box(diff_dist, list_fur)
    plot_bar(converg_data, list_vers)

    plt.legend()
    plt.show()


def compare_drag_vs_nodrag():

    diff_dist = []
    list_algo = ["drag", "nodrag"]
    list_vers = ["v2"]
    list_fur = [2, 3, 5, 6, 7, 8]
    converg_data = np.zeros((len(list_algo), len(list_fur)))
    nb_folds = number_scen(3, list_algo[0], list_vers[0])

    for ll, nb_fur in enumerate(list_fur):
        dist_data = np.zeros((2, 100))

        for version in list_vers:
            kk = 0

            for jj, algo in enumerate(list_algo):
                data = json.load(
                    open(f"new_seed_gen/distance_nb{nb_fur}_{algo}_{version}.json", "r")
                )

                dist = np.zeros(nb_folds)

                # for agent in data.keys():
                for ii in range(nb_fur):
                    dist = dist + data[f"agent_{ii}"]["total_dist"]
                    # sum_direct_dist += data[f"agent_{ii}"]["direct_dist"]
                    # times += data[f"agent_{ii}"]["time_conv"]

                dist /= nb_fur

                dist_data[kk, :] = dist
                converg_data[jj, ll] = data["converged"].count(True)
                kk += 1

                conv_data.append(data["converged"].count(True))

            temp = dist_data[0, :] - dist_data[1, :]
            diff_dist.append(dist_data[0, :] - dist_data[1, :])

    plot_box(diff_dist, list_fur)
    plot_bar(converg_data, list_algo)

    plt.legend()
    plt.show()


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
    data = json.load(open(f"new_seed_gen/distance_nb{nb_fur}_{algo}_{vers}.json", "r"))
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

    if save:
        plt.savefig(
            f"autonomous_furniture/metrics/distance_nb{nb_fur}.png", format="png"
        )


def plot_bar(data, label):
    barWidth = 0.25
    fig2 = plt.subplots(figsize=(12, 8))

    drag = data[0, :]
    nodrag = data[1, :]

    br1 = np.arange(len(drag))
    br2 = [x + barWidth for x in br1]

    plt.bar(br1, drag, color="r", width=barWidth, edgecolor="grey", label=label[0])
    plt.bar(br2, nodrag, color="g", width=barWidth, edgecolor="grey", label=label[1])

    plt.xlabel("Number of furnitures", fontweight="bold", fontsize=15)
    plt.ylabel("Converged scenario[%]", fontweight="bold", fontsize=15)
    plt.xticks([r + barWidth for r in range(len(drag))], [2, 3, 5, 6, 7, 8])


def plot_collisions():

    list_nb_fur = [3, 4]
    list_version = ["v1"]
    list_algo = ["drag", "nodrag"]

    collisions_data = []

    for idx, nb_fur in enumerate(list_nb_fur):
        collisions_data_temp = []

        for algo in list_algo:
            for vers in list_version:
                data = json.load(open(f"distance_nb{nb_fur}_{algo}_{vers}.json", "r"))
                collisions_data_temp.append(data["collisions"])

        collisions_data.append(collisions_data_temp)
    for ii in range(2):
        boxplot(collisions_data[ii], positions=[1, 2], widths=0.6)
    print("Coucou")


if __name__ == "__main__":
    # compare_v2_vs_v1()
    # compare_drag_vs_nodrag()
    plot_collisions()
