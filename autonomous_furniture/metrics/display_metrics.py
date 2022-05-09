import json
import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())



for nb_fur in [2,3,5,6,7]:
    dist_data = []
    time_data = []
    conv_data = []
    for algo in ["drag", "nodrag"]:
        data = json.load(open(
            f"autonomous_furniture/metrics/newmetrics/distance_nb{nb_fur}_{algo}_seed10.json", "r"))

        nb_folds = len(data["agent_0"]["total_dist"])

        vel = np.zeros(nb_folds)
        sum_direct_dist = np.zeros(nb_folds)
        times = np.zeros(nb_folds)

        # for agent in data.keys():
        for ii in range(nb_fur):
            vel = vel + data[f"agent_{ii}"]["total_dist"]
            sum_direct_dist += data[f"agent_{ii}"]["direct_dist"]
            #times += data[f"agent_{ii}"]["time_conv"]
        vel /= nb_fur
        #times /= nb_fur

        time_data.append(times)
        dist_data.append(vel)

        conv_data.append(data["converged"].count(True))

    labels = ["Drag", "No Drag"]
    fig,ax = plt.subplots(figsize=(10, 7))
    ax.set_xticklabels(labels)
    ax.set(ylabel="Distance[m]",
            title=f"{nb_fur} Furnitures")
    # Creating plot
    bp = plt.boxplot(dist_data)


    plt.savefig(f"autonomous_furniture/metrics/distance_nb{nb_fur}.png", format="png")
    plt.show()
    # if any(dist_over_direct < 1):
    #     breakpoint()
    #     print("error")
    del fig
    del ax

print("Hello")
