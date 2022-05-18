import json
from unittest.util import _MIN_DIFF_LEN
import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())



for nb_fur in [3]:
    dist_data = np.zeros((2,100))
    time_data = []
    conv_data = []
    kk=0
    for algo in ["drag", "nodrag"]:
        
        data = json.load(open(
            f"autonomous_furniture/metrics/newmetrics/distance_nb{nb_fur}_{algo}_seed10.json", "r"))

        nb_folds = len(data["agent_0"]["total_dist"])

        dist = np.zeros(nb_folds)
        sum_direct_dist = np.zeros(nb_folds)
        times = np.zeros(nb_folds)

        # for agent in data.keys():
        for ii in range(nb_fur):
            dist = dist + data[f"agent_{ii}"]["total_dist"]
            sum_direct_dist += data[f"agent_{ii}"]["direct_dist"]
            #times += data[f"agent_{ii}"]["time_conv"]
        dist /= nb_fur
        #times /= nb_fur

        time_data.append(times)
        dist_data[kk,:]=dist
        kk += 1

        conv_data.append(data["converged"].count(True))

    diff_dist = dist_data[0,:]-dist_data[1,:]
    
    best_try = min(diff_dist)
    worst_try = max(diff_dist)

    sort_dist = sorted(enumerate(diff_dist), key=lambda i: i[1]) # Sort the list and keep their old index which corresponds to the number of the scenario

    print(f"drag performed better during scenario :")
    print(sort_dist[:5])
    print(f"drag performed worst during scenario ")
    print(sort_dist[-5:])


    labels = ["Drag", "No Drag"]
    fig,ax = plt.subplots(figsize=(10, 7))
    ax.set_xticklabels(labels)
    ax.set(ylabel="Distance[m]",
            title=f"{nb_fur} Furnitures")
    # Creating plot
    bp = plt.boxplot(dist_data.T)


    plt.savefig(f"autonomous_furniture/metrics/distance_nb{nb_fur}.png", format="png")
    plt.show()
    # if any(dist_over_direct < 1):
    #     breakpoint()
    #     print("error")
    del fig
    del ax
    
print("Hello")
