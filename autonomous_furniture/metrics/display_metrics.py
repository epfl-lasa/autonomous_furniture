import json
import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())

dist_data = []
time_data = []
conv_data = []

nb_fur = 3
for algo in ["drag", "nodrag"]:
    data = json.load(open(
        f"metrics/newmetrics/distance_nb{nb_fur}_{algo}_seed10.json", "r"))

    nb_folds = len(data["agent_0"]["total_dist"])

    vel = np.zeros(nb_folds)
    sum_direct_dist = np.zeros(nb_folds)
    times = np.zeros(nb_folds)

    # for agent in data.keys():
    for ii in range(nb_fur):
        vel = vel + data[f"agent_{ii}"]["total_dist"]
        sum_direct_dist += data[f"agent_{ii}"]["direct_dist"]
        times += data[f"agent_{ii}"]["time_conv"]

    time_data.append(times)
    dist_data.append(vel/sum_direct_dist)

    conv_data.append(data["converged"].count(True))

fig = plt.figure(figsize=(10, 7))

# Creating plot
bp = plt.boxplot(time_data)


plt.show()

# if any(dist_over_direct < 1):
#     breakpoint()
#     print("error")

print("Hello")
