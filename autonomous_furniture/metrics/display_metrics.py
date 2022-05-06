import json
import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())

data = json.load(open("autonomous_furniture/metrics/100fold/distance_nb2_drag_seed10.json", "r"))

nb_folds = len(data["agent_0"]["total_dist"])

vel = np.zeros(nb_folds)
sum_direct_dist = 0

for agent in data.keys() :
    vel = vel + data[agent]["total_dist"]
    sum_direct_dist += data[agent]["direct_dist"]

dist_over_direct = vel/sum_direct_dist

if any(dist_over_direct < 1):
    breakpoint()
    print("error")

print("Hello")