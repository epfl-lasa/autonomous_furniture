# imports used throughout this example
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import unary_union
from itertools import combinations
import matplotlib.pyplot as plt
import math
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--num_ctl_x",
    action="store",
    default=2,
    help="int of number of control points in the furniture on x axis",
)
parser.add_argument(
    "--num_ctl_y",
    action="store",
    default=1,
    help="int of number of control points in the furniture on y axis",
)
parser.add_argument(
    "--rect_size",
    action="store",
    default="2,1",
    help="x,y of the max size of the furniture",
)
args = parser.parse_args()

num_ctl_x = int(args.num_ctl_x)
num_ctl_y = int(args.num_ctl_y)
str_axis = args.rect_size.split(",")
axis = [float(str_axis[0]), float(str_axis[1])]
ax_min = min(axis)
ax_max = max(axis)
div_x = ax_max / (num_ctl_x + 1)
div_y = ax_min / (num_ctl_y + 1)
radius = math.sqrt((div_y**2) + (div_x**2))
rectangle = Polygon([[0, 0], [ax_max, 0], [ax_max, ax_min], [0, ax_min], [0, 0]])

shapes = []
for j in range(num_ctl_y):
    for i in range(num_ctl_x):
        shapes.append(Point(div_x * (i + 1), div_y * (j + 1)).buffer(radius))

# All intersections
inter = unary_union([pair[0].intersection(pair[1]) for pair in combinations(shapes, 2)])
# Remove from union of all shapes
nonoverlap = unary_union(shapes).difference(rectangle)

plt.close("all")
plt.ion()
fig, ax = plt.subplots()
ax.set_aspect(1.0)
plt.xlim([-0.2, 0.8])
plt.ylim([-0.2, 0.7])

plt.plot(*rectangle.exterior.xy, "r")

for geom in nonoverlap.geoms:
    xs, ys = geom.exterior.xy
    ax.fill(xs, ys, "b", alpha=0.5)
for j in range(num_ctl_y):
    for i in range(num_ctl_x):
        plt.plot(div_x * (i + 1), div_y * (j + 1), "o", color="g")

area_circ = nonoverlap.area
print(f"Radius: {radius}")
print(f"Dead area: {area_circ}")
print(f"Max vertical margin: {radius-div_y}")
print(f"Max horizontal margin: {radius-div_x}")
