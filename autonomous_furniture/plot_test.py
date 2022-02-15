# imports used throughout this example
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import unary_union
from itertools import combinations
import matplotlib.pyplot as plt
import math
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--num_ctl", action="store", default=2, help="int of number of control points in the furniture")
parser.add_argument("--rect_size", action="store", default="1.6,0.7", help="x,y of the max size of the furniture")
args = parser.parse_args()

num_ctl = int(args.num_ctl)
str_axis = args.rect_size.split(",")
axis = [float(str_axis[0]), float(str_axis[1])]
ax_min = min(axis)
ax_max = max(axis)
div = ax_max / (num_ctl + 1)
radius = math.sqrt(((ax_min / 2) ** 2) + (div ** 2))
rectangle = Polygon([[0, -(ax_min / 2)], [0 + ax_max, -(ax_min / 2)], [0 + ax_max, -(ax_min / 2) + ax_min],
                     [0, -(ax_min / 2) + ax_min], [0, -(ax_min / 2)]])
shapes = []
for i in range(num_ctl):
    shapes.append(Point(div * (i + 1), 0).buffer(radius))

# Here are your input shapes (circles A, B, C)
# A = Point(3, 6).buffer(4)
# B = Point(6, 2).buffer(4)
# C = Point(1, 2).buffer(4)

# list the shapes so they are iterable
# shapes = [A, B, C]

# All intersections
inter = unary_union([pair[0].intersection(pair[1]) for pair in combinations(shapes, 2)])
# Remove from union of all shapes
nonoverlap = unary_union(shapes).difference(rectangle)

plt.close("all")
plt.ion()
fig, ax = plt.subplots()
ax.set_aspect(1.0)

plt.plot(*rectangle.exterior.xy, "r")

for geom in nonoverlap.geoms:
    xs, ys = geom.exterior.xy
    ax.fill(xs, ys, alpha=0.5)
for i in range(num_ctl):
    plt.plot(div * (i + 1), 0, "o", color="g")

area_circ = nonoverlap.area
print(f"Radius: {radius}")
print(f"Dead area: {area_circ}")
print(f"Max vertical margin: {radius-(ax_min/2)}")
print(f"Max horizontal margin: {radius-div}")
