from collections import namedtuple
import math
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import time

Circle = namedtuple("Circle", "x y r")
Polygone = namedtuple("Polygone", " x y")

# circles = [
#     Circle(1.6417233788, 1.6121789534, 0.0848270516),
#     Circle(-1.4944608174, 1.2077959613, 1.1039549836),
#     Circle(0.6110294452, -0.6907087527, 0.9089162485),
#     Circle(0.3844862411, 0.2923344616, 0.2375743054),
#     Circle(-0.2495892950, -0.3832854473, 1.0845181219),
#     Circle(1.7813504266, 1.6178237031, 0.8162655711),
#     Circle(-0.1985249206, -0.8343333301, 0.0538864941),
#     Circle(-1.7011985145, -0.1263820964, 0.4776976918),
#     Circle(-0.4319462812, 1.4104420482, 0.7886291537),
#     Circle(0.2178372997, -0.9499557344, 0.0357871187),
#     Circle(-0.6294854565, -1.3078893852, 0.7653357688),
#     Circle(1.7952608455, 0.6281269104, 0.2727652452),
#     Circle(1.4168575317, 1.0683357171, 1.1016025378),
#     Circle(1.4637371396, 0.9463877418, 1.1846214562),
#     Circle(-0.5263668798, 1.7315156631, 1.4428514068),
#     Circle(-1.2197352481, 0.9144146579, 1.0727263474),
#     Circle(-0.1389358881, 0.1092805780, 0.7350208828),
#     Circle(1.5293954595, 0.0030278255, 1.2472867347),
#     Circle(-0.5258728625, 1.3782633069, 1.3495508831),
#     Circle(-0.1403562064, 0.2437382535, 1.3804956588),
#     Circle(0.8055826339, -0.0482092025, 0.3327165165),
#     Circle(-0.6311979224, 0.7184578971, 0.2491045282),
#     Circle(1.4685857879, -0.8347049536, 1.3670667538),
#     Circle(-0.6855727502, 1.6465021616, 1.0593087096),
#     Circle(0.0152957411, 0.0638919221, 0.9771215985)]

# rectangles = [
#     Polygone(1.6, 0.8)
# ]


def calculate_area(circ, pres):
    x_min = min(c.x - c.r for c in circ)
    x_max = max(c.x + c.r for c in circ)
    y_min = min(c.y - c.r for c in circ)
    y_max = max(c.y + c.r for c in circ)

    box_side = pres

    dx = (x_max - x_min) / box_side
    dy = (y_max - y_min) / box_side

    count = 0

    for r in range(box_side):
        y = y_min + r * dy
        for c in range(box_side):
            x = x_min + c * dx
            if any((x - circle.x) ** 2 + (y - circle.y) ** 2 <= (circle.r ** 2)
                   for circle in circ):
                count += 1

    return count * dx * dy


def populate_control_points(rect, num_control_points):
    ax_min = min(rect[0].x, rect[0].y)
    ax_max = max(rect[0].x, rect[0].y)
    div = ax_max / (num_control_points + 1)
    radius = math.sqrt(((ax_min/2) ** 2) + (div ** 2))
    circle = []
    for i in range(num_control_points):
        circle.append(Circle(div * i, 0, radius))

    return circle, div


# def calculate_dead_zone(rect, num_control_points, pres):
#     circles = populate_control_points(rect, num_control_points)
#     circles_area = calculate_area(circles, pres)
#     rect_area = rect[0].x * rect[0].y
#     dead_zone_area = circles_area - rect_area
#     return dead_zone_area


if __name__ == "__main__":
    rectangle = [
        Polygone(1.6, 0.8)
    ]
    num_ctl = 2
    precision = 2000

    circles, margin = populate_control_points(rectangle, num_ctl)
    circles_area = calculate_area(circles, precision)
    rect_area = rectangle[0].x * rectangle[0].y
    dead_zone_area = circles_area - rect_area

    print(dead_zone_area)

    plt.close("all")
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim([-1, 1.5])
    ax.set_ylim([-1, 1])
    ax.set_aspect(1.0)
    ax.add_patch(ptc.Rectangle((-margin, -0.4), rectangle[0].x, rectangle[0].y, ec="r", fc="none", lw=1))

    for c in circles:
        ax.add_patch(plt.Circle((c.x, c.y), c.r, ec="b", lw=1, fc="none"))
        plt.plot(c.x, c.y, "o", markersize=5, color="g")

