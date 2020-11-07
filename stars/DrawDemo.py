import matplotlib.pyplot as plt
import random
import math

def draw_polygon(star):
    print(star)

    for index in range(0, len(star) - 2, 2):
        plt.plot([star[index], star[index + 2]], [star[index + 1], star[index + 3]])
    plt.plot([star[len(star) - 2], star[0]], [star[len(star) - 1], star[1]])

    plt.xlim(0, 1), plt.ylim(0, 1)
    plt.show()


def gen_star():
    # [10, 10, 20, 30, 30, 10, 10, 20, 40, 20]
    border = 0.01
    width = 1
    height = 1

    rx = border + ((width - (border * 2)) * random.random())
    ry = border + ((width - (border * 2)) * random.random())

    top_distance = height - ry
    bottom_distance = ry
    left_distance = rx
    right_distance = width - rx

    max_radius = min(top_distance, bottom_distance, left_distance, right_distance)
    radius = max(border, max_radius * random.random())

    r_ang = random.random() * math.pi * 2
    increase = (math.tau * 3) / 5

    starting_point_x = rx + (math.cos(r_ang) * radius)
    starting_point_y = ry + (math.sin(r_ang) * radius)

    result = [starting_point_x, starting_point_y]

    for i in range(4):
        r_ang = r_ang + increase
        result.append(rx + (math.cos(r_ang) * radius))
        result.append(ry + (math.sin(r_ang) * radius))

    return result

draw_polygon(gen_star())