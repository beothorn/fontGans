# tamanho minimo
# Ponto aleatrio nessa area de 50x50
# mas a partir da borda esquerda + minimo ate a borda direita - minimo
# topo + minimo e fundo - minimo
# Mede a distancia pra borda mais proxima
# tamanho aleatorio entre o minimo e a distancia medida
# angulo aleatorio entre 0 360
# acha o ponto do circulo nesse angulo
# repete 4 vezes: soma 144 + variaçao pequena entre -3 e +3 (vai dar 5 segmentos formando uma estrela)
# da pra usar PILLOW pra desenhar os testes

import matplotlib.pyplot as plt
import random
import math


def draw_polygon(star):
    for i in range(0, len(star) - 2, 2):
        plt.plot([star[i], star[i + 2]], [star[i + 1], star[i + 3]])
    plt.plot([star[len(star) - 2], star[0]], [star[len(star) - 1], star[1]])

    plt.xlim(0, 50), plt.ylim(0, 50)
    plt.show()


def gen_star():
    # [10, 10, 20, 30, 30, 10, 10, 20, 40, 20]
    border = 10
    width = 50
    height = 50

    rx = random.randint(0 + border, width - border)
    ry = random.randint(0 + border, height - border)

    top_distance = height - ry
    bottom_distance = ry
    left_distance = rx
    right_distance = width - rx

    max_radius = min(top_distance, bottom_distance, left_distance, right_distance)
    radius = random.randint(border, max_radius)

    r_ang = random.random() * math.pi * 2
    increase = (math.tau * 3) / 5

    starting_point_x = rx + (math.cos(r_ang) * radius)
    starting_point_y = ry + (math.sin(r_ang) * radius)

    result = [starting_point_x, starting_point_y]

    for i in range(5):
        r_ang = r_ang + increase
        result.append(rx + (math.cos(r_ang) * radius))
        result.append(ry + (math.sin(r_ang) * radius))

    return result


draw_polygon(gen_star())
