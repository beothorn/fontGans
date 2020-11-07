import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Reshape
import numpy as np
from numpy import random
import time

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

generator = tf.keras.models.Sequential([
    Dense(10, activation='relu'),
    Dense(20, activation='relu'),
    Dense(20, activation='relu'),
    Dense(20, activation='relu'),
    Dense(30, activation='relu'),
    Dense(30, activation='relu'),
    Dense(30, activation='relu'),
    Dense(30, activation='relu'),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(80, activation='relu'),
    Dense(80, activation='relu'),
    Dense(80, activation='relu'),
    Dense(80, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(10, activation='relu'),
    Dense(10, activation='sigmoid')
])

generator.load_weights('./weights/StarDisc')

draw_polygon(np.asarray(generator(np.asarray([[-1., -2., 12, 2, 4, 5]]))).tolist()[0])

