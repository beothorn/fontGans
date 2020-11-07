import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Reshape
import numpy as np
from numpy import random
import time

import matplotlib.pyplot as plt
import random
import math
from starsGans import generator, draw_polygon


generator.load_weights('./weights/StarDisc')

draw_polygon(np.asarray(generator(np.asarray([[-1., -2., 12, 2, 4, 5]]))).tolist()[0])

