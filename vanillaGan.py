import tensorflow as tf

from tensorflow.keras.layers import Activation, Dense, Softmax

import numpy as np
from numpy import random
from matplotlib import pyplot as plt


def view_samples(samples, m, n):
    fig, axes = plt.subplots(figsize=(10, 10), nrows=m, ncols=n, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(1-img.reshape((2,2)), cmap='Greys_r')
    plt.show()


# Examples of faces
good_faces_x = np.array([np.array([1, 0, 0, 1]),
         np.array([0.9, 0.1, 0.2, 0.8]),
         np.array([0.9, 0.2, 0.1, 0.8]),
         np.array([0.8, 0.1, 0.2, 0.9]),
         np.array([0.8, 0.2, 0.1, 0.9])])
good_faces_y = np.column_stack((np.ones(5), np.zeros(5)))

bad_faces_x = np.array([np.random.randn(4) for i in range(20)])
bad_faces_y = np.column_stack((np.zeros(20), np.ones(20)))

faces_x = np.concatenate((good_faces_x, bad_faces_x))
faces_y = np.concatenate((good_faces_y, bad_faces_y))


model = tf.keras.models.Sequential([
  #tf.keras.layers.Flatten(input_shape=(2, 2)),
  Dense(4, activation='relu'),
  Dense(2, activation='relu'),
  Softmax()
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(faces_x, faces_y, epochs=30)

# print(faces_x)
# print(faces_y)

print(model(faces_x))
