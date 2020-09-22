import tensorflow as tf

from tensorflow.keras.layers import Activation, Dense, Softmax

import numpy as np
from numpy import random
from matplotlib import pyplot as plt


def good_face_gen():
    return [random.uniform(0.8, 1), random.uniform(0, 0.2), random.uniform(0, 0.2), random.uniform(0.8, 1)]


def bad_face_gen():
    return np.array([random.uniform(0, 0.7), random.uniform(0.3, 1), random.uniform(0.3, 1), random.uniform(0, 0.7)])




# Examples of faces
# good_faces_x = np.array([np.array([1, 0, 0, 1]),
#                         np.array([0.9, 0.1, 0.2, 0.8]),
#                         np.array([0.9, 0.2, 0.1, 0.8]),
#                         np.array([0.8, 0.1, 0.2, 0.9]),
#                         np.array([0.8, 0.2, 0.1, 0.9])])


faces_x = np.array([good_face_gen()])
faces_y = np.array([1])



for i in range(2000):
    if random.uniform(0, 1) > 0.5:
        faces_x = np.append(faces_x, [good_face_gen()], axis = 0)
        faces_y = np.append(faces_y, [1], axis=0)
    else:
        faces_x = np.append(faces_x, [bad_face_gen()], axis = 0)
        faces_y = np.append(faces_y, [0], axis=0)


test_faces_x = np.array([good_face_gen()])
test_faces_y = np.array([1])

for i in range(200):
    if random.uniform(0, 1) > 0.5:
        test_faces_x = np.append(test_faces_x, [good_face_gen()], axis = 0)
        test_faces_y = np.append(test_faces_y, [1], axis=0)
    else:
        test_faces_x = np.append(test_faces_x, [bad_face_gen()], axis = 0)
        test_faces_y = np.append(test_faces_y, [0], axis=0)

model = tf.keras.models.Sequential([
  Dense(4, activation='relu'),
  Dense(4, activation='relu'),
  Dense(4, activation='relu'),
  Dense(4, activation='relu'),
  Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(faces_x, faces_y, epochs=30)

result = model.predict(test_faces_x)

for i, r in enumerate(result):
    print(f"Was {r[0] < r[1]} and should be {test_faces_y[i] == 1} : {test_faces_x[i]} ")
