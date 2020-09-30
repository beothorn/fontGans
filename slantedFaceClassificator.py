import tensorflow as tf

from tensorflow.keras.layers import Activation, Dense, Softmax

import numpy as np
from numpy import random

def good_face_gen():
    return [random.uniform(0.8, 1), random.uniform(0, 0.2), random.uniform(0, 0.2), random.uniform(0.8, 1)]


def bad_face_gen():
    return np.array([random.uniform(0, 0.7), random.uniform(0.3, 1), random.uniform(0.3, 1), random.uniform(0, 0.7)])

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

# End generation ==========================================

generator = tf.keras.models.Sequential([
  Dense(1, activation='relu'),
  Dense(4, activation='relu'),
  Dense(4, activation='relu'),
  Dense(4, activation='relu')
])

discrimitnator = tf.keras.models.Sequential([
  Dense(4, activation='relu'),
  Dense(4, activation='relu'),
  Dense(4, activation='relu'),
  Dense(4, activation='relu'),
  Dense(1, activation='softmax')
])

for i in range(200):
    if random.uniform(0, 1) > 0.5:
        test_faces_x = np.append(test_faces_x, [good_face_gen()], axis = 0)
        test_faces_y = np.append(test_faces_y, [1], axis=0)
    else:
        test_faces_x = np.append(test_faces_x, [bad_face_gen()], axis = 0)
        test_faces_y = np.append(test_faces_y, [0], axis=0)

generator_discriminator = tf.keras.models.Sequential([
  Dense(1, activation='relu'),
  Dense(4, activation='relu'),
  Dense(4, activation='relu'),
  Dense(4, activation='relu'),
# =============================
  Dense(4, activation='relu'),
  Dense(4, activation='relu'),
  Dense(4, activation='relu'),
  Dense(4, activation='relu'),
  Dense(1, activation='softmax')
])


discriminator = tf.keras.models.Sequential([
  Dense(4, activation='relu'),
  Dense(2, activation='softmax')
])

discriminator.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

discriminator.fit(faces_x, faces_y, epochs=5)

result = discriminator.predict(test_faces_x)

for i in discriminator.trainable_variables:
    print(i)

for i, r in enumerate(result):
    print(f"Was {r[0] < r[1]} and should be {test_faces_y[i] == 1} : {test_faces_x[i]} ")
