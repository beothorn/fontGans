import tensorflow as tf
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
faces_x = np.array([np.array([1, 0, 0, 1]),
         np.array([0.9, 0.1, 0.2, 0.8]),
         np.array([0.9, 0.2, 0.1, 0.8]),
         np.array([0.8, 0.1, 0.2, 0.9]),
         np.array([0.8, 0.2, 0.1, 0.9])])
faces_y = np.ones(5)


faces_x.concatenate(np.array([np.random.randn(2,2) for i in range(20)]))
faces_y.concatenate(np.zeros(20))

#view_samples(noise, 4,5)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(2, 2)),
  tf.keras.layers.Dense(4, activation='relu'),
  tf.keras.layers.Dense(2, activation='relu')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(faces_x, faces_y, epochs=5)

model.evaluate(faces_x,  faces_y, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(probability_model(faces_x[:2]))
print(probability_model(faces_x[:10]))