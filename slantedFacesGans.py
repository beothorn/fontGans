import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Reshape
import numpy as np
from numpy import random
import time


# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

# BUFFER_SIZE = 60000
# BATCH_SIZE = 256

# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

BATCH_SIZE = 3

def good_face_gen():
    return [[random.uniform(0.8, 1), random.uniform(0, 0.2)], [random.uniform(0, 0.2), random.uniform(0.8, 1)]]


def bad_face_gen():
    return np.array(
        [[random.uniform(0, 0.7), random.uniform(0.3, 1)], [random.uniform(0.3, 1), random.uniform(0, 0.7)]])


faces_x = np.array([good_face_gen()])
faces_y = np.array([1])

for i in range(2000):
    if random.uniform(0, 1) > 0.5:
        faces_x = np.append(faces_x, [good_face_gen()], axis=0)
        faces_y = np.append(faces_y, [1], axis=0)
    else:
        faces_x = np.append(faces_x, [bad_face_gen()], axis=0)
        faces_y = np.append(faces_y, [0], axis=0)

test_faces_x = np.array([good_face_gen()])
test_faces_y = np.array([1])

train_dataset = tf.data.Dataset.from_tensor_slices((faces_x, faces_y)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_faces_x, test_faces_y)).batch(BATCH_SIZE)

data_width = 2
data_height = 2

# ==========================================================
EPOCHS = 10
noise_array_size = 10

discriminator = tf.keras.models.Sequential([
    Dense(4, activation='relu', input_shape=(2, 2)),
    Dense(4, activation='relu'),
    Dense(1)
])

generator = tf.keras.models.Sequential([
    Dense(4, activation='relu'),
    Dense(4, activation='relu'),
    Dense(data_width * data_height, activation='relu'),
    Reshape((data_width, data_height, 1))
])


def generator_loss(generated_output):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(generated_output), logits=generated_output)


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(generated_output),
                                                             logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


generator_optimizer = tf.optimizers.Adam(1e-4)
discriminator_optimizer = tf.optimizers.Adam(1e-4)


def train_step(images, old_gen_loss, old_disc_loss):
    # generating noise from a normal distribution
    number_of_samples_on_batch = len(images[0])
    noise_batch = tf.random.normal([number_of_samples_on_batch, noise_array_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise_batch, training=True)
        real_input = images[0]
        real_output = discriminator(real_input, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        gen_loss = 500
        disc_loss = 500
        for images in dataset:
            gen_loss, disc_loss = train_step(images, gen_loss, disc_loss)

        if (epoch + 1) % 15 == 0:
            generator.save_weights('./weights/slanted')

        print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


train(train_dataset, EPOCHS)
generated = generator(np.random.rand(1, noise_array_size)).numpy().reshape(data_width, data_height)
generator.save_weights('./weights/slanted')

result = discriminator.predict(test_faces_x)

for i, r in enumerate(result):
    print(f"Was {r} and should be {test_faces_y[i]}")

plt.imshow(generated, cmap='gray', vmin=0, vmax=1)
plt.show()
