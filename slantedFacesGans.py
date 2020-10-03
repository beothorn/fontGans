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


faces_x = np.array([good_face_gen()])
faces_y = np.array([1])

for i in range(BATCH_SIZE*100):
    faces_x = np.append(faces_x, [good_face_gen()], axis=0)
    faces_y = np.append(faces_y, [1], axis=0)

test_faces_x = np.array([good_face_gen()])
test_faces_y = np.array([1])

train_dataset = tf.data.Dataset.from_tensor_slices((faces_x, faces_y)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_faces_x, test_faces_y)).batch(BATCH_SIZE)

data_width = 2
data_height = 2

# ==========================================================
EPOCHS = 30
noise_array_size = 2

generator = tf.keras.models.Sequential([
    Dense(4, activation='relu'),
    Dense(4, activation='relu'),
    Dense(data_width * data_height, activation='relu'),
    Reshape((data_width, data_height, 1))
])


discriminator = tf.keras.models.Sequential([
    Dense(4, activation='relu', input_shape=(2, 2)),
    Dense(4, activation='relu'),
    Flatten(),
    Dense(1, activation='sigmoid')
])


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


discriminator_optimizer = tf.optimizers.Adam(0.1)
generator_optimizer = tf.optimizers.Adam(0.1)


#@tf.function
def train_step(images):
    # generating noise from a normal distribution
    #number_of_samples_on_batch = len(images[0])
    noise_batch = tf.random.normal([BATCH_SIZE, noise_array_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise_batch, training=True)
        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs):
    start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        gen_loss_list = []
        disc_loss_list = []

        for image_batch in dataset:
            t = train_step(image_batch)
            gen_loss_list.append(t[0])
            disc_loss_list.append(t[1])

        g_loss = sum(gen_loss_list) / len(gen_loss_list)
        d_loss = sum(disc_loss_list) / len(disc_loss_list)

        epoch_elapsed = time.time()-epoch_start
        print(f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss}, {epoch_elapsed}')

    elapsed = time.time()-start
    print(f'Training time: {elapsed}')


should_train = False
should_continue = True

if should_train:
    if should_continue:
        generator.load_weights('./weights/slanted')
    train(train_dataset, EPOCHS)
    generator.save_weights('./weights/slanted')
else:
    generator.load_weights('./weights/slanted')

print(generator(np.random.rand(1, noise_array_size)).numpy().reshape(data_width, data_height))
print(generator(np.random.rand(1, noise_array_size)).numpy().reshape(data_width, data_height))
print(generator(np.random.rand(1, noise_array_size)).numpy().reshape(data_width, data_height))
print(generator(np.random.rand(1, noise_array_size)).numpy().reshape(data_width, data_height))
print(generator(np.random.rand(1, noise_array_size)).numpy().reshape(data_width, data_height))

# plt.imshow(generated, cmap='gray', vmin=0, vmax=1)
# plt.show()
