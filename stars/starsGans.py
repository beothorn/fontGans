import tensorflow as tf
import datetime
from tensorflow.keras.layers import Flatten, Dense, Dropout, Reshape
import numpy as np
from numpy import random
import time

import matplotlib.pyplot as plt
import random
import math

# Hyperparams

EPOCHS = 2
UPDATE_RATE_FOR_EPOCH = 10
noise_array_size = 6
BATCH_SIZE = 2
NUMBER_OF_BATCHES = 5
GEN_LEARNING_RATE = 0.01
DISCRIMINATOR_LEARNING_RATE = 0.001
NAME = "Star"


# ====================


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


values_x = np.array([gen_star()])
values_y = np.array([1])

print("Generating values")
number_of_values = BATCH_SIZE * NUMBER_OF_BATCHES
print(f"Will generate {number_of_values}")

for i in range((BATCH_SIZE * NUMBER_OF_BATCHES) - 1):
    values_x = np.append(values_x, [gen_star()], axis=0)
    values_y = np.append(values_y, [1], axis=0)

print("Done generating values")

test_values_x = np.array([gen_star()])
test_values_y = np.array([1])

print("Creating datasets")
train_dataset = tf.data.Dataset.from_tensor_slices((values_x, values_y)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_values_x, test_values_y)).batch(BATCH_SIZE)

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

discriminator = tf.keras.models.Sequential([
    Dense(10, activation='relu', input_shape=(10,)),
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
    Dense(100, activation='relu'),
    Dense(1, activation='sigmoid')
])


def discriminator_loss(real_output, fake_output):
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output))
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output))
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.ones_like(fake_output)))


discriminator_optimizer = tf.optimizers.Adam(DISCRIMINATOR_LEARNING_RATE)
generator_optimizer = tf.optimizers.Adam(GEN_LEARNING_RATE)


# @tf.function
def train_step(images):
    # generating noise from a normal distribution
    # number_of_samples_on_batch = len(images[0])
    noise_batch = tf.random.normal([len(images[0]), noise_array_size])

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


fixed_random = np.random.rand(1, noise_array_size)


def train(dataset, epochs):
    start = time.time()
    average_time = 0

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
        d_loss = sum(d_loss) / len(d_loss)

        epoch_elapsed = time.time() - epoch_start
        if average_time == 0:
            average_time = epoch_elapsed
        average_time = (epoch_elapsed + average_time) / 2
        if epoch % UPDATE_RATE_FOR_EPOCH == 0:
            print("SAVING..")
            generator.save_weights(f"./weights/{NAME}Gen")
            discriminator.save_weights(f"./weights/{NAME}Disc")
            print("TESTING:")
            print("====================")
            print(f"Sample \n {np.asarray(generator(fixed_random)).tolist()}")
            # draw_polygon(generator(fixed_random)[0])
            print("====================")
            print(f'Epoch {epoch + 1}, gen loss={g_loss},disc loss={d_loss}, elapsed: {epoch_elapsed}')
            expected_seconds = math.ceil(average_time * (EPOCHS - (epoch + 1)))
            print(f"Expected time = {math.floor(expected_seconds / 3600)} "
                  f"Hours {math.floor((expected_seconds / 60) % 60)} "
                  f"Minutes {expected_seconds % 60} seconds")
            print("##############################################################################################")

    elapsed = time.time() - start
    print(f'Training time: {elapsed}')


def print_test():
    print("====================")
    x = np.random.rand(1, noise_array_size)
    print(f"Random seed {x}")
    print()
    generated = np.asarray(generator(x)).tolist()
    draw_polygon(generated[0])


# print_test_fixed()
# print_test()

try:
    generator.load_weights(f"./weights/slantedGen")
    discriminator.load_weights(f"./weights/slantedDisc")
except:
    pass

print("Will start training")
train(train_dataset, EPOCHS)

# generator.summary()
# discriminator.summary()

generator.save_weights(f"./weights/{NAME}Gen")
discriminator.save_weights(f"./weights/{NAME}Disc")

def test_discriminator(star):
    print( discriminator(np.asarray([star])) )

test_discriminator(gen_star())
test_discriminator([0.5, 0.6, 0.7, 0.8, 0.5, 0.5, 0.6, 0.7, 0.8, 0.5])

#draw_polygon(np.asarray(generator(np.asarray([[-1., -2., 0.5, 65, 52, 65]]))).tolist()[0])
draw_polygon(np.asarray(generator(np.random.rand(1, noise_array_size)).tolist()[0]))
draw_polygon(np.asarray(generator(np.random.rand(1, noise_array_size)).tolist()[0]))
draw_polygon(np.asarray(generator(np.random.rand(1, noise_array_size)).tolist()[0]))