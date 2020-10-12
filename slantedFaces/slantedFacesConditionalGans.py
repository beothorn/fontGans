import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Reshape
import numpy as np
from numpy import random
import time

# Hyperparams

EPOCHS = 4
UPDATE_RATE_FOR_EPOCH = 10
noise_array_size = 2
BATCH_SIZE = 30
FACE_COUNT = 100
GEN_LEARNING_RATE = 0.01
DISCRIMINATOR_LEARNING_RATE = 0.01


def good_face_left_gen():
    big_1 = random.uniform(0.8, 1)
    big_2 = random.uniform(0.8, 1)
    sma_1 = random.uniform(0, 0.2)
    sma_2 = random.uniform(0, 0.2)
    return [
        [big_1, sma_1],
        [sma_2, big_2]
    ]


def good_face_right_gen():
    big_1 = random.uniform(0.8, 1)
    big_2 = random.uniform(0.8, 1)
    sma_1 = random.uniform(0, 0.2)
    sma_2 = random.uniform(0, 0.2)
    return [
        [sma_1, big_1],
        [big_2, sma_2]
    ]


faces_left_x = np.array([good_face_left_gen()])
faces_left_y = np.array([1])

faces_right_x = np.array([good_face_right_gen()])
faces_right_y = np.array([1])

for i in range(BATCH_SIZE * FACE_COUNT):
    faces_left_x = np.append(faces_left_x, [good_face_left_gen()], axis=0)
    faces_left_y = np.append(faces_left_y, [1], axis=0)


train_dataset_left = tf.data.Dataset.from_tensor_slices((faces_left_x, faces_left_y)).shuffle(FACE_COUNT).batch(BATCH_SIZE)
train_dataset_right = tf.data.Dataset.from_tensor_slices((faces_right_x, faces_right_y)).shuffle(FACE_COUNT).batch(BATCH_SIZE)

data_width = 2
data_height = 2


generator = tf.keras.models.Sequential([
    Dense(data_width * data_height, activation='relu'),
    Dense(data_width * data_height, activation='relu'),
    Dense(data_width * data_height, activation='sigmoid'),
    Reshape((data_width, data_height, 1))
])


discriminator = tf.keras.models.Sequential([
    Flatten(input_shape=(2, 2)),
    Dense(data_width * data_height, activation='relu'),
    Dense(data_width * data_height, activation='relu'),
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
def train_step(images, noise_left):
    # generating noise from a normal distribution
    # number_of_samples_on_batch = len(images[0])
    noise_batch = tf.random.normal([len(images[0]), noise_array_size])
    for n in noise_batch:
        if noise_left:
            n[0] = 1
        else:
            n[0] = 0

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


def train(dataset_left, dataset_right, epochs):
    start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        gen_loss_list = []
        disc_loss_list = []

        for image_batch in dataset_left:
            t = train_step(image_batch, True)
            gen_loss_list.append(t[0])
            disc_loss_list.append(t[1])

        for image_batch in dataset_right:
            t = train_step(image_batch, False)
            gen_loss_list.append(t[0])
            disc_loss_list.append(t[1])

        g_loss = sum(gen_loss_list) / len(gen_loss_list)
        d_loss = sum(disc_loss_list) / len(disc_loss_list)

        epoch_elapsed = time.time() - epoch_start
        if epoch % UPDATE_RATE_FOR_EPOCH == 0:
            print("SAVING..")
            generator.save_weights('./weights/slantedGen')
            discriminator.save_weights('./weights/slantedDisc')
            print("TESTING:")
            print("====================")
            print(f"Sample \n {np.asarray(generator(fixed_random)).tolist()}")
            print("====================")
            print(f'Epoch {epoch + 1}, gen loss={g_loss},disc loss={d_loss}, elapsed: {epoch_elapsed}')
            print("##############################################################################################")

    elapsed = time.time() - start
    print(f'Training time: {elapsed}')


def print_test():
    print("====================")
    x = np.random.rand(1, noise_array_size)
    x[0] = 1
    print(f"Random seed LEFT {x}")
    print()
    generated = np.asarray(generator(x)).tolist()
    for item in generated:
        for row in item:
            for col in row:
                print(f" {round(col[0])} ")

    x = np.random.rand(1, noise_array_size)
    x[0] = 0
    print(f"Random seed RIGHT {x}")
    print()
    generated = np.asarray(generator(x)).tolist()
    for item in generated:
        for row in item:
            for col in row:
                print(f" {round(col[0])} ")
    print("====================")

# print_test_fixed()
# print_test()

try:
    generator.load_weights('./weights/slantedGen')
    discriminator.load_weights('./weights/slantedDisc')
except:
    pass


train(train_dataset_left, train_dataset_right, EPOCHS)

generator.summary()
discriminator.summary()

generator.save_weights('./weights/slantedGen')
discriminator.save_weights('./weights/slantedDisc')

for _ in range(10):
    print_test()
