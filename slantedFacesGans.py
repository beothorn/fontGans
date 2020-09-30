import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Reshape
import numpy as np
import time

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

noise_dim = 100


def make_generator_model():
    return tf.keras.models.Sequential([
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(28 * 28, activation='relu'),
        Reshape((28, 28, 1))
    ])


def make_discriminator_model():
    return tf.keras.models.Sequential([
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(1)
    ])


generator = make_generator_model()
discriminator = make_discriminator_model()


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

EPOCHS = 50

num_examples_to_generate = 16

# We'll re-use this random vector used to seed the generator so
# it will be easier to see the improvement over time.
random_vector_for_generation = tf.random.normal([num_examples_to_generate,
                                                 noise_dim])


def train_step(images, old_gen_loss, old_disc_loss):
    # generating noise from a normal distribution
    noise = tf.random.normal([len(images[0]), noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images[0], training=True)
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

        # saving (checkpoint) the model every 15 epochs
        # if (epoch + 1) % 15 == 0:
        #    checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                         time.time() - start))
    # generating after the final epoch
    # display.clear_output(wait=True)
    # generate_and_save_images(generator,
    #                       epochs,
    #                       random_vector_for_generation)


train(train_dataset, EPOCHS)
generated = generator(np.random.rand(1, 100)).numpy().reshape(28, 28)
plt.imshow(generated)
plt.show()
