import tensorflow as tf

# https://www.tensorflow.org/guide/autodiff

x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x + x
dy_dx = g.gradient(y, x) # Will compute to 6.0

print(dy_dx)