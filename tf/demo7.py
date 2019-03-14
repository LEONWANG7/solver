import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('layer'):
            w = tf.Variable(tf.random_normal([in_size, out_size]), name='weight.....')
        b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        y0 = tf.matmul(inputs, w) + b
        if activation_function is None:
            return y0
        else:
            return activation_function(y0)


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(layer1, 10, 1, activation_function=None)

with tf.name_scope('error....'):
    loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.square(ys - prediction),
            reduction_indices=[1]
        )
    )
optimizer = tf.train.GradientDescentOptimizer(0.1)
with tf.name_scope('train...'):
    train_step = optimizer.minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('/Users/heyulong/projects/Github/solver/tf/', sess.graph)
    sess.run(init)

