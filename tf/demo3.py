import tensorflow as tf

state = tf.Variable(0, name='counter')
print(state.name)

one = tf.constant(1)

new_val = tf.add(state, one)
update = tf.assign(state, new_val)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        sess.run(update)
        print(sess.run(state))
