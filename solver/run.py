import tensorflow as tf

a = tf.constant(2)
b = tf.constant(4)
c = tf.add(a, b)

sess = tf.Session()

print(sess.run(c))

sess.close()