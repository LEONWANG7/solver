import tensorflow as tf

m1 = tf.constant([[1, 2]])
m2 = tf.constant([[3], [4]])
prod = tf.matmul(m1, m2)

sess = tf.Session()
result = sess.run(prod)
print('法1：', result)
sess.close()

with tf.Session() as sess:
    result2 = sess.run(prod)
    print('法2：', result2)
