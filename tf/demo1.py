# 一个线性函数的拟合
import tensorflow as tf
import numpy as np

# 【1】创建数据
# 100个0~1的随机数
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
# 【2】创建结构
# 定义变量：一维随机向量，-1~1之间
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# b 初始值是 0
b = tf.Variable(tf.zeros([1]))
# 预测值
y = w * x_data + b
# 均方作为误差
loss = tf.reduce_mean(tf.square(y-y_data))
# 优化器：梯度下降，学习效率0.5
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)
# 【3】定义会话
sess = tf.Session()
# 初始化
init = tf.initialize_all_variables()
sess.run(init)

for step in range(1001):
    sess.run(train)
    if step % 50 == 0:
        print(step, sess.run(w), sess.run(b))
