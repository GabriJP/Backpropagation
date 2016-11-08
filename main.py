import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder("float", [None, 35])
y_ = tf.placeholder("float", [None, 35])

W1 = tf.Variable(np.float32(np.random.rand(35,10))*0.1)
W2 = tf.Variable(np.float32(np.random.rand(10,35))*0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(35)) * 0.1)
y1 = tf.sigmoid(tf.matmul(x, W1) + b1)
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

cross_entropy = tf.reduce_sum(tf.square(y_ - y2))
train = tf.train.GradientDescentOptimizer(0.03).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
