import tensorflow as tf
import numpy as np
import string

TARGET_ERROR = 0.2
INPUT_FILE = "Input/letters.txt"
DELIMITER = " "
INPUT_LAYER_WIDTH = 7
INPUT_LAYER_HEIGHT = 5
HIDDEN_LAYER_WIDTH = 1
HIDDEN_LAYER_HEIGHT = 10
OUTPUT_LAYER_WIDTH = 1
OUTPUT_LAYER_HEIGHT = 26
ALPHABET = list(string.ascii_uppercase)


def one_hot(l, n):
    if type(l) == list:
        l = np.array(l)
    l = l.flatten()
    o_h = np.zeros((len(l), n))
    o_h[np.arange(len(l)), l] = 1
    return o_h


data = np.genfromtxt(INPUT_FILE, delimiter=DELIMITER, dtype=int)
np.random.shuffle(data)
x_data = data[:, 0:35]
y_data = one_hot(data[:, 35], 26)

x = tf.placeholder(tf.float32, [None, 35])
y_ = tf.placeholder(tf.float32, [None, 26])

W1 = tf.Variable(
    np.float32(np.random.rand(INPUT_LAYER_WIDTH * INPUT_LAYER_HEIGHT, HIDDEN_LAYER_WIDTH * HIDDEN_LAYER_HEIGHT)) * 0.1)
W2 = tf.Variable(np.float32(
    np.random.rand(HIDDEN_LAYER_WIDTH * HIDDEN_LAYER_HEIGHT, OUTPUT_LAYER_WIDTH * OUTPUT_LAYER_HEIGHT)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(HIDDEN_LAYER_WIDTH * HIDDEN_LAYER_HEIGHT)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(OUTPUT_LAYER_WIDTH * OUTPUT_LAYER_HEIGHT)) * 0.1)
y1 = tf.sigmoid(tf.matmul(x, W1) + b1)
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

cross_entropy = tf.reduce_sum(tf.square(y_ - y2))
train = tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")

errors = [sess.run(cross_entropy, feed_dict={x: x_data, y_: y_data})]

while errors[-1] > TARGET_ERROR:
    sess.run(train, feed_dict={x: x_data, y_: y_data})
    errors.append(sess.run(cross_entropy, feed_dict={x: x_data, y_: y_data}))

    if len(errors) % 50 is 0:
        print("Iteration #:", len(errors), "Error: ", errors[-1])

if len(errors) % 50 is not 0:
    print("Iteration #:", len(errors), "Error: ", errors[-1])
    print(errors[-20:])
