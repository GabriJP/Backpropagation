# coding=utf-8
import tensorflow as tf
import numpy as np
import string
import random

TARGET_ERROR = 0.2
INPUT_FILE = "Input/letters.txt"
DELIMITER = " "
INPUT_LAYER_WIDTH = 5
INPUT_LAYER_HEIGHT = 7
OUTPUT_LAYER_WIDTH = 1
OUTPUT_LAYER_HEIGHT = 26
ALPHABET = list(string.ascii_uppercase)
MAX_ITERATIONS = 10000


def one_hot(l, n):
    """Returns an array of arrays of zeros, each internal array with one 1 (the element of the list)"""
    if type(l) == list:
        l = np.array(l)
    l = l.flatten()
    o_h = np.zeros((len(l), n))
    o_h[np.arange(len(l)), l] = 1
    return o_h


def get_pattern(pattern, width, height):
    """Returns the pattern in a string of width*height figure"""
    if len(pattern) != width * height:
        raise Exception("len(pattern) != width*height")
    result = ""
    for row in range(height):
        for column in range(width):
            result += ' ' if pattern[row * width + column] == 0 else '·'
        result += '\n'
    return result


def add_noise(data_list, noise_ratio):
    result = []
    for data in data_list:
        current_pattern = []
        result.append(current_pattern)
        for index in range(len(data)):
            if random.random() < noise_ratio:
                current_pattern.append(1 if data[index] is 0 else 0)
            else:
                current_pattern.append(data[index])
    return result


def net_gradiente(learning_rate, number_of_hidden_elements, noise):
    print("net(%f, %d)" % (learning_rate, number_of_hidden_elements))

    data = np.genfromtxt(INPUT_FILE, delimiter=DELIMITER, dtype=int)
    np.random.shuffle(data)
    x = data[:, 0:35]
    x_data = add_noise(x, noise)
    # y_data = x_data
    y_data = one_hot(data[:, 35], 26)

    x = tf.placeholder(tf.float32, [None, INPUT_LAYER_WIDTH * INPUT_LAYER_HEIGHT])
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_LAYER_WIDTH * OUTPUT_LAYER_HEIGHT])

    "Initialization of weights for hidden and output layers"
    w1 = tf.Variable(
        np.float32(np.random.rand(INPUT_LAYER_WIDTH * INPUT_LAYER_HEIGHT, number_of_hidden_elements)) * 0.1)
    w2 = tf.Variable(
        np.float32(np.random.rand(number_of_hidden_elements, OUTPUT_LAYER_WIDTH * OUTPUT_LAYER_HEIGHT)) * 0.1)

    "Initialization ob bias for hidden and output layers"
    b1 = tf.Variable(np.float32(np.random.rand(number_of_hidden_elements)) * 0.1)
    b2 = tf.Variable(np.float32(np.random.rand(OUTPUT_LAYER_WIDTH * OUTPUT_LAYER_HEIGHT)) * 0.1)

    "Output of hidden and output layers (activation function)"
    y1 = tf.sigmoid(tf.matmul(x, w1) + b1)
    y2 = tf.nn.softmax(tf.matmul(y1, w2) + b2)

    "Function to reduce"
    cross_entropy = tf.reduce_sum(tf.square(y_ - y2))
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    "TF initialization"
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    errors = [sess.run(cross_entropy, feed_dict={x: x_data, y_: y_data})]

    "Training"
    while errors[-1] > TARGET_ERROR and len(errors) < MAX_ITERATIONS:
        sess.run(train, feed_dict={x: x_data, y_: y_data})
        errors.append(sess.run(cross_entropy, feed_dict={x: x_data, y_: y_data}))
    if len(errors) >= MAX_ITERATIONS:
        return -1
    else:
        return len(errors)
