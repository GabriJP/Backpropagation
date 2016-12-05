# coding=utf-8
import tensorflow as tf
import numpy as np
import string
import random

TARGET_ERROR = 0.1
INPUT_FILE = "Input/letters.txt"
DELIMITER = " "
INPUT_LAYER_WIDTH = 5
INPUT_LAYER_HEIGHT = 7
OUTPUT_LAYER_WIDTH = 1
OUTPUT_LAYER_HEIGHT = 26
ALPHABET = list(string.ascii_uppercase)
MAX_ITERATIONS = 30000


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


def add_noise(data, noise_ratio):
    swap_value = (lambda x: 1 if x == 0 else 0)
    swap_if_random = (lambda x: swap_value(x) if random.random() < noise_ratio else x)
    check_every_value = (lambda x: list(map(swap_if_random, x)))
    return list(map(check_every_value, data))


def net_1capa(learning_rate, number_of_hidden_elements, noise, func, momentum=0.9, ele=2):
    print("net(%f, %d)" % (learning_rate, number_of_hidden_elements))

    data = np.genfromtxt(INPUT_FILE, delimiter=DELIMITER, dtype=int)
    np.random.shuffle(data)
    x_data = data[:, 0:35]

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
    if func == 1:
        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    else:
        train = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cross_entropy)

    "TF initialization"
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    errors = [sess.run(cross_entropy, feed_dict={x: add_noise(x_data, noise), y_: y_data})]

    "Training"
    while errors[-1] > TARGET_ERROR and len(errors) < MAX_ITERATIONS:
        sess.run(train, feed_dict={x: add_noise(x_data, noise), y_: y_data})
        errors.append(sess.run(cross_entropy, feed_dict={x: add_noise(x_data, noise), y_: y_data}))
    if len(errors) >= MAX_ITERATIONS:
        return -1
    else:
        return len(errors)


def net_2capas(learning_rate, number_of_elements_layer1, noise, func, momentum=0.9, number_of_elements_layer2=None):
    if number_of_elements_layer2 is None:
        number_of_elements_layer2 = number_of_elements_layer1

    print("net(%f, %d,%d)" % (learning_rate, number_of_elements_layer1, number_of_elements_layer2))

    data = np.genfromtxt(INPUT_FILE, delimiter=DELIMITER, dtype=int)
    np.random.shuffle(data)
    x_data = data[:, 0:35]

    # y_data = x_data
    y_data = one_hot(data[:, 35], 26)

    x = tf.placeholder(tf.float32, [None, INPUT_LAYER_WIDTH * INPUT_LAYER_HEIGHT])
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_LAYER_WIDTH * OUTPUT_LAYER_HEIGHT])

    "Initialization of weights for hidden and output layers"
    w1 = tf.Variable(
        np.float32(np.random.rand(INPUT_LAYER_WIDTH * INPUT_LAYER_HEIGHT, number_of_elements_layer1)) * 0.1)
    w2 = tf.Variable(
        np.float32(np.random.rand(number_of_elements_layer1, number_of_elements_layer2)) * 0.1)
    w3 = tf.Variable(
        np.float32(np.random.rand(number_of_elements_layer2, OUTPUT_LAYER_WIDTH * OUTPUT_LAYER_HEIGHT)) * 0.1)

    "Initialization ob bias for hidden and output layers"
    b1 = tf.Variable(np.float32(np.random.rand(number_of_elements_layer1)) * 0.1)
    b2 = tf.Variable(np.float32(np.random.rand(number_of_elements_layer2)) * 0.1)
    b3 = tf.Variable(np.float32(np.random.rand(OUTPUT_LAYER_WIDTH * OUTPUT_LAYER_HEIGHT)) * 0.1)

    "Output of hidden and output layers (activation function)"
    y1 = tf.sigmoid(tf.matmul(x, w1) + b1)
    y2 = tf.sigmoid(tf.matmul(y1, w2) + b2)
    y3 = tf.nn.softmax(tf.matmul(y2, w3) + b3)

    "Function to reduce"
    cross_entropy = tf.reduce_sum(tf.square(y_ - y3))
    if func == 1:
        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    else:
        train = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cross_entropy)

    "TF initialization"
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    errors = [sess.run(cross_entropy, feed_dict={x: add_noise(x_data, noise), y_: y_data})]

    "Training"
    while errors[-1] > TARGET_ERROR and len(errors) < MAX_ITERATIONS:
        sess.run(train, feed_dict={x: add_noise(x_data, noise), y_: y_data})
        errors.append(sess.run(cross_entropy, feed_dict={x: add_noise(x_data, noise), y_: y_data}))
    if len(errors) >= MAX_ITERATIONS:
        return -1
    else:
        return len(errors)
