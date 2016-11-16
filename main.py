# coding=utf-8
import tensorflow as tf
import numpy as np
import string
import statistics
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
NOISE = 0.1


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
    for data in data_list:
        for index in range(len(data)):
            if random.random() < noise_ratio:
                # data[index] = random.choice((0, 1))
                data[index] = 1 if data[index] is 0 else 0


def net(learning_rate, number_of_hidden_elements):
    print("net(%f, %d)" % (learning_rate, number_of_hidden_elements))

    data = np.genfromtxt(INPUT_FILE, delimiter=DELIMITER, dtype=int)
    np.random.shuffle(data)
    x_data = data[:, 0:35]
    add_noise(x_data, NOISE)
    # y_data = x_data
    y_data = one_hot(data[:, 35], 26)

    # for index in range(20):
    #     input_data = get_pattern(x_data[index], INPUT_LAYER_WIDTH, INPUT_LAYER_HEIGHT)
    #     output_data = get_pattern(y_data[index], INPUT_LAYER_WIDTH, INPUT_LAYER_HEIGHT)
    #     print(input_data[:-1] + '  -> ' + output_data[:-1] + '\n')

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

        # if len(errors) % 50 is 0:
        #     print("Iteration #:", len(errors), "Error: ", errors[-1])

    # if len(errors) % 50 is not 0:
    #     print("Iteration #:", len(errors), "Error: ", errors[-1])
    #     print(errors[-20:])
    if len(errors) == MAX_ITERATIONS:
        print("Max iterations reached, retrying")
        return net(learning_rate, number_of_hidden_elements)
    else:
        return len(errors)


for rate in np.arange(0.05, 1.5, 0.05):
    for number_of_elements in range(7, 15):
        current = []
        for i in range(10):
            current.append(net(rate, number_of_elements))
        print(
            "For %d elements in the hidden layer and a learning rate of %f, it was needed a median of %f iterations" % (
                number_of_elements, rate, statistics.median(current)))
