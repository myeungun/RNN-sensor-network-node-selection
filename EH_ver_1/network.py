import tensorflow as tf
import random
import os
import numpy as np

def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var

def inference(keep_prob, logs_dir, weight, input, N_USER):

    with tf.variable_scope("inference"):

        var_dict = {}

        if os.path.exists(logs_dir + weight):
            weight_dict = np.load(logs_dir + weight, encoding='latin1').item()
            load_flag = True
        else:
            load_flag = False
            print ("No weight!!")

        if load_flag == False: # initialize weights
            FC1_W = weight_variable([4*N_USER, 40], name="FC1_W")
            FC1_B = bias_variable([40], name="FC1_B")
        else:
            FC1_W = get_variable(weight_dict[("FC1_W")], name="FC1_W")
            FC1_B = get_variable(weight_dict[("FC1_B")], name="FC1_B")
        input_flat = tf.reshape(input,[-1,4*N_USER])
        h_1 = tf.nn.relu(tf.matmul(input_flat, FC1_W) + FC1_B)
        h_1_drop = tf.nn.dropout(h_1, keep_prob)

        var_dict[("FC1_W")] = FC1_W
        var_dict[("FC1_B")] = FC1_B

        if load_flag == False: # initialize weights
            FC2_W = weight_variable([40, 40], name="FC2_W")
            FC2_B = bias_variable([40], name="FC2_B")
        else:
            FC2_W = get_variable(weight_dict[("FC2_W")], name="FC2_W")
            FC2_B = get_variable(weight_dict[("FC2_B")], name="FC2_B")
        h_2 = tf.nn.relu(tf.matmul(h_1_drop, FC2_W) + FC2_B)
        h_2_drop = tf.nn.dropout(h_2, keep_prob)

        var_dict[("FC2_W")] = FC2_W
        var_dict[("FC2_B")] = FC2_B

        if load_flag == False: # initialize weights
            FC3_W = weight_variable([40, 40], name="FC3_W")
            FC3_B = bias_variable([40], name="FC3_B")
        else:
            FC3_W = get_variable(weight_dict[("FC3_W")], name="FC3_W")
            FC3_B = get_variable(weight_dict[("FC3_B")], name="FC3_B")
        h_3 = tf.nn.relu(tf.matmul(h_2_drop, FC3_W) + FC3_B)
        h_3_drop = tf.nn.dropout(h_3, keep_prob)

        var_dict[("FC3_W")] = FC3_W
        var_dict[("FC3_B")] = FC3_B

        if load_flag == False: # initialize weights
            FC4_W = weight_variable([40, 40], name="FC4_W")
            FC4_B = bias_variable([40], name="FC4_B")
        else:
            FC4_W = get_variable(weight_dict[("FC4_W")], name="FC4_W")
            FC4_B = get_variable(weight_dict[("FC4_B")], name="FC4_B")
        h_4 = tf.nn.relu(tf.matmul(h_3_drop, FC4_W) + FC4_B)
        h_4_drop = tf.nn.dropout(h_4, keep_prob)

        var_dict[("FC4_W")] = FC4_W
        var_dict[("FC4_B")] = FC4_B

        if load_flag == False: # initialize weights
            FC5_W = weight_variable([40, 40], name="FC5_W")
            FC5_B = bias_variable([40], name="FC5_B")
        else:
            FC5_W = get_variable(weight_dict[("FC5_W")], name="FC5_W")
            FC5_B = get_variable(weight_dict[("FC5_B")], name="FC5_B")
        h_5 = tf.matmul(h_4_drop, FC5_W) + FC5_B
        h_5_drop = tf.nn.dropout(h_5, keep_prob)

        var_dict[("FC4_W")] = FC4_W
        var_dict[("FC4_B")] = FC4_B

        if load_flag == False: # initialize weights
            FC6_W = weight_variable([40, 40], name="FC6_W")
            FC6_B = bias_variable([40], name="FC6_B")
        else:
            FC6_W = get_variable(weight_dict[("FC6_W")], name="FC6_W")
            FC6_B = get_variable(weight_dict[("FC6_B")], name="FC6_B")
        h_6 = tf.matmul(h_5_drop, FC6_W) + FC6_B
        h_6_drop = tf.nn.dropout(h_6, keep_prob)

        var_dict[("FC5_W")] = FC5_W
        var_dict[("FC5_B")] = FC5_B

        if load_flag == False: # initialize weights
            FC7_W = weight_variable([40, 40], name="FC7_W")
            FC7_B = bias_variable([40], name="FC7_B")
        else:
            FC7_W = get_variable(weight_dict[("FC7_W")], name="FC7_W")
            FC7_B = get_variable(weight_dict[("FC7_B")], name="FC7_B")
        h_7 = tf.matmul(h_6_drop, FC7_W) + FC7_B

        var_dict[("FC7_W")] = FC7_W
        var_dict[("FC7_B")] = FC7_B


        softmax_selection = tf.nn.softmax(h_7)

        selection = tf.one_hot(tf.argmax(softmax_selection, axis=1, name="prediction"), 2 * N_USER)
        selection = tf.reshape(selection, shape=(2, N_USER))


    return h_7, selection, var_dict