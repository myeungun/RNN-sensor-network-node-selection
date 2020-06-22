'''
This script shows how to predict next decisions using a basic RNN
'''
import tensorflow as tf
import numpy as np
import matplotlib
import os
import itertools
import random
from blackbox_module_for_test_ver2 import BB_module

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


class par:
    UL = 0
    DL = 1
    N_UE = 2 # number of UEs = 2, number of cases = 2*2
    N_time = 3
    N_realization = int(5)
    N_quantize_type = 3 # Low=0, Middle=1, High=2
    N_schemes = 3
    Harvesting_efficiency = 1.0

# train Parameters
seq_length = 2
data_dim = 2
hidden_dim = 64
output_dim = 64
learning_rate = 0.01
iterations = 20000

x = np.loadtxt('/home/mukim/Desktop/EH_ver_2/training_data/training_channel_ver3_reali_5_itr_5000_eff_0_7.csv', delimiter=',')
y = np.loadtxt('/home/mukim/Desktop/EH_ver_2/training_data/training_decision_ver3_reali_5_itr_5000_eff_0_7.csv', delimiter=',')
z = np.loadtxt('/home/mukim/Desktop/EH_ver_2/test_data/ver2/validation_channel_time_UE_ver2.csv', delimiter=',')
h = np.loadtxt('/home/mukim/Desktop/EH_ver_2/test_data/ver2/validation_3_timeslot_channel_UE_time_ver2.csv', delimiter=',')

# Open, High, Low, Volume, Close
#xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
#xy = xy[::-1]  # reverse order (chronically ordered)
#xy = MinMaxScaler(xy)
#x = xy
#y = xy[:, [-1]]  # Close as label

# build a training dataset
dataX = []
dataY = []
for i in range(0, len(x) - seq_length, seq_length):
    _x = x[i:i + seq_length]
    _y = y[i/2]
    #print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# build a test dataset
test_2time = []
test_3time = []
for i in range(0, len(z) - seq_length, seq_length):
    _z = z[i:i + seq_length]
    _h = h[i:i + seq_length]
    test_2time.append(_z)
    test_3time.append(_h)

# train/test split
train_size = int(len(dataY))
test_size = int(len(test_2time))
trainX, testX = np.array(dataX[0:train_size]), np.array(
    test_2time[0:test_size])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    test_3time[0:test_size])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    decision_combination = list(itertools.product(range(par.N_UE * 2), repeat=par.N_time))
    battery_combination = list(itertools.product(np.arange(0, 1, 1 / 3.0), repeat=par.N_UE))
    delay_combination = list(itertools.product(range(4), repeat=par.N_UE * 2))

    # initialization
    aaa=battery_combination[random.randrange(0,len(battery_combination))]
    bbb=delay_combination[random.randrange(0,len(delay_combination))]
    battery = [aaa,aaa,aaa]
    delay = [bbb,bbb,bbb]

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})

    total_existence_of_penalty = np.zeros(shape=(test_predict.shape[0], par.N_schemes))
    total_each_time_existence_of_penalty = np.zeros(shape=(test_predict.shape[0], par.N_time, par.N_schemes))
    total_each_UE_each_penalty_info = np.zeros(shape=(par.N_UE, 3, test_predict.shape[0], par.N_time, par.N_schemes))  # 3 means UL battery, UL deadline, DL deadline

    for i in range(test_predict.shape[0]):
    #for i in range(10):
        for s in range(par.N_schemes):
            if s==0:      #machine
                decision_idx = np.argmax(test_predict[i,:])
                decision = decision_combination[decision_idx]
                battery[0], delay[0], each_time_existence_of_penalty, existence_of_penalty, each_UE_each_penalty_info = BB_module(par, test_3time[i], decision, battery[0], delay[0])
            if s==1:      #random
                decision = decision_combination[random.randrange(0,len(decision_combination))]
                battery[1], delay[1], each_time_existence_of_penalty, existence_of_penalty, each_UE_each_penalty_info = BB_module(par, test_3time[i], decision, battery[1], delay[1])
            if s==2:      #round robin
                decision = decision_combination[i % len(decision_combination)]
                battery[2], delay[2], each_time_existence_of_penalty, existence_of_penalty, each_UE_each_penalty_info = BB_module(par, test_3time[i], decision, battery[2], delay[2])

            total_existence_of_penalty[i,s]=existence_of_penalty
            total_each_time_existence_of_penalty[i,:,s]=each_time_existence_of_penalty
            total_each_UE_each_penalty_info[:,:,i,:,s] = each_UE_each_penalty_info

    abstract_total_existence_of_penalty = np.zeros(shape=(test_predict.shape[0]/100, par.N_schemes))
    for k in range(test_predict.shape[0]/100):
        for i in range(par.N_schemes):
                abstract_total_existence_of_penalty[k,i] = np.sum(total_existence_of_penalty[100*k:100*k+100,i])
    # Plot predictions
    plt.plot(abstract_total_existence_of_penalty[:,0],label='machine')
    plt.plot(abstract_total_existence_of_penalty[:,1],label='random')
    plt.plot(abstract_total_existence_of_penalty[:,2],label='roundrobin')
    #plt.legend([abstract_total_existence_of_penalty[:,0], abstract_total_existence_of_penalty[:,1], abstract_total_existence_of_penalty[:,2]], ['machine', 'random', 'roundrobin'])
    #plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Penalty count")
    plt.show()

    np.save('/home/mukim/Desktop/EH_ver_2/ver3_5000_total_existence_of_penalty.npy',total_existence_of_penalty)
    np.save('/home/mukim/Desktop/EH_ver_2/ver3_5000_total_each_time_existence_of_penalty.npy', total_each_time_existence_of_penalty)
    np.save('/home/mukim/Desktop/EH_ver_2/ver3_5000_total_each_UE_each_penalty_info.npy', total_each_UE_each_penalty_info)
