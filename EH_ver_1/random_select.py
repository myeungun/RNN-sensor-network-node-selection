from __future__ import print_function
import tensorflow as tf
import numpy as np
from six.moves import xrange
import datetime
import scipy.io
import random
from info_update_ver3 import info_update


MAX_ITERATION = int(30)

class sys_param:
    N_USER = 20  # number of users
    epsilon = 0.0000001 # Energy harvesting energy efficiency (ratio)
    Tx_power = pow(10,-0.3) # at least -3dBm == 10^(-0.3)mW
    N_subcarrier = 96

def cal_penalty(penalty):
    SUM = penalty.UL_bat_level_penalty+penalty.UL_deadline_penalty
    total_penalty = np.concatenate((SUM, penalty.DL_deadline_penalty), axis=1)
    return total_penalty

def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    X = tf.placeholder(tf.float32, [4, sys_param.N_USER])
    Y = tf.placeholder(tf.int32, [None,2*sys_param.N_USER])

    # initial UL & DL deadline in time slot
    class deadline:
        True_UL_deadline     = np.random.randint(low=1, high=sys_param.N_USER * 5, size=20)
        #Expected_UL_deadline = np.random.randint(low=1, high=sys_param.N_USER, size=20)
        Expected_UL_deadline = np.zeros(shape=True_UL_deadline.shape)
        True_DL_deadline     = np.random.randint(low=1, high=sys_param.N_USER * 5, size=20)

    # initial battery level
    class bat_level:
        #Expected_bat_level = np.random.randint(low=0, high=40, size=20)
        True_bat_level     = np.random.uniform(low=0.5, high=0.5, size=20)
        #True_bat_level = np.zeros(shape=[20,])
        Expected_bat_level = np.zeros(shape=True_bat_level.shape)

    # initial penalty
    class penalty:
        UL_deadline_penalty  = np.zeros(shape=(1, 20), dtype=int)
        DL_deadline_penalty = np.zeros(shape=(1, 20), dtype=int)
        UL_bat_level_penalty = np.zeros(shape=(1, 20), dtype=int)
        count = np.zeros(shape=(3,20), dtype=int)

    # load channel information
    mat = scipy.io.loadmat('/home/mukim/Desktop/EH/H_H_hermitian.mat')
    channel = mat['H_H_hermitian']
    train_data_size = channel.shape[0]

    total_penalty = np.zeros([1,40])
    cumul_expected_penalty = np.zeros([1,40])

    save_penalty_count = np.zeros([MAX_ITERATION,3,sys_param.N_USER])

    for itr in xrange(MAX_ITERATION):

        print("Process: %d iteration, Current time: %s" % (itr+1, datetime.datetime.now()))
        penalty.count = np.zeros(shape=(3, 20), dtype=int)

        for train_count in xrange(train_data_size):

            # Make training dataset
            input = np.array([channel[train_count,:],deadline.Expected_UL_deadline,deadline.True_DL_deadline,bat_level.Expected_bat_level])
            input = np.reshape(input,newshape=(4,sys_param.N_USER))

            nd_array_selection = np.zeros(shape=[1,2*sys_param.N_USER])
            aaa=random.randint(0, 39)
            nd_array_selection[0][aaa]=1
            nd_array_selection=np.reshape(nd_array_selection,newshape=[2,sys_param.N_USER])
            deadline, bat_level, penalty, Expected_label, Expected_total_penalty = \
                info_update(sys_param,deadline,bat_level,penalty,nd_array_selection,input)

        save_penalty_count[itr,:,:]=penalty.count
        #np.save('/home/mukim/Desktop/EH/loss.npy', save_loss)
        np.save('/home/mukim/Desktop/EH/random_penalty_count.npy', save_penalty_count)





if __name__ == "__main__":
    tf.app.run()