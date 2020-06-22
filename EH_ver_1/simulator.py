from __future__ import print_function
import tensorflow as tf
import numpy as np
from six.moves import xrange
import datetime
import network
import scipy.io
import random
from info_update_ver3 import info_update

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
tf.flags.DEFINE_string("logs_dir", "/fast/mukim/Copy/BigFaceReplacement/Minimum_n_Target_10x10/Train/weight/", "path to logs directory")
tf.flags.DEFINE_string("weight", "01_____.npy","the latest weight saved")

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

def train(loss_val,var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    #return optimizer.apply_gradients(grads)
    return optimizer.minimize(loss_val)

def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    X = tf.placeholder(tf.float32, [4, sys_param.N_USER])
    Y = tf.placeholder(tf.int32, [None,2*sys_param.N_USER])

    # initial UL & DL deadline in time slot
    class deadline:
        True_UL_deadline     = np.zeros(shape=[sys_param.N_USER,])
        for i in range(sys_param.N_USER):
            True_UL_deadline[i] = random.randint(5*i+1,5*i+6)
        Expected_UL_deadline = np.zeros(shape=True_UL_deadline.shape)
        True_DL_deadline     = np.zeros(shape=[sys_param.N_USER,])
        for i in range(sys_param.N_USER):
            True_DL_deadline[i] = random.randint(5*i+1,5*i+6)

    # initial battery level
    class bat_level:
        #Expected_bat_level = np.random.randint(low=0, high=40, size=20)
        True_bat_level     = sys_param.Tx_power*np.ones(shape=[sys_param.N_USER,])
        Expected_bat_level = np.zeros(shape=True_bat_level.shape)

    # initial penalty
    class penalty:
        UL_deadline_penalty  = np.zeros(shape=(1, 20), dtype=int)
        DL_deadline_penalty = np.zeros(shape=(1, 20), dtype=int)
        UL_bat_level_penalty = np.zeros(shape=(1, 20), dtype=int)
        count = np.zeros(shape=(3,20), dtype=int)

    h_5, selection, var_dict = network.inference(keep_probability, FLAGS.logs_dir, FLAGS.weight, X, sys_param.N_USER)

    # loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_5, labels=Y))

    trainable_var = tf.trainable_variables()

    train_op = train(loss,trainable_var)

    print("Session Open")

    sess = tf.Session()

    print("Weight Initialization")
    sess.run(tf.global_variables_initializer())

    if FLAGS.mode == "train":
        print("Start Training")

        # load channel information
        mat = scipy.io.loadmat('/home/mukim/Desktop/EH/H_H_hermitian.mat')
        channel = mat['H_H_hermitian']
        train_data_size = channel.shape[0]

        total_penalty = np.zeros([1,40])
        cumul_expected_penalty = np.zeros([1,40])

        save_loss = np.zeros([MAX_ITERATION, train_data_size/100])
        save_penalty_count = np.zeros([MAX_ITERATION,3,sys_param.N_USER])

        for itr in xrange(MAX_ITERATION):

            print("Process: %d iteration, Current time: %s" % (itr+1, datetime.datetime.now()))
            penalty.count = np.zeros(shape=(3, 20), dtype=int)

            for train_count in xrange(train_data_size):

                # Make training dataset

                UL_d_p = np.reshape(penalty.UL_deadline_penalty,newshape=[sys_param.N_USER,])
                DL_d_p = np.reshape(penalty.DL_deadline_penalty, newshape=[sys_param.N_USER, ])
                UL_b_p = np.reshape(penalty.UL_bat_level_penalty, newshape=[sys_param.N_USER, ])

                channel_info = np.zeros(shape=[sys_param.N_USER,])

                for i in range(sys_param.N_USER):
                    ind = np.argsort(channel[train_count, :])[i]
                    channel_info[ind] = i # the largest channel gain means sys_param.N-1, the smallest channel gain means 0

                # input = np.array([channel[train_count,:],deadline.Expected_UL_deadline,deadline.True_DL_deadline,bat_level.Expected_bat_level])
                input = np.array([channel_info, UL_d_p, DL_d_p, UL_b_p])
                input = np.reshape(input,newshape=(4,sys_param.N_USER))

                if (train_count%1000)<2*sys_param.N_USER:
                    nd_array_selection = np.zeros(shape=[1,2*sys_param.N_USER])
                    nd_array_selection[0][train_count%1000]=1
                    nd_array_selection=np.reshape(nd_array_selection,newshape=[2,sys_param.N_USER])
                    deadline, bat_level, penalty, Expected_label, Expected_total_penalty = \
                        info_update(sys_param,deadline,bat_level,penalty,nd_array_selection,input)
                else:
                    feed_dict = {keep_probability:0.7, X:input}
                    nd_array_selection =sess.run(selection, feed_dict=feed_dict)
                    deadline, bat_level, penalty, Expected_label, Expected_total_penalty = \
                        info_update(sys_param, deadline, bat_level, penalty, nd_array_selection, input)

                    feed_dict = {keep_probability: 0.7, X: input, Y: Expected_label}
                    sess.run(train_op, feed_dict=feed_dict)

                    if train_count % (train_data_size-1) == 0:
                        if (itr+1)%10==0:
                            weight_dict_ = sess.run(var_dict, feed_dict=feed_dict)
                            np.save("/home/mukim/Desktop/EH/weight"+ "_" + str(itr + 1) + ".npy", weight_dict_)
                            print("Weight saved!")
                    if train_count % 100 == 0:
                        train_loss, weight_dict_ = sess.run([loss,var_dict], feed_dict=feed_dict)
                        #print("-----------------Penalty count-------------------")
                        #print (penalty.count)
                        #print("------------True battery level----------------")
                        #print (bat_level.True_bat_level)
                        #print("------------Expected battery level----------------")
                        #print(bat_level.Expected_bat_level)
                        #current_penalty = cal_penalty(penalty)
                        #total_penalty = total_penalty + current_penalty
                        #cumul_expected_penalty = cumul_expected_penalty+Expected_total_penalty

                        #print (current_penalty)
                        #print("--------------EXPECTATION---------------")
                        #print (Expected_total_penalty)
                        save_loss[itr][train_count/100]=train_loss
                        print("Time: %s, Round: %d, Batch: %d, Train_loss:%g" % (datetime.datetime.now(), itr + 1, train_count, train_loss))
            save_penalty_count[itr,:,:]=penalty.count
            np.save('/home/mukim/Desktop/EH/h7_channelinfo_loss.npy', save_loss)
            np.save('/home/mukim/Desktop/EH/h7_channelinfo_penalty_count.npy', save_penalty_count)

    elif FLAGS.mode == "test":
        print("To be continue...")



if __name__ == "__main__":
    tf.app.run()