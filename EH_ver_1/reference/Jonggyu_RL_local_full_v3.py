# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 23:13:46 2018

@author: jongg
"""
## maximizing the sum-rate with local CSI information
#basically, only user who has maximum SNR can be candidates of the RB.

import math
import tensorflow as tf
import numpy as np
import scipy.io as sio
model_path = "/tmp/model.ckpt"

partition_vec = np.array([0,2,5,7,10],dtype=int)
UE = 40 #per BS
RB = 100
BS = 4
distance=100
sess = tf.Session()
## Neural network
size_nn = tf.placeholder(tf.int32)
X = tf.placeholder(tf.float32,[None, 26])
y_input = tf.placeholder(tf.int32,[None, 2])
y_input_32 = tf.cast(y_input, tf.float32)
step_size = tf.placeholder(tf.float32)
h_1 = tf.contrib.layers.fully_connected(X,32)
h_7 = tf.contrib.layers.fully_connected(h_1,32)
h_8 = tf.contrib.layers.fully_connected(h_7,32)
h_9 = tf.contrib.layers.fully_connected(h_8,16)
h_10 = tf.contrib.layers.fully_connected(h_9,8)
h_11= tf.contrib.layers.fully_connected(h_10,4)
h_final = tf.contrib.layers.fully_connected(h_11,2,activation_fn=None)
y_output = tf.nn.softmax(h_final,1)
#loss = tf.reduce_sum(-y_input*tf.log(y_output))
loss = tf.reduce_sum(tf.square(y_input_32-y_output))
optim = tf.train.AdamOptimizer(learning_rate=step_size) # There are function that optimizing train
train = optim.minimize(loss)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess,'./my_model_RL_v3_2')
def input_gen(SNR, BS_ind,user_quantized):
    desired = SNR[BS_ind*10:(BS_ind+1)*10,BS_ind]
    a = np.max(desired)
    b = (np.sum(SNR[:,BS_ind])-np.sum(desired))/30.0
    undesired = np.concatenate((SNR[0:BS_ind*10,BS_ind],SNR[BS_ind*10+10:40,BS_ind]),axis=0)
    tmp = np.mod([BS_ind+1,BS_ind+2,BS_ind+3],4)
    quan_max = np.zeros(3)
    quan_min = np.zeros(3)
    quan_mean = np.zeros(3)
    quan_var = np.zeros(3)
    for ind_tmp in range(3):
        SNR_quan = SNR[tmp[ind_tmp]*10+user_quantized[0,tmp[ind_tmp]]:tmp[ind_tmp]*10+user_quantized[1,tmp[ind_tmp]],tmp[ind_tmp]]
        quan_max[ind_tmp] = np.max(SNR_quan)
        quan_min[ind_tmp] = np.min(SNR_quan)
        quan_mean[ind_tmp] = np.mean(SNR_quan)
        quan_var[ind_tmp] = np.var(SNR_quan)
    avg_val = np.array([a, b])
    und_res = np.reshape(undesired,[3,10])
    max_val = np.max(und_res,axis=1)
    min_val = np.min(und_res,axis=1)
    var_val = np.var(und_res,axis=1)
    mean_val = np.mean(und_res,axis=1)
    return np.concatenate((avg_val,max_val,mean_val,min_val,var_val,quan_max,quan_min,quan_mean,quan_var),axis=0), np.argmax(desired)

def quantized(SNR,partition_vec):
    avg_SNR = np.mean(SNR,axis=0)
    avg_part = np.zeros([np.shape(partition_vec)[0]-1,4])
    for BS_tmp in range(4):
        for ind_tmp in range(np.shape(partition_vec)[0]-1):
            avg_part[ind_tmp,BS_tmp] = np.mean(avg_SNR[BS_tmp*10+partition_vec[ind_tmp]:BS_tmp*10+partition_vec[ind_tmp+1],BS_tmp])
    ind_arg = np.argmax(avg_part,axis=0)
    out_range = np.zeros([2,4],dtype=int)
    for BS_tmp in range(4):
        out_range[:,BS_tmp] = partition_vec[ind_arg[BS_tmp]:ind_arg[BS_tmp]+2]
    return out_range

start = 0
end = 13
for j in np.arange(start,end):
    #SNR_map_overall = sio.loadmat('SNR_1.mat')
    SNR_map_overall =eval("sio.loadmat('SNR_"+str(j+1)+".mat')")
    SNR_map_overall = np.array(SNR_map_overall['SNR_map'])
    if j ==start:
        SNR_total=SNR_map_overall
    if j !=start:
        SNR_total = np.concatenate([SNR_total,SNR_map_overall],axis=0)
for j in range(SNR_total.shape[0]):
    SNR_j = SNR_total[j,:,:,:]
    UA = np.zeros([40],dtype=int)
    SNR_avg = np.mean(SNR_j,axis=0)
    for total_ind in range(10):
        for BS_ind in range(4):
            UE_selected = np.argmax(SNR_avg[:,BS_ind])
            UA[BS_ind*10+total_ind] = UE_selected
            SNR_avg[UE_selected,:]=0
    SNR_j = SNR_j[:,UA,:]
    SNR_total[j,:,:,:] = SNR_j
            
for total_index in range(100):
    ordered = np.random.permutation(SNR_total.shape[0])
    SR_result = np.zeros([SNR_total.shape[0]])
    SR_SNR = np.zeros([SNR_total.shape[0]])
    SR_SLNR = np.zeros([SNR_total.shape[0]])
    for j in range(SNR_total.shape[0]):
        SNR_map = SNR_total[ordered[j],:,:,:]
        user_range = quantized(SNR_map,partition_vec) 
        Alloc_index =np.zeros([100,40,4])
        SR=0
        loss_val = 0 
        data_rate_proposed = 0.0
        data_rate_SLNR = 0.0
        data_Rate_maxSNR = 0.0
        Alloc_index =np.zeros([100,40,4])
        for RB_ind in range(100):
            for BS_ind in range(4):
                input_1, max_1 = input_gen(SNR_map[RB_ind,:,:],BS_ind,user_range)
                leng = input_1.shape[0]
                output_1 = sess.run(y_output, feed_dict={X:input_1.reshape([1,-1])})
                user = np.argmax(output_1)
                user_selected = (1-user)*10+user*max_1
                if user_selected != 10:
                    Alloc_index[RB_ind,10*BS_ind+user_selected,BS_ind]=1
            SNR_map_after = np.multiply(SNR_map[RB_ind,:,:],np.tile(np.sum(Alloc_index[RB_ind,:,:],axis=0,keepdims=True),[40,1]))
            SINR_map_after = np.divide(SNR_map_after,1+np.tile(np.sum(SNR_map_after,axis=1,keepdims=True),[1,4])-SNR_map_after)
            data_rate_proposed += np.sum(np.log(1+SINR_map_after)/np.log(2)*Alloc_index[RB_ind,:,:])
        for rep in range(1):
            for RB_ind in range(100):
                for BS_ind in range(4):
                    Alloc_index[RB_ind,:,:]=0
                    for BS_ind_2 in range(4):
                        input_1, max_1 = input_gen(SNR_map[RB_ind,:,:],BS_ind_2,user_range)
                        output_1 = sess.run(y_output, feed_dict={X:input_1.reshape([1,-1]),size_nn : leng})
                        user = np.argmax(output_1)
                        user_selected = (1-user)*10+user*max_1
                        if user_selected != 10:
                            Alloc_index[RB_ind,10*BS_ind_2+user_selected,BS_ind_2]=1
                    Alloc_index_tmp = Alloc_index
                    input_1, max_1 = input_gen(SNR_map[RB_ind,:,:],BS_ind,user_range)
                    Alloc_index_tmp[RB_ind,:,BS_ind]=0
                    SNR_tmp = np.multiply(SNR_map[RB_ind,:,:],np.tile(np.sum(Alloc_index_tmp[RB_ind,:,:],axis=0,keepdims=True),[40,1]))
                    SINR = np.divide(SNR_tmp,1+np.tile(np.sum(SNR_tmp,axis=1,keepdims=True),[1,4])-SNR_tmp)
                    rate = np.log(1+SINR)/np.log(2)
                    SR_0 = np.sum(np.multiply(rate,Alloc_index_tmp[RB_ind,:,:]))
                    Alloc_index_tmp[RB_ind,max_1+BS_ind*10,BS_ind]=1
                    SNR_tmp = np.multiply(SNR_map[RB_ind,:,:],np.tile(np.sum(Alloc_index_tmp[RB_ind,:,:],axis=0,keepdims=True),[40,1]))
                    SINR = np.divide(SNR_tmp,1+np.tile(np.sum(SNR_tmp,axis=1,keepdims=True),[1,4])-SNR_tmp)
                    rate = np.log(1+SINR)/np.log(2)
                    SR_1 = np.sum(np.multiply(rate,Alloc_index_tmp[RB_ind,:,:]))
                    y_max = np.array([1-(SR_1>SR_0),(SR_1>SR_0)]).reshape(1,2)
                    _, A = sess.run([train,loss], feed_dict = {X:input_1.reshape(1,-1), y_input_32:y_max,step_size : 0.0000000001/(1+0.1*np.power(1.05,total_index))})
                    loss_val += A
            loss_val /= 400
        SR_result[j]=data_rate_proposed
        if j==649:
            print('%i -th iteration || Sumrate (Mbps):%f' %(total_index+1,0.18*np.mean(SR_result)))
            saver.save(sess, './my_model_RL_v3_2')
