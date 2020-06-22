# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 01:06:06 2018

@author: Jonggyu
"""
import tensorflow as tf
import numpy as np
import scipy.io as sio
# Data 
num_action =2
epsilon = 0.2
RB_num = 100
partition_vec = np.array([0,2,4,6,8,10],dtype=int)
partition_vec_tf = tf.convert_to_tensor(partition_vec)
gamma= 0
# DQN module
def _qvalues(input_1,BS_ind, reuse_ = True):
    with tf.variable_scope(BS_ind, reuse= reuse_):
#        input_1 = tf.layers.batch_normalization(input_1,axis=1)
        h1 = tf.contrib.layers.fully_connected(input_1,64)
        h2 = tf.contrib.layers.fully_connected(h1,128)
        h3 = tf.contrib.layers.fully_connected(h2,256)
        h4_1 = tf.contrib.layers.fully_connected(h3,512)
        h4_2= tf.contrib.layers.fully_connected(h4_1,1024)
        h4_3 = tf.contrib.layers.fully_connected(h4_2,1024)
        h4_4 = tf.contrib.layers.fully_connected(h4_3,1024)
        h4_5 = tf.contrib.layers.fully_connected(h4_4,512)
        h5 = tf.contrib.layers.fully_connected(h4_5,256)
        h6 = tf.contrib.layers.fully_connected(h3,128)
        h7 = tf.contrib.layers.fully_connected(h3,64)
        h8 = tf.contrib.layers.fully_connected(h3,32)
        h9 = tf.contrib.layers.fully_connected(h3,16)
        h10 = tf.contrib.layers.fully_connected(h3,8)
        h11 = tf.contrib.layers.fully_connected(h3,4)
        output = tf.contrib.layers.fully_connected(h3,num_action,activation_fn = None)
        return output
def input_gen(SNR, BS_ind,user_quantized):
    desired = SNR[:,BS_ind*10:(BS_ind+1)*10,BS_ind]
    #a = tf.reduce_max(desired,axis=1,keepdims=True)
    #b = (tf.reduce_sum(SNR[:,:,BS_ind],axis=1,keepdims=True)-tf.reduce_sum(desired,axis=1,keepdims=True))/30.0
    undesired = tf.concat((SNR[:,0:BS_ind*10,BS_ind],SNR[:,BS_ind*10+10:40,BS_ind]),axis=1)
    full = tf.concat((desired,undesired),axis=1)
    tmp = np.mod([BS_ind+1,BS_ind+2,BS_ind+3],4)
    SNR_quan = tf.zeros([100,3,np.floor_divide(10,partition_vec.shape[0]-1)])
    for ind_tmp in range(3):
        for RB_tmp in range(100):
            RB_sel = np.floor_divide(RB_tmp,RB_num)
            SNR_quan_tmp = tf.Variable(tf.zeros_like(SNR_quan))
            SNR_quan_tmp = SNR_quan_tmp[RB_tmp,ind_tmp,:].assign(SNR[RB_tmp,tmp[ind_tmp]*10+user_quantized[0,tmp[ind_tmp],RB_sel]:tmp[ind_tmp]*10+user_quantized[1,tmp[ind_tmp],RB_sel],BS_ind])
            SNR_quan = tf.add(SNR_quan,SNR_quan_tmp)
    SNR_quan = tf.reshape(SNR_quan,[100,-1])
    return tf.concat((full,SNR_quan),axis=1), tf.argmax(desired,axis=1)

def quantized(SNR,partition_vec):
    out_range = tf.zeros([2,4,np.floor_divide(100,RB_num)],dtype=tf.int32)
    for RB_q in range(np.floor_divide(100,RB_num)):
        avg_SNR = tf.reduce_mean(SNR[RB_q*RB_num:RB_q*RB_num+RB_num,:,:],axis=0)
        avg_part = tf.zeros([np.shape(partition_vec)[0]-1,4])
        for BS_tmp in range(4):
            for ind_tmp in range(np.shape(partition_vec)[0]-1):
                avg_tmp = tf.Variable(tf.zeros_like(avg_part))
                avg_tmp = avg_tmp[ind_tmp,BS_tmp].assign(tf.reduce_mean(avg_SNR[BS_tmp*10+partition_vec[ind_tmp]:BS_tmp*10+partition_vec[ind_tmp+1],BS_tmp]))
                avg_part = tf.add(avg_part,avg_tmp) 
        ind_arg = tf.argmax(avg_part,axis=0)
        for BS_tmp in range(4):
            out_tmp = tf.Variable(tf.zeros_like(out_range))
            out_tmp = out_tmp[:,BS_tmp,RB_q].assign(partition_vec_tf[ind_arg[BS_tmp]:ind_arg[BS_tmp]+2])
            out_range = tf.add(out_range,out_tmp)
    return out_range
def random_action(action,epsilon):
    action = tf.argmax(action,axis=1)# 1이면 할당 0이면 할당안함
    random_action = tf.random_uniform([100],0,num_action,tf.int64)
    should_explore = tf.random_uniform([100],0,1)<epsilon
    action_learning = tf.where(should_explore,random_action, action)
    return action_learning
## Q leanring model
# DNN input
SNR_map = tf.placeholder(tf.float32,[100,40,4])
user_quantized_current = quantized(SNR_map,partition_vec)
for BS_ind in range(4):
    exec("BS_input_c" +str(BS_ind+1)+ ", max_ue_c" +str(BS_ind+1)+" = input_gen(SNR_map,BS_ind,user_quantized_current)")
    exec("action_tmp_"+ str(BS_ind+1) +"=_qvalues(BS_input_c"+ str(BS_ind+1) +",str(BS_ind),False)")
    exec("random_tmp_"+ str(BS_ind+1) +" = random_action(action_tmp_"+ str(BS_ind+1) +",epsilon)")
    exec("BS_alloc_tmp" +str(BS_ind+1) + "=tf.reshape(random_tmp_" + str(BS_ind+1)+"/(num_action-1),[100,1])")
    exec("BS_" + str(BS_ind+1) + "_alloc = tf.reshape(tf.one_hot(max_ue_c"+str(BS_ind+1) +"+BS_ind*10,depth=40)*tf.cast(tf.tile(BS_alloc_tmp"+str(BS_ind+1) +",[1,40]),tf.float32),[100,40,1])")

# Rate (reward)
Alloc_ind = tf.concat((BS_1_alloc,BS_2_alloc,BS_3_alloc,BS_4_alloc),axis=2)
SINR = tf.divide(SNR_map*Alloc_ind,(1+tf.tile(tf.reduce_sum(tf.tile(tf.reduce_sum(Alloc_ind,axis=1,keepdims=True),[1,40,1])*SNR_map,axis=2,keepdims=True),[1,1,4])-SNR_map))
Rate = tf.log(1+SINR)*180*1000
reward = tf.reduce_sum(tf.reduce_sum(Rate,axis=2),axis=1) # Actual sum rate를 reward로 받아
# Q_current + Action
for BS_ind in range(4):
    exec("action" + str(BS_ind+1) + " = tf.placeholder(tf.int64,[100])")
    exec("input_" + str(BS_ind+1) + " = tf.placeholder(tf.float32,[None,46])")
    exec("current_" + str(BS_ind+1) + " = tf.reduce_sum(_qvalues(input_"+ str(BS_ind+1) +",str(BS_ind))*tf.one_hot(action"+ str(BS_ind+1) +",depth=num_action),axis=1)") #현재 선택한 action별 q value들
#    exec("current_" + str(BS_ind+1) + " = tf.gather(_qvalues(input_"+ str(BS_ind+1) +",str(BS_ind)),action"+ str(BS_ind+1) +")[:,0]") #현재 선택한 action별 q value들
# Q_target
SNR_next = tf.placeholder(tf.float32,[100,40,4])
rate_t = tf.placeholder(tf.float32,[100])
user_quantized_target = quantized(SNR_next,partition_vec)
for BS_ind in range(4):
    exec("BS_input_t" +str(BS_ind+1)+ ", max_ue_t" +str(BS_ind+1)+" = input_gen(SNR_next,BS_ind,user_quantized_target)")
    exec("action_next"+str(BS_ind+1)+" = _qvalues(BS_input_t"+ str(BS_ind+1)+ ",str(BS_ind))")
    exec("Qmax_next"+str(BS_ind+1)+ " =  tf.reduce_max(action_next"+str(BS_ind+1)+",axis=1)")# 1이면 할당 0이면 할당안함
    exec("target_"+str(BS_ind+1) + " = rate_t + gamma*Qmax_next"+str(BS_ind+1))
Q_target_1 = tf.placeholder(tf.float32,[100])
Q_target_2 = tf.placeholder(tf.float32,[100])
Q_target_3 = tf.placeholder(tf.float32,[100])
Q_target_4 = tf.placeholder(tf.float32,[100])
loss = tf.reduce_sum(tf.add(tf.add(tf.square(current_1-Q_target_1),tf.square(current_2-Q_target_2)),tf.add(tf.square(current_3-Q_target_3),tf.square(current_4-Q_target_4))))
optim = tf.train.AdamOptimizer(learning_rate=0.0005) # There are function that optimizing train
train = optim.minimize(loss)
#loss = tf.square(current-target)
        
# Initialization 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
## USer association
start =0
end = 10
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
# learning
for total_iter in range(1000):
    reward_total= 0
    for iteration in range(10):
        for time_slot in range(49):
            # neural network input(state) +  action (with epsilon-greedy)
            for BS_ind in range(4):
                exec("input_current_"+str(BS_ind+1)+" , Action_L"+str(BS_ind+1)+"= sess.run([BS_input_c"+str(BS_ind+1)+",random_tmp_"+str(BS_ind+1)+"],feed_dict = {SNR_map : SNR_total[iteration*50+time_slot,:,:,:]})")
            # reward
            Reward_L = sess.run(reward,feed_dict={SNR_map : SNR_total[iteration*50+time_slot,:,:,:]})/1000000
            reward_total+=np.sum(Reward_L)
            # current 
            for BS_ind in range(4):
                exec("current_L"+str(BS_ind+1)+"=sess.run(current_" + str(BS_ind+1)+",feed_dict={input_"+str(BS_ind+1)+":input_current_"+str(BS_ind+1)+", action"+str(BS_ind+1)+":Action_L"+str(BS_ind+1)+"})") 
            # target
            for BS_ind in range(4):
                exec("Q_target_L_"+str(BS_ind+1)+"= sess.run(target_"+str(BS_ind+1)+",feed_dict = {SNR_next :SNR_total[iteration*50+time_slot+1,:,:,:],rate_t : Reward_L})")
            sess.run([train], feed_dict = {input_1:input_current_1, action1 : Action_L1,input_2:input_current_2, action2 : Action_L2, input_3:input_current_3, action3: Action_L3, input_4:input_current_4, action4: Action_L4, Q_target_1 : Q_target_L_1,Q_target_2 : Q_target_L_2,Q_target_3 : Q_target_L_3,Q_target_4 : Q_target_L_4})
    #         learning
    #        _, SR = sess.run([train,reward],feed_dict={SNR_map: SNR_total[iteration*50+time_slot,:,:,:] ,SNR_next :SNR_total[iteration*50+time_slot+1,:,:,:] })
    print("total_iter : %i  , Sum_rate : %f" %(total_iter,reward_total/49/10))  