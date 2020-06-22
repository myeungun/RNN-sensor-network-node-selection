import numpy as np
import itertools
import pandas as pd
import csv

class par:
    UL = 0
    DL = 1
    N_UE = 2 # number of UEs = 2, number of cases = 2*2
    N_time = 3
    N_realization = int(5)
    N_quantize_type = 3 # Low=0, Middle=1, High=2

iterations = 20000

def make_a_realization(h_UE_time, par):
    for i in range(1, par.N_time+1):
        while True:
            while True:
                h_tmp = np.random.randn(par.N_UE, ) + 1j * np.random.randn(par.N_UE, )
                h_UE_time[:, i] = (h_tmp*np.conjugate(h_tmp)).real
                if np.mean(abs(h_UE_time[:, i - 1] - h_UE_time[:, i])) < 1:
                    break
            if np.corrcoef(h_UE_time[:, i-1],h_UE_time[:, i])[0][1] > 0.99:
                break

    h_UE_time = h_UE_time.real.astype(float)

    return h_UE_time

def quantize_H(h_UE_time,par):
    h_flatten = np.reshape(h_UE_time, newshape=[6, ])
    h_quantized_tmp = np.zeros(shape=h_flatten.shape)
    h_quantized = np.zeros(shape=h_flatten.shape)
    for a in range(h_flatten.shape[0]):
        if h_flatten[a] < 1:
            h_quantized_tmp[a] = 0.5
        if h_flatten[a] >= 1 and h_flatten[a] < 2:
            h_quantized_tmp[a] = 1.5
        if h_flatten[a] >= 2:
            h_quantized_tmp[a] = 2.5
    h_quantized[0:par.N_time-1] = h_quantized_tmp[0:par.N_time-1]
    h_quantized[par.N_time-1:par.N_time+1] = h_quantized_tmp[par.N_time:par.N_time+2]
    h_quantized[par.N_time+1] = h_quantized_tmp[par.N_time-1]
    h_quantized[par.N_time*2-1] = h_quantized_tmp[par.N_time*2-1]

    return h_quantized

f = open('/home/mukim/Desktop/EH_ver_2/training_data/training_channel_ver3_label_30timeslot_reali_5_itr_5000_eff_0_7.csv', 'w')
f_d = open('/home/mukim/Desktop/EH_ver_2/training_data/training_decision_ver3_label_30timeslot_reali_5_itr_5000_eff_0_7.csv', 'w')
wr = csv.writer(f)
wr_d = csv.writer(f_d)

decision_table = np.load('/home/mukim/Desktop/EH_ver_2/label/ver3_label_30timeslot_reali_5_itr_5000_eff_0_7.npy')
one_hot_Y_table = np.identity(64)

# make new start point of a realization
# Only dL channel information
h_UE_time = np.zeros(shape=(par.N_UE, par.N_time + 1))
h_tmp = np.random.randn(par.N_UE, ) + 1j * np.random.randn(par.N_UE, )
count = 0
while count <= iterations:
    print (count)
    if count==0:
        h_UE_time[:, 0] = (h_tmp * np.conjugate(h_tmp)).real
    else:
        # the last time slot of the previous channel info
        h_UE_time[:, 0] = h_UE_time[:,3]

    # make a realization with 4 time slots
    h_UE_time = make_a_realization(h_UE_time, par)

    h_quantized = quantize_H(h_UE_time[:,:3], par)

    # save the decision
    quantization_combination = list(itertools.product(range(par.N_quantize_type), repeat=par.N_UE * (par.N_time - 1)))
    channel_ind = quantization_combination.index(tuple([int(k) for k in h_quantized[0:4]]))
    decision = decision_table[channel_ind]
    one_hot_decision = one_hot_Y_table[int(decision[0])]
    wr_d.writerow(one_hot_decision)

    # save 2 timeslot input
    h_time_UE = np.transpose(np.reshape(h_quantized[:4], newshape=(2, 2)))
    for x in range(h_time_UE.shape[0]):
        wr.writerow(h_time_UE[x,:])

    count += 1

f.close()
f_d.close()


