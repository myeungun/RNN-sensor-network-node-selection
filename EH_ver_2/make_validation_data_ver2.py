import numpy as np
import itertools
import pandas as pd
import csv
from blackbox_module import BB_module

class par:
    UL = 0
    DL = 1
    N_UE = 2 # number of UEs = 2, number of cases = 2*2
    N_time = 3
    N_realization = int(5)
    N_quantize_type = 3 # Low=0, Middle=1, High=2

iterations = 5000

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

def rearrange_H(h_UE_time,par):
    h_flatten = np.reshape(h_UE_time, newshape=[6, ])
    h_rearranged = np.zeros(shape=h_flatten.shape)
    h_rearranged[0:par.N_time-1] = h_flatten[0:par.N_time-1]
    h_rearranged[par.N_time-1:par.N_time+1] = h_flatten[par.N_time:par.N_time+2]
    h_rearranged[par.N_time+1] = h_flatten[par.N_time-1]
    h_rearranged[par.N_time*2-1] = h_flatten[par.N_time*2-1]

    return h_rearranged

f = open('/home/mukim/Desktop/EH_ver_2/validation_channel_time_UE_ver2.csv', 'w')
f_3 = open('/home/mukim/Desktop/EH_ver_2/validation_3_timeslot_channel_UE_time_ver2.csv', 'w')
#f_d = open('/home/mukim/Desktop/EH_ver_2/validation_decision.csv', 'w')
wr = csv.writer(f)
wr_3 = csv.writer(f_3)
#wr_d = csv.writer(f_d)

#one_hot_Y_table = np.identity(64)

# make new start point of a realization
# Only dL channel information
h_UE_time = np.zeros(shape=(par.N_UE,par.N_time+1))
h_tmp = np.random.randn(par.N_UE,)+1j*np.random.randn(par.N_UE,)
h_UE_time[:,0] = (h_tmp*np.conjugate(h_tmp)).real
count = 0
while count <= iterations:
    print (count)
    # make a realization with 4 time slots
    h_UE_time = make_a_realization(h_UE_time, par)
    h_UE_time[:, 0] = h_UE_time[:,3]

    # save 3 timeslot channel info for performance measurement
    for y in range(h_UE_time.shape[0]):
        wr_3.writerow(h_UE_time[y,:3])

    # save 2 timeslot input
    h_rearranged = rearrange_H(h_UE_time[:,:3], par)
    h_time_UE = np.transpose(np.reshape(h_rearranged[:4], newshape=(2, 2)))
    for x in range(h_time_UE.shape[0]):
        wr.writerow(h_time_UE[x,:])

    count += 1

f.close()
f_3.close()
#f_d.close()


