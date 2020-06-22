import numpy as np
import itertools
from blackbox_module_ver3 import BB_module
import random

def make_a_realization(h_UE_time, par):
    for i in range(1, 15):
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

def find_the_best_decision_for_the_realization(h_UE_time,par,ind, battery, delay):
    # decision / battery level / delay combinations
    decision_combination = list(itertools.product(range(par.N_UE * 2), repeat=par.N_time))
    battery_combination = list(itertools.product(np.arange(0, 1, 1 / 3.0), repeat=par.N_UE))
    delay_combination = list(itertools.product([1, 3, 5, 7], repeat=par.N_UE * 2))
    penalty_per_decision = np.zeros(shape=pow(par.N_UE * 2, par.N_time))
    if ind == 0:
        tmp_battery=np.zeros(shape=(len(decision_combination),len(battery_combination),len(delay_combination),par.N_UE))
        tmp_delay = np.zeros(shape=(len(decision_combination), len(battery_combination), len(delay_combination),par.N_UE*2))

        for deci_num in range(len(decision_combination)):
            decision = decision_combination[deci_num]
            for bat_num in range(len(battery_combination)):
                battery = battery_combination[bat_num]
                for delay_num in range(len(delay_combination)):
                    delay = delay_combination[delay_num]

                    total_penalty, battery, delay = BB_module(par, h_UE_time, decision, battery, delay)

                    tmp_battery[deci_num,bat_num,delay_num,:] = battery
                    tmp_delay[deci_num, bat_num, delay_num,:] = delay

                    penalty_per_decision[deci_num] += total_penalty
        f_index = np.argmin(penalty_per_decision)
        s_index = random.randrange(0,len(battery_combination)-1)
        t_index = random.randrange(0,len(delay_combination)-1)


        battery = tmp_battery[f_index,s_index,t_index,:]
        delay = tmp_delay[f_index,s_index,t_index,:]

    else:
        for deci_num in range(len(decision_combination)):
            decision = decision_combination[deci_num]
            total_penalty, battery, delay= BB_module(par, h_UE_time, decision, battery, delay)
            penalty_per_decision[deci_num] += total_penalty

    num_final_decision = np.argmin(penalty_per_decision)


    return battery, delay, num_final_decision

def quantize_H(h_UE_time,par):
    h_flatten = np.reshape(h_UE_time, newshape=[6, ])
    h_quantized_tmp = np.zeros(shape=h_flatten.shape)
    h_quantized = np.zeros(shape=h_flatten.shape)
    for a in range(h_flatten.shape[0]):
        if h_flatten[a] < 1:
            h_quantized_tmp[a] = 0
        if h_flatten[a] >= 1 and h_flatten[a] < 2:
            h_quantized_tmp[a] = 1
        if h_flatten[a] >= 2:
            h_quantized_tmp[a] = 2
    h_quantized[0:par.N_time-1] = h_quantized_tmp[0:par.N_time-1]
    h_quantized[par.N_time-1:par.N_time+1] = h_quantized_tmp[par.N_time:par.N_time+2]
    h_quantized[par.N_time+1] = h_quantized_tmp[par.N_time-1]
    h_quantized[par.N_time*2-1] = h_quantized_tmp[par.N_time*2-1]

    return h_quantized

def save_result(h_quantized, num_final_decision, result_table, par):
    quantization_combination = list(itertools.product(range(par.N_quantize_type), repeat=par.N_UE*(par.N_time-1)))
    last_quantization_combination = list(itertools.product(range(par.N_quantize_type), repeat=par.N_UE))

    d_1_ind = quantization_combination.index(tuple([int(k) for k in h_quantized[0:4]]))
    d_2_ind = last_quantization_combination.index(tuple([int(j) for j in h_quantized[4:6]]))
    d_3_ind = num_final_decision

    result_table[d_1_ind, d_2_ind,d_3_ind] += 1

    return result_table