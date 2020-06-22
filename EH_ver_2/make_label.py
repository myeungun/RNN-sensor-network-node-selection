from modules import *
import datetime

MAX_ITERATION = int(5000)

class par:
    UL = 0
    DL = 1
    N_UE = 2 # number of UEs = 2, number of cases = 2*2
    N_time = 3
    N_realization = int(5)
    N_quantize_type = 3 # Low=0, Middle=1, High=2
    Harvesting_efficiency = 1.0

# Total case results table
d_1 = pow(par.N_quantize_type,(par.N_UE*(par.N_time-1)))
d_2 = pow(par.N_quantize_type,par.N_UE)
d_3 = pow(par.N_UE*2,par.N_time)
result_table = np.zeros(shape=(d_1, d_2, d_3))

for itr_1 in range(MAX_ITERATION):
    print("MAX_ITERATION: %d, Harvesting_efficiency: %f" % (MAX_ITERATION,par.Harvesting_efficiency))
    print("current time: %s, itr_1: %d" % (datetime.datetime.now(), itr_1))

    # make new start point of a realization
    # Only dL channel information
    h_UE_time = np.zeros(shape=(par.N_UE,par.N_time))
    h_tmp = np.random.randn(par.N_UE,)+1j*np.random.randn(par.N_UE,)
    h_UE_time[:,0] = (h_tmp*np.conjugate(h_tmp)).real

    # make multiple realizations with the same start point
    for itr_2 in range(par.N_realization):

        h_UE_time = make_a_realization(h_UE_time,par)

        num_final_decision = find_the_best_decision_for_the_realization(h_UE_time,par)

        h_quantized = quantize_H(h_UE_time,par) # UE1_past, UE1_current, UE2_past, UE2_current, UE1_future, UE2_future

        result_table = save_result(h_quantized, num_final_decision, result_table, par)
        print (num_final_decision)

real_label = np.zeros(shape=(d_1,1))
# make real labels
for itr_3 in range(d_1):
    predicted_future_h_ind = np.argmax(np.sum(result_table[itr_3],axis=1))
    final_decision_ind = np.argmax(result_table[itr_3,predicted_future_h_ind])

    real_label[itr_3] = final_decision_ind

np.save('/home/mukim/Desktop/EH_ver_2/label/ver3_label_reali_5_itr_5000_eff_1.npy',real_label)








