import numpy as np
import matplotlib
import matplotlib.pyplot as plt

total_existence_of_penalty=np.load('/home/mukim/Desktop/EH_ver_2/ver3_1_5000_15time_total_existence_of_penalty.npy')
# (total test num) X (scheme num)
total_each_UE_each_penalty_info=np.load('/home/mukim/Desktop/EH_ver_2/ver3_1_5000_15time_total_each_UE_each_penalty_info.npy')
# (UE num) X (kind of penalty) X (total test num) X (time slot num) X (scheme num)
total_each_time_existence_of_penalty=np.load('/home/mukim/Desktop/EH_ver_2/ver3_1_5000_15time_total_each_time_existence_of_penalty.npy')
# (total test num) X (time slot num) X (scheme num)

for i in range(total_existence_of_penalty.shape[1]):
    print ("total penalty count for scheme # %d : %d" %((i+1),np.sum(total_existence_of_penalty[:,i])))

for i in range(total_each_UE_each_penalty_info.shape[4]): #scheme num
    for j in range(total_each_UE_each_penalty_info.shape[1]): # kind of penalty
        for k in range(total_each_UE_each_penalty_info.shape[3]): #time slot num
            print ("scheme #: %d, penalty #: %d, timeslot #: %d, count: %d" %((i+1), (j+1), (k+1), np.sum(total_each_UE_each_penalty_info[:,j,:,k,i])))

for i in range(total_each_time_existence_of_penalty.shape[2]): #scheme num
    for j in range(total_each_time_existence_of_penalty.shape[1]): #time slot num
        print ("scheme #: %d, timeslot #: %d, count: %d" %((i+1),(j+1),np.sum(total_each_time_existence_of_penalty[:,j,i])))



#abstract_total_existence_of_penalty = np.zeros(shape=(total_existence_of_penalty.shape[0] / 100, total_existence_of_penalty.shape[1]))
#for k in range(total_existence_of_penalty.shape[0] / 100):
#     for i in range(total_existence_of_penalty.shape[1]):
#         abstract_total_existence_of_penalty[k, i] = np.sum(total_existence_of_penalty[100 * k:100 * k + 100, i])
#         # Plot predictions
# plt.plot(abstract_total_existence_of_penalty[:, 0], label='machine')
# plt.plot(abstract_total_existence_of_penalty[:, 1], label='random')
# plt.plot(abstract_total_existence_of_penalty[:, 2], label='roundrobin')
# plt.xlabel("Time Period")
# plt.ylabel("Penalty count")
# plt.show()

