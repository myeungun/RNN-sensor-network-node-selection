import numpy as np

RR      = np.load('/home/mukim/Desktop/EH/round_robin_penalty_count.npy')
Rand    = np.load('/home/mukim/Desktop/EH/random_penalty_count.npy')
Machine = np.load('/home/mukim/Desktop/EH/h5_penalty_penalty_count.npy')
MD      = np.load('/home/mukim/Desktop/EH/minimum_deadline_penalty_count.npy')

MAX_ITERATION=RR.shape[0]
KIND         =RR.shape[1]
N_USER       =RR.shape[2]

for case in range(0,4):
    if case==0:
        file_1 = open("/home/mukim/Desktop/EH/RR_UL_deadline.txt", 'w')
        file_2 = open("/home/mukim/Desktop/EH/RR_DL_deadline.txt", 'w')
        file_3 = open("/home/mukim/Desktop/EH/RR_bat_level.txt", 'w')

        result = RR

    if case==1:
        file_1 = open("/home/mukim/Desktop/EH/Rand_UL_deadline.txt", 'w')
        file_2 = open("/home/mukim/Desktop/EH/Rand_DL_deadline.txt", 'w')
        file_3 = open("/home/mukim/Desktop/EH/Rand_bat_level.txt", 'w')

        result = Rand

    if case==2:
        file_1 = open("/home/mukim/Desktop/EH/Machine_UL_deadline.txt", 'w')
        file_2 = open("/home/mukim/Desktop/EH/Machine_DL_deadline.txt", 'w')
        file_3 = open("/home/mukim/Desktop/EH/Machine_bat_level.txt", 'w')

        result = Machine

    if case==3:
        file_1 = open("/home/mukim/Desktop/EH/MD_UL_deadline.txt", 'w')
        file_2 = open("/home/mukim/Desktop/EH/MD_DL_deadline.txt", 'w')
        file_3 = open("/home/mukim/Desktop/EH/MD_bat_level.txt", 'w')

        result = MD

    for itr in range(0,MAX_ITERATION):
        for kind in range(0,KIND):
            for ue_ind in range(0,N_USER):
                data = "%d, " % (result[itr][kind][ue_ind])
                if kind==0:
                    file_1.write(data)
                if kind==1:
                    file_2.write(data)
                if kind==2:
                    file_3.write(data)
            data = "\n"
            if kind == 0:
                file_1.write(data)
            if kind == 1:
                file_2.write(data)
            if kind == 2:
                file_3.write(data)

    file_1.close()
    file_2.close()
    file_3.close()