import random
import tensorflow as tf
import numpy as np

def info_update(sys_param, deadline, bat_level, penalty, selection, input):
    Expected_UL_deadline_penalty  = np.zeros(shape=(1, 20), dtype=int)
    Expected_UL_bat_level_penalty = np.zeros(shape=(1, 20), dtype=int)
    Expected_DL_deadline_penalty  = np.zeros(shape=(1, 20), dtype=int)

    ###########################
    # UL deadline info update #
    ###########################
    for slot_N in range(0, sys_param.N_USER):
        # not selected
        if selection[0][slot_N] == 0:
            deadline.True_UL_deadline[slot_N] = deadline.True_UL_deadline[slot_N] - 1

            # impose a Expected_UL_deadline_penalty
            if deadline.Expected_UL_deadline[slot_N] < 0:
                Expected_UL_deadline_penalty[0][slot_N] += 1000
            else:
                Expected_UL_deadline_penalty[0][slot_N] += 0

            if deadline.Expected_UL_deadline[slot_N] == 3:
                whether_3_or_over = random.randrange(start=0, stop=2)
                if whether_3_or_over == 1:  # deadline=3
                    deadline.Expected_UL_deadline[slot_N] = deadline.Expected_UL_deadline[slot_N] - 1
                else:  # deadline>3
                    deadline.Expected_UL_deadline[slot_N] = deadline.Expected_UL_deadline[slot_N]
            if deadline.Expected_UL_deadline[slot_N] != 3:
                deadline.Expected_UL_deadline[slot_N] = deadline.Expected_UL_deadline[slot_N] - 1
        # selected
        elif selection[0][slot_N] == 1:
            if deadline.True_UL_deadline[slot_N] < 0:
                # Only for performance evaluation
                penalty.count[0][slot_N] = penalty.count[0][slot_N] + 1
            # generate the next UL deadline
            deadline.True_UL_deadline[slot_N] = random.randrange(start=1,
                                                                        stop=5 * sys_param.N_USER + 1)  # range: 1<=deadline<(2*N_USER+1)
            # impose a UL_deadline_penalty
            if deadline.True_UL_deadline[slot_N] == 1:
                penalty.UL_deadline_penalty[0][slot_N] = 500
            else:
                penalty.UL_deadline_penalty[0][slot_N] = 0

            Expected_UL_deadline_penalty[0][slot_N] = penalty.UL_deadline_penalty[0][slot_N]

            # 2 bit feedback
            if deadline.True_UL_deadline[slot_N] >= 3:
                deadline.Expected_UL_deadline[slot_N] = 3
            elif deadline.True_UL_deadline[slot_N] == 2:
                deadline.Expected_UL_deadline[slot_N] = 2
            elif deadline.True_UL_deadline[slot_N] == 1:
                deadline.Expected_UL_deadline[slot_N] = 1

    ###########################
    # DL deadline info update #
    ###########################
    for slot_N in range(0, sys_param.N_USER):
        # not selected
        if selection[1][slot_N] == 0:
            if deadline.True_DL_deadline[slot_N] < 0:
                penalty.DL_deadline_penalty[0][slot_N] += 1000
                # Only for performance evaluation
                penalty.count[1][slot_N] = penalty.count[1][slot_N] + 1
            else:
                penalty.DL_deadline_penalty[0][slot_N] += 0

            deadline.True_DL_deadline[slot_N] = deadline.True_DL_deadline[slot_N] - 1
        # selected
        elif selection[1][slot_N] == 1:
            if deadline.True_DL_deadline[slot_N] < 0:
                # Only for performance evaluation
                penalty.count[1][slot_N] = penalty.count[1][slot_N] + 1
            deadline.True_DL_deadline[slot_N] = random.randrange(start=1, stop=5 * sys_param.N_USER + 1)

        Expected_DL_deadline_penalty[0][slot_N] = penalty.DL_deadline_penalty[0][slot_N]


    ########################
    # Battery level update #
    ########################
    for slot_N in range(0, sys_param.N_USER):
        # not selected
        if selection[0][slot_N] == 0:
            bat_level.True_bat_level[slot_N] = bat_level.True_bat_level[slot_N] + sys_param.epsilon * (
            sys_param.Tx_power / sys_param.N_subcarrier) * input[0][slot_N] + 0.001 * sys_param.Tx_power

            # impose a UL_bat_level_penalty
            if bat_level.Expected_bat_level[slot_N] < 1:
                Expected_UL_bat_level_penalty[0][slot_N] -= 1000
            else:
                Expected_UL_bat_level_penalty[0][slot_N] = 0

            if bat_level.Expected_bat_level[slot_N] >= 3:
                bat_level.Expected_bat_level[slot_N] = 3
            elif bat_level.Expected_bat_level[slot_N] < 3:
                whether_over_txpower_or_not = random.randrange(start=0, stop=2)
                if whether_over_txpower_or_not == 1:  # harvested energy >= tx power
                    bat_level.Expected_bat_level[slot_N] = bat_level.Expected_bat_level[slot_N] + 1
                else:  # harvested energy < tx power
                    bat_level.Expected_bat_level[slot_N] = bat_level.Expected_bat_level[slot_N]
        # selected
        elif selection[0][slot_N] == 1:
            if bat_level.True_bat_level[slot_N] - sys_param.Tx_power > 0:
                bat_level.True_bat_level[slot_N] = bat_level.True_bat_level[slot_N] - sys_param.Tx_power # Transmit data + ACK
            else:
                if bat_level.True_bat_level[slot_N] - 0.1*sys_param.Tx_power > 0:
                    bat_level.True_bat_level[slot_N] = bat_level.True_bat_level[slot_N] - 0.1*sys_param.Tx_power # Only transmit NACK
                else:
                    bat_level.True_bat_level[slot_N] = 0
                # Only for performance evaluation
                penalty.count[2][slot_N] = penalty.count[2][slot_N] + 1

            # impose a UL_bat_level_penalty before updating True_bat_level
            if bat_level.True_bat_level[slot_N] < 1 * sys_param.Tx_power:
                penalty.UL_bat_level_penalty[0][slot_N] -= 1000
            else:
                penalty.UL_bat_level_penalty[0][slot_N] = 0

            Expected_UL_bat_level_penalty[0][slot_N] = penalty.UL_bat_level_penalty[0][slot_N]

            # 2 bit feedback
            if bat_level.True_bat_level[slot_N] >= 3 * sys_param.Tx_power:
                bat_level.Expected_bat_level[slot_N] = 3
            elif bat_level.True_bat_level[slot_N] >= 2 * sys_param.Tx_power and bat_level.True_bat_level[slot_N] < 3 * sys_param.Tx_power:
                bat_level.Expected_bat_level[slot_N] = 2
            elif bat_level.True_bat_level[slot_N] >= 1 * sys_param.Tx_power and bat_level.True_bat_level[slot_N] < 2 * sys_param.Tx_power:
                bat_level.Expected_bat_level[slot_N] = 1
            elif bat_level.True_bat_level[slot_N] < 1 * sys_param.Tx_power:
                bat_level.Expected_bat_level[slot_N] = 0

    SUM = 0.5*(Expected_UL_deadline_penalty+Expected_UL_bat_level_penalty)
    Expected_total_penalty = np.concatenate((SUM,Expected_DL_deadline_penalty),axis=1)
    total_ind = np.argmax(Expected_total_penalty)

    DL_ind = np.where(deadline.True_DL_deadline == 1)

    if DL_ind[0].shape==(0,):
        ind = total_ind
    else:
        ind = DL_ind[0][0]
        #coin=random.randrange(start=0, stop=2)
        #if coin == 0:
        #    ind=total_ind
        #else:
        #    ind=DL_ind[0][0]

    Expected_label = np.zeros(shape=[1,40])
    Expected_label[0][ind]=1


    return deadline, bat_level, penalty, Expected_label, Expected_total_penalty