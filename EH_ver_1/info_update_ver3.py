import random
import tensorflow as tf
import numpy as np

def info_update(sys_param, deadline, bat_level, penalty, selection, input):

    ###########################
    # UL deadline info update #
    ###########################
    for slot_N in range(0, sys_param.N_USER):
        # not selected
        if selection[0][slot_N] == 0:
            deadline.True_UL_deadline[slot_N] = deadline.True_UL_deadline[slot_N] - 1

            deadline.Expected_UL_deadline[slot_N] = deadline.Expected_UL_deadline[slot_N] - 1

            # impose a Expected_UL_deadline_penalty
            if deadline.Expected_UL_deadline[slot_N] <= 0:
                penalty.UL_deadline_penalty[0][slot_N] += 1
            else:
                penalty.UL_deadline_penalty[0][slot_N] += 0

        # selected
        elif selection[0][slot_N] == 1:
            if deadline.True_UL_deadline[slot_N] < 0:
                # Only for performance evaluation
                penalty.count[0][slot_N] += (-1)*deadline.True_UL_deadline[slot_N]

            # generate the next UL deadline
            deadline.True_UL_deadline[slot_N] = random.randint(5*slot_N+1,5*slot_N+6)
            # feedback
            deadline.Expected_UL_deadline[slot_N] = deadline.True_UL_deadline[slot_N]

    ###########################
    # DL deadline info update #
    ###########################
    for slot_N in range(0, sys_param.N_USER):
        if deadline.True_DL_deadline[slot_N] < 0:
            # Only for performance evaluation
            penalty.count[1][slot_N] = penalty.count[1][slot_N] + 1
            penalty.DL_deadline_penalty[0][slot_N] += 1
        else:
            penalty.DL_deadline_penalty[0][slot_N] = 0
        # not selected
        if selection[1][slot_N] == 0:
            deadline.True_DL_deadline[slot_N] = deadline.True_DL_deadline[slot_N] - 1
        # selected
        elif selection[1][slot_N] == 1:
            deadline.True_DL_deadline[slot_N] = random.randint(5*slot_N+1,5*slot_N+6)

    ########################
    # Battery level update #
    ########################
    for slot_N in range(0, sys_param.N_USER):
        # not selected
        if selection[0][slot_N] == 0:
            bat_level.True_bat_level[slot_N] = bat_level.True_bat_level[slot_N] + sys_param.epsilon * (
            sys_param.Tx_power / sys_param.N_subcarrier) * input[0][slot_N] + 0.001 * sys_param.Tx_power

            whether_over_txpower_or_not = random.randrange(start=0, stop=2)
            if whether_over_txpower_or_not == 1:  # harvested energy >= tx power
                bat_level.Expected_bat_level[slot_N] = bat_level.Expected_bat_level[slot_N] + 1
            else:  # harvested energy < tx power
                bat_level.Expected_bat_level[slot_N] = bat_level.Expected_bat_level[slot_N]

            # impose a UL_bat_level_penalty
            if bat_level.Expected_bat_level[slot_N] < 1:
                penalty.UL_bat_level_penalty[0][slot_N] -= 1
            else:
                penalty.UL_bat_level_penalty[0][slot_N] += 0

        # selected
        elif selection[0][slot_N] == 1:
            if bat_level.True_bat_level[slot_N] < sys_param.Tx_power:
                # Only for performance evaluation
                penalty.count[2][slot_N] = penalty.count[2][slot_N] + 1

                if bat_level.True_bat_level[slot_N] - 0.1*sys_param.Tx_power > 0:
                    bat_level.True_bat_level[slot_N] = bat_level.True_bat_level[slot_N] - 0.1*sys_param.Tx_power # Only transmit NACK
                else:
                    bat_level.True_bat_level[slot_N] = 0
            else:
                bat_level.True_bat_level[slot_N] = bat_level.True_bat_level[slot_N] - sys_param.Tx_power  # Transmit data +Nack

            # feedback
            bat_level.Expected_bat_level[slot_N] = int(bat_level.True_bat_level[slot_N]/sys_param.Tx_power)



    SUM = 0.5*(penalty.UL_deadline_penalty+penalty.UL_bat_level_penalty)
    Expected_total_penalty = np.concatenate((SUM,penalty.DL_deadline_penalty),axis=1)
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