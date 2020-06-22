import random
import numpy as np

def UL_bat_update(i, battery, ind, penalty):
    UL_success_ind = 0
    UL_battery_penalty = 0
    if ind==1:
        # enough battery level
        if battery[i]>1:
            battery = list(battery)
            battery[i]-=1
            battery = tuple(battery)
            UL_success_ind = 1
        # not enough battery level
        else:
            penalty += 1
            UL_battery_penalty = 1
            # enough for only NACK
            if battery[i] > 0.1:
                battery = list(battery)
                battery[i]-=0.1
                battery = tuple(battery)
            #else:
            #    battery[i] = 0
    else:
        battery = list(battery)
        battery[i] += 1/3.0
        battery = tuple(battery)
    return penalty, UL_battery_penalty, battery, UL_success_ind

def UL_deadline_update(i, delay, ind, penalty, UL_success_ind):
    UL_deadline_penalty = 0
    if ind==1:
        if UL_success_ind==0: # battery is not enough --> fail to transmit UL
            if delay[i] <= 0:
                penalty += 1
                UL_deadline_penalty = 1
            delay = list(delay)
            delay[i] = delay[i]-1
            delay = tuple(delay)
        else: # battery is enough --> success to transmit UL
            if delay[i] < 0:
                penalty += 1
                UL_deadline_penalty = 1
            delay = list(delay)
            num_list=[1,3,5,7]
            delay[i]=random.sample(num_list,1)[0]
            delay = tuple(delay)
    else:
        if delay[i] <= 0:
            penalty += 1
            UL_deadline_penalty = 1
        delay = list(delay)
        delay[i] = delay[i]-1
        delay = tuple(delay)

    return penalty, UL_deadline_penalty, delay

def DL_bat_update(i, battery, H, ind, par):
    if ind==0:
        battery = list(battery)
        battery[i-par.N_UE] += par.Harvesting_efficiency*H[i-par.N_UE]
        battery = tuple(battery)
    return battery

def DL_deadline_update(i, delay, ind, penalty):
    DL_deadline_penalty = 0
    if ind==1:
        if delay[i] < 0:
            penalty += 1
            DL_deadline_penalty = 1
        delay = list(delay)
        num_list = [1, 3, 5, 7]
        delay[i] = random.sample(num_list, 1)[0]
        delay = tuple(delay)
    else:
        if delay[i] <= 0:
            penalty += 1
            DL_deadline_penalty = 1
        delay = list(delay)
        delay[i] = delay[i]-1
        delay = tuple(delay)
    return penalty, DL_deadline_penalty,delay

def info_update(par, time_ind, decision, battery, H, delay, each_UE_each_penalty_info):
    penalty = 0

    #UL
    if decision[time_ind]<par.N_UE:
        for i in range(par.N_UE):
            # selected
            if decision[time_ind]==i:
                selected_ind=1
                penalty, UL_bat_penalty, battery, UL_success_ind = UL_bat_update(i, battery, selected_ind, penalty)
                penalty, UL_deadline_penalty, delay = UL_deadline_update(i, delay, selected_ind, penalty, UL_success_ind)

                if UL_bat_penalty == 1:
                    each_UE_each_penalty_info[i,0,time_ind]=1
                if UL_deadline_penalty ==1:
                    each_UE_each_penalty_info[i,1,time_ind]=1

                selected_ind=0
                for DL_i in range(par.N_UE, par.N_UE*2):
                    penalty, DL_deadline_penalty, delay = DL_deadline_update(DL_i, delay, selected_ind, penalty)
                    if DL_deadline_penalty == 1:
                        each_UE_each_penalty_info[DL_i-par.N_UE,2,time_ind]=1

            # not selected
            else:
                selected_ind=0
                penalty, UL_bat_penalty,battery, UL_success_ind = UL_bat_update(i, battery, selected_ind, penalty)
                penalty,UL_deadline_penalty, delay = UL_deadline_update(i, delay, selected_ind, penalty, UL_success_ind)

                if UL_bat_penalty == 1:
                    each_UE_each_penalty_info[i, 0, time_ind] = 1
                if UL_deadline_penalty == 1:
                    each_UE_each_penalty_info[i, 1, time_ind] = 1

    #DL
    if decision[time_ind]>=par.N_UE:
        for i in range(par.N_UE, par.N_UE*2):
            # selected
            if decision[time_ind]==i:
                selected_ind=1
                penalty,DL_deadline_penalty, delay = DL_deadline_update(i, delay, selected_ind, penalty)

                if DL_deadline_penalty == 1:
                    each_UE_each_penalty_info[i - par.N_UE, 2, time_ind] = 1

                selected_ind = 0
                UL_success_ind = 0
                for UL_i in range(par.N_UE):
                    penalty, UL_deadline_penalty, delay = UL_deadline_update(UL_i, delay, selected_ind, penalty, UL_success_ind)

                    if UL_deadline_penalty == 1:
                        each_UE_each_penalty_info[UL_i, 1, time_ind] = 1

            # not selected
            else:
                selected_ind=0
                battery = DL_bat_update(i, battery, H[:, time_ind], selected_ind, par)
                penalty, DL_deadline_penalty, delay = DL_deadline_update(i, delay, selected_ind, penalty)

                if DL_deadline_penalty == 1:
                    each_UE_each_penalty_info[i - par.N_UE, 2, time_ind] = 1

    return battery, delay, penalty, each_UE_each_penalty_info

def BB_module(par, H, decision, battery, delay):
    each_time_existence_of_penalty=np.zeros((3,))
    each_UE_each_penalty_info = np.zeros(shape=(par.N_UE, 3, par.N_time))  # 3 means the number of kind of penalties
    for time_ind in range(len(decision)):
        battery, delay, existence_of_penalty, each_UE_each_penalty_info = info_update(par, time_ind, decision, battery, H, delay, each_UE_each_penalty_info)
        if existence_of_penalty >= 1:
            each_time_existence_of_penalty[time_ind]=1
        else:
            each_time_existence_of_penalty[time_ind]=0

    if np.sum(each_time_existence_of_penalty)>0:
        existence_of_penalty=1

    return battery, delay, each_time_existence_of_penalty, existence_of_penalty, each_UE_each_penalty_info # UL battery, UL deadline, DL deadline