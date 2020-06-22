import random

def UL_bat_update(i, battery, ind, penalty):
    UL_success_ind = 0
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
    return penalty, battery, UL_success_ind

def UL_deadline_update(i, delay, ind, penalty, UL_success_ind):
    if ind==1:
        if UL_success_ind==0: # battery is not enough --> fail to transmit UL
            if delay[i] <= 0:
                penalty += 1
            delay = list(delay)
            delay[i] -= 1
            delay = tuple(delay)
        else: # battery is enough --> success to transmit UL
            if delay[i] < 0:
                penalty += 1
            delay = list(delay)
            numlist = [1,3,5,7]
            delay[i] = random.sample(numlist,1)[0]
            delay = tuple(delay)
    else:
        if delay[i] <= 0:
            penalty += 1
        delay = list(delay)
        delay[i] -= 1
        delay = tuple(delay)

    return penalty, delay

def DL_bat_update(i, battery, H, ind, par):
    if ind==0:
        battery = list(battery)
        battery[i-par.N_UE] += par.Harvesting_efficiency*H[i-par.N_UE]
        battery = tuple(battery)
    return battery

def DL_deadline_update(i, delay, ind, penalty):
    if ind==1:
        if delay[i] < 0:
            penalty += 1
        delay = list(delay)
        numlist = [1, 3, 5, 7]
        delay[i] = random.sample(numlist, 1)[0]
        delay = tuple(delay)
    else:
        if delay[i] <= 0:
            penalty += 1
        delay = list(delay)

        delay[i] -=1
        delay = tuple(delay)
    return penalty, delay

def info_update(par, time_ind, decision, battery, H, delay, penalty):
    #UL
    if decision[time_ind]<par.N_UE:
        for i in range(par.N_UE):
            # selected
            if decision[time_ind]==i:
                selected_ind=1
                penalty, battery, UL_success_ind = UL_bat_update(i, battery, selected_ind, penalty)
                penalty, delay = UL_deadline_update(i, delay, selected_ind, penalty, UL_success_ind)

                selected_ind=0
                for DL_i in range(par.N_UE, par.N_UE*2):
                    penalty, delay = DL_deadline_update(DL_i, delay, selected_ind, penalty)

            # not selected
            else:
                selected_ind=0
                penalty, battery, UL_success_ind = UL_bat_update(i, battery, selected_ind, penalty)
                penalty, delay = UL_deadline_update(i, delay, selected_ind, penalty, UL_success_ind)

    #DL
    if decision[time_ind]>=par.N_UE:
        for i in range(par.N_UE, par.N_UE*2):
            # selected
            if decision[time_ind]==i:
                selected_ind=1
                penalty, delay = DL_deadline_update(i, delay, selected_ind, penalty)

                selected_ind = 0
                UL_success_ind = 0
                for UL_i in range(par.N_UE):
                    penalty, delay = UL_deadline_update(UL_i, delay, selected_ind, penalty, UL_success_ind)

            # not selected
            else:
                selected_ind=0
                battery = DL_bat_update(i, battery, H[:, time_ind], selected_ind, par)
                penalty, delay = DL_deadline_update(i, delay, selected_ind, penalty)

    return battery, delay, penalty

def BB_module(par, H, decision, battery, delay):
    total_penalty = 0
    this_time_penalty = 0
    for time_ind in range(len(decision)):
        battery, delay, this_time_penalty = info_update(par, time_ind, decision, battery, H, delay, this_time_penalty)
        if this_time_penalty >=1:
            total_penalty += 1
    return total_penalty
    # Maximum penalty = 3