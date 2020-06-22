import numpy as np
import matplotlib
import matplotlib.pyplot as plt


t1=np.load('/home/mukim/Desktop/EH_ver_2/0.npy')
#t2=np.load('/home/mukim/Desktop/EH_ver_2/4.npy')
#t3=np.load('/home/mukim/Desktop/EH_ver_2/3.npy')
#t4=np.load('/home/mukim/Desktop/EH_ver_2/2.npy')
#t5=np.load('/home/mukim/Desktop/EH_ver_2/1.npy')

a=np.mean(t1,axis=2)

abstract_total_existence_of_penalty = np.zeros(shape=(51, 3))
for k in range(50):
    for i in range(3):
            abstract_total_existence_of_penalty[k+1,i] = np.sum(a[100*k:100*k+100,i])


plt.plot(abstract_total_existence_of_penalty[:,0],label='machine')
plt.plot(abstract_total_existence_of_penalty[:,1],label='random')
plt.plot(abstract_total_existence_of_penalty[:,2],label='roundrobin')
#plt.legend([abstract_total_existence_of_penalty[:,0], abstract_total_existence_of_penalty[:,1], abstract_total_existence_of_penalty[:,2]], ['machine', 'random', 'roundrobin'])

plt.xlabel("Time Period")
plt.ylabel("Penalty count")
plt.grid(True)
plt.ylim((82,100))
plt.xlim((1,50))
plt.savefig('penalty.eps', format='eps', dpi=1000)
plt.show()