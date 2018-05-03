import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data/urbain_fonc_ondes_continuum.txt")

evals = data[:,0]
R = data[:,1]
evecs = data[:,2]

eval_1 = 3.674984321553838E-002
eval_2 = 7.349968643107677E-002
eval_3 = 1.102495296466152E-001
eval_4 = 1.469993728621535E-001

select_one = evals == eval_1
select_two = evals == eval_2
select_three = evals == eval_3
select_four = evals == eval_4

#plt.plot(R[select_one], evecs[select_one])


plt.plot(R[select_two], evecs[select_two])
#plt.plot(R[select_three], evecs[select_three])
#plt.plot(R[select_four], evecs[select_four])

data = np.loadtxt("data/free_evec.txt")
R = data[:,0]
v = data[:,36]
plt.plot(R,v)
#for i in range(1, data[0].size):

print(np.amax(evecs[select_two])/np.amax(v))

plt.show()
