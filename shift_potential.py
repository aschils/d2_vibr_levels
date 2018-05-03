
import numpy as np

potential_data = np.loadtxt("data/pot_d2_b.txt")
shift = -27.211
R = potential_data[:,0]
potential = potential_data[:,1]

out_file = "data/pot_d2b_shifted.txt"
out_file = open(out_file, "w")

for i in range(0,R.size):
    line = str(R[i])+" "+str(potential[i]+shift)+"\n"
    out_file.write(line)

out_file.close()
