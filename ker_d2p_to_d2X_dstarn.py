from numerov_and_ker_api import *

ker_exp = np.loadtxt(EXP_KER_PATH)
energies = ker_exp[:,0]
events_nbr_exp = ker_exp[:,1]


ker = comp_ker_vector(D2P_POTENTIAL_FILE, D2X_1SG_NUMEROV_PARAMS,
D2_plus_vib_level_distrib, energies)

plt.plot(energies, ker)
plt.show()
