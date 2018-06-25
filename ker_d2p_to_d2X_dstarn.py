from numerov_and_ker_api import *

ker_exp = np.loadtxt(EXP_KER_PATH)
energies = ker_exp[:,0]
events_nbr_exp = ker_exp[:,1]

#numerov_res = numerov(D2X_1SG_NUMEROV_PARAMS)
#numerov_res.plot()

for n in range(2,7):
    energy_shift = -0.754-energy_diff_with_ground_state(n)
    ker = comp_ker_vector(D2P_NUMEROV_PARAMS, MolecularState(D2X_1SG_NUMEROV_PARAMS),
    D2_plus_vib_level_distrib, energies, energy_shift)
    plt.plot(energies, ker*10**3, label=str(n))

plt.plot(energies, events_nbr_exp)
plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)

plt.show()
