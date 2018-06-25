from numerov_and_ker_api import *

#compute KER From triplet of D2+ to D2(n=2)

numerov_res_D2P = numerov(D2P_NUMEROV_PARAMS)
numerov_res_D2STAR_3_3SG = numerov(D2STAR_3_3SG_NUMEROV_PARAMS)

FCM_D2P_D2STAR_3_3SG = comp_franck_condon_matrix(numerov_res_D2P, numerov_res_D2STAR_3_3SG)
D2STAR_3_3SG_pop = final_pop_from_franck_condon(D2P_vib_levels_distr, FCM_D2P_D2STAR_3_3SG)
D2STAR_3_3SG_pop = build_proba_of_E_fun(D2STAR_3_3SG_pop,
numerov_res_D2STAR_3_3SG.eigen_values)

numerov_res_D2B = numerov(D2B_NUMEROV_PARAMS)

FCM = comp_franck_condon_matrix(numerov_res_D2STAR_3_3SG, numerov_res_D2B)

energies = np.linspace(0.02,15,500)
proba = []
for e in energies:
    p_e = ker_dissoc(numerov_res_D2STAR_3_3SG, numerov_res_D2B,
    D2STAR_3_3SG_pop, e, FCM)
    proba.append(p_e)
proba = np.array(proba)
plt.plot(energies, proba)
plt.show()



eval = numerov_res_D2STAR_3_3SG.eigen_values
evec = numerov_res_D2STAR_3_3SG.eigen_vectors

dissoc_level = -0.625*au_to_ev

energies = np.zeros((eval >= dissoc_level).size)
ker = np.zeros(energies.size)

for i in range(0, eval.size):
    if eval[i] >= dissoc_level:
        energies[i] = eval[i]-dissoc_level
        ker[i] = D2STAR_3_3SG_pop(eval[i])

print(energies)
print(ker)

plt.plot(energies, ker)
plt.show()


print(numerov_res_D2STAR_3_3SG.eigen_values)

# numerov_res_D2X_1SG_FREE_PART = numerov(D2X_1SG_FREE_PART_NUMEROV_PARAMS)
# FCM = comp_franck_condon_matrix(numerov_res_D2STAR_3_3SG, numerov_res_D2X_1SG_FREE_PART)
#
# proba = []
# for e in energies:
#     p_e = ker_dissoc(numerov_res_D2STAR_3_3SG, numerov_res_D2X_1SG_FREE_PART,
#     D2STAR_3_3SG_pop, e, FCM)
#     proba.append(p_e)
# proba = np.array(proba)
# plt.plot(energies, proba)
# plt.show()
