from numerov_and_ker_api import *


################### Compute KER D2_+, D2_b #############################

# Note on D2_+ numerov:
# Lowest energy bounded state E_min 10.8198528842
# Set E=11 to have 1 computed bound state
# Set E=12.9 to have 27 computed bound states


# #Compute bounded stats of D_2+ molecules: wave functions and their energies
# (r_bohr_bound, V_bound, eigen_values_bound, eigen_vectors_bound) = numerov(
# "pot_d2+.txt", True, E_ground_state, reduced_mass_d2, refine=3)
# #Compute free states of dissociated molecule
# (r_bohr_free, V_free, eigen_values_free, eigen_vectors_free) = numerov(
# "pot_d2_b.txt", False, 10, reduced_mass_d2, refine=1)
#
# plot_wave_fun(eigen_vectors_bound, eigen_values_bound, r_bohr_bound, V_bound)
# plot_wave_fun(eigen_vectors_free, eigen_values_free, r_bohr_free, V_free)
#
# proba_e = []
# energies = np.linspace(0.001, 9, 2000)
# delta_e = energies[1]-energies[0]
# normalization = 0
# i = 0
# delta_disp = energies.size/100
# disp_count = delta_disp
# for e in energies:
#     p_e = ker(D2_plus_vib_level_distrib(eigen_values_bound), eigen_values_bound,
#     eigen_vectors_bound, eigen_values_free, eigen_vectors_free, r_bohr_bound,
#     r_bohr_free, e, ionisation_potential_d2, V_free)
#     proba_e.append(p_e)
#
#     #normalization = normalization+p_e*delta_e
#
#     progress = i/energies.size*100
#     if i / disp_count == 10:
#         print("Progress  "+str(progress)+"%")
#         disp_count = disp_count+delta_disp
#     i = i+1
# proba_e = proba_e#/normalization
#
# #np.set_printoptions(threshold=np.nan)
#
# # ker_out_file = open("ker.txt", "w")
# # for i in range(0, energies.size):
# #     ker_out_file.write(str(energies[i])+" ")
# #     ker_out_file.write(str(proba_e[i])+" ")
# #     ker_out_file.write("\n")
# # ker_out_file.close()
#
# plt.plot(energies, proba_e)
# plt.show()

#ker_exp_f_path = "data/0eV_pos_ker_dissoc.whist"
ker_exp_f_path = "data/d2_ker_distribution.txt"
ker_exp = np.loadtxt(ker_exp_f_path, skiprows=1)
#Drop values below 0.2eV and above 7ev
ker_exp = ker_exp[ker_exp[:,0] >= 0.2]
#ker_exp = ker_exp[ker_exp[:,0] <= 7]

numerov_res_D2P = numerov(D2P_NUMEROV_PARAMS_DISSOC)
numerov_res_D2B = numerov(D2B_NUMEROV_PARAMS)
numerov_res_D2B.write_eigen_vector("data/free_evec.txt")
#numerov_res_D2P.plot()
#numerov_res_D2B.plot()
#
numerov_res_i = numerov_res_D2P
numerov_res_f = numerov_res_D2B
# plot_bound_and_free(numerov_res_i.eigen_vectors, numerov_res_i.eigen_values,
# numerov_res_i.r_bohr, numerov_res_i.V,
# numerov_res_f.eigen_vectors, numerov_res_f.eigen_values,
# numerov_res_f.r_bohr, numerov_res_f.V, plot_ratio_bound=1, plot_ratio_free=1,
# V_free_shift=0)

FCM = comp_franck_condon_matrix(numerov_res_i, numerov_res_f)

energies = ker_exp[:,0]
proba_exp = ker_exp[:,1]
energies_theo = numerov_res_D2B.eigen_values #np.linspace(0.0, 10, 200)
energies_theo = energies_theo[energies_theo > 0.12]
#energies_theo = energies_theo[energies_theo < 5]


proba_exp_f = interp1d(energies,proba_exp,kind="linear",fill_value="extrapolate")


#
res = sp.optimize.curve_fit(ker_to_fit, (energies_theo, numerov_res_D2P,
numerov_res_D2B, FCM), proba_exp_f(energies_theo), p0=(10e29))
#res = sp.optimize.curve_fit(ker_to_fit, energies, proba_exp, p0=(35*10**8))#, 3))
#print(res)
#res = [(0.000146889116013, 8.00268053247)]
#(alpha, gamma) = (222521.324352, -4455915.30647)#res[0]
#(alpha, gamma) = res[0]
alpha = res[0]

print("minimum error for alpha "+str(alpha))#+" and gamma "+str(gamma))
#print(res[1])
#
# #energies_theo = np.linspace(0, 0.5, 2000)
proba_theo = []
for e in energies_theo:
    p_e = ker_dissoc(numerov_res_D2P, numerov_res_D2B,
    D2_plus_vib_level_distrib(numerov_res_D2P.eigen_values), e,
    FCM)
    proba_theo.append(p_e)
proba_theo_fit = alpha*np.array(proba_theo)
#print(proba_theo)
#
plt.plot(energies, proba_exp)
plt.plot(energies_theo, proba_theo_fit)
plt.show()


#q(r)
