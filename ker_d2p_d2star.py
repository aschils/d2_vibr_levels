# -*- coding: utf-8 -*-

from numerov_and_ker_api import *

########################## BOUND state D_2^* potential ########################

#E_max = 1-.663091017*27.211

#eigen_values_bound = eigen_values_bound-27.211

#numerov_res_D2STAR_GK1SG = numerov(
#D2STAR_GK1SG_POTENTIAL_FILE, True, E_max, reduced_mass_d2, refine=5, pot_in_au=True)

#plot_wave_fun(eigen_vectors_bound_exc, eigen_values_bound_exc, r_bohr_bound_exc, V_bound_exc)


# proba_e = []
# energies = np.linspace(0, 6, 2000)
# delta_e = energies[1]-energies[0]
# normalization = 0
# i = 0
# delta_disp = energies.size/100
# disp_count = delta_disp
# for e in energies:
#     p_e = ker2(e, D2_plus_vib_level_distrib(eigen_values_bound))
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
# plt.plot(energies, proba_e)
# plt.show()

ker_exp = np.loadtxt(EXP_KER_PATH)
energies_exp = ker_exp[:,0]
energies = energies_exp
events_nbr_exp = ker_exp[:,1]
energy_shift = -0.754

plt.plot(energies_exp, events_nbr_exp)
plt.xlabel("Energy (eV)")
frame = plt.gca()
frame.axes.get_yaxis().set_visible(False)
plt.show()


# proba = comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_GK1SG_MS,
# only_ground_state_vib_level_distrib, energies, energy_shift)
# plt.plot(energies, proba)
# plt.show()
#
# ker_to_file(energies, proba, "data/numeric_ker_GK1SG_v0only_with_qr.txt")
#

# end_states_no_q_r = [(D2STAR_GK1SG_NUMEROV_PARAMS, 10**2*3, "GK1SG"),
# (D2STAR_2_3SG_NUMEROV_PARAMS, 150*3/4, "2_3SG"),
# (D2STAR_3_3SG_NUMEROV_PARAMS, 140, "3_3SG"),
# (D2STAR_1_SU_B_NUMEROV_PARAMS, 125, "1_SU_B"),
# (D2STAR_1_SU_BP_NUMEROV_PARAMS, 450, "1_SU_BP"),
# (D2STAR_1_SU_BPP_NUMEROV_PARAMS, 100, "1_SU_BPP"),
# (D2STAR_2_1SG_NUMEROV_PARAMS, 90, "2_1SG"),
# (D2STAR_3_1SG_NUMEROV_PARAMS, 300, "3_1SG"),
# (D2STAR_4_1SG_NUMEROV_PARAMS, 125, "4_1SG"),
# (D2STAR_2_3SU_NUMEROV_PARAMS, 125*1.5, "2_3SU"),
# (D2STAR_3_3SU_NUMEROV_PARAMS, 10, "3_3SU"),
# (D2STAR_4_3SU_NUMEROV_PARAMS, 125, "4_3SU")]

# Symétrie autorisée:
# Sigma_g^+, Pi_u, Delta_g
# Sigma_u^+, Pi_g, Delta_u
#
# end_states = [(D2STAR_GK1SG_NUMEROV_PARAMS, 10**2*4*2.7, "GK1SG"),
# (D2STAR_2_3SG_NUMEROV_PARAMS, 150*4/5, "2_3SG")
# (D2STAR_3_3SG_NUMEROV_PARAMS, 140*3/5, "3_3SG"),
# (D2STAR_1_SU_B_NUMEROV_PARAMS, 125, "1_SU_B"),
# (D2STAR_1_SU_BP_NUMEROV_PARAMS, 450, "1_SU_BP"),
# (D2STAR_1_SU_BPP_NUMEROV_PARAMS, 10, "1_SU_BPP"), #ici j avais mis 0
# (D2STAR_2_1SG_NUMEROV_PARAMS, 90, "2_1SG"),
# (D2STAR_3_1SG_NUMEROV_PARAMS, 300*1.5*2.7, "3_1SG"),
# (D2STAR_4_1SG_NUMEROV_PARAMS, 125*5/6, "4_1SG"),
# (D2STAR_2_3SU_NUMEROV_PARAMS, 125, "2_3SU"),
# (D2STAR_3_3SU_NUMEROV_PARAMS, 10, "3_3SU"),
# (D2STAR_4_3SU_NUMEROV_PARAMS, 10, "4_3SU"), #ici j avais mis 0
# (D2STAR_1PI_U_C_NUMEROV_PARAMS,50, "1PI_U_C"),
# (D2STAR_1PI_U_D_NUMEROV_PARAMS,50, "1PI_U_D"),
# (D2STAR_1PI_U_C_NUMEROV_PARAMS,50, "1PI_U_C"),
# (D2STAR_1PI_U_D_NUMEROV_PARAMS,50, "1PI_U_D"),
# (D2STAR_1_3PI_G_NUMEROV_PARAMS,100, "1_3PI_G"),
# (D2STAR_1_3PI_U_NUMEROV_PARAMS,100, "1_3PI_U"),
# (D2STAR_2_3PI_G_NUMEROV_PARAMS,100, "2_3PI_G"),
# (D2STAR_2_3PI_U_NUMEROV_PARAMS,100, "2_3PI_U"),
# (D2STAR_3_3PI_G_NUMEROV_PARAMS,100, "3_3PI_G"),
# (D2STAR_3_3PI_U_NUMEROV_PARAMS,100, "3_3PI_U"),
# (D2STAR_1_PI_GI_NUMEROV_PARAMS, 100, "1_PI_GI"),
# (D2STAR_1_PI_GR_NUMEROV_PARAMS, 100, "1_PI_GR")
# ]

#
# end_states_q_r_quad = [(D2STAR_GK1SG_NUMEROV_PARAMS, 10**2*4*5/6, "GK1SG"),
# (D2STAR_2_3SG_NUMEROV_PARAMS, 150*8/15, "2_3SG"),
# (D2STAR_3_3SG_NUMEROV_PARAMS, 20, "3_3SG"),
# (D2STAR_1_SU_B_NUMEROV_PARAMS, 125, "1_SU_B"),
# (D2STAR_1_SU_BP_NUMEROV_PARAMS, 233, "1_SU_BP"),
# (D2STAR_1_SU_BPP_NUMEROV_PARAMS, 10, "1_SU_BPP"),
# (D2STAR_2_1SG_NUMEROV_PARAMS, 90, "2_1SG"),
# #(D2STAR_3_1SG_NUMEROV_PARAMS, 300*1.5*2.7, "3_1SG"),
# (D2STAR_4_1SG_NUMEROV_PARAMS, 70, "4_1SG"),
# (D2STAR_2_3SU_NUMEROV_PARAMS, 125*3/4, "2_3SU"),
# (D2STAR_3_3SU_NUMEROV_PARAMS, 10, "3_3SU"),
# (D2STAR_4_3SU_NUMEROV_PARAMS, 10, "4_3SU"),
# (D2STAR_1PI_U_C_NUMEROV_PARAMS,50, "1PI_U_C"),
# (D2STAR_1PI_U_D_NUMEROV_PARAMS,20, "1PI_U_D")]
#
#
def ker_vec_fit(energies,alpha1, alpha2, alpha3):#, alpha3, alpha4, alpha5):

    #return gaussian(a, b, c, energies)*( \
    return comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_GK1SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies,energy_shift)*alpha1+ \
    comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_1_SU_BP_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies, energy_shift)*alpha2+ \
    comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_2_3SU_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies, energy_shift)*alpha3
    #comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_3_3SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha3 + \
    #comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_2_3SU_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha4 + \
    #comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_4_3SU_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha5)
    #comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_2_3SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha4)
    # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_2_3SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha3 + \
    # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_1_SU_B_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha4 + \
    # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_2_1SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha5 + \
    # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_4_1SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha6 + \
    # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_2_3SU_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha7 + \
    # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_3_3SU_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha8 + \
    # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_3_3SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha9
    #)
#
#energies_fit_idx = np.logical_and(energies > 1.2, energies < 2)
# energies_fit_idx = np.logical_and(energies > 0.5, energies < 2)
#
#energies_fit = energies[energies_fit_idx]
#events_nbr_exp_fit = events_nbr_exp[energies_fit_idx]

#Bound for width at mid height of gaussian 1eV

#b_bot_bound = 1.15
#b_up_bound = 1.5
#c_up_bound = 1/math.sqrt(2*math.log(2))

#res = sp.optimize.curve_fit(ker_vec_fit, energies_fit,
#events_nbr_exp_fit, p0=(1, 1.3, 0.5, 1, 1, 1),
#bounds = ((0,b_bot_bound,0,0,0,0), (math.inf,b_up_bound,c_up_bound,math.inf,math.inf,math.inf))
#)

#res = sp.optimize.curve_fit(ker_vec_fit, energies_fit,
#events_nbr_exp_fit, p0=(1, 1, 1),
#bounds = ((0,0,0), (math.inf,math.inf,math.inf))
#)

#print(res)
#(alpha1, alpha2, alpha3) = res[0]


#numerov_params_list = [D2STAR_GK1SG_NUMEROV_PARAMS, D2STAR_1_SU_BP_NUMEROV_PARAMS,
#D2STAR_3_3SG_NUMEROV_PARAMS, D2STAR_2_3SU_NUMEROV_PARAMS, D2STAR_4_3SU_NUMEROV_PARAMS]

#pop = pop_from_coef(energies, numerov_params_list, alpha1, alpha2)
#alpha3, alpha4, alpha5)
#
# print("GK1SG "+str(pop[0]))
# print("1_SU_BP "+str(pop[1]))
# print("3_3SG "+str(pop[2]))
# print("2_3SU "+str(pop[3]))
# print("4_3SU "+str(pop[4]))






#plt.plot(energies, events_nbr_exp)
#plt.plot(energies, ker_vec_fit(energies,alpha1, alpha2, alpha3))#, alpha3, alpha4, alpha5))
#plt.plot(energies, alpha1*gaussian(a, b, c,energies))
#plt.show()

#not_zero_idx = energies > 0.5
#energies =  energies[not_zero_idx]

ker_states = [(D2STAR_GK1SG_NUMEROV_PARAMS, 0.03/5, "GK1SG", 2, 0)]
#(D2STAR_1_SU_BP_NUMEROV_PARAMS, 0.001, "1_SU_BP", 1, 0),
#(D2STAR_2_3SU_NUMEROV_PARAMS, 0.001, "2_3SU", 1, 1),
#(D2STAR_1PI_U_C_NUMEROV_PARAMS, 0.001, "1PI_U_C", 1, 0),
#(D2STAR_3_3SG_NUMEROV_PARAMS, 0.001, "3_3SG", 0, 1),
#(D2STAR_1_SU_B_NUMEROV_PARAMS, 0.001, "1_SU_B", 1, 0),
#(D2STAR_1_SU_BPP_NUMEROV_PARAMS, 0.001, "1_SU_BPP", 1, 0),
#(D2STAR_2_1SG_NUMEROV_PARAMS, 0.001, "2_1SG", 0, 0),
#(D2STAR_3_1SG_NUMEROV_PARAMS, 0.001, "3_1SG", 0, 0),
#(D2STAR_4_1SG_NUMEROV_PARAMS, 0.001, "4_1SG", 0, 0), # pas sur pour L S
#(D2STAR_3_3SU_NUMEROV_PARAMS, 0.001, "3_3SU", 1, 1),
#(D2STAR_4_3SU_NUMEROV_PARAMS, 0.001, "4_3SU", 3, 1),
#(D2STAR_1PI_U_D_NUMEROV_PARAMS,0.001, "1PI_U_D", 1, 0),
#(D2STAR_1PI_U_C_NUMEROV_PARAMS,0.001, "1PI_U_C", 1, 0),
#(D2STAR_1_3PI_G_NUMEROV_PARAMS,0.001, "1_3PI_G", 2, 0),
#(D2STAR_1_3PI_U_NUMEROV_PARAMS,0.001, "1_3PI_U", 1, 0),
#(D2STAR_2_3PI_G_NUMEROV_PARAMS,0.001, "2_3PI_G", 2, 1),
#(D2STAR_2_3PI_U_NUMEROV_PARAMS,0.001, "2_3PI_U", 1, 1),
#(D2STAR_3_3PI_G_NUMEROV_PARAMS,0.001, "3_3PI_G", 2, 1),
#(D2STAR_3_3PI_U_NUMEROV_PARAMS,0.001, "3_3PI_U", 1, 1),
#(D2STAR_1_PI_GI_NUMEROV_PARAMS, 0.001, "1_PI_GI", 0, 0),# pas sur pour L S
#(D2STAR_1_PI_GR_NUMEROV_PARAMS, 0.001, "1_PI_GR", 0, 0)]# pas sur pour L S

# def ker_f(params):
#     (numerov_params, scale_coef, label, L, S) = params
#     return comp_ker_vector(D2P_NUMEROV_PARAMS,
#     MolecularState(numerov_params, 0, L, S), D2_plus_vib_level_distrib, energies,energy_shift, use_cache = True)
#
# executor = concurrent.futures.ProcessPoolExecutor(1)
# futures = [executor.submit(ker_f, ker_state) for ker_state in ker_states]
# concurrent.futures.wait(futures)
#
# plt.plot(energies_exp, ker_exp)
# for i in range(0, len(futures)):
#     (numerov_params, scale_coef, label, L, S) = ker_states[i]
#     ker = futures[i].result()
#     plt.plot(energies, ker*scale_coef, label=label+" L "+str(L)+" S "+str(S))
# plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)
# plt.show()


#print(ker_f)
##energy_shift = -0.754
#plt.plot(energies, events_nbr_exp)
## for (numerov_params, scale_coef, label, l, v) in ker_states:
##     events_nbr = comp_ker_vector(D2P_NUMEROV_PARAMS,
##     MolecularState(numerov_params), D2_plus_vib_level_distrib,
##     energies,energy_shift)*scale_coef
##     plt.plot(energies, events_nbr, label=label)
## plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)
##
## plt.show()

#essayer largeur 1, 2, 3 eV entre 0.7 et 2.7 1/e^2 largeur à mi hauter +/- 1eV

#ker_lz = comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_GK1SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies,energy_shift)
#not_zero_idx = energies > 0.5


#ker_lz = ker_lz[not_zero_idx]
#energies =  energies[not_zero_idx]
#R = au_to_ev/(energies)

#R = 30.493

#energy = 1/R*au_to_ev

#lz = landau_zener(D2STAR_GK1SG_MS, energy, 1.646)

#print(transition_proba_at_one_curve_crossing_pt(30.493, 0.005/(au_to_ev), 3.87*10**-4, 15, 2*6.06*10**-3/au_to_ev, R**2))


#plt.plot(R, lz)
#plt.show()

#plt.plot(energies, lz)
#plt.show()


# R = np.arange(15,40)
# energies = 1/R*au_to_ev
#
#
# result = landau_zener(D2STAR_GK1SG_MS, energies, 1.6)
#
# plt.plot(R, result)
# plt.show()

# R = 30.493
# energy = 1/R*au_to_ev
# E_A = 1.646
#
# landau_zener(D2STAR_GK1SG_MS, energy, E_A)

#def transition_proba_at_one_curve_crossing_pt(R, 0.005/(au_to_ev), 3.87*10**-4, 15, deltaE_ic, deltaF):


#plt.plot(energies_exp, ker_exp)
#plt.plot(energies, comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_GK1SG_MS,
#D2_plus_vib_level_distrib, energies,energy_shift)*10**16)
#
#print(comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_GK1SG_MS,
#D2_plus_vib_level_distrib, energies, energy_shift))
#plt.show()
# #print(R)
# #print(lz)
# #fit = interp1d(R, lz,kind="linear",fill_value="extrapolate")
# #print(fit(30)*(0.529*10**-8)**2)

# R_au = 30.493
# E_As = np.linspace(0.1,10, num=1000)
# H_ic = H(D2STAR_GK1SG_MS, E_As, R_au)
# lz = np.zeros(E_As.size)
# for i in range(0, lz.size):
#     lz[i] = landau_zener(H_ic[i], R_au)
# plt.plot(E_As, lz)
# plt.xlabel('E_A (eV)')
# plt.ylabel("sigma landau zener (au) for R = 30.493 au")
# plt.show()
#
# E_A = 1.646
# R_au_s = np.linspace(0,50, num=1000)
# H_ic = H(D2STAR_GK1SG_MS, np.array([E_A]), R_au_s)
# lz = np.zeros(R_au_s.size)
# for i in range(0, lz.size):
#     lz[i] = landau_zener(H_ic[i], R_au_s[i])
#
# plt.plot(R_au_s, lz)
# plt.xlabel('R (au)')
# plt.ylabel("sigma landau zener (au) for E_A = 1.646 eV")
# plt.show()




#numerov_res_gk1sg = numerov(D2STAR_GK1SG_NUMEROV_PARAMS, use_cache=False)
#numerov_res_sub = numerov(D2STAR_1_SU_B_NUMEROV_PARAMS, use_cache=False)
#print("numeric domain of numerov started at: "+str(numerov_res.r_bohr[0])+"bohr")
#ev = numerov_res.eigen_values
#print("eigen_values: "+str(ev+17.93289676+14.02655))
#numerov_res.plot()
#Ei_minus_Ef_matrix = energy_diff_matrix(numerov_res_gk1sg, numerov_res_sub,
#0)
#FCM = comp_landau_zener_matrix(numerov_res_gk1sg, MolecularState(D2STAR_1_SU_B_NUMEROV_PARAMS), numerov_res_sub,
#Ei_minus_Ef_matrix, None, 0)
#for j in range(0,FCM[0].size):
#    print("v'' = "+str(j)+" "+str(FCM[0][j]))


#Find transitions responsible for right pic in experimental KER
# v_i <-> v_f such that KER is between 1.7 and 2.4 eV
#(f_state, v_i, v_f) <-> count

def ker_find_resp_trans(E, bound_vib_level_distrib, numerov_res_i, molecular_state, numerov_res_f,
landau_zener_matrix, Ei_minus_Ef):

    (r_bohr_i, V_i, eigen_values_i, eigen_vectors_i) = (numerov_res_i.r_bohr,
    numerov_res_i.V, numerov_res_i.eigen_values, numerov_res_i.eigen_vectors)
    (r_bohr_f, V_f, eigen_values_f, eigen_vectors_f) = (numerov_res_f.r_bohr,
    numerov_res_f.V, numerov_res_f.eigen_values, numerov_res_f.eigen_vectors)

    ker_of_E_contrib = np.zeros((len(eigen_vectors_i), len(eigen_vectors_f)))
    proba_v_f = bound_vib_level_distrib(eigen_values_i)

    for i in range(0,len(eigen_vectors_i)):
        for j in range(0,len(eigen_vectors_f)):
            proba_v = proba_v_f(eigen_values_i[i])

            sigma = 0.05/math.sqrt(2*math.log(2))

            if Ei_minus_Ef[i,j] > 0:

                ker_of_E_contrib[i,j] = proba_v* \
                math.exp(-(E-Ei_minus_Ef[i,j])**2/(2*sigma**2))/0.106447* \
                landau_zener_matrix[i,j]


    return ker_of_E_contrib


def comp_ker_vector_find_resp_trans(numerov_params_i, molecular_state_f,
vib_level_distrib_i, energies, energy_shift=0, use_cache = True, lz_formule_id=0):

    numerov_params_f = molecular_state_f.get_numerov_params()

    numerov_res_i = numerov(numerov_params_i)
    numerov_res_f = numerov(numerov_params_f)

    Ei_minus_Ef_matrix = energy_diff_matrix(numerov_res_i, numerov_res_f,
    energy_shift)
    E_As = electron_binding_energy_of_final_state_matrix(numerov_res_i,
    numerov_res_f)
    landau_zener_matrix = comp_landau_zener_matrix(numerov_res_i, molecular_state_f,
    numerov_res_f, Ei_minus_Ef_matrix, E_As, lz_formule_id)

    ker_of_E_contrib = {}
    for i in range(0,energies.size):
        ker_of_E_contrib[energies[i]] = ker_find_resp_trans(energies[i], vib_level_distrib_i,
        numerov_res_i, molecular_state_f, numerov_res_f, landau_zener_matrix,
        Ei_minus_Ef_matrix)

    return ker_of_E_contrib


#
# @Returns: the position of the N greater numbers in the matrix in descending
# order.
#
# @Pre: - all matrix elements are >= 0
#       - N < number of lines X number of columns
#
def x_max_idx(matrix, N):

    dim = matrix.shape
    positions = []

    for i in range(0,N):

        max_pos = np.argmax(matrix)
        i = max_pos//dim[1]
        j = max_pos%dim[1]
        positions.append((i,j))
        matrix[i,j] = 0
    return positions

energy_shift = -0.754
energies = np.array([1])#np.linspace(1.5, 2.5, 10)
#plt.plot(energies, events_nbr_exp)
for (numerov_params, scale_coef, label, l, v) in ker_states:
    ker_of_E_contrib = comp_ker_vector_find_resp_trans(D2P_NUMEROV_PARAMS,
    MolecularState(numerov_params), D2_plus_vib_level_distrib,
    energies,energy_shift)

ker_of_E_contrib_matrix = ker_of_E_contrib[1]
max_elements_pos = x_max_idx(ker_of_E_contrib_matrix, 5)
print(max_elements_pos)




    #print(ker_of_E_contrib)
    #plt.plot(energies, events_nbr, label=label)
#plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)

#plt.show()

#Compute the R^2 for the various vib levels of D2+

numerov_D2P = numerov(D2P_NUMEROV_PARAMS)
D2P_ev = numerov_D2P.eigen_vectors
r_bohr_squared_D2P = numerov_D2P.r_bohr**2
dr = (numerov_D2P.r_bohr[1]-numerov_D2P.r_bohr[0])*bohr_to_meter

Rsquared = []
for i in range(0,len(D2P_ev)):

    psi_i = D2P_ev[i]
    #Compute <Psi | R^2 | Psi > = <R^2>
    Rsquared.append(wave_fun_scalar_prod(psi_i*numerov_D2P.r_bohr, psi_i, dr))
print(Rsquared)
