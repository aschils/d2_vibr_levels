from numerov_and_ker_api import *

def clean_plot():
    plt.clf()
    plt.cla()
    plt.close()


FIGURES_DIR = "../figures/"

#Generates examples of unbound and bound potentials

pot_file_path = POTENTIALS_DIR+"/"+D2X_1SG_POTENTIAL_FILE
pot_file = np.loadtxt(pot_file_path)
r = pot_file[:,0]
V = pot_file[:,1]*au_to_ev+32

plt.rcParams.update({"font.size": 24})

plt.figure()
plt.plot(r,V)
plt.axis([0,6, -0.5, 7])
plt.xlabel("Inter-atomic distance (Bohr)")
plt.ylabel("Potential (eV)")
plt.savefig(FIGURES_DIR+"D2X_1SG_POTENTIAL.pgf", bbox_inches="tight")
clean_plot()


pot_file_path = POTENTIALS_DIR+"/"+D2DISSOC_POTENTIAL_FILE
pot_file = np.loadtxt(pot_file_path)
r = pot_file[:,0]
V = pot_file[:,1]

plt.figure()
plt.plot(r,V)
plt.axis([0,6, -2, 20])
plt.xlabel("Inter-atomic distance (Bohr)")
plt.ylabel("Potential (eV)")
plt.savefig(FIGURES_DIR+"D2DISSOC_POTENTIAL_FILE.pgf", bbox_inches="tight")
clean_plot()


#Generates plots of wavefunctions for dissociative potential with different
#numerical domain sizes

#pot_file_path, is_bound_potential, E_max, reduced_mass,
#r_end_for_unbound_potential=10, refine=1, pot_in_au, auto_E_max

D2B_NUMEROV_PARAMS_local = NumerovParams(D2DISSOC_POTENTIAL_FILE, False, E_max_bound_states_D2b,
reduced_mass_d2, 7, refine, False, False)
numerov_result = numerov(D2B_NUMEROV_PARAMS_local, use_cache = True)
numerov_result.plot(r_bohr_max=float("inf"), plot_ratio=1, save=True, out_path=FIGURES_DIR+"numerov_dissoc_rend7.pgf",
	show=False)
plt.clf()
plt.cla()
plt.close()


figsize_p=(7,7)

D2B_NUMEROV_PARAMS_local = NumerovParams(D2DISSOC_POTENTIAL_FILE, False, E_max_bound_states_D2b,
reduced_mass_d2, 5, refine, False, False)
numerov_result = numerov(D2B_NUMEROV_PARAMS_local, use_cache = True)
numerov_result.plot(r_bohr_max=float("inf"), plot_ratio=1, save=True, out_path=FIGURES_DIR+"numerov_dissoc_rend5.pgf",
	show=False, figsize=figsize_p)
print(numerov_result.eigen_values.size)
clean_plot()


D2B_NUMEROV_PARAMS_local = NumerovParams(D2DISSOC_POTENTIAL_FILE, False, E_max_bound_states_D2b,
reduced_mass_d2, 3, refine, False, False)
numerov_result = numerov(D2B_NUMEROV_PARAMS_local, use_cache = True)
numerov_result.plot(r_bohr_max=float("inf"), plot_ratio=1, save=True, out_path=FIGURES_DIR+"numerov_dissoc_rend3.pgf",
	show=False, figsize=figsize_p)
clean_plot()

print(numerov_result.eigen_values.size)

#Plot to explain algorithm to find amplitude at x=infinity of continuum wavefunctions

D2B_NUMEROV_PARAMS_local = NumerovParams(D2DISSOC_POTENTIAL_FILE, False, 0.5,
reduced_mass_d2, 14, 10, False, False)
numerov_result = numerov(D2B_NUMEROV_PARAMS_local, use_cache = True)

dr = numerov_result.r_bohr[1]-numerov_result.r_bohr[0]

figsize_p=(12,8)

plot_wave_fun(np.abs(np.array([np.append(numerov_result.eigen_vectors[10],0)])),
[numerov_result.eigen_values[10]],
np.append(numerov_result.r_bohr, numerov_result.r_bohr[-1]+dr), numerov_result.V,
        1, True, FIGURES_DIR+"detail_computation_A.pgf", False, 
figsize_p, (0,0.25), "none", "x (Bohr)")

clean_plot()

#Outputs all figures and tables, Numerov
figsize_p = (12,10)

plot_x_end = 12
states = [(D2P_NUMEROV_PARAMS, 15 ,"$D_2^+$",figsize_p),
(D2STAR_GK1SG_NUMEROV_PARAMS, 8 , "GK1SG",figsize_p),
(D2STAR_2_3SG_NUMEROV_PARAMS, 15 , "2_3SG",figsize_p), #6
(D2STAR_3_3SG_NUMEROV_PARAMS, 15 , "3_3SG",figsize_p),
(D2STAR_1_SU_B_NUMEROV_PARAMS, 20 , "1_SU_B",figsize_p),
(D2STAR_1_SU_BP_NUMEROV_PARAMS,18 , "1_SU_BP",figsize_p),

(D2STAR_1_SU_BPP_NUMEROV_PARAMS, 36 , "1_SU_,BPP",figsize_p),
(D2STAR_2_1SG_NUMEROV_PARAMS, 17.5 , "2_1SG",figsize_p),
(D2STAR_3_1SG_NUMEROV_PARAMS, 8 , "3_1SG",figsize_p),
(D2STAR_4_1SG_NUMEROV_PARAMS, 36 , "4_1SG",figsize_p),

(D2STAR_2_3SU_NUMEROV_PARAMS, 5.5, "2_3SU",(14,8)), #figsize too small

(D2STAR_3_3SU_NUMEROV_PARAMS, 5 , "3_3SU",figsize_p),
(D2STAR_4_3SU_NUMEROV_PARAMS, 10 , "4_3SU",figsize_p),
(D2STAR_1PI_U_C_NUMEROV_PARAMS,9 , "1PI_U_C",figsize_p),
(D2STAR_1PI_U_D_NUMEROV_PARAMS,17.5 , "1PI_U_D",figsize_p),
(D2STAR_1PI_U_C_NUMEROV_PARAMS,9 , "1PI_U_C",figsize_p),
(D2STAR_1PI_U_D_NUMEROV_PARAMS,17.5 , "1PI_U_D",figsize_p),
(D2STAR_1_3PI_G_NUMEROV_PARAMS,5 , "1_3PI_G", (14,8)), #figsize too small
(D2STAR_1_3PI_U_NUMEROV_PARAMS,24 , "1_3PI_U",figsize_p),
(D2STAR_2_3PI_G_NUMEROV_PARAMS,20 , "2_3PI_G",figsize_p),
(D2STAR_2_3PI_U_NUMEROV_PARAMS,18 , "2_3PI_U",figsize_p),
(D2STAR_3_3PI_G_NUMEROV_PARAMS,6 , "3_3PI_G",figsize_p),
(D2STAR_3_3PI_U_NUMEROV_PARAMS,15 , "3_3PI_U",figsize_p),
(D2STAR_1_PI_GI_NUMEROV_PARAMS, 4.5 , "1_PI_GI",figsize_p),
(D2STAR_1_PI_GR_NUMEROV_PARAMS, 11 , "1_PI_GR",figsize_p)
]

#D2STAR_2_3SG_NUMEROV_PARAMS.plot_potential(15)

idx_to_consider = [0,1,2,3,4,5,10]

#file_nbr = 0
nbr_ev_per_line = 7
for file_nbr in idx_to_consider:
    state = states[file_nbr]
    #if state[2] == "2_3SU" or state[2] == "1_3PI_G":
    numerov_result = numerov(state[0])
    numerov_result.plot(r_bohr_max=state[1],save=True, out_path=FIGURES_DIR+"vib_states/file"+str(file_nbr)+".pgf",
	show=False, figsize=state[3])#, title=state[2], useTex=True)
    clean_plot()

    evs = numerov_result.eigen_values

    tex_tab = "\\begin{tabular}{|c|"
    for j in range(0, nbr_ev_per_line):
        tex_tab = tex_tab+"c|"
    tex_tab = tex_tab+"}"
    
    for i in range(0,evs.size):

        if i % nbr_ev_per_line == 0:

            if i != 0:
                tex_tab = tex_tab + "\\\\"            

            tex_tab = tex_tab+" \hline \n level"
            for j in range(1, nbr_ev_per_line+1):
                tex_tab = tex_tab+"&"+str(i+j)
            tex_tab = tex_tab+"\\\\ \hline \n energy (eV)"

        
        tex_tab = tex_tab + "&"+"%.3f" % evs[i]

    nbr_empty_to_add = nbr_ev_per_line - i % nbr_ev_per_line-1
    for j in range(0,nbr_empty_to_add):
        tex_tab = tex_tab+"&"

    tex_tab = tex_tab+"\\\\ \hline \n \end{tabular}"

    tex_file = open(FIGURES_DIR+"vib_states/file"+str(file_nbr)+".tex", "w")
    tex_file.write(tex_tab)
    tex_file.close()
    file_nbr = file_nbr+1


#Plot KER for D2+ to dissociated state comp_franck_condon_matrix

d2p_numerov_result = numerov(D2P_NUMEROV_PARAMS)
d2p_pot_data = np.loadtxt(POTENTIALS_DIR+"/"+D2P_POTENTIAL_FILE)
V_min = np.min(d2p_pot_data[:,1])
plt.figure(figsize=(12,12))
plt.rc("text", usetex=True)
#r_bohr_min = 0.9
#r_bohr_max = 10
V = lambda r: d2p_numerov_result.V(r) - V_min + ionisation_potential_d2
ev = d2p_numerov_result.eigen_values - V_min + ionisation_potential_d2
#r_bohr = d2p_numerov_result.r_bohr[d2p_numerov_result.r_bohr <= r_bohr_max]
r_bohr = d2p_numerov_result.r_bohr


plot_ref1 = plot_pot_and_ev(r_bohr, V, ev, d2p_numerov_result.min_of_V, label="$D_2^+$")


#under_plot_wave_fun(d2p_numerov_result.eigen_vectors, ev,
#r_bohr , V, plot_ratio=1, y_axis="energy eV")

d2b_pot_data = np.loadtxt(POTENTIALS_DIR+"/"+D2DISSOC_POTENTIAL_FILE)
r = pot_file[:,0]
#r_temp = r_init[r_init <= r_bohr_max]
#r = r_temp[r_temp >= r_bohr_min]

V = pot_file[:,1]
V = V+4.747300146159997
#V = V[r_init <= r_bohr_max]
#V = V[r_temp >= r_bohr_min]

plot_ref2 = plt.plot(r,V, label="$D_2(b_3 \ \Sigma_u^+$)")


r_bohr_max = 8
plot_ref3 = D2X_1SG_NUMEROV_PARAMS.plot_potential(r_bohr_max,show=False, min_at_zero=True,
label ="$D_2(X)$")
plt.legend(handles=[plot_ref1[0], plot_ref2[0], plot_ref3[0]])


plt.ylim((0, 18.38))
plt.xlim((0, r_bohr_max))
plt.xlabel("Inter-nuclear distance (Bohr)")
plt.ylabel("Energy (eV)")
plt.savefig(FIGURES_DIR+"d2p_d2b_d2x.pgf")
clean_plot()


#Plot KER exp et theo pour interaction avec gaz residuel D2^+ + H2 -> D+D+H2^+

ker_exp_f_path = "data/d2_ker_distribution.txt"
ker_exp = np.loadtxt(ker_exp_f_path, skiprows=1)
#Drop values below 0.2eV and above 7ev
ker_exp = ker_exp[ker_exp[:,0] >= 0.2]
numerov_res_D2P = numerov(D2P_NUMEROV_PARAMS_DISSOC)
numerov_res_D2B = numerov(D2B_NUMEROV_PARAMS)
numerov_res_D2B.write_eigen_vector("data/free_evec.txt")
numerov_res_i = numerov_res_D2P
numerov_res_f = numerov_res_D2B
FCM = comp_franck_condon_matrix(numerov_res_i, numerov_res_f)
energies = ker_exp[:,0]
proba_exp = ker_exp[:,1]
energies_theo = numerov_res_D2B.eigen_values #np.linspace(0.0, 10, 200)
energies_theo = energies_theo[energies_theo > 0.12]
energies_theo = energies_theo[energies_theo < 10]
alpha = 5.81308826e+14
alpha_no_cosh = 3.97075003e+13
proba_theo = []
proba_theo_no_cosh = []
for e in energies_theo:
    p_e = ker_dissoc(numerov_res_D2P, numerov_res_D2B,
    D2_plus_vib_level_distrib(numerov_res_D2P.eigen_values), e,
    FCM, q_I_cosh=True)
    p_e_no_cosh = ker_dissoc(numerov_res_D2P, numerov_res_D2B,
    D2_plus_vib_level_distrib(numerov_res_D2P.eigen_values), e,
    FCM)
    proba_theo.append(p_e)
    proba_theo_no_cosh.append(p_e_no_cosh)
proba_theo_fit = alpha*np.array(proba_theo)
proba_theo_fit_no_cosh = alpha_no_cosh*np.array(proba_theo_no_cosh)

plt.figure(figsize=(12,12))
plot_exp, = plt.plot(energies, proba_exp, linewidth=0.5, color="black", label="Exp")
plot_theo, = plt.plot(energies_theo, proba_theo_fit, color="red", label="$q(\Delta I) = \cosh^{-2}$")
plot_theo_no_cosh, = plt.plot(energies_theo, proba_theo_fit_no_cosh, color="green", label="$q(\Delta I)$ linear")
plt.xlabel("Kinetic Energy Released (eV)")
plt.ylabel("Relative Intensity")
plt.legend(handles=[plot_exp, plot_theo, plot_theo_no_cosh])

#frame = plt.gca()
#frame.axes.get_yaxis().set_visible(False)
plt.savefig(FIGURES_DIR+"ker_d2p_to_d2b_exp_vs_theo.eps")
#plt.show()

clean_plot()
plt.rcParams.update({"font.size": 30})
#Plot q(delta I)
plt.figure(figsize=(12,12))
delta_I_ev = np.linspace(-12,3,1000)
q_I_l = []
for d in delta_I_ev:
    q_I_l.append(q_I(d,0, q_I_cosh=False))
plt.plot(delta_I_ev, q_I_l)
plt.xlabel("$\Delta I (eV)$")
plt.ylabel("$q(\Delta I)$")
plt.show()
plt.savefig(FIGURES_DIR+"q_delta_I_de_bruijn.eps")











