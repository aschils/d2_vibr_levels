from numerov_and_ker_api import *
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(0, "../detector_acceptance_3_bodies")
from three_bodies_acceptance import *
from ker_and_dalitz_from_exp_data import *



def clean_plot():
    plt.clf()
    plt.cla()
    plt.close()


FIGURES_DIR = "../figures/"

plot_numerov_stuffs=False
if plot_numerov_stuffs:
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

    (D2STAR_1_SU_BPP_NUMEROV_PARAMS, 36 , "1_SU_BPP",figsize_p),
    (D2STAR_2_1SG_NUMEROV_PARAMS, 17.5 , "2_1SG",figsize_p),
    #(D2STAR_3_1SG_NUMEROV_PARAMS, 8 , "3_1SG",figsize_p),
    (D2STAR_4_1SG_NUMEROV_PARAMS, 36 , "4_1SG",figsize_p),

    (D2STAR_2_3SU_NUMEROV_PARAMS, 5.5, "2_3SU",(14,8)), #figsize too small

    (D2STAR_3_3SU_NUMEROV_PARAMS, 5 , "3_3SU",figsize_p),
    (D2STAR_4_3SU_NUMEROV_PARAMS, 10 , "4_3SU",figsize_p),

    (D2STAR_1PI_U_C_NUMEROV_PARAMS,9 , "1PI_U_C",figsize_p),
    (D2STAR_1PI_U_D_NUMEROV_PARAMS,17.5 , "1PI_U_D",figsize_p),

    (D2STAR_1_3PI_G_NUMEROV_PARAMS,5 , "1_3PI_G", (14,8)), #figsize too small
    (D2STAR_1_3PI_U_NUMEROV_PARAMS,24 , "1_3PI_U",figsize_p),
    (D2STAR_2_3PI_G_NUMEROV_PARAMS,20 , "2_3PI_G",figsize_p),
    (D2STAR_2_3PI_U_NUMEROV_PARAMS,18 , "2_3PI_U",figsize_p),
    (D2STAR_3_3PI_G_NUMEROV_PARAMS,6 , "3_3PI_G",figsize_p),
    (D2STAR_3_3PI_U_NUMEROV_PARAMS,15 , "3_3PI_U",figsize_p),
    (D2STAR_1_PI_GI_NUMEROV_PARAMS, 4.5 , "1_PI_GI",figsize_p),
    (D2STAR_1_PI_GR_NUMEROV_PARAMS, 11 , "1_PI_GR",figsize_p),
    (D2X_1SG_NUMEROV_PARAMS, 15, "D2X", figsize_p)
    ]

    #D2STAR_2_3SG_NUMEROV_PARAMS.plot_potential(15)

    #idx_to_consider = [0,1,2,3,4,5,10]
    idx_to_consider = range(0,23)
    plot_ratios = np.ones(23)
    start_plot_ratios_ev = np.ones(23)
    plot_ratios[6] = 5
    start_plot_ratios_ev[6] = -15.5
    plot_ratios[8] = 5
    start_plot_ratios_ev[8] = -16

    #file_nbr = 0
    nbr_ev_per_line = 7
    for file_nbr in idx_to_consider:
        state = states[file_nbr]
        #if state[2] == "2_3SU" or state[2] == "1_3PI_G":
        numerov_result = numerov(state[0])
        numerov_result.plot(r_bohr_max=state[1], plot_ratio=plot_ratios[file_nbr],save=True, out_path=FIGURES_DIR+"vib_states/file"+str(file_nbr)+".pgf",
    	show=False, figsize=state[3], start_applying_plot_ratio_from_ev=start_plot_ratios_ev[file_nbr])#, title=state[2], useTex=True)
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
plot_D2_H2_stuffs = False
if plot_D2_H2_stuffs:
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


    plot_ref1 = plot_pot_and_ev(r_bohr, V, ev, d2p_numerov_result.min_of_V, label="D$_2^+$")


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

    plot_ref2 = plt.plot(r,V, label="D$_2($b$_3 \ \Sigma_u^+$)")


    r_bohr_max = 8
    plot_ref3 = D2X_1SG_NUMEROV_PARAMS.plot_potential(r_bohr_max,show=False, min_at_zero=True,
    label ="D$_2($X$)$")
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
    #alpha_no_cosh = 3.97075003e+13
    alpha_no_cosh = 3.97075003e+13/5645.0*5400
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

    plt.rcParams.update({"font.size": 50})
    #Plot q(delta I)
    plt.figure(figsize=(18,16))
    delta_I_ev = np.linspace(-12,3,1000)
    q_I_l = []
    for d in delta_I_ev:
        q_I_l.append(q_I(d,0, q_I_cosh=False))
    plt.plot(delta_I_ev, q_I_l, color="black", linewidth=2)
    plt.xlabel("$\Delta I (eV)$")
    plt.ylabel("$q(\Delta I)$")
    plt.savefig(FIGURES_DIR+"q_delta_I_de_bruijn.eps")
    clean_plot()


    #D2STAR_2_1SG_NUMEROV_PARAMS.plot_potential(30, show=True, min_at_zero=False, label="")


    #pot_file_path = POTENTIALS_DIR+"/"+D2STAR_2_1SG_POTENTIAL_FILE
    #pot_file = np.loadtxt(pot_file_path)
    #ri = pot_file[:,0]
    #r = ri[ri < 30]
    #V = pot_file[:,1]*au_to_ev
    #V = V[ri < 30]

    #plt.plot(r,V, ".", markersize=1)
    #plt.show()


    # Plot to explain ker around 8eV due to 2p3 Pi_u(c) predissociation
    plt.rcParams.update({"font.size": 24})
    plt.figure(figsize=(12,12))
    plt.rc("text", usetex=True)

    #plot_ref1 = D2P_NUMEROV_PARAMS.plot_potential(r_bohr_max,show=False, min_at_zero=True,
    #shift=ionisation_potential_d2, label ="D$_2^+$")

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


    plot_ref1 = plot_pot_and_ev(r_bohr, V, ev, d2p_numerov_result.min_of_V, label="D$_2^+$")



    d2b_pot_data = np.loadtxt(POTENTIALS_DIR+"/"+D2DISSOC_POTENTIAL_FILE)
    r = pot_file[:,0]
    #r_temp = r_init[r_init <= r_bohr_max]
    #r = r_temp[r_temp >= r_bohr_min]

    V = pot_file[:,1]
    V = V+4.747300146159997
    #V = V[r_init <= r_bohr_max]
    #V = V[r_temp >= r_bohr_min]

    plot_ref2 = plt.plot(r,V, label="D$_2($b$_3 \ \Sigma_u^+$)", color="black")


    r_bohr_max = 8
    plot_ref3 = D2X_1SG_NUMEROV_PARAMS.plot_potential(r_bohr_max,show=False, min_at_zero=True,
    label ="D$_2($X$)$", color="black")


    plt.ylim((0, 18.38))
    plt.xlim((0, r_bohr_max))
    plt.xlabel("Inter-nuclear distance (Bohr)")
    plt.ylabel("Energy (eV)")


    d2s_numerov_result = numerov(D2STAR_1_3PI_U_NUMEROV_PARAMS)
    d2s_pot_data = np.loadtxt(POTENTIALS_DIR+"/"+D2STAR_1_3PI_U_POTENTIAL_FILE)
    V = au_to_ev*d2s_pot_data[:,1]
    V_min = np.min(V)
    V = lambda r: d2s_numerov_result.V(r) - V_min + 11.9
    ev = d2s_numerov_result.eigen_values - V_min + 11.9
    #r_bohr = d2p_numerov_result.r_bohr[d2p_numerov_result.r_bohr <= r_bohr_max]
    r_bohr = np.linspace(0.5, 8.5, 1000) #d2s_numerov_result.r_bohr


    plot_ref4 = plot_pot_and_ev(r_bohr, V, ev, d2s_numerov_result.min_of_V, label="D$_2^*$($2$p${}^3 \Pi_u($c$)$)")
    #plt.legend(handles=[plot_ref1[0], plot_ref2[0], plot_ref3[0], plot_ref4[0]])
    plt.savefig(FIGURES_DIR+"d2p_d2b_d2x_d2star.pgf")
    #plt.show()
    clean_plot()


plot_3body_stuff = True
if plot_3body_stuff:
    #Draw 3D view of probability distribution for v_3

    plt.rcParams.update({"font.size": 24})
    plt.figure(figsize=(12,12))
    plt.rc("text", usetex=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def triangular_distrib(x):
        if x <= 0:
            return 0.5*x+1
        else:
            return -0.5*x+1


    u = np.linspace(-2,2,100)
    v = np.linspace(-2,2,100)
    x, y = np.meshgrid(u,v)
    zs = np.array([triangular_distrib(x)*triangular_distrib(y) for x,y in zip(np.ravel(x), np.ravel(y))])
    z = zs.reshape(x.shape)
    ax.plot_surface(x,y,z, alpha=0.5)

    radius = math.sqrt(1/3.0)
    # Make data
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi/2, 100)
    # x = radius * np.outer(np.cos(u), np.sin(v))
    # y = radius * np.outer(np.sin(u), np.sin(v))
    # z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    u = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, 1, 100)
    x = radius * np.outer(np.cos(u), np.ones(u.size))
    y = radius * np.outer(np.sin(u), np.ones(u.size))
    z = np.outer(np.ones(np.size(u)), z)


    ax.set_xlabel("$v_{3_x}$", labelpad=15)
    ax.set_ylabel("$v_{3_y}$", labelpad=15)
    ax.set_zlabel("$P(v_{3_x}, v_{3_y})$", labelpad=5)
    ax.set_zticks([])

    # Plot the surface
    ax.plot_surface(x, y, z, color='r', alpha=0.5)
    plt.savefig(FIGURES_DIR+"triangular_distrib_cut.pgf")

    clean_plot()

    #Plot acceptance
    #Parse acceptance from file
    acceptance_f_path = "../detector_acceptance_3_bodies/acceptance.txt"
    acceptance_f = open(acceptance_f_path, "r")
    acceptance_f_str = acceptance_f.readlines()
    acceptance_str = acceptance_f_str[-1]
    acceptance_kers =  np.linspace(0.01, 6, 500)
    acceptance = np.zeros(acceptance_kers.size)
    number_str = ""
    i = 0
    for char in acceptance_str:
        if char == ",":
            acceptance[i] = float(number_str)
            number_str = ""
            i = i+1
        elif char != " " and char != "[":
            number_str = number_str+char

    def chi_squared(x_v, shift_x, scale_x, scale_y):
        k = 3
        res = np.zeros(x_v.size)

        for i in range(0, x_v.size):
            x = scale_x*x_v[i]+shift_x
            if x <= 0:
                res[i] = 0
            else:
                half_k = k/2.0
                num = x**(half_k-1)*math.exp(-x/2.0)
                den = 2**half_k*sp.special.gamma(half_k)
                res[i] = scale_y*num/den
        return res

    (params, err) = sp.optimize.curve_fit(chi_squared, acceptance_kers, acceptance)
    shift_x, scale_x, scale_y = params
    acceptance_f = lambda x_v: chi_squared(x_v, shift_x, scale_x, scale_y)

    plt.rc("text", usetex=True)
    p1, = plt.plot(acceptance_kers, acceptance, label="Simulation")
    p2, = plt.plot(acceptance_kers,acceptance_f(acceptance_kers), label="$\chi^2_{k=3}$ fit")
    plt.legend(handles=[p1, p2])
    plt.xlabel("Kinetic energy released (eV)")
    plt.ylabel("Acceptance")
    plt.savefig(FIGURES_DIR+"acceptance.pgf")
    clean_plot()

    #Plot KER dissoc 3 corps
    exp_res_path = "../detector_acceptance_3_bodies/20180622_3body.txt"
    exp_res = np.loadtxt(exp_res_path)
    X1 = exp_res[:,0]*10**-1
    Y1 = exp_res[:,1]*10**-1
    X2 = exp_res[:,2]*10**-1
    Y2 = exp_res[:,3]*10**-1
    dT12 = exp_res[:,4]*10**-9
    dT13 = -exp_res[:,5]*10**-9
    #XU: jeter les évènements pour lesquels x2 est inférieur à 12
    events_to_keep = X2 >= 12*10**-1
    X1 = X1[events_to_keep]
    Y1 = Y1[events_to_keep]
    X2 = X2[events_to_keep]
    Y2 = Y2[events_to_keep]
    dT12 = dT12[events_to_keep]
    dT13 = dT13[events_to_keep]
    BEAM_ENERGY = 18000 #eV
    print("Total number of events: "+str(X1.size))
    #Velocity of center of mass in laboratory frame
    V_cm = np.sqrt(BEAM_ENERGY/3)*speed_to_SI_cm*np.array([0,0,1])

    X1_i, Y1_i, X2_i, Y2_i, dT12_i, dT13_i = (X1, Y1, X2, Y2, dT12, dT13)
    signal_plus_noise, noises = noise_dT12_dT13(X1, Y1, X2, Y2, dT12, dT13)
    X1, Y1, X2, Y2, dT12, dT13 = signal_plus_noise
    print("Number of events removing pure noise: "+str(X1.size))
    val_events = keep_valid_events(X1, Y1, X2, Y2, dT12, dT13,V_cm)
    (X1v,Y1v,X2v,Y2v,X3v,Y3v,dT12v,dT13v,t1v) = val_events
    print("Number of events after keep valid events: "+str(X1v.size))
    (V1, V2, V3) = compute_v_lab(t1v, X1v, Y1v, X2v, Y2v, dT12v, X3v, Y3v, dT13v)

    (total_ker, kerp1, kerp2, kerp3) = compute_ker(V1, V2, V3,V_cm)
    plt.rc("text", usetex=True)
    kin_energies_list = []
    for i in range(0, len(kerp1)):
        kin_energies_list.append((kerp1[i], kerp2[i], kerp3[i]))
    plt.figure(figsize=(12,12))
    dalitz_plot(kin_energies_list, show=False)

    plt.xlabel("$\\frac{E_2-E_1}{\\sqrt{3}}$")
    plt.ylabel("$E_3-\\frac{1}{3}$")
    plt.savefig(FIGURES_DIR+"dalitz_plot_valid_events.jpg")
    clean_plot()

    nbr_bars = 200

    #Ker of p1 p2 and P3 seperately
    spb_p1, bins = np.histogram(kerp1, np.linspace(0, 6, nbr_bars))
    centers = (bins[:-1] + bins[1:])/2
    plt.figure(figsize=(12,12))
    p1, = plt.plot(centers, spb_p1, label="TODO")
    plt.savefig(FIGURES_DIR+"ker_part1.pgf")
    clean_plot()

    spb_p2, bins = np.histogram(kerp2, np.linspace(0, 6, nbr_bars))
    centers = (bins[:-1] + bins[1:])/2
    plt.figure(figsize=(12,12))
    p2, = plt.plot(centers, spb_p2, label="TODO")
    plt.savefig(FIGURES_DIR+"ker_part2.pgf")
    clean_plot()

    spb_p3, bins = np.histogram(kerp3, np.linspace(0, 6, nbr_bars))
    centers = (bins[:-1] + bins[1:])/2
    plt.figure(figsize=(12,12))
    p3, = plt.plot(centers, spb_p3, label="TODO")
    plt.savefig(FIGURES_DIR+"ker_part3.pgf")
    clean_plot()


    spb, bins = np.histogram(total_ker, np.linspace(0, 6, nbr_bars))
    centers = (bins[:-1] + bins[1:])/2
    spb_accept = acceptance_correction(centers, spb, acceptance_kers, acceptance,
    acceptance_f)

    plt.figure(figsize=(12,12))
    p1, = plt.plot(centers, spb, label="S+B without acceptance")
    p2, = plt.plot(centers, spb_accept, label="S+B with acceptance")
    plt.xlabel("Kinetic energy released (eV)")
    plt.ylabel("Relative intensity")
    plt.legend(handles=[p1, p2])
    plt.savefig(FIGURES_DIR+"SB_with_wo_acceptance.pgf")
    clean_plot()

    b_tot = np.zeros(nbr_bars-1)
    for noise in noises:
        n_X1, n_Y1, n_X2, n_Y2, n_dT12, n_dT13 = noise
        (X1vs1,Y1vs1,X2vs1,Y2vs1,X3vs1,Y3vs1,dT12vs1,dT13vs1,t1vs1) = \
        keep_valid_events(n_X1, n_Y1, n_X2, n_Y2, n_dT12, n_dT13,V_cm)
        (V1, V2, V3) = compute_v_lab(t1vs1, X1vs1, Y1vs1, X2vs1, Y2vs1, dT12vs1, X3vs1, Y3vs1, dT13vs1)
        (ker_list_noise, n_kerp1, n_kerp2, n_kerp3) = compute_ker(V1, V2, V3,V_cm)
        #ker_hist_noise = np.histogram(ker_list_noise, np.linspace(0, 60, nbr_bars))
        # # ker_list_noise = ker_of_gen_noise(X1, X2, Y1, Y2, dT12, dT13)
        b, bins = np.histogram(ker_list_noise, np.linspace(0, 6, nbr_bars))
        b_tot = b_tot+b
    b_tot = b_tot/len(noises)
    centers = (bins[:-1] + bins[1:])/2

    b_accept = acceptance_correction(centers, b, acceptance_kers, acceptance,
    acceptance_f)
    plt.figure(figsize=(12,12))
    p1, = plt.plot(centers, b, label="B without acceptance")
    p2, = plt.plot(centers, b_accept, label="B with acceptance")
    plt.xlabel("Kinetic energy released (eV)")
    plt.ylabel("Relative intensity")
    plt.legend(handles=[p1, p2])
    plt.savefig(FIGURES_DIR+"B_with_wo_acceptance.pgf")
    clean_plot()

    s = spb-b

    centers = (bins[:-1] + bins[1:]) / 2
    width = 0.7 * (bins[1] - bins[0])
    s_accept = acceptance_correction(centers, s, acceptance_kers, acceptance,
    acceptance_f)
    plt.figure(figsize=(12,12))
    p1, = plt.plot(centers, b_accept, label="B with acceptance")
    p2, = plt.plot(centers, spb_accept, label="S+B with acceptance")
    p3, = plt.plot(centers, s_accept, color="k", linewidth=2,label="S+B-B with acceptance")
    plt.xlabel("Kinetic energy released (eV)")
    plt.ylabel("Relative intensity")
    plt.legend(handles=[p1, p2, p3])
    plt.savefig(FIGURES_DIR+"S_SPB_with_wo_acceptance.pgf")
    clean_plot()

    #Plot des events -> D+D+D dans avec en x dT12 et en y dT13
    point_size = 0.25
    sg_x_left = -0.25*10**-7
    sg_x_right = 2.75*10**-7
    sg_y_bot = -4.2*10**-7-0.75*10**-7
    sg_y_up = -2.4*10**-7+0.75*10**-7
    xlim = (sg_x_left*10**9, sg_x_right*10**9)
    ylim = ((sg_y_bot-10**-7)*10**9, (sg_y_up+10**-7)*10**9)
    plt.figure(figsize=(12,12))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.scatter(dT12_i*10**9, dT13_i*10**9,  color="black", s=point_size)
    plt.plot([80, 212], [-438, -340], color="red")
    plt.plot([80, 44], [-438, -301], color="red")
    plt.plot([44, 178], [-301, -199], color="red")
    plt.plot([178, 212], [-199, -340], color="red")

    plt.plot([73, 200], [-574, -574], color="green")
    plt.plot([73, 200], [-458, -458], color="green")
    plt.plot([73, 73], [-574, -458], color="green")
    plt.plot([200, 200], [-574, -458], color="green")

    #plt.scatter(sig_bloc_dT12, sig_bloc_dT13, s=point_size, color="g")
    #plt.scatter(noise_bu_dT12, noise_bu_dT13, s=point_size, color="r")
    plt.xlabel("$dT_{12}$(ns)")
    plt.ylabel("$dT_{13}$(ns)")

    plt.savefig(FIGURES_DIR+"events_in_dT12_dT13_space.pgf")
    clean_plot()

#Plot speed distribution of events, Dalitz


    gen_events = False
    if gen_events:
        kers = np.array([3])
        nbr_events_per_ker = 100000
        proton_mass = 1.0
        deuterium_mass = 2.0*proton_mass
        speed_to_SI_cm = 978897.1372228841
        gen_valid_events_params = (kers, deuterium_mass, nbr_events_per_ker, V_cm, speed_to_SI_cm)
        (kin_energies_list, velocities_cm_list, velocities_lab_list) = \
        gen_valid_events(gen_valid_events_params)
        output = open("pickle_kin_energies_list", "wb")
        pickle.dump(kin_energies_list, output, pickle.HIGHEST_PROTOCOL)
        output.close()
    else:
        input = open("pickle_kin_energies_list", 'rb')
        kin_energies_list = pickle.load(input)
        input.close()

    dalitz_plot(kin_energies_list, show=True)

    dalitz_x = np.zeros(len(kin_energies_list))
    dalitz_y = np.zeros(len(kin_energies_list))

    idx = 0
    for kin_energies in kin_energies_list:
        x = (kin_energies[1]-kin_energies[0])/math.sqrt(3.0)
        y = kin_energies[2]-1.0/3.0
        dalitz_x[idx] = x
        dalitz_y[idx] = y
        idx = idx+1
    plt.rcParams.update({"font.size": 30})
    plt.figure(figsize=(14,14))
    plt.rc("text", usetex=True)
    plt.hist2d(x=dalitz_x,y=dalitz_y,bins=(50,50), cmap=plt.cm.Greys)
    plt.colorbar()
    plt.xlabel("$\\frac{E_2-E_1}{\\sqrt{3}}$")
    plt.ylabel("$E_3-\\frac{1}{3}$")
    plt.savefig(FIGURES_DIR+"dalitz_plot_generated_events.jpg")
    plt.show()
