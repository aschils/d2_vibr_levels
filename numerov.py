#!/usr/bin/env python3

###############################
# Author: Arnaud Schils (NAPS)#
# arnaud.schils@gmail.com     #
###############################

#The Numerov numerical method allows to solve second order differential
#equations of the form: Psi''(r) = f(r) Psi(r)
#This script uses this method to solve the Schrodinger equation for the
#molecules D_2. The potential is radial and the schrodinger equation
#is 1D in this case. The goal is to find the vibrational energies of these
#molecules.

import math
import numpy as np
import scipy as sp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import concurrent.futures

#Definition of constants
h = 6.626*10**-34 #Planck Constant [J s]
hbar = h/(2*math.pi)
m_p = 1.672*10**-27 #Proton mass [kg]
reduced_mass_d2 = 4*m_p #approximation of D_2 reduced mass [kg]
reduced_mass_he2 = 2*m_p
electron_charge = 1.60217662*10**-19 #coulomb
bohr_to_meter = 5.291772*10**-11

ionisation_potential_d2 = 15.466 #eV

# <psi1 | psi2> = int_{-\infty}^{\infty} psi1(r)^* psi2(r) dr
def wave_fun_scalar_prod(psi1, psi2, dr):
    return np.dot(psi1,psi2)*dr-dr/2.0*psi1[0]*psi2[0]-dr/2.0*psi1[-1]*psi2[-1]

def normalization_of_continuum_wave_fun(psi, energy, r):
    mu = m_p #masse raduite = masse deuterium/2 = mp
    k = math.sqrt(2*mu*energy)/hbar
    s = math.sin(k*r[-1])
    c = math.cos(k*r[-1])
    sd = k*c
    cd = -k*s

    dr = r[1]-r[0]
    F = psi[-1]
    Fd = (3*psi[-1]-4*psi[-2]+psi[-3])/(2*dr)

    A = (F*sd-Fd*s)/(c*sd-cd*s)
    B = (F*cd-Fd*c)/(s*cd-sd*c)
    print("ormalization_of_continuum_wave_fun "+str(math.sqrt((A**2+B**2)*math.pi*k)))
    return math.sqrt((A**2+B**2)*math.pi*k)
    #return np.sqrt(k)

def normalize_continuum_wave_fun(psi, energy, r):
    psi_normalized = psi/normalization_of_continuum_wave_fun(psi, energy, r)
    return psi_normalized

def wave_fun_norm(psi, dr):
    return math.sqrt(wave_fun_scalar_prod(psi, psi, dr))

def wave_fun_scalar_prod_from_fun(psi1, psi2, r_0, r_e):
    f = lambda r: psi1(r)*psi2(r)
    i, abserr = sp.integrate.quad(f, r_0, r_e,limit=200)
    return i

#
# Shift potential having at least one negative value to positive values.
# Substracting the minimum of the potential to all values in V.
# pre: V is a numpy array
def shift_neg_potential(V):

    if not (V >= 0).all():
        min_V = np.amin(V)
        return V-min_V
    else:
        return V

#
# @Preconditions:
# First column of pot file is inter-atomic distance in Bohr
# Second column is the potential in eV
#
# is_bound_potential must be True if the molecular potential has a minimum,
# False otherwise
#
# E_max: search energy levels from potential minimum up to this energy
#
# reduced_mass: of molecule in kg
#
# r_end_for_unbound_potential: set end of the numerical domain for unbound
# potential (for bound potential is it the second intersection between the
# E_max line and the potential V(r))
#
# refine: initial step dr will be decreased as dr/refine
#
# @Return: (r_bohr, V, eigen_values, eigen_vectors)
# r_bohr : points of the numerical domain in Bohr (1D numpy array)
# V, a function of r, is the potential in Joule as a function of the interatomic
# distance r IN METERS
# eigen_values: a 1D numpy array, energy levels (eigenvalues of Hamiltonian)
# in eV, sorted by increasing energy
# eigen_vectors: 2D numpry array, eigen_vectors[i] is the Psi_i
#   eigen_vectors[i][j] is Psi_i(r_j), r_j are the values in t_bohr
#
def numerov(pot_file_path, is_bound_potential, E_max, reduced_mass,
r_end_for_unbound_potential=10, refine=1, au_to_ev=False):

    pot_file = np.loadtxt(pot_file_path)

    r = pot_file[:,0]*bohr_to_meter
    V_shifted = shift_neg_potential(pot_file[:,1])
    if au_to_ev:
        V_shifted = V_shifted*27
    V_discr = V_shifted*electron_charge #Joule
    V = interp1d(r,V_discr,kind="linear",fill_value="extrapolate")
    #plt.plot(r,pot_file[:,1])
    #plt.show()
    r_inf = 10**6 #purely arbitrary

    if is_bound_potential:
        i_min = np.argmin(V_discr)
        E_min = V_discr[i_min]
        r_min_of_V = r[i_min]
    else:
        E_min = V(r_inf)
        r_min_of_V = r_inf

    print("E_min "+str(E_min/electron_charge))

    E_max = E_max*electron_charge #Joule

    print("Will search vibrational levels between E_min "+str(E_min/electron_charge)+" and E_max "+str(E_max/electron_charge))

    f = lambda x: V(x)-E_max
    V_E_intersect_left = sp.optimize.brentq(f, 0, r_min_of_V)

    if is_bound_potential:
        V_E_intersect_right = sp.optimize.brentq(f, r_min_of_V,10*r_min_of_V)
    else:
        V_E_intersect_right = r_end_for_unbound_potential*bohr_to_meter

    #De Broglie wavelength (indicator of appropriate dr)
    wavelength = h/math.sqrt(2*reduced_mass*E_max)

    dr = wavelength/(2.0*math.pi*refine)
    print("Step dr is "+str(dr/bohr_to_meter)+" bohr")

    n = round((V_E_intersect_right-V_E_intersect_left)/dr+5)

    print("Number of points in the grid: "+str(n))

    i = np.arange(1,n+1)
    r = (V_E_intersect_left-2*dr)+dr*i

    ones = np.ones(n)
    ones_short = np.ones(n-1)
    B = (sp.sparse.diags(ones_short, offsets=-1)
    +10*sp.sparse.diags(ones, offsets=0)
    +sp.sparse.diags(ones_short, offsets=1))/12.0
    A = (sp.sparse.diags(ones_short, offsets=-1)
    -2*sp.sparse.diags(ones, offsets=0)
    +sp.sparse.diags(ones_short, offsets=1))/dr**2

    A = sp.sparse.csc_matrix(A)
    B = sp.sparse.csc_matrix(B)

    print("Building Hamiltonian matrix.")

    KE = -sp.sparse.linalg.inv(B).dot(A)*hbar**2/(2.0*reduced_mass)

    H = KE + sp.sparse.diags(V(r), offsets=0)

    print("Computing eigenvalues and eigenvectors of Hamiltonian.")
    eigen_values, eigen_vectors = sp.sparse.linalg.eigs(H, k=n-2, which="SM")
    print("Eigenvalues and eigenvectors of Hamiltonian computed.")

    eigen_values_sorted_idx = np.argsort(eigen_values)
    eigen_values = np.asarray(list(eigen_values[i] for i in eigen_values_sorted_idx))
    eigen_values = np.real(eigen_values)

    eigen_vectors = eigen_vectors.T
    eigen_vectors_temp = np.asarray(list(eigen_vectors[i] for i in eigen_values_sorted_idx))

    higher_idx_to_keep = np.abs(eigen_values-E_max).argmin()
    eigen_values = np.delete(eigen_values, np.arange(higher_idx_to_keep, eigen_values.size))
    eigen_vectors_temp = np.delete(eigen_vectors_temp,
    np.arange(higher_idx_to_keep, eigen_vectors_temp.size),axis=0)

    eigen_vectors = []

    c = 0
    for ev in eigen_vectors_temp:
        ev = np.real(ev)
        if is_bound_potential:
            eigen_vectors.append(ev/wave_fun_norm(ev, dr))
        else:
            eigen_vectors.append(normalize_continuum_wave_fun(ev, eigen_values[c], r))
        c = c+1

    eigen_vectors = np.asarray(eigen_vectors)
    eigen_values = eigen_values/electron_charge

    print("Eigenvalues:")
    print(eigen_values)

    r_bohr = r/bohr_to_meter
    return (r_bohr, V, eigen_values, eigen_vectors)

#
# @pre:
# -  ev_array is a 2D numpy array or a list of 1D numpy array
# - r is a 1D numpy array
# @return: each element i of returned list is a polynomial interpolation by parts
# of the points (r, ev_array[i])
#
def interpolate_eigen_vec_array(ev_array, r):
    ev_fun = []
    for i in range(0,r.size):
        fun = interp1d(r,ev_array[i],kind=0,fill_value="extrapolate")
        ev_fun.append(fun)
    return ev_fun

#
# eigen_values_free (eigen_vectors_free): eigen values (vectors) obtained
# solving Schrodinger with dissociative potential
# E: a float (energy)
# r_bohr: numerical domaine used in numerov to obtained eval and evec.
#
# @return: (eigen value, eigen vector) pair for which eigen_value nearest to E
#
def final_dissoc_state(eigen_values_free, eigen_vectors_free, E, r_bohr):

    delta_E = eigen_values_free[1]-eigen_values_free[0]

    #if E < eigen_values_free[0]-delta_E:
    #    return (0, np.zeros(eigen_vectors_free[0].size))

    if E < eigen_values_free[0]:
        eigen_vec = eigen_vectors_free[0]
        r_meter = r_bohr*bohr_to_meter
        eigen_vec = eigen_vec*normalization_of_continuum_wave_fun(eigen_vec, eigen_values_free[0], r_meter)
        eigen_vec = eigen_vec/normalization_of_continuum_wave_fun(eigen_vec, E, r_meter)
        #print("in final dissoc state np.linalg.norm(eigen_vec) "+str(np.linalg.norm(eigen_vec)))
        #print(np.linalg.norm(eigen_vec))
        return (E, eigen_vec)

    i = np.abs(eigen_values_free-E).argmin()
    return (eigen_values_free[i], eigen_vectors_free[i])

#
# bound_vib_level_distrib: a function, gives proba for bound molecule to be
#               in vibrational state of energy E
# r_bohr_bound: numerov numerical domain used to computed bounded wave functions
#              solutions of Schrodinger with bound potential (bohr)
# r_bohr_free: numerov numerical domain used to computed free wave functions
#              solutions of Schrodinger with unbound potential (bohr)
# E: energy for which we want p(E), proba to observe kinetic energy E
# bounded_state_IP: ionisation potential of molecule bounded state in eV
#
def ker_dissoc(bound_vib_level_distrib, eigen_values_bound,
eigen_vectors_bound, eigen_values_free, eigen_vectors_free, r_bohr_bound,
r_bohr_free, E, bounded_state_IP, V_free, gamma=1.547):

    p_E = 0
    dr_bound = r_bohr_bound[1]-r_bohr_bound[0]

    #Find dissociated state from the inter-atomic distance in the
    #bound state
    (E_final_state, final_free_state) = final_dissoc_state(eigen_values_free,
    eigen_vectors_free, E, r_bohr_free)
    final_free_statef = interp1d(r_bohr_free, final_free_state,
    kind=0,fill_value=0, bounds_error = False)
    final_free_state = final_free_statef(r_bohr_bound)

    for ev_b_idx in range(0, eigen_values_bound.size):

        e_val_b = eigen_values_bound[ev_b_idx]
        e_vec_b = eigen_vectors_bound[ev_b_idx]

        #Proba to be in vibrational bound state v
        proba_v = bound_vib_level_distrib(e_val_b)

        #Transition probability = scalar product between initial and
        #final states
        #trans_proba_squared = wave_fun_scalar_prod(e_vec_b, final_free_state, dr_bound)**2
        #trans_proba_squared = wave_fun_scalar_prod(e_vec_b*r_bohr_bound, final_free_state, dr_bound)**2

        #divide by cosh: remove from probability the fact that you can form a
        #bounded D2 molecule 1/cosh((E(eV)-E_res)/1.547)^2
        #E_res = e_val_b-bounded_state_IP
        #p_E = p_E+proba_v*trans_proba_squared/math.cosh((E-E_res)/gamma)**2

        f = lambda x: V_free(x)/electron_charge-E
        r_star = sp.optimize.brentq(f, 0, 10)
        r_star_bohr = r_star/bohr_to_meter

        dr = (r_bohr_free[1]-r_bohr_free[0])
        V_free_deriv = np.abs((V_free(r_star+dr)-V_free(r_star))/dr)

        e_vec_b_f = interp1d(r_bohr_bound, e_vec_b, kind=0,fill_value="extrapolate")
        #p_E = p_E + proba_v*r_star_bohr*e_vec_b_f(r_star_bohr)**2/V_free_deriv
        p_E = p_E + proba_v*e_vec_b_f(r_star_bohr)**2/V_free_deriv

        #if E < 0.02:
        #    print("proba_v "+str(proba_v))
        #    print("trans_proba_squared "+str(trans_proba_squared))
        #    print("math.cosh((E-E_res)/gamma) "+str(math.cosh((E-E_res)/gamma)))

    return p_E

#def bound_vib_level_distrib(mol_energy):
#    return 1/eigen_values_bound.size

def under_plot_wave_fun(eigen_vectors, eigen_values, r_bohr, V, plot_ratio=1):

    r = r_bohr*bohr_to_meter
    plt.plot(r_bohr, V(r))

    for i in range(0, len(eigen_vectors)):
        if i%plot_ratio == 0:
            psi = eigen_vectors[i]
            #if i < len(eigen_vectors)-1:
            #    delta_E = eigen_values[i+1]-eigen_values[i]
            #else:
            #    delta_E = V(r_bohr[0])*electron_charge/10
            #plt.plot(r_bohr, psi**2/np.linalg.norm(psi**2)+eigen_values[i])
            delta_E = 1
            plt.plot(r_bohr, psi/np.linalg.norm(psi)*delta_E+eigen_values[i])

def plot_wave_fun(eigen_vectors, eigen_values, r_bohr, V, plot_ratio=1):
    V_ev = lambda r: V(r)/electron_charge
    under_plot_wave_fun(eigen_vectors, eigen_values, r_bohr, V_ev, plot_ratio=1)
    plt.show()

def plot_bound_and_free(eigen_vectors_bound, eigen_values_bound, r_bohr_bound, V_bound,
eigen_vectors_free, eigen_values_free, r_bohr_free, V_free, plot_ratio_bound=1, plot_ratio_free=1,
    V_free_shift=0):
    V_bound_ev = lambda r: V_bound(r)/electron_charge
    under_plot_wave_fun(eigen_vectors_bound, eigen_values_bound, r_bohr_bound, V_bound_ev, plot_ratio_bound)
    V_free_shifted = lambda r: V_free(r)/electron_charge + V_free_shift
    under_plot_wave_fun(eigen_vectors_free, eigen_values_free+V_free_shift, r_bohr_free, V_free_shifted, plot_ratio_free)
    plt.show()

#He2 Raghed ###################################################################
# (r_bohr_free, V_free, eigen_values_free, eigen_vectors_free) = numerov("pot_rag_free.txt", False, 3, mr_he2, refine=3)
# (r_bohr_bound, V_bound, eigen_values_bound, eigen_vectors_bound) = numerov("pot_rag.txt", True, 2.3, mr_he2, refine=7)
#
# plot_bound_and_free(eigen_vectors_bound, eigen_values_bound, r_bohr_bound, V_bound,
# eigen_vectors_free, eigen_values_free, r_bohr_free, V_free,3,10,0.0899*27.2)

#eigen_values_free_cm = eigen_values_free*8065.54429

# article_cm = np.array([839.53,
# 2465.81,
# 4021.51,
# 5506.77,
# 6920.80,
# 8264.30,
# 9538.59,
# 10744.29,
# 11881.04,
# 12947.69,
# 13942.75,
# 14864.74,
# 15712.07,
# 16483.21,
# 17176.57,
# 17790.42,
# 18322.96
# #18772.77
# #19138.51,
# #19422.68,
# #19626.76,
# #19729.67
# ])
#
# diff_with_art = article_cm-eigen_values_free_cm
# print(diff_with_art/article_cm)
#
# print(article_cm)
# print(eigen_values_free_cm)

#END He2 Raghed ###################################################################


#pre: eigen values sorted by increasing energies
#     eigen_values_bound size is <= 27
def D2_plus_vib_level_distrib(eigen_values_bound):

    #Provided by X. Urbain
    proba_of_levels = np.array([0.080937,
    0.0996592,
    0.117558,
    0.117339,
    0.109159,
    0.0917593,
    0.0799601,
    0.0670289,
    0.0498184,
    0.0420871,
    0.0310552,
    0.0230351,
    0.0206377,
    0.0139325,
    0.0117896,
    0.00925865,
    0.00724063,
    0.00603588,
    0.00578535,
    0.00419046,
    0.00285177,
    0.00245263,
    0.00167712,
    0.00132409,
    0.00161653,
    0.000861533,
    0.000948806])

    #From von Busch et al.
    # proba_of_levels = np.array([0.0448,
    # 0.1038,
    # 0.1407,
    # 0.1476,
    # 0.1337,
    # 0.1106,
    # 0.085,
    # 0.063,
    # 0.042,
    # 0.034,
    # 0.024,
    # 0.017,
    # 0.0122,
    # 0.0088,
    # 0.0064,
    # 0.0048,
    # 0.0036,
    # 0.0025,
    # 0.00185,
    # 0.0038,
    # 0.00102,
    # 0.00075,
    # 0.00054,
    # 0.00037,
    # 0.00023,
    # 0.00011,
    # 0.00002])

    proba_of_E = lambda E: proba_of_levels[np.abs(eigen_values_bound-E).argmin()]
    return proba_of_E


################### Compute KER D2_+, D2_b #############################

#
# #E_min 10.8198528842
#
# # Set E=11 to have 1 computed bound state
# # Set E=12.9 to have 27 computed bound states
#
E_max_bound_states = 12.9
# E_ground_state = 11
# E_two_states = 11.1
# E_three_states = 11.2
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

################### END Compute KER D2_+, D2_b #############################


########### In this part will fit exp ker with theo ker #######################

# ker_exp_f_path = "d2_ker_distribution.txt"
# ker_exp = np.loadtxt(ker_exp_f_path, skiprows=1)
# #Drop values below 0.2eV and above 7ev
# ker_exp = ker_exp[ker_exp[:,0] >= 0.5]
# ker_exp = ker_exp[ker_exp[:,0] <= 7]
#
#Avoid to compute numerov at each run
numerov_save_path = "numerov_save.pkl"
try:
    input = open(numerov_save_path, 'rb')
    numerov_res_bound = pickle.load(input)
    numerov_res_free = pickle.load(input)
except:
    numerov_res_bound = numerov("pot_d2+.txt", True, E_max_bound_states,
    reduced_mass_d2, refine=3)
    numerov_res_free = numerov("pot_d2_b.txt", False, 10, reduced_mass_d2, refine=1)
    try:
        output = open(numerov_save_path, "wb")
        pickle.dump(numerov_res_bound, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(numerov_res_free, output, pickle.HIGHEST_PROTOCOL)
    except:
        print("pickle fail")

#
# ker_exp: 2D numpy array, first column contains energies in eV.
#           second column contains the number of events with corresponding energy
# numerov_res_bound: tuple, output of numerov computation for bound molecule
# numerov_res_free:  tuple, output of numerov computation for unbound state
# alpha:  coefficient multiplying the theoretical KER
# gamma: the gamma factor of 1/cosh(.../gamma), factor taking into account
#        proba to produce the bound molecule (D2 in this case)
#
# return: sum of mean squared error, \sum_i=1^N (nbr_event_exp_i-nbr_event_theo_i)^2
#
def exp_theo_error(ker_exp, numerov_res_bound, numerov_res_free, alpha, gamma):

    (r_bohr_bound, V_bound, eigen_values_bound, eigen_vectors_bound) = numerov_res_bound
    (r_bohr_free, V_free, eigen_values_free, eigen_vectors_free) = numerov_res_free

    energies = ker_exp[:,0]
    proba_exp = ker_exp[:,1]

    error = 0
    for i in range(0,proba_exp.size):
        e = energies[i]
        proba_theo = alpha*ker(D2_plus_vib_level_distrib(eigen_values_bound),
        eigen_values_bound, eigen_vectors_bound, eigen_values_free,
        eigen_vectors_free, r_bohr_bound, r_bohr_free, e, ionisation_potential_d2, V_free, gamma)
        error = error+(proba_exp[i]-proba_theo)**2
    return error

def exp_theo_error_wrap(tuple_params):
    (ker_exp, numerov_res_bound, numerov_res_free, alpha, gamma) = tuple_params
    return exp_theo_error(ker_exp, numerov_res_bound, numerov_res_free, alpha, gamma)

#f = lambda alpha: exp_theo_error(ker_exp, numerov_res_bound, numerov_res_free, alpha, 1.5)

#Search bracket alpha
# alpha = 2300000000
# delta_alpha = 10**8
# error = 10**10
# prev_error = 10**20
# while prev_error > error:
#     prev_error = error
#     error = f(alpha)
#     alpha = alpha+delta_alpha
# alpha_right = alpha
# alpha_left = alpha-2*delta_alpha


# error = 10**10
# prev_error = 10**20
#
# while np.abs(error-prev_error) > 1:
#     prev_error = error
#     print(prev_error)
#     alpha = alpha_left+(alpha_right-alpha_left)/2.0
#     error = f(alpha)
#     print(error)
#     error_delta_right = f(alpha+0.5)
#     error_delta_left = f(alpha-0.5)
#     if error_delta_left < error_delta_right:
#         alpha_right = alpha
#     else:
#         alpha_left = alpha
# print(alpha)

#alpha = 2429711914.0625

# gammas = np.linspace(1, 2, 100)
# errors = []
# for gamma in gammas:
#     errors.append(exp_theo_error(ker_exp, numerov_res_bound, numerov_res_free, alpha, gamma))
# errors = np.array(errors)
# i = errors.argmin()
# print(gammas[i])
#gamma = 1.55555555556

(r_bohr_bound, V_bound, eigen_values_bound, eigen_vectors_bound) = numerov_res_bound
(r_bohr_free, V_free, eigen_values_free, eigen_vectors_free) = numerov_res_free

#plot_wave_fun(eigen_vectors_free, eigen_values_free, r_bohr_free, V_free, plot_ratio=1)


#Try to fit the curve at best, grid search alpha - gamma
#gammas = np.linspace(1.35, 2, 20)
#alphas = np.linspace(1*10**8, 35*10**8, 20)

# gammas = np.linspace(0.5, 4, 100)
# alphas = np.linspace(10**7, 25*10**8, 100)
# errors = np.zeros((gammas.size, alphas.size))
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# items = []
# #kept = []
# for i in range(0,gammas.size):
#     for j in range(0,alphas.size):
#         #if not (gammas[i] > 2.0 and alphas[i] > 0.5):
#         items.append((ker_exp, numerov_res_bound, numerov_res_free, alphas[j], gammas[i]))
#         #kept.append((i,j))
#
# executor = concurrent.futures.ProcessPoolExecutor(4)
# futures = [executor.submit(exp_theo_error_wrap, item) for item in items]
# concurrent.futures.wait(futures)
#
# for future_idx in range(0,len(futures)):
#     i = int(future_idx/alphas.size)
#     j = int(future_idx%alphas.size)
#     errors[i][j] = futures[future_idx].result()
#
# #k = 0
# #for (i,j) in kept:
# #    errors[i][j] = futures[k].result()
# #    k = k+1
#
#
# #for i in range(0,gammas.size):
# #    for j in range(0,alphas.size):
# #        errors[i][j] = exp_theo_error(ker_exp, numerov_res_bound, numerov_res_free, alphas[j], gammas[i])
#
# #errors = np.array(errors)
# flatten_idx = errors.argmin()
#
# i = int(flatten_idx/alphas.size)
# j = int(flatten_idx%alphas.size)
#
# print(i)
# print(j)
#
# alpha = alphas[j]
# gamma = gammas[i]
#
# print("minimum error for alpha "+str(alpha)+" and gamma "+str(gamma))
#
# ax = fig.add_subplot(111, projection='3d')
#
# X, Y = np.meshgrid(alphas, gammas)
# Z = errors
#
# ax.plot_surface(X, Y, Z, color='b')
#
# plt.show()

#minimum error for alpha 433030303.03 and gamma 2.24242424242 starting from 0.2eV
#minimum error for alpha 451951333.367 and gamma 2.21656477901 starting from 0.2ev (scipy)
#minimum error for alpha 60303030.303 and gamma 3.89393939394 starting from 0.8eV
#minimum error for alpha 66708983.1675 and gamma 3.74454792294 from 0.8eV (scipy)
#minimum error for alpha 60303030.303 and gamma 3.82323232323 starting from 1.11eV
#minimum error for alpha 41688581.6395 and gamma 4.4093264698 from 1.11eV (scipy)


######### Optimize with scipy #################


def ker_to_fit(energies, alpha, gamma=1.54):

    p_e = np.zeros(energies.size)
    for i in range(0, p_e.size):
        p_e[i] = alpha*ker(D2_plus_vib_level_distrib(eigen_values_bound), eigen_values_bound,
        eigen_vectors_bound, eigen_values_free, eigen_vectors_free, r_bohr_bound,
        r_bohr_free, energies[i], ionisation_potential_d2, V_free, gamma)

    return p_e
#
# energies = ker_exp[:,0]
# proba_exp = ker_exp[:,1]
#
# #res = sp.optimize.curve_fit(ker_to_fit, energies, proba_exp, p0=(433030303, 2.24))
# res = sp.optimize.curve_fit(ker_to_fit, energies, proba_exp, p0=(35*10**8))#, 3))
# #print(res)
# (alpha, gamma) = (res[0],1.54)
#
# print("minimum error for alpha "+str(alpha)+" and gamma "+str(gamma))
# print(res[1])
#
# energies_theo = np.linspace(0.001, 10, 200)
# #energies_theo = np.linspace(0, 0.5, 2000)
# proba_theo = []
# for e in energies_theo:
#     p_e = ker(D2_plus_vib_level_distrib(eigen_values_bound), eigen_values_bound,
#     eigen_vectors_bound, eigen_values_free, eigen_vectors_free, r_bohr_bound,
#     r_bohr_free, e, ionisation_potential_d2, V_free, gamma)
#     proba_theo.append(p_e)
# proba_theo = alpha*np.array(proba_theo)
#
# plt.plot(energies, proba_exp)
# plt.plot(energies_theo, proba_theo)
# plt.show()


#q(r)


########################## BOUND state D_2^* potential ########################

E_max = 1

eigen_values_bound = eigen_values_bound-27.211

(r_bohr_bound_exc, V_bound_exc, eigen_values_bound_exc, eigen_vectors_bound_exc) = numerov(
"gk1sg.dat", True, E_max, reduced_mass_d2, refine=5, au_to_ev=True)
eigen_values_bound_exc = eigen_values_bound_exc-.663091017*27.211

#plot_wave_fun(eigen_vectors_bound_exc, eigen_values_bound_exc, r_bohr_bound_exc, V_bound_exc)

#Compute numerical domain
r_left = 0
r_right = max(r_bohr_bound[-1], r_bohr_bound_exc[-1])
dr = min(r_bohr_bound[1]-r_bohr_bound[0], r_bohr_bound_exc[1]-r_bohr_bound_exc[0])
r_bohr = np.linspace(r_left, r_right, (r_right-r_left)/dr)
dr = (r_bohr[1]-r_bohr[0])*bohr_to_meter

#Compute Franck Candom D2+, D2*

franck_condon_matrix = np.zeros((len(eigen_vectors_bound), len(eigen_vectors_bound_exc)))
Ei_minus_Ef = np.zeros((len(eigen_vectors_bound), len(eigen_vectors_bound_exc)))

for i in range(0,len(eigen_vectors_bound)):
    for j in range(0,len(eigen_vectors_bound_exc)):

        ev_d2p = interp1d(r_bohr_bound, eigen_vectors_bound[i],
        kind=0, fill_value=0, bounds_error = False)
        ev_d2star = interp1d(r_bohr_bound_exc, eigen_vectors_bound_exc[j],
        kind=0, fill_value=0, bounds_error = False)
        ev_d2p_data = ev_d2p(r_bohr)
        ev_d2star_data = ev_d2star(r_bohr)

        franck_condon_matrix[i,j] = wave_fun_scalar_prod(ev_d2p_data, ev_d2star_data, dr)**2
        Ei_minus_Ef[i,j] = eigen_values_bound[i]-eigen_values_bound_exc[j]-0.754

#print("*********")
#print(np.sum(franck_condon_matrix, axis=0))

# fig, axis = plt.subplots()
# heatmap = axis.pcolor(franck_condon_matrix)
# plt.colorbar(heatmap)
# plt.show()


#Compute ker but for case such as D^- + D_2^+ -> D_2^* + D
#ker is different w.r.t to prev case
def ker2(E, bound_vib_level_distrib):
 #largeur de 50meV pour la gaussienne
    p_E = 0

    for i in range(0,len(eigen_vectors_bound)):
        for j in range(0,len(eigen_vectors_bound_exc)):
            proba_v = bound_vib_level_distrib(i)
            sigma = 0.05/math.sqrt(2*math.log(2))
            #print(sigma)
            p_E = p_E + proba_v*franck_condon_matrix[i, j]*math.exp(-(E-Ei_minus_Ef[i,j])**2/(2*sigma**2))/0.125331

    return p_E


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

####### Now compare ker2 result with experimental ker #####################

ker_exp_f_path = "ker_d2_d.txt"
ker_exp = np.loadtxt(ker_exp_f_path)

energies = ker_exp[:,0]
events_nbr_exp = ker_exp[:,1]
energies_theo = np.zeros(energies.size)
events_nbr_theo = np.zeros(energies.size)
for i in range(0,energies.size):
    events_nbr_theo[i] = ker2(energies[i], D2_plus_vib_level_distrib(eigen_values_bound))*80000

plt.plot(energies, events_nbr_exp)
plt.plot(energies, events_nbr_theo)
plt.show()
