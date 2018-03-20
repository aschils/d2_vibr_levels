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
import matplotlib.pyplot as plt

h = 6.626*10**-34 #Planck Constant [J s]
hbar = h/(2*math.pi)
m_p = 1.672*10**-27 #Proton mass [kg]
reduced_mass_d2 = 4*m_p #approximation of D_2 reduced mass [kg]
reduced_mass_he2 = 2*m_p

electron_charge = 1.60217662*10**-19 #coulomb
bohr_to_meter = 5.291772*10**-11

# <psi1 | psi2> = int_{-\infty}^{\infty} psi1(r)^* psi2(r) dr
def wave_fun_scalar_prod(psi1, psi2, dr):
    return np.dot(psi1,psi2)*dr-dr/2.0*psi1[0]*psi2[0]-dr/2.0*psi1[-1]*psi2[-1]

def normalize_continuum_wave_fun(psi, energy, r):
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

    psi_normalized = psi/math.sqrt((A**2+B**2)*math.pi*k)
    return psi_normalized

def wave_fun_norm(psi, dr):
    return math.sqrt(wave_fun_scalar_prod(psi, psi, dr))

def wave_fun_scalar_prod_from_fun(psi1, psi2, r_0, r_e):
    f = lambda r: psi1(r)*psi2(r)
    i, abserr = sp.integrate.quad(f, r_0, r_e,limit=200)
    return i

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
# V, a function of r, is the potential as a function of the interatomic
# distance r IN METERS
# eigen_values: a 1D numpy array, energy levels (eigenvalues of Hamiltonian)
# in eV, sorted by increasing energy
# eigen_vectors: 2D numpry array, eigen_vectors[i] is the Psi_i
#   eigen_vectors[i][j] is Psi_i(r_j), r_j are the values in t_bohr
#
def numerov(pot_file_path, is_bound_potential, E_max, reduced_mass,
r_end_for_unbound_potential=10, refine=1):

    pot_file = np.loadtxt(pot_file_path)

    r = pot_file[:,0]*bohr_to_meter
    V_discr = pot_file[:,1]*electron_charge #Joule
    V = interp1d(r,V_discr,kind="linear",fill_value="extrapolate")
    #plt.plot(r,V(r))
    #plt.show()
    r_inf = 10**6 #purely arbitrary

    if is_bound_potential:
        #min_V_result = sp.optimize.minimize_scalar(V)
        #if not min_V_result.success:
        #    raise Exception("Minimum of potential V not found.")
        #r_min_of_V = min_V_result.x
        #E_min = V(r_min_of_V)
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

    #De Broglie wavelength
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


    #print(eigen_values/electron_charge)


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
        #ev = ev_fat[0]
        ev = np.real(ev)
        if is_bound_potential:
        #if True:
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

def interpolate_eigen_vec_array(ev_array, r):
    ev_fun = []
    for i in range(0,r.size):
        fun = interp1d(r,ev_array[i],kind=0,fill_value="extrapolate")
        ev_fun.append(fun)
    return ev_fun

def final_dissoc_state(eigen_values_free, eigen_vectors_free, E):

    delta_E = eigen_values_free[1]-eigen_values_free[0]

    if E < eigen_values_free[0]-delta_E:
        return np.zeros(eigen_vectors_free[0].size)
    
    i = np.abs(eigen_values_free-E).argmin()
    return eigen_vectors_free[i]

def ker(bound_vib_level_distrib, eigen_values_bound,
eigen_vectors_bound, eigen_values_free, eigen_vectors_free, r_bohr_bound,
r_bohr_free, E):

    p_E = 0
    dr_bound = r_bohr_bound[1]-r_bohr_bound[0]

    #Find dissociated state from the inter-atomic distance in the
    #bound state
    final_free_state = final_dissoc_state(eigen_values_free,
    eigen_vectors_free, E)
    final_free_statef = interp1d(r_bohr_free, final_free_state,
    kind=0,fill_value="extrapolate")
    final_free_state = final_free_statef(r_bohr_bound)

    for ev_b_idx in range(0, eigen_values_bound.size):

        e_val_b = eigen_values_bound[ev_b_idx]
        e_vec_b = eigen_vectors_bound[ev_b_idx]

        #Proba to be in vibrational bound state v
        proba_v = bound_vib_level_distrib(e_val_b)

        #proba_inter_atomic_dist = e_vec_b(r_bohr)**2

        #plt.plot()

        #Transition probability = scalar product between initial and
        #final states


        trans_proba = wave_fun_scalar_prod(e_vec_b, final_free_state, dr_bound)**2

        #trans_proba = wave_fun_scalar_prod_from_fun(final_free_state, e_vec_b,
        #r_bohr_bound[0], r_bohr_bound[-1])**2

        #eigen_vectors_boundf = interpolate_eigen_vec_array(eigen_vectors_bound, r_bohr_bound)

        #print("proba_v "+str(proba_v))
        #print("proba_inter_atomic_dist "+str(proba_inter_atomic_dist))
        #print("trans_proba "+str(trans_proba))
        #proba_to_dissociate_from_r = proba_to_dissociate_from_r
        #+proba_v*proba_inter_atomic_dist*trans_proba
        p_E = p_E+proba_v*trans_proba

    #print("proba_to_dissociate_E "+str(p_E))
    return p_E

#def bound_vib_level_distrib(mol_energy):
#    return 1/eigen_values_bound.size

def under_plot_wave_fun(eigen_vectors, eigen_values, r_bohr, V, plot_ratio=1):

    r = r_bohr*bohr_to_meter
    plt.plot(r_bohr, V(r))

    for i in range(0, len(eigen_vectors)):
        if i%plot_ratio == 0:
        #if True:
            psi = eigen_vectors[i]
            #print(eigen_values[i])
            #print(psi)

            #plt.plot(r_bohr, psi**2/np.linalg.norm(psi**2)+eigen_values[i])
            plt.plot(r_bohr, psi/np.linalg.norm(psi)+eigen_values[i])
        #print(energies[i])
    #plt.show()

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

#He2
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

#pre: eigen values sorted by increasing energies
#     eigen_values_bound size is <= 27
def D2_plus_vib_level_distrib(eigen_values_bound):

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

    proba_of_E = lambda E: proba_of_levels[np.abs(eigen_values_bound-E).argmin()]
    return proba_of_E


#E_min 10.8198528842

#Compute bounded stats of D_2+ molecules: wave functions and their energies
(r_bohr_bound, V_bound, eigen_values_bound, eigen_vectors_bound) = numerov(
"pot_d2+.txt", True, 12.9, reduced_mass_d2, refine=3)
#Compute free states of dissociated molecule
(r_bohr_free, V_free, eigen_values_free, eigen_vectors_free) = numerov(
"pot_d2_b.txt", False, 10, reduced_mass_d2, refine=1)

plot_wave_fun(eigen_vectors_bound, eigen_values_bound, r_bohr_bound, V_bound)
plot_wave_fun(eigen_vectors_free, eigen_values_free, r_bohr_free, V_free)

proba_e = []
energies = np.linspace(0, 10, 2000)
delta_e = energies[1]-energies[0]
normalization = 0
i = 0
delta_disp = energies.size/100
disp_count = delta_disp
for e in energies:
    p_e = ker(D2_plus_vib_level_distrib(eigen_values_bound), eigen_values_bound,
    eigen_vectors_bound, eigen_values_free, eigen_vectors_free, r_bohr_bound,
    r_bohr_free, e)
    proba_e.append(p_e)
    normalization = normalization+p_e*delta_e

    progress = i/energies.size*100
    if i / disp_count == 10:
        print("Progress  "+str(progress)+"%")
        disp_count = disp_count+delta_disp
    i = i+1

proba_e = proba_e/normalization

plt.plot(energies, proba_e)
plt.show()

#f1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1,1,1])
#f2 = np.array([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
#print(wave_fun_scalar_prod(f1, f2, 0.5))
