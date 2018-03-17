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
m = 4*m_p #approximation of D_2 mass [kg]

electron_charge = 1.60217662*10**-19 #coulomb
bohr_to_meter = 5.291772*10**-11

def wave_fun_scalar_prod(psi1, psi2, dr):
    return np.dot(psi1,psi2)*dr

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
# r_end_for_unbound_potential: set end of the numerical domain for unbound
# potential (for bound potential is it the second intersection between the
# E_max line and the potential V(r))
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
def numerov(pot_file_path, is_bound_potential, E_max, r_end_for_unbound_potential=10):

    pot_file = np.loadtxt(pot_file_path)

    r = pot_file[:,0]*bohr_to_meter
    V = pot_file[:,1]*electron_charge #Joule
    V = interp1d(r,V,kind="linear",fill_value="extrapolate")

    r_inf = 10**6 #purely arbitrary

    if is_bound_potential:
        min_V_result = sp.optimize.minimize_scalar(V)
        if not min_V_result.success:
            raise Exception("Minimum of potential V not found.")
        r_min_of_V = min_V_result.x
        E_min = V(r_min_of_V)
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
    wavelength = h/math.sqrt(2*m*E_max)

    dr = wavelength/(2.0*math.pi)
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
    KE = -sp.linalg.inv(B.todense()).dot(A.todense())*hbar**2/(2.0*m)

    H = KE + sp.sparse.diags(V(r), offsets=0)
    eigen_values, eigen_vectors = np.linalg.eig(H)

    eigen_values_sorted_idx = np.argsort(eigen_values)
    eigen_values = np.asarray(list(eigen_values[i] for i in eigen_values_sorted_idx))

    eigen_vectors = eigen_vectors.T
    eigen_vectors_temp = np.asarray(list(eigen_vectors[i] for i in eigen_values_sorted_idx))
    eigen_vectors = []

    c = 0
    for ev_fat in eigen_vectors_temp:
        ev = ev_fat[0]
        if is_bound_potential:
        #if True:
            eigen_vectors.append(ev/wave_fun_norm(ev, dr))
        else:
            eigen_vectors.append(normalize_continuum_wave_fun(ev, eigen_values[c], r))
        c = c+1

    eigen_vectors = np.asarray(eigen_vectors)

    eigen_values = eigen_values/electron_charge

    r_bohr = r/bohr_to_meter

    return (r_bohr, V, eigen_values, eigen_vectors)

def interpolate_eigen_vec_array(ev_array, r):
    ev_fun = []
    for i in range(0,r.size):
        fun = interp1d(r,ev_array[i],kind=0,fill_value="extrapolate")
        ev_fun.append(fun)
    return ev_fun

def final_dissoc_state(eigen_values_free, eigen_vectors_free, E):
    i = np.abs(eigen_values_free-E).argmin()
    return eigen_vectors_free[i]

def ker(bound_vib_level_distrib, eigen_values_bound,
eigen_vectors_bound, eigen_values_free, eigen_vectors_free, r_bohr_bound,
r_bohr_free, E):

    p_E = 0
    dr_bound = r_bohr_bound[1]-r_bohr_bound[0]

    for ev_b_idx in range(0, eigen_values_bound.size):

        e_val_b = eigen_values_bound[ev_b_idx]
        e_vec_b = eigen_vectors_bound[ev_b_idx]
        #Proba to be in vibrational bound state v
        proba_v = bound_vib_level_distrib(e_val_b)
        #proba_inter_atomic_dist = e_vec_b(r_bohr)**2

        #Find dissociated state from the inter-atomic distance in the
        #bound state
        final_free_state = final_dissoc_state(eigen_values_free,
        eigen_vectors_free, E)
        #Transition probability = scalar product between initial and
        #final states

        final_free_statef = interp1d(r_bohr_free, final_free_state,
        kind=0,fill_value="extrapolate")
        final_free_state = final_free_statef(r_bohr_bound)
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

    print("proba_to_dissociate_E "+str(p_E))
    return p_E


# (r_bohr, V, eigen_values, eigen_vectors) = numerov("pot_d2_b.txt", False, 2)
# (r_bohr, V, eigen_values, eigen_vectors) = numerov("pot_d2+.txt", True, 13)
#
#

#E_min 10.8198528842

(r_bohr_bound, V_bound, eigen_values_bound, eigen_vectors_bound) = numerov("pot_d2+.txt", True, 10.9)
(r_bohr_free, V_free, eigen_values_free, eigen_vectors_free) = numerov("pot_d2_b.txt", False, 6)

def bound_vib_level_distrib(mol_energy):
    return 1/eigen_values_bound.size



#r_m_bound = r_bohr_bound*bohr_to_meter
#r_m_free = r_bohr_free*bohr_to_meter

#eigen_vectors_boundf = interpolate_eigen_vec_array(eigen_vectors_bound, r_bohr_bound)
#eigen_vectors_freef = interpolate_eigen_vec_array(eigen_vectors_free, r_bohr_free)

# eigen_vectors = eigen_vectors_free
# eigen_values = eigen_values_free
# r_bohr = r_bohr_free
# V = V_free
# #code to plot wave functions above potential
# for i in range(0,eigen_values.size):
#     if i%10==0:
#         psi = eigen_vectors[i]
#         #print(psi)
#         r = r_bohr*bohr_to_meter
#         plt.plot(r_bohr, V(r)/electron_charge)
#         #plt.plot(r_bohr, psi**2/np.linalg.norm(psi**2)+eigen_values[i])
#         plt.plot(r_bohr, psi/np.linalg.norm(psi)+eigen_values[i])
#     #print(energies[i])
# plt.show()


#print(prob_to_dissociate_from_r(bound_vib_level_distrib, eigen_values_bound,
#eigen_vectors_bound, eigen_values_free, eigen_vectors_free, r_bohr_bound, 3))


proba_e = []
delta_e = 0.01
energies = np.linspace(1.5, 4.5, 200)
delta_e = energies[1]-energies[0]
I = 0
i = 0
for e in energies:
    p_e = ker(bound_vib_level_distrib, eigen_values_bound,
    eigen_vectors_bound, eigen_values_free, eigen_vectors_free, r_bohr_bound,
    r_bohr_free, e)
    proba_e.append(p_e)
    I = I+p_e*delta_e
    print("Progress  "+str(i/e.size)+"%")
    i = i+1

print(I)

plt.plot(energies, proba_e)
plt.show()



#
#
# print(eigen_values)
#print(wave_fun_norm(eigen_vectors[0],dr))
#print(wave_fun_scalar_prod(eigen_vectors[0], eigen_vectors[10], dr))
