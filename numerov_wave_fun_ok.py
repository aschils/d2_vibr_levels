
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
m = m_p #approximation of D_2 reduced mass [kg]

electron_charge = 1.60217662*10**-19 #coulomb
bohr_to_meter = 5.291772*10**-11

def wave_fun_scalar_prod(psi1, psi2, dr):
    return np.dot(psi1,psi2)*dr

def wave_fun_norm(psi, dr):
    return math.sqrt(wave_fun_scalar_prod(psi, psi, dr))

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

    eigen_values = eigen_values/electron_charge
    eigen_values_sorted_idx = np.argsort(eigen_values)
    eigen_values = np.asarray(list(eigen_values[i] for i in eigen_values_sorted_idx))

    eigen_vectors = eigen_vectors.T
    eigen_vectors_temp = np.asarray(list(eigen_vectors[i] for i in eigen_values_sorted_idx))
    eigen_vectors = []
    for ev_fat in eigen_vectors_temp:
        ev = ev_fat[0]
        eigen_vectors.append(ev/wave_fun_norm(ev, dr))
    eigen_vectors = np.asarray(eigen_vectors)

    r_bohr = r/bohr_to_meter

    return (r_bohr, V, eigen_values, eigen_vectors)

#(r_bohr, V, eigen_values, eigen_vectors) = numerov("pot_d2_b.txt", False, 2)
(r_bohr, V, eigen_values, eigen_vectors) = numerov("data/pot_d2+.txt", True, 13)

print(eigen_values)
#code to plot wave functions above potential
for i in range(0,10):
    if True:
        psi = eigen_vectors[i]
        #print(psi)
        r = r_bohr*bohr_to_meter
        plt.plot(r_bohr, V(r)/electron_charge)
        plt.plot(r_bohr, psi**2/np.linalg.norm(psi**2)+eigen_values[i])
        #plt.plot(r_bohr, psi/np.linalg.norm(psi)+eigen_values[i])
    #print(energies[i])
plt.show()




#
#
#print(wave_fun_norm(eigen_vectors[0],dr))
#print(wave_fun_scalar_prod(eigen_vectors[0], eigen_vectors[10], dr))
