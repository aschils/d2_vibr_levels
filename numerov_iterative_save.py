#!/usr/bin/env python3

###############################
# Author: Arnaud Schils (NAPS)#
# arnaud.schils@gmail.com     #
###############################

#The Numerov numerical method allows to solve second order differential
#equations of the form: Psi''(r) = f(r) Psi(r)
#This script uses this method to solve the Schrodinger equation for the
#molecules H_2 or D_2. The potential is radial and the schrodinger equation
#is 1D in this case. The goal is to find the vibrational energies of these
#molecules.

import math
import numpy as np
import scipy as sp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


#Function "f" in Psi''(r) = f(r) Psi(r)
def f(r, V, E):
    return 2*(V(r)-E)

#Compute phi_i defined as phi_i = Psi_i (1-delta^2 f_i/12)
#from Psi_i
def phi(psi, delta_squared, r, V, E):
    return psi*(1-delta_squared*f(r,V,E)/12)

# phi_i = Psi_i (1-delta^2 f_i/12)
# phi_{i+1} = 2 phi_i - phi_{i-1} + delta^2 f_i Psi_i
# compute phi_{i+1}
def comp_phi_next(phi_cur, phi_prev, delta_squared, f_cur, psi_cur):
    return 2*phi_cur-phi_prev+delta_squared*f_cur*psi_cur

#
# Compute Psi_i for all i (i.e. all points of the numerical domaine r_i)
#
def comp_psi(r, delta_squared, V, E):

    psi_0 = 0 #We are on the potential wall
    psi_1 = 1

    psi = np.zeros(r.size)
    psi[0] = psi_0
    psi[1] = psi_1

    phi_prev = phi(psi[0], delta_squared, r[0], V, E)
    phi_cur = phi(psi[1], delta_squared, r[1], V, E)

    f_cur = f(r[1], V, E)

    for i in range(1, r.size-1):
        phi_next = comp_phi_next(phi_cur, phi_prev, delta_squared, f_cur, psi[i])
        phi_prev = phi_cur
        phi_cur = phi_next
        f_next = f(r[i+1], V, E)
        psi[i+1] = phi_next/(1-delta_squared*f_next/12)
        f_cur = f_next
    return psi/wave_fun_norm(r, psi)

def search_bracket_energies(r, delta_squared, V, psi_e, delta_E, E_max,
    E_start):

    bracket_E_found = False
    left_found = False
    right_found = False
    E_left = E_start
    E_right = E_start
    E = E_start

    while not bracket_E_found and E <= E_max:

        psi = comp_psi(r, delta_squared, V, E)
        error_on_psi_e = psi[-1]-psi_e

        if not left_found and error_on_psi_e > 0:
            E_left = E
            left_found = True
        elif not right_found and error_on_psi_e < 0:
            E_right = E
            right_found = True

        bracket_E_found = left_found and right_found
        E = E + delta_E

    if bracket_E_found:
        return (E_left, E_right)
    else:
        return None

    #print("E_left: "+str(E_left))
    #print("E_right: "+str(E_right))

    #psi = comp_psi(r, psi, delta_squared, V, E_min)
    #error_on_psi_e = psi[-1]-psi_e
    #print(error_on_psi_e)

#Find the two intersections between the potential and the energy line.
def find_V_E_intersect(r_min_of_V, V, E):
    f = lambda x: V(x)-E
    root_left = sp.optimize.brentq(f, 0, r_min_of_V)
    r_infinity = r_min_of_V*100
    root_right = sp.optimize.brentq(f, r_min_of_V, r_infinity)
    return (root_left,root_right)


def search_exact_energy_shoot_method(psi_e, stop_error_on_psi_e, E_left, E_right,
r, delta_squared, V):

    error_on_psi_e = stop_error_on_psi_e+1
    prev_error_on_psi_e = 999999999

    while np.abs(error_on_psi_e) > stop_error_on_psi_e:

        E = E_left + (E_right-E_left)/2.0
        psi = comp_psi(r, delta_squared, V, E)
        error_on_psi_e = psi[-1]-psi_e

        if math.fabs(error_on_psi_e-prev_error_on_psi_e) < 0.0001:
            return None

        #print("("+str(E_left)+","+str(E_right)+")")
        #print(error_on_psi_e)
        if error_on_psi_e > 0:
            E_left = E
        else:
            E_right = E
        prev_error_on_psi_e = error_on_psi_e

    return (psi, E)

# Schrodinger
# Psi''(r) = f(r) Psi(r)
#
def numerov_bound(V, stop_error_on_psi_e, delta_energy):

    nbr_of_pts = 1000
    min_V_result = sp.optimize.minimize_scalar(V)
    psi_e = 0 #We are on the potential wall

    if not min_V_result.success:
        raise Exception("Minimum of potential V not found.")

    r_min_of_V = min_V_result.x
    #E_min = V(r_min_of_V)
    E_min = 0
    r_infinity = r_min_of_V*100

    print("Minimum of potential is r_min="+str(r_min_of_V)+".")
    print("Value of potential at infinity taken at r = 100 times r_min, r_infinity="+str(r_infinity)+".")

    E_max = V(r_infinity)
    E_max = E_min+1

    print("Searching vibrational energies between "+str(E_min)+" and "+str(E_max)+".")

    error_on_psi_e = stop_error_on_psi_e+1

    energies = []
    wave_functions = []

    E = E_min+delta_energy #At E_min, single intersection between V and E

    while E <= E_max:

        #[r_0, r_e] = [0,8]
        #[r_0, r_e] = find_V_E_intersect(r_min_of_V, V, E)
        r_0 = 0
        r_e = 1
        print("For energy E="+str(E)+" potential walls found at r_left="+str(r_0)+" and r_right="+str(r_e)+".")

        r = np.linspace(r_0, r_e, nbr_of_pts)

        delta = r[1]-r[0]
        delta_squared = delta**2

        bracket_energies = search_bracket_energies(r, delta_squared, V,
        psi_e, delta_E, E_max, E)

        print(bracket_energies)

        if bracket_energies is not None:

            (E_left, E_right) = bracket_energies
            shoot_result = search_exact_energy_shoot_method(psi_e,
            stop_error_on_psi_e, E_left, E_right, r, delta_squared, V)

            if shoot_result is not None:
                (psi_eigenvec, E_eigenval) = shoot_result
                energies.append(E_eigenval)
                wave_functions.append(psi_eigenvec)

            #else:
                #print("SHOOT RESULT EST NONE")
            E = E_right+delta_E

        else:
            E = E+delta_E


    return (r, wave_functions, energies)

    #print(error_on_psi_e)
    #print(psi)



def numerov_bound_simple(V, stop_error_on_psi_e, delta_energy):

    nbr_of_pts = 100
    #min_V_result = sp.optimize.minimize_scalar(V)
    psi_e = 0 #We are on the potential wall

    #if not min_V_result.success:
    #    raise Exception("Minimum of potential V not found.")

    #r_min_of_V = min_V_result.x
    #E_min = V(r_min_of_V)
    E_min = 0
    #r_infinity = r_min_of_V*100

    #print("Minimum of potential is r_min="+str(r_min_of_V)+".")
    #print("Value of potential at infinity taken at r = 100 times r_min, r_infinity="+str(r_infinity)+".")

    #E_max = V(r_infinity)
    E_max = 2

    print("Searching vibrational energies between "+str(E_min)+" and "+str(E_max)+".")

    error_on_psi_e = stop_error_on_psi_e+1

    energies = []
    wave_functions = []

    E = E_min+delta_energy #At E_min, single intersection between V and E

    while E <= E_max:

        [r_0, r_e] = [0,1]
        #[r_0, r_e] = find_V_E_intersect(r_min_of_V, V, E)
        print("For energy E="+str(E)+" potential walls found at r_left="+str(r_0)+" and r_right="+str(r_e)+".")

        r = np.linspace(r_0, r_e, nbr_of_pts)

        delta = r[1]-r[0]
        delta_squared = delta**2

        psi = comp_psi(r, delta_squared, V, E)
        error_on_psi_e = math.fabs(psi[-1]-psi_e)

        if error_on_psi_e <= stop_error_on_psi_e:
            energies.append(E)
            wave_functions.append(psi)

        E = E+delta_E

    return (r, wave_functions, energies)

    #print(error_on_psi_e)
    #print(psi)



#Compute int_{r_0}^_{r_e} |psi|^2 dr
def wave_fun_norm(r, psi):
    dr = r[1]-r[0]
    return np.sqrt(np.sum(psi**2*dr))



#r_0 = -1

pot_file = np.loadtxt("pot_d2+.txt")
r = pot_file[:,0]
V = pot_file[:,1]
#E_max = np.amax(V)
#E_min = np.amin(V)
V =  interp1d(r,V,kind="linear",fill_value="extrapolate")

#E_max = 12

#r_0 = 0
#r_e = 10

#r_e = 8
#r_e = 1

#nbr_of_pts = 100
delta_E = 0.1

#V = lambda r: (r-1.5)**2

def potential_well(r):
    if r > 0 and r < 1:
        return 0
    else:
        return 99999999

#V = lambda r: (0, 99999999)[np.abs(r) > 1]
V = potential_well
stop_error_on_psi_e = 0.1 #10**-2

#print(V)


(r, wave_functions, energies) = numerov_bound_simple(V, stop_error_on_psi_e, delta_E)
#print(wave_fun_norm(r[0], r[-1], nbr_of_pts, psi))
#print(psi)

for i in range(0,len(wave_functions)):
    psi = wave_functions[i]
    plt.plot(r, psi)
    print(energies[i])
    #plt.plot(r, V(r))
plt.show()

#Solution for potential well
#E = k^2 hbar^2 /(2m)
