#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Symétrie autorisée:
# Sigma_g^+, Pi_u, Delta_g
# Sigma_u^+, Pi_g, Delta_u
#

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
import scipy.interpolate as interpolate

#Some constants
NUMEROV_SAVE_PKL_DIR = "cache_pkl"
KER_SAVE_FILE = "cache_pkl/computed_ker.pkl"
Q_R_PATH = "data/q_wrt_r.txt"

POTENTIALS_DIR = "data"
D2P_POTENTIAL_FILE_DISSOC = "pot_d2+.txt"
D2P_POTENTIAL_FILE = "pot_d2+_shifted.txt"
D2DISSOC_POTENTIAL_FILE = "pot_d2_b.txt"
D2X_1SG_POTENTIAL_FILE = "d2_x_1sg.txt" #Ground state
D2X_1SG_FREE_PART_POTENTIAL_FILE = "d2_x_1sg_free.txt"
D2STAR_GK1SG_POTENTIAL_FILE = "gk1sg.dat"
D2STAR_2_3SG_POTENTIAL_FILE = "2_3Sigma_g"#_truncat"
D2STAR_3_3SG_POTENTIAL_FILE = "3_3Sigma_g"
D2STAR_1_SU_B_POTENTIAL_FILE = "1Sigma_u_B.txt"
D2STAR_1_SU_BP_POTENTIAL_FILE = "1Sigma_u_Bp.txt"
D2STAR_1_SU_BPP_POTENTIAL_FILE = "1Sigma_u_Bpp.txt"
D2STAR_2_1SG_POTENTIAL_FILE = "2_1Sigma_g"
D2STAR_3_1SG_POTENTIAL_FILE = "3_1Sigma_g"
D2STAR_4_1SG_POTENTIAL_FILE = "4_1Sigma_g"
D2STAR_2_3SU_POTENTIAL_FILE = "2_3Sigma_u"
D2STAR_3_3SU_POTENTIAL_FILE = "3_3Sigma_u"
D2STAR_4_3SU_POTENTIAL_FILE = "4_3Sigma_u"
D2STAR_1PI_U_C_POTENTIAL_FILE = "1Piu_C.txt"
D2STAR_1PI_U_D_POTENTIAL_FILE = "1Piu_D.txt"
D2STAR_1_3PI_G_POTENTIAL_FILE = "1_3pig.txt"
D2STAR_1_3PI_U_POTENTIAL_FILE = "1_3piu.txt"
D2STAR_2_3PI_G_POTENTIAL_FILE = "2_3pig.txt"
D2STAR_2_3PI_U_POTENTIAL_FILE = "2_3piu.txt"
D2STAR_3_3PI_G_POTENTIAL_FILE = "3_3pig.txt"
D2STAR_3_3PI_U_POTENTIAL_FILE = "3_3piu.txt"
D2STAR_1_PI_GI_POTENTIAL_FILE = "1_pi_g_I.txt"
D2STAR_1_PI_GR_POTENTIAL_FILE = "1_pi_g_R.txt"


EXP_KER_PATH = "data/ker_d2_d.txt"

#Definition of physical constants
h = 6.626*10**-34 #Planck Constant [J s]
hbar = h/(2*math.pi)
m_e = 9.10938356*10**-31 #electron mass [kg]
m_p = 1.672*10**-27 #Proton mass [kg]
reduced_mass_d2 = m_p #approximation of D_2 reduced mass [kg]
reduced_mass_he2 = 2*m_p
reduced_mass_h2 = m_p/2
electron_charge = 1.60217662*10**-19 #coulomb
bohr_to_meter = 5.291772*10**-11
ionisation_potential_d2 = 15.466 #eV
au_to_ev = 27.211396
electron_binding_energy_in_H_minus = 0.754 #eV

E_max_bound_states_D2p = 12.9-27.211
E_ground_state_D2p = 11-27.211
E_two_states_D2p = 11.1-27.211
E_three_states_D2p = 11.2-27.211
E_max_bound_states_D2b = 12#0

E_max_bound_states_D2STAR_GK1SG = 1-.663091017*27.211
E_max_bound_states_D2STAR_2_3SG = 1-.663091017*27.211
E_max_bound_states_D2STAR_3_3SG = -15.2 #E_max -15.1174389834 eV
E_max_bound_states_D2STAR_1_SU_B = -17.5

class NumerovParams:

    def __init__(self, pot_file_path, is_bound_potential, E_max, reduced_mass,
    r_end_for_unbound_potential=10, refine=1, pot_in_au=False, auto_E_max=True):

        self.pot_file_path = pot_file_path
        self.is_bound_potential = is_bound_potential
        self.E_max = E_max
        self.reduced_mass = reduced_mass
        self.r_end_for_unbound_potential = r_end_for_unbound_potential
        self.refine = refine
        self.pot_in_au = pot_in_au
        self.auto_E_max = auto_E_max

    def plot_potential(self, r_end, show=True, min_at_zero=False, label=""):
        pot_file = np.loadtxt(POTENTIALS_DIR+"/"+self.pot_file_path)
        r_init = pot_file[:,0]
        r = r_init[r_init < r_end]
        V_init = pot_file[:,1]
        V = V_init[r_init < r_end]
        if self.pot_in_au:
            V = V*au_to_ev

        if min_at_zero:
            min_V = np.min(V)
            V = V-min_V
        plot_ref = plt.plot(r, V, label=label)
        if show:
            plt.show()
        return plot_ref
        

    def to_string(self):
        return NUMEROV_SAVE_PKL_DIR+"/"+self.pot_file_path+"_" \
        +str(self.is_bound_potential)+"_"+str(self.E_max)+"_"+str(self.reduced_mass)+"_" \
        +str(self.r_end_for_unbound_potential)+"_"+str(self.refine)+"_"+str(self.pot_in_au)+"_" \
        +str(self.auto_E_max)

    def as_tuple(self):
        return (self.pot_file_path, self.is_bound_potential, self.E_max,
        self.reduced_mass, self.r_end_for_unbound_potential, self.refine,
        self.pot_in_au, self.auto_E_max)

class NumerovResult:

    def __init__(self, r_bohr, V, eigen_values, eigen_vectors, min_of_V):
        self.r_bohr = r_bohr
        self.V = V
        self.eigen_values = eigen_values
        self.eigen_vectors = eigen_vectors
        self.min_of_V = min_of_V

    def plot(self, r_bohr_max=float("inf"), plot_ratio=1, save=False, out_path=None,
	show=True, figsize=(10,10), ylim=None, y_axis="energy eV", xlabel="Bohr", title=None,
    useTex=False):
        r_bohr = self.r_bohr[self.r_bohr <= r_bohr_max]
        plot_wave_fun(self.eigen_vectors, self.eigen_values, r_bohr, self.V,
        plot_ratio, save, out_path, show, figsize, ylim, y_axis, xlabel, title,
        useTex)

    def write_eigen_vector(self, path):
        of = open(path, "w")
        for i in range(0, self.r_bohr.size):
            of.write(str(self.r_bohr[i])+" ")
            for j in range(0, len(self.eigen_vectors)):
                of.write(str(self.eigen_vectors[j][i])+" ")
            of.write("\n")
        of.close()

def plot_pot_and_ev(r_bohr, V, eigen_values, min_of_V, label=""):        
    pot_plot_ref = plt.plot(r_bohr, V(r_bohr), label=label)
    for ev in eigen_values:
        f = lambda x: V(x)-ev    
        V_E_intersect_left = sp.optimize.brentq(f, 0, min_of_V)
        try:
            V_E_intersect_right = sp.optimize.brentq(f, min_of_V, r_bohr[-1])
        except ValueError as e:
            V_E_intersect_right = r_bohr[-1]

        plt.plot([V_E_intersect_left, V_E_intersect_right], [ev, ev], color="black")
    return pot_plot_ref
            

class MolecularState:

    def __init__(self, numerov_params, electron_binding_energy = 0,
    electron_angular_momentum = 0, total_spin = 0):
        self.numerov_params = numerov_params
        self.electron_binding_energy = electron_binding_energy #E_A
        self.electron_angular_momentum = electron_angular_momentum #lambda
        self.total_spin = total_spin

    #Getters
    def get_numerov_params(self):
        return self.numerov_params

    def get_electron_binding_energy(self):
        return self.electron_binding_energy

    def get_electron_angular_momentum(self):
        return self.electron_angular_momentum

    def get_total_spin(self):
        return self.total_spin

refine = 2
#pot_file_path, is_bound_potential, E_max, reduced_mass,
#r_end_for_unbound_potential=10, refine=1, pot_in_au, auto_E_max
D2P_NUMEROV_PARAMS = NumerovParams(D2P_POTENTIAL_FILE, True, E_max_bound_states_D2p,
reduced_mass_d2, 10, refine, False, True)
H2P_NUMEROV_PARAMS = NumerovParams(D2P_POTENTIAL_FILE, True, E_max_bound_states_D2p,
reduced_mass_h2, 10, refine, False, True)
D2P_NUMEROV_PARAMS_DISSOC = NumerovParams(D2P_POTENTIAL_FILE_DISSOC, True, E_max_bound_states_D2p,
reduced_mass_d2, 10, refine, False, True)
D2B_NUMEROV_PARAMS = NumerovParams(D2DISSOC_POTENTIAL_FILE, False, E_max_bound_states_D2b,
reduced_mass_d2, 30, refine, False, False)
D2X_1SG_NUMEROV_PARAMS = NumerovParams(D2X_1SG_POTENTIAL_FILE, True,
0, reduced_mass_d2, 0, refine, True, True)
D2X_1SG_FREE_PART_NUMEROV_PARAMS = NumerovParams(D2X_1SG_FREE_PART_POTENTIAL_FILE, False,
0, reduced_mass_d2, 10, refine, True, False)
D2STAR_GK1SG_NUMEROV_PARAMS = NumerovParams(D2STAR_GK1SG_POTENTIAL_FILE, True,
E_max_bound_states_D2STAR_GK1SG, reduced_mass_d2, 10, refine, True, True)
H2STAR_GK1SG_NUMEROV_PARAMS = NumerovParams(D2STAR_GK1SG_POTENTIAL_FILE, True,
E_max_bound_states_D2STAR_GK1SG, reduced_mass_h2, 10, refine, True, True)
D2STAR_2_3SG_NUMEROV_PARAMS = NumerovParams(D2STAR_2_3SG_POTENTIAL_FILE, True,
E_max_bound_states_D2STAR_2_3SG, reduced_mass_d2, 10, refine, True, True)
D2STAR_3_3SG_NUMEROV_PARAMS = NumerovParams(D2STAR_3_3SG_POTENTIAL_FILE, True,
E_max_bound_states_D2STAR_3_3SG, reduced_mass_d2, 10, refine, True, True)
D2STAR_1_SU_B_NUMEROV_PARAMS = NumerovParams(D2STAR_1_SU_B_POTENTIAL_FILE, True,
E_max_bound_states_D2STAR_1_SU_B, reduced_mass_d2, 10, refine, True, True)
D2STAR_1_SU_BP_NUMEROV_PARAMS = NumerovParams(D2STAR_1_SU_BP_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
H2STAR_1_SU_BP_NUMEROV_PARAMS = NumerovParams(D2STAR_1_SU_BP_POTENTIAL_FILE, True,
0, reduced_mass_h2, 10, refine, True, True)
D2STAR_1_SU_BPP_NUMEROV_PARAMS = NumerovParams(D2STAR_1_SU_BPP_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_2_1SG_NUMEROV_PARAMS = NumerovParams(D2STAR_2_1SG_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_3_1SG_NUMEROV_PARAMS = NumerovParams(D2STAR_3_1SG_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_4_1SG_NUMEROV_PARAMS = NumerovParams(D2STAR_4_1SG_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_2_3SU_NUMEROV_PARAMS = NumerovParams(D2STAR_2_3SU_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_3_3SU_NUMEROV_PARAMS = NumerovParams(D2STAR_3_3SU_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_4_3SU_NUMEROV_PARAMS = NumerovParams(D2STAR_4_3SU_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_1PI_U_C_NUMEROV_PARAMS = NumerovParams(D2STAR_1PI_U_C_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_1PI_U_D_NUMEROV_PARAMS = NumerovParams(D2STAR_1PI_U_D_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_1_3PI_G_NUMEROV_PARAMS = NumerovParams(D2STAR_1_3PI_G_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_1_3PI_U_NUMEROV_PARAMS = NumerovParams(D2STAR_1_3PI_U_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_2_3PI_G_NUMEROV_PARAMS = NumerovParams(D2STAR_2_3PI_G_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_2_3PI_U_NUMEROV_PARAMS = NumerovParams(D2STAR_2_3PI_U_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_3_3PI_G_NUMEROV_PARAMS = NumerovParams(D2STAR_3_3PI_G_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_3_3PI_U_NUMEROV_PARAMS = NumerovParams(D2STAR_3_3PI_U_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_1_PI_GI_NUMEROV_PARAMS = NumerovParams(D2STAR_1_PI_GI_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
D2STAR_1_PI_GR_NUMEROV_PARAMS = NumerovParams(D2STAR_1_PI_GR_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)

D2STAR_GK1SG_MS = MolecularState(D2STAR_GK1SG_NUMEROV_PARAMS, 1.646/au_to_ev, 2, 0)
D2STAR_1_SU_BP_MS = MolecularState(D2STAR_1_SU_BP_NUMEROV_PARAMS, 0, 0, 0)


q_r_data = np.loadtxt(Q_R_PATH)
r = q_r_data[:,0]
q = q_r_data[:,1]
q_r_bound_to_bound = interp1d(r,q,kind="linear",fill_value="extrapolate")

q_r_bound_to_bound = lambda x: 1

def gaussian(a, b, c, x):
    return a*np.exp(-((x-b)/c)**2/2)

# <psi1 | psi2> = int_{-\infty}^{\infty} psi1(r)^* psi2(r) dr
#Integrate using trapeze method (MUCH faster than using scipy.quad...)
def wave_fun_scalar_prod(psi1, psi2, dr):
    # f = psi1*psi2
    #
    # #for i in range(0,psi1.size):
    # I = dr/2*(2*np.sum(f)-f[0]-f[-1])
    # return I
    return np.dot(psi1,psi2)*dr-dr/2.0*psi1[0]*psi2[0]-dr/2.0*psi1[-1]*psi2[-1]

#def normalization_of_continuum_wave_fun(psi, energy):




    #mu = reduced_mass_d2
    #k = math.sqrt(2*mu*energy)/hbar
    # s = math.sin(k*r[-1])
    # c = math.cos(k*r[-1])
    # sd = k*c
    # cd = -k*s
    #
    # dr = r[1]-r[0]
    # F = psi[-1]
    # Fd = (3*psi[-1]-4*psi[-2]+psi[-3])/(2*dr)
    #
    # A = (F*sd-Fd*s)/(c*sd-cd*s)
    # B = (F*cd-Fd*c)/(s*cd-sd*c)
    # #print("ormalization_of_continuum_wave_fun "+str(math.sqrt((A**2+B**2)*math.pi*k)))
    # return math.sqrt((A**2+B**2)*math.pi*k)
    #return 1

# def normalize_continuum_wave_fun(psi, energy):
#     psi_normalized = psi/normalization_of_continuum_wave_fun(psi, energy)
#     return psi_normalized

def normalize_continuum_wave_fun(r, psi, energy):

    #defense
    energy = np.abs(energy)

    #Find amplitude of sin in psi "at infinity"
    r = r/bohr_to_meter
    psi_f = interp1d(r, psi, kind=0, fill_value=0, bounds_error = False)

    root_found = False
    root1 = 0
    root2 = r[-1]
    r_idx = r.size-1

    while not root_found and r_idx > 0:

        try:
            #print("r_left "+str(r[r_idx-1])+"r_right "+str(r[r_idx]))
            root1 = sp.optimize.brentq(psi_f, r[r_idx-1], r[r_idx])
            root_found = True
        except:
            pass
        r_idx = r_idx-1

    #print("root1 "+str(root1)+" root2 "+str(root2))

    f = lambda r: -np.abs(psi_f(r))
    opt_res = sp.optimize.minimize_scalar(f, bracket=(root1, root2))
    psi_amplitude = -f(opt_res.x)
    #psi_amplitude = np.abs(psi_f((root2-root1)/2.0+root1))
    # plt.plot(r, f(r))
    # plt.plot(r, psi_f(r))
    # print("Amplitude "+str(psi_amplitude))
    # plt.plot(r, psi_amplitude*np.ones(r.size))
    # plt.show()


    psi = psi/psi_amplitude
    k = math.sqrt(2*reduced_mass_d2*energy)/hbar
    #print("k "+str(k))
    # k dimension 1/L
    # hbar**2 k**2/(2m) = E
    # hbar = E.T
    # k = sqrt(M/(E T^2)) = sqrt(1/L^2)  = 1/L
    # E = M*L^2/T^2

    #psi = math.sqrt(2*reduced_mass_d2/(math.pi*hbar**2*k))*psi

    psi = 1/math.sqrt(hbar)*math.sqrt(math.sqrt(2*reduced_mass_d2)/(math.pi*math.sqrt(energy)))*psi
    psi = psi*math.sqrt(bohr_to_meter*electron_charge)#*au_to_ev)

    #psi = psi/math.sqrt(k)

    #dim  of norm factor sqrt(M L/(E^2 T^2)) = sqrt(T^2/(M L^3)) = sqrt(1/(E L))

    return psi




# def normalize_continuum_wave_fun(idx, wave_fun, eigen_values,r_max):
#
#     if idx == 0:
#         idx = idx+1
#     elif idx == (eigen_values.size-1):
#         idx = idx-1
#
#     #bsplines_dos = r_max/(math.pi*np.sqrt(2*eigen_values[idx]))
#     # print(bsplines_dos)
#     azero = 0.529*10**-10
#     adim = hbar*3*10**8/azero
#     density_of_states = 2*adim/(eigen_values[idx+1]-eigen_values[idx-1])#/electron_charge
#     # print(density_of_states)
#     #return wave_fun*bsplines_dos/density_of_states*np.sqrt(density_of_states)*electron_charge
#     return wave_fun*np.sqrt(density_of_states)#*1.89794045821*10**-8 #/density_of_states*np.sqrt(density_of_states)*electron_charge

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
        return (V-min_V, -min_V)
    else:
        return (V, 0)

#
# This function is useful to find a maximum energy E_max for the search of
# wave functions solution of Schrodinger with a bounded potential.
# If we just take E_max as the maximum of the right part of provided potential well,
# the numerical domain is unecessary too big. Therefore this function finds
# an energy in provided potential vector, "x" percent less than the true top
# of the well.
#
# It is easier not to interpolate and take any energy because restricting
# ourselves to the discrete values in the potential vector avoids to search
# intersection between selected E_max and the potential well afterwareds
# (root finding algorithm in scipy is somewhat capricious).
#
# @pre: - sorted_values is a numpy 1D array of float sorted in increasing order
#       - value is one of the values in sorted_values
#       - percent is a number (int or float...) such as 0 <= percent <= 100
#
def idx_of_lower_E_max(values, value, percent_less):

    min_val = values[np.argmin(values)]
    delta = value-min_val
    target_val = value-delta/100.0*percent_less
    nearest_of_target_idx = np.abs(values-target_val).argmin()
    nearest_val = values[nearest_of_target_idx]

    if nearest_val >= target_val or nearest_of_target_idx == (values.size-1):
        return nearest_of_target_idx
    else:
        return nearest_of_target_idx+1

#
# You must pass to this function a NumerovParams object. This object contains
# all info required to perform Numerov.
#
# @Preconditions:
# First column of pot file is inter-atomic distance in Bohr
# Second column is the potential in eV (if in au you can set pot_in_au to True)
#
# is_bound_potential must be True if the molecular potential has a minimum (potential well),
# False otherwise
#
# E_max: search energy levels from potential minimum up to this energy
#       (not considered if auto_E_max is set to True)
#
# reduced_mass: of molecule in kg
#
# r_end_for_unbound_potential: set end of the numerical domain for unbound
# potential
#
# refine: initial step dr will be decreased as dr/refine
#
# @Return: NumerovResult object having following attribues:
# - r_bohr : points of the numerical domain in Bohr (1D numpy array)
# - V, a function of r, is the potential in eV as a function of the interatomic
# distance r in bohr
# - eigen_values: a 1D numpy array, energy levels (eigenvalues of Hamiltonian)
# in eV, sorted by increasing energy
# - eigen_vectors: 2D numpry array, eigen_vectors[i] is the Psi_i
#   eigen_vectors[i][j] is Psi_i(r_j), r_j are the values in r_bohr
#
#def numerov(pot_file_path, is_bound_potential, E_max, reduced_mass,
#r_end_for_unbound_potential=10, refine=1, pot_in_au=False, auto_E_max=True):

def numerov(NumerovParams, use_cache = True):

    (pot_file_path, is_bound_potential, E_max, reduced_mass,
    r_end_for_unbound_potential, refine, pot_in_au, auto_E_max) = \
    NumerovParams.as_tuple()

    print("Starting Numerov with potential file: "+pot_file_path)

    numerov_save_path = NumerovParams.to_string()+".pkl"

    if use_cache:
        try:
            print("Checking Numerov cache...")
            input = open(numerov_save_path, 'rb')
            numerov_res = pickle.load(input)
            print("Numerov results retrieved from cache.")
            input.close()
            return numerov_res
        except:
            print("Numerov results not found in cache, I will now compute them.")

    pot_file = np.loadtxt(POTENTIALS_DIR+"/"+pot_file_path)

    r_init = pot_file[:,0]
    V_init = pot_file[:,1]
    r = r_init*bohr_to_meter
    if pot_in_au:
        V_init = V_init*au_to_ev
    V_discr = V_init*electron_charge #Joule
    V = interp1d(r,V_discr,kind="linear",fill_value="extrapolate")
    r_inf = 10**6 #purely arbitrary

    E_max = E_max*electron_charge #Joule

    if is_bound_potential:
        i_min = np.argmin(V_discr)
        E_min = V_discr[i_min]
        r_min_of_V = r[i_min]
        if auto_E_max:
            V_after_min = V_discr[i_min:]
            i_max = np.argmax(V_after_min)+i_min
            E_max = V_discr[i_max]
            percent_below_E_max = 0.028646113 #Taking E_max as real max implies too long computation
            i_max = idx_of_lower_E_max(V_discr, E_max, percent_below_E_max)
            E_max = V_discr[i_max]
    else:
        #E_min = V(r_inf)
        #r_min_of_V = r_inf
        r_min_of_V = r[-1]
        E_min = V(r_min_of_V)

    print("Will search vibrational levels between E_min "+str(E_min/electron_charge)+ \
    " eV and E_max "+str(E_max/electron_charge)+" eV")

    f = lambda x: V(x)-E_max
    V_E_intersect_left = sp.optimize.brentq(f, 0, r_min_of_V)

    if is_bound_potential:
        if auto_E_max:
            V_E_intersect_right = r[i_max]
        else:
            V_E_intersect_right = sp.optimize.brentq(f, r_min_of_V,10*r_min_of_V)
    else:
        V_E_intersect_right = r_end_for_unbound_potential*bohr_to_meter

    #De Broglie wavelength (indicator of appropriate dr)
    wavelength = h/math.sqrt(2*reduced_mass*(E_max-E_min))

    dr = wavelength/(2.0*math.pi*refine)
    print("Step dr is "+str(dr/bohr_to_meter)+" bohr")

    box_left_boundary = 0

    n = int(round((V_E_intersect_right-V_E_intersect_left)/dr+5))
    #n = int(round((V_E_intersect_right-box_left_boundary)/dr+1))
    #print("n "+str(n)+" V_E_intersect_right "+str(V_E_intersect_right/bohr_to_meter)+" V_E_intersect_left "+str(V_E_intersect_left/bohr_to_meter)+" dr "+str(dr/bohr_to_meter))

    print("Number of points in the grid: "+str(n))

    i = np.arange(1,n+1)
    #i = np.arange(0,n)
    #r = box_left_boundary+dr*i
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
    H = (KE + sp.sparse.diags(V(r), offsets=0)).todense()

    ###### Change basis to bsplines

    # #n+1 nodes, n-p functions of degree p in basis, n-p control points
    # curve = gaussian(1, (r[-1]-r[0])/2.0, 10**-10, r)
    # #plt.plot(r, curve)
    # #plt.show()
    # (t, c, k) = interpolate.splrep(r, curve)
    #
    # change_basis_m = np.zeros((n,n))
    #
    # def eval_bs(i, r):
    #     coef = np.zeros(c.size)
    #     coef[i] = 1
    #     bs = interpolate.BSpline(t,coef,k)
    #     return bs(r)
    #
    # for i in range(0,n):
    #     change_basis_m[i] = eval_bs(i, r)
    #
    # H = change_basis_m*H#*np.linalg.inv(change_basis_m)

    print("Computing eigenvalues and eigenvectors of Hamiltonian.")
    #eigen_values, eigen_vectors = sp.sparse.linalg.eigs(H, k=n-2, which="SM")
    eigen_values, eigen_vectors = np.linalg.eig(H)
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

    #r_max = V_E_intersect_right/bohr_to_meter

    idx = 0
    for ev_fat in eigen_vectors_temp:
        ev = ev_fat[0]
        ev = np.real(ev) #Required using eig for sparce matrix
        if is_bound_potential:
            eigen_vectors.append(ev/wave_fun_norm(ev, dr))
        else: #r, psi, energy
            if eigen_values[idx] >= -1*au_to_ev:
                eigen_vectors.append(normalize_continuum_wave_fun(r, ev, eigen_values[idx]-E_min))
                #eigen_vectors.append(normalize_continuum_wave_fun(idx, ev, eigen_values, V_E_intersect_right))
        idx = idx+1

    eigen_vectors = np.asarray(eigen_vectors)
    print(len(eigen_vectors))
    eigen_values = eigen_values/electron_charge#-shift

    print("Eigenvalues:")
    print(eigen_values)

    r_bohr = r/bohr_to_meter
    V = interp1d(r_init,V_init,kind="linear",fill_value="extrapolate")
    res = NumerovResult(r_bohr, V, eigen_values, eigen_vectors, r_min_of_V/bohr_to_meter)

    #print("norm in numerov "+str(wave_fun_scalar_prod(eigen_vectors[3],
    #eigen_vectors[3], r[1]-r[0])))

    #Save Numerov result in cache
    if use_cache:
        try:
            output = open(numerov_save_path, "wb")
            pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
            output.close()
        except Exception as e:
            print("Pickle failed to save Numerov results.")
            print(e)

    return res

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

    # delta_E = eigen_values_free[1]-eigen_values_free[0]
    #
    # if E < eigen_values_free[0]-delta_E:
    #    return (0, np.zeros(eigen_vectors_free[0].size))

    # if E < eigen_values_free[0]:
    #     eigen_vec = eigen_vectors_free[0]
    #     r_meter = r_bohr*bohr_to_meter
    #     eigen_vec = eigen_vec*normalization_of_continuum_wave_fun(eigen_vec, eigen_values_free[0], r_meter)
    #     eigen_vec = eigen_vec/normalization_of_continuum_wave_fun(eigen_vec, E, r_meter)
    #     #print("in final dissoc state np.linalg.norm(eigen_vec) "+str(np.linalg.norm(eigen_vec)))
    #     #print(np.linalg.norm(eigen_vec))
    #     return (E, eigen_vec)

    i = np.abs(eigen_values_free-E).argmin()
    return i

def q_I(delta_I_eV,bound_eval, q_I_cosh=False):

    if q_I_cosh:
        return 1/math.cosh(0.191*delta_I_eV)**2

    b = 1.4
    ang_coef = (7.1-b)/(-5+9)
    if delta_I_eV < -9:
        return b
    else:
        return b+(delta_I_eV+9)*ang_coef
    
    #return 1/math.cosh(0.191*delta_I_eV)**2
    #return 1/math.cosh(17.1/math.sqrt(bound_eval)*delta_I_eV)**2



#
# bound_vib_level_distrib: a function, gives proba for bound molecule to be
#               in vibrational state of energy E
# E: energy for which we want p(E), proba to observe kinetic energy E
# bounded_state_IP: ionisation potential of molecule bounded state in eV
#
def ker_dissoc(numerov_res_bound, numerov_res_free, bound_vib_level_distrib, E,
franck_condon_matrix, q_I_cosh=False):

    eigen_values_bound = numerov_res_bound.eigen_values
    eigen_vectors_bound = numerov_res_bound.eigen_vectors
    r_bohr_bound = numerov_res_bound.r_bohr
    eigen_values_free = numerov_res_free.eigen_values
    eigen_vectors_free = numerov_res_free.eigen_vectors
    r_bohr_free = numerov_res_free.r_bohr
    V_free = numerov_res_free.V

    p_E = 0
    dr_bound = r_bohr_bound[1]-r_bohr_bound[0]

    #Find dissociated state from the inter-atomic distance in the
    #bound state
    ev_f_idx = final_dissoc_state(eigen_values_free,
    eigen_vectors_free, E, r_bohr_free)
    #final_free_statef = interp1d(r_bohr_free, final_free_state,
    #kind=0,fill_value=0, bounds_error = False)
    #final_free_state = final_free_statef(r_bohr_bound)


    f = lambda x: V_free(x)-E
    r_star_bohr = sp.optimize.brentq(f, 0, 10)

    for ev_b_idx in range(0, eigen_values_bound.size):

        e_val_b = eigen_values_bound[ev_b_idx]
        e_vec_b = eigen_vectors_bound[ev_b_idx]
        #
        #Proba to be in vibrational bound state v
        proba_v = bound_vib_level_distrib(e_val_b)
        #
        #Transition probability = scalar product between initial and
        #final states
        #trans_proba_squared = wave_fun_scalar_prod(e_vec_b, final_free_state, dr_bound)**2
        #trans_proba_squared = wave_fun_scalar_prod(e_vec_b*r_bohr_bound, final_free_state, dr_bound)**2

        #sigma = 0.05/math.sqrt(2*math.log(2))

        #divide by cosh: remove from probability the fact that you can form a
        #bounded D2 molecule 1/cosh((E(eV)-E_res)/1.547)^2
        #E_res = e_val_b-bounded_state_IP
        #p_E = p_E+proba_v*trans_proba_squared/math.cosh((E-E_res)/gamma)**2

        #exp_exp = math.exp(-(E-E_final_state)**2/(2*sigma**2))/0.106447

        # print("E "+str(E))
        # print("E final state "+str(E))
        # print(math.exp(-(E-E_final_state)**2/(2*sigma**2))/0.106447)

        dr = (r_bohr_free[1]-r_bohr_free[0])*25
        #V_free_deriv = np.abs((V_free(r_star_bohr+dr)-V_free(r_star_bohr-dr))/(2*dr))
        V_free_deriv = np.abs(-V_free(r_star_bohr+2*dr)+8*V_free(r_star_bohr+dr) \
        -8*V_free(r_star_bohr-dr)+V_free(r_star_bohr-2*dr))/(12*dr)

        e_vec_b_f = interp1d(r_bohr_bound, e_vec_b, kind=0, fill_value=0, bounds_error = False)
        #plt.plot(r_bohr_free, e_vec_b_f(r_bohr_free))
        #plt.show()
        #p_E = p_E + proba_v*r_star_bohr*e_vec_b_f(r_star_bohr)**2/V_free_deriv
        #delta_I_eV = E-e_val_b
        #delta_I_eV = E - e_val_b + 0.754
        #delta_I_eV = E - e_val_b - 0.754

        #X.U.: Dans le cas D2+ + gaz résiduel = D2, l'écart est la différence
        #d'énergie entre l'état initial, D2+(v+) +D2(v=0) et l'état final,
        #D2 dissociatif + D2+(v+).
        #Dans l'hypothèse minimale, selon laquelle l'ionisation de D2 se fait
        #vers D2+(v+=0), l'écart en énergie entre état initial et final est la
        #différence d'énergie entre l'état v+ de départ et l'énergie de
        #dissociation finale, moins le potentiel d'ionisation de D2, 15.4 eV environ.

        delta_I_eV = e_val_b-E-15.4
        #print(delta_I_eV)
        #E_res = e_val_b-bounded_state_IP

        #if (e_vec_b_f(r_star_bohr)**2) < 0.01:
        #    print(e_vec_b_f(r_star_bohr)**2)
        
        p_E = p_E + proba_v*franck_condon_matrix[ev_b_idx,ev_f_idx]*q_I(delta_I_eV, e_val_b, q_I_cosh)

        #p_E = p_E + proba_v*e_vec_b_f(r_star_bohr)**2/V_free_deriv*q_I(delta_I_eV, e_val_b)

        #p_E = p_E + proba_v*e_vec_b_f(r_star_bohr)**2/V_free_deriv

        #if E < 0.02:
        #    print("proba_v "+str(proba_v))
        #    print("trans_proba_squared "+str(trans_proba_squared))
        #    print("math.cosh((E-E_res)/gamma) "+str(math.cosh((E-E_res)/gamma)))

    return p_E

#def bound_vib_level_distrib(mol_energy):
#    return 1/eigen_values_bound.size

def under_plot_wave_fun(eigen_vectors, eigen_values, r_bohr, V, plot_ratio=1,
y_axis="energy eV"):

    plt.plot(r_bohr, V(r_bohr), color="black")

    for i in range(0, len(eigen_vectors)):
        if i%plot_ratio == 0:
            psi = eigen_vectors[i]
            #if i < len(eigen_vectors)-1:
            #    delta_E = eigen_values[i+1]-eigen_values[i]
            #else:
            #    delta_E = V(r_bohr[0])*electron_charge/10
            #plt.plot(r_bohr, psi**2/np.linalg.norm(psi**2)+eigen_values[i])
            delta_E = 1
            plt.plot(r_bohr, psi[0:r_bohr.size]/np.linalg.norm(psi)*delta_E+eigen_values[i])

    if y_axis == "none":
        plt.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)

def plot_wave_fun(eigen_vectors, eigen_values, r_bohr, V_eV, plot_ratio=1,
save=False, out_path=None, show=True, figsize_param=(10,10), ylim=None,
y_axis="energy eV", xlabel="Bohr", title=None, useTex=False):
    plt.figure(figsize=figsize_param)
    plt.xlabel(xlabel)
    plt.rc("text", usetex=useTex)
    if y_axis != "none":
        plt.ylabel("eV")

    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])
    
    if title is not None:
        plt.title(title)
    
    under_plot_wave_fun(eigen_vectors, eigen_values, r_bohr, V_eV, plot_ratio, y_axis)
    if save:
        plt.savefig(out_path)
    if show:
    	plt.show()

def plot_bound_and_free(eigen_vectors_bound, eigen_values_bound, r_bohr_bound, V_bound,
eigen_vectors_free, eigen_values_free, r_bohr_free, V_free, plot_ratio_bound=1, plot_ratio_free=1,
    V_free_shift=0, r_bohr_max=float("inf"), save=False, out_path=None):
    plt.xlabel("Bohr")
    plt.ylabel("eV")
    r_bohr_bound = r_bohr_bound[r_bohr_bound <= r_bohr_max]
    r_bohr_free = r_bohr_free[r_bohr_free <= r_bohr_max]
    V_bound_ev = lambda r: V_bound(r)
    under_plot_wave_fun(eigen_vectors_bound, eigen_values_bound, r_bohr_bound, V_bound_ev, plot_ratio_bound)
    V_free_shifted = lambda r: V_free(r)+ V_free_shift
    under_plot_wave_fun(eigen_vectors_free, eigen_values_free, r_bohr_free, V_free_shifted, plot_ratio_free)
    if save:
        plt.savefig(out_path)
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

def build_proba_of_E_fun(proba_of_levels, eigen_values_bound):
    def proba_of_E(E):
        i = np.abs(eigen_values_bound-E).argmin()
        if i >= proba_of_levels.size:
            return 0
        else:
            return proba_of_levels[i]
    return proba_of_E


#Provided by X. Urbain
D2P_vib_levels_distr = np.array([0.080937,
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

H2P_vib_levels_distr = np.array([0.101072,
0.149683,
0.136775,
0.125093,
0.0943612,
0.0778076,
0.0695653,
0.055193,
0.0425999,
0.0349292,
0.0308706,
0.0223014,
0.0204149,
0.0137707,
0.00944833,
0.00648526,
0.0027797,
0.00335822,
0.00349202
])

#pre: eigen values sorted by increasing energies
#     eigen_values_bound size is <= 27
def D2_plus_vib_level_distrib(eigen_values_bound):
    return build_proba_of_E_fun(D2P_vib_levels_distr, eigen_values_bound)

def H2_plus_vib_level_distrib(eigen_values_bound):
    return build_proba_of_E_fun(H2P_vib_levels_distr, eigen_values_bound)

def only_ground_state_vib_level_distrib(eigen_values_bound):
    def proba_of_E(E):
        i = np.abs(eigen_values_bound-E).argmin()
        if i == 0:
            return 1
        else:
            return 0
    #proba_of_E = lambda E: proba_of_levels[np.abs(eigen_values_bound-E).argmin()]
    return proba_of_E



########### In this part will fit exp ker D_2^+ + D^- -> D+D +D ? with theo ker #######################

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

def ker_to_fit(fixed_params, alpha):

    (energies, numerov_res_D2P, numerov_res_D2B, franck_condon_matrix) = fixed_params

    p_e = np.zeros(energies.size)
    for i in range(0, p_e.size):
        p_e[i] = alpha*ker_dissoc(numerov_res_D2P, numerov_res_D2B,
        D2_plus_vib_level_distrib(numerov_res_D2P.eigen_values), energies[i],
        franck_condon_matrix)
    return p_e

def comp_franck_condon_matrix(numerov_res_i, numerov_res_f, q_r = lambda r: 1, show=False):

    (r_bohr_i, V_i, eigen_values_i, eigen_vectors_i) = (numerov_res_i.r_bohr,
    numerov_res_i.V, numerov_res_i.eigen_values, numerov_res_i.eigen_vectors)
    (r_bohr_f, V_f, eigen_values_f, eigen_vectors_f) = (numerov_res_f.r_bohr,
    numerov_res_f.V, numerov_res_f.eigen_values, numerov_res_f.eigen_vectors)

    #Compute numerical domain
    r_left = 0
    r_right = max(r_bohr_i[-1], r_bohr_f[-1])
    dr = min(r_bohr_i[1]-r_bohr_i[0], r_bohr_f[1]-r_bohr_f[0])
    r_bohr = np.linspace(r_left, r_right, (r_right-r_left)/dr)
    dr = (r_bohr[1]-r_bohr[0])*bohr_to_meter

    franck_condon_matrix = np.zeros((len(eigen_vectors_i), len(eigen_vectors_f)))

    q_r_data = q_r(r_bohr)

    for i in range(0,len(eigen_vectors_i)):

        ev_i = interp1d(r_bohr_i, eigen_vectors_i[i],
        kind=0, fill_value=0, bounds_error = False)
        ev_i_data = ev_i(r_bohr)

        for j in range(0,len(eigen_vectors_f)):

            ev_f = interp1d(r_bohr_f, eigen_vectors_f[j],
            kind=0, fill_value=0, bounds_error = False)
            ev_f_data = ev_f(r_bohr)

            franck_condon_matrix[i,j] = wave_fun_scalar_prod(ev_i_data*q_r_data,
            ev_f_data, dr)**2

    if show:
        print("*********")
        print(np.sum(franck_condon_matrix, axis=0))
        fig, axis = plt.subplots()
        heatmap = axis.pcolor(franck_condon_matrix)
        plt.colorbar(heatmap)
        plt.xlabel("D2(GK) vib level indices")
        plt.ylabel("D2+ vib level indices")
        plt.savefig('../plots/fcm.pdf', bbox_inches='tight')
        plt.show()


    return franck_condon_matrix

def final_pop_from_franck_condon(init_pop_dist, franck_condon_matrix):

    nbr_of_i_states = franck_condon_matrix.shape[0]
    nbr_of_f_states = franck_condon_matrix.shape[1]
    final_pop = np.zeros(nbr_of_f_states)

    for i in range(0, nbr_of_f_states):
        for j in range(0, init_pop_dist.size):
            final_pop[i] = final_pop[i] + init_pop_dist[j]*franck_condon_matrix[j,i]

    return final_pop

def energy_diff_matrix(numerov_res_i, numerov_res_f, energy_shift=0):

    (r_bohr_i, V_i, eigen_values_i, eigen_vectors_i) = (numerov_res_i.r_bohr,
    numerov_res_i.V, numerov_res_i.eigen_values, numerov_res_i.eigen_vectors)
    (r_bohr_f, V_f, eigen_values_f, eigen_vectors_f) = (numerov_res_f.r_bohr,
    numerov_res_f.V, numerov_res_f.eigen_values, numerov_res_f.eigen_vectors)

    Ei_minus_Ef = np.zeros((len(eigen_vectors_i), len(eigen_vectors_f)))

    for i in range(0,len(eigen_vectors_i)):
        for j in range(0,len(eigen_vectors_f)):
            Ei_minus_Ef[i,j] = eigen_values_i[i]-eigen_values_f[j]+energy_shift#-0.754

    return Ei_minus_Ef

#Compute ker but for case such as D^- + D_2^+ -> D_2^* + D
#ker is different w.r.t to prev case
def ker(E, bound_vib_level_distrib, numerov_res_i, molecular_state, numerov_res_f,
landau_zener_matrix, Ei_minus_Ef):

    (r_bohr_i, V_i, eigen_values_i, eigen_vectors_i) = (numerov_res_i.r_bohr,
    numerov_res_i.V, numerov_res_i.eigen_values, numerov_res_i.eigen_vectors)
    (r_bohr_f, V_f, eigen_values_f, eigen_vectors_f) = (numerov_res_f.r_bohr,
    numerov_res_f.V, numerov_res_f.eigen_values, numerov_res_f.eigen_vectors)

    ker_of_E = 0
    proba_v_f = bound_vib_level_distrib(eigen_values_i)
    #states_comb_list = []
    #val_list = []

    for i in range(0,len(eigen_vectors_i)):
        for j in range(0,len(eigen_vectors_f)):
            proba_v = proba_v_f(eigen_values_i[i])

            #Sigma of Gaussian such as width at half-height is 0.05eV
            sigma = 0.05/math.sqrt(2*math.log(2))

            #states_comb_list.append((i, j))
            #val_list.append(proba_v*franck_condon_matrix[i, j]* \
            #math.exp(-(E-Ei_minus_Ef[i,j])**2/(2*sigma**2))/0.106447)

            if Ei_minus_Ef[i,j] > 0:

                #new_contrib = proba_v*franck_condon_matrix[i, j]* \
                # new_contrib = proba_v* \
                # math.exp(-(E-Ei_minus_Ef[i,j])**2/(2*sigma**2))/0.106447
                #
                # if enable_landau_zener:
                #     lz = landau_zener(molecular_state, Ei_minus_Ef[i,j], E_As[i,j],
                #     franck_condon_matrix[i, j])
                #     new_contrib = new_contrib*lz
                #
                # ker_of_E = ker_of_E + new_contrib
                ker_of_E = ker_of_E + proba_v* \
                math.exp(-(E-Ei_minus_Ef[i,j])**2/(2*sigma**2))/0.106447* \
                landau_zener_matrix[i,j]
                #landau_zener(H_interaction_matrix[i,j], R_au)

    # val_list = np.array(val_list)
    # max_index = np.argmax(val_list)
    # print("Max contrib for "+str(states_comb_list[max_index]))
    # print("proba_v "+str(proba_v))
    # print("franck_condon_matrix[i, j] "+str(franck_condon_matrix[i, j]))
    # print("Gaussian "+str(math.exp(-(E-Ei_minus_Ef[i,j])**2/(2*sigma**2))/0.106447))
    # print("E "+str(E))
    # print("Ei_minus_Ef[i,j] "+str(Ei_minus_Ef[i,j]))

    return ker_of_E

def H(molecular_state, E_A, R):
    #noramlization constant of the asymptotic radial wavefunction of active
    #electron in initial H^-
    A_i = 1.12
    #Electron binding energy of final state H_2 in H_2^+ + H_^- -> H_2 + H
    #E_A = molecular_state.get_electron_binding_energy()

    L = molecular_state.get_electron_angular_momentum()
    S = molecular_state.get_total_spin()

    #Factor resulting from coupling of initial and final state angular and spin
    #momenta
    D = math.sqrt((2*L+1)*(2*S+1))
    E_A = E_A/au_to_ev
    gamma = np.sqrt(2*E_A)


    #R = au_to_ev/energy

    #R = 30.493

    # normalization constant of asymptotic radial wavefunction of active
    #electron in final H_2
    A_c = np.zeros(gamma.size)
    for i in range(0, gamma.size):
        if (1.0/gamma[i]-L) < 0:
            #print("Warning (1.0/gamma-L) < 0 in landau_zener, A_c set to 0.")
            A_c[i] = 0
        else:
            A_c[i] = gamma[i]*(2*gamma[i])**(1.0/gamma[i])/np.sqrt(
            sp.special.gamma(1.0/gamma[i]+L+1)*sp.special.gamma(1.0/gamma[i]-L))

    #Calcul de H_ic
    H_12 = 0.5*A_i*A_c*D*R**(1.0/gamma-1)*np.exp(-gamma*R)
    return H_12


def comp_landau_zener_matrix(numerov_res_i, molecular_state_f, numerov_res_f,
Ei_minus_Ef_matrix, E_As, formule_id=0):

    (r_bohr_i, V_i, eigen_values_i, eigen_vectors_i) = (numerov_res_i.r_bohr,
    numerov_res_i.V, numerov_res_i.eigen_values, numerov_res_i.eigen_vectors)
    (r_bohr_f, V_f, eigen_values_f, eigen_vectors_f) = (numerov_res_f.r_bohr,
    numerov_res_f.V, numerov_res_f.eigen_values, numerov_res_f.eigen_vectors)

    #Compute numerical domain
    r_left = 0
    r_right = max(r_bohr_i[-1], r_bohr_f[-1])
    dr = min(r_bohr_i[1]-r_bohr_i[0], r_bohr_f[1]-r_bohr_f[0])
    r_bohr = np.linspace(r_left, r_right, (r_right-r_left)/dr)
    dr = (r_bohr[1]-r_bohr[0])*bohr_to_meter

    landau_zener_matrix = np.zeros((len(eigen_vectors_i), len(eigen_vectors_f)))

    for i in range(0,len(eigen_vectors_i)):

        ev_i = interp1d(r_bohr_i, eigen_vectors_i[i],
        kind=0, fill_value=0, bounds_error = False)
        ev_i_data = ev_i(r_bohr)

        for j in range(0,len(eigen_vectors_f)):

            ev_f = interp1d(r_bohr_f, eigen_vectors_f[j],
            kind=0, fill_value=0, bounds_error = False)
            ev_f_data = ev_f(r_bohr)

            #Only FCF
            if formule_id == 0:
                FCF = wave_fun_scalar_prod(ev_i_data, ev_f_data, dr)**2
                landau_zener_matrix[i,j] = FCF

            #Formula 1
            if formule_id == 1:
                R_au = 30.493
                H_ic = 6.06*10**-3
                FCF = wave_fun_scalar_prod(ev_i_data, ev_f_data, dr)**2
                landau_zener_matrix[i,j] = landau_zener(H_ic, R_au)*FCF

            #Formula 2
            if formule_id == 2:
                R_au = 30.493
                H_r = H(molecular_state_f, np.array([E_As[i,j]]), R_au)
                S = wave_fun_scalar_prod(ev_i_data, ev_f_data, dr)
                H_ic = H_r*S
                landau_zener_matrix[i,j] = landau_zener(H_ic, R_au)

            #Formula 3
            if formule_id == 3:
                R_au = au_to_ev/Ei_minus_Ef_matrix[i,j]
                H_r = H(molecular_state_f, np.array([E_As[i,j]]), R_au)
                S = wave_fun_scalar_prod(ev_i_data, ev_f_data, dr)
                H_ic = H_r*S
                landau_zener_matrix[i,j] = landau_zener(H_ic, R_au)

            #Formula 4
            if formule_id == 4:
                R_au = 30.493
                E_A = V_i(r_bohr)-V_f(r_bohr)
                H_r = H(molecular_state_f, E_A, R_au)
                H_ic = wave_fun_scalar_prod(ev_i_data*H_r, ev_f_data, dr)
                landau_zener_matrix[i,j] = landau_zener(H_ic, R_au)

            #Formula 5
            if formule_id == 5:
                R_au = au_to_ev/Ei_minus_Ef_matrix[i,j]
                E_A = V_i(r_bohr)-V_f(r_bohr)
                H_r = H(molecular_state_f, E_A, R_au)
                H_ic = wave_fun_scalar_prod(ev_i_data*H_r, ev_f_data, dr)
                landau_zener_matrix[i,j] = landau_zener(H_ic, R_au)

            #Formula 6
            if formule_id == 6:
                R_au = au_to_ev/Ei_minus_Ef_matrix[i,j]
                H_ic = H(molecular_state_f, np.array([E_As[i,j]]), R_au)
                FCF = wave_fun_scalar_prod(ev_i_data, ev_f_data, dr)**2
                landau_zener_matrix[i,j] = landau_zener(H_ic, R_au)*FCF

    return landau_zener_matrix




def comp_ker_vector(numerov_params_i, molecular_state_f,
vib_level_distrib_i, energies, energy_shift=0, use_cache = True, lz_formule_id=0):

    numerov_params_f = molecular_state_f.get_numerov_params()

    if use_cache:

        L =  molecular_state_f.get_electron_angular_momentum()
        S = molecular_state_f.get_total_spin()
        ker_cache_key = numerov_params_i.to_string()+" "+numerov_params_f.to_string() \
        +" "+vib_level_distrib_i.__name__+" "+str(energies)+" "+str(energy_shift) \
        +" "+str(L)+" "+str(S)

        try:
            print("Checking KER cache...")
            input = open(KER_SAVE_FILE, 'rb')
            ker_cache = pickle.load(input)
            print("KER cache loaded. Searching for key.")
            events_nbr = ker_cache[ker_cache_key]
            input.close()
            return events_nbr
        except IOError:
            print("Unable to load KER cache from file "+KER_SAVE_FILE)
        except KeyError:
            print("Key not found in cache. Computing KER now then...")


    numerov_res_i = numerov(numerov_params_i)
    numerov_res_f = numerov(numerov_params_f)
    #plt.show()
    #numerov_res_f.plot()

    # plot_bound_and_free(numerov_res_i.eigen_vectors, numerov_res_i.eigen_values,
    # numerov_res_i.r_bohr, numerov_res_i.V,
    # numerov_res_f.eigen_vectors, numerov_res_f.eigen_values,
    # numerov_res_f.r_bohr, numerov_res_f.V, plot_ratio_bound=1, plot_ratio_free=1,
    #     V_free_shift=0)



    #vib_level_distrib_i = vib_level_distrib_i(numerov_res_i.eigen_values)
    #franck_condon_matrix = comp_franck_condon_matrix(numerov_res_i,
    #numerov_res_f, q_r_bound_to_bound)
    Ei_minus_Ef_matrix = energy_diff_matrix(numerov_res_i, numerov_res_f,
    energy_shift)
    E_As = electron_binding_energy_of_final_state_matrix(numerov_res_i,
    numerov_res_f)
    landau_zener_matrix = comp_landau_zener_matrix(numerov_res_i, molecular_state_f,
    numerov_res_f, Ei_minus_Ef_matrix, E_As, lz_formule_id)


    #executor = concurrent.futures.ProcessPoolExecutor(4)
    #ker_f = lambda e: ker(e, vib_level_distrib_i, numerov_res_i,
    #molecular_state_f, numerov_res_f, franck_condon_matrix,
    #Ei_minus_Ef_matrix, E_As, enable_landau_zener)
    #futures = [executor.submit(ker_f, energy) for energy in energies]
    #concurrent.futures.wait(futures)


    events_nbr = np.zeros(energies.size)
    for i in range(0,energies.size):
        events_nbr[i] = ker(energies[i], vib_level_distrib_i, numerov_res_i,
        molecular_state_f, numerov_res_f, landau_zener_matrix,
        Ei_minus_Ef_matrix)
        #events_nbr[i] = futures[i].result()

    if use_cache:
        try:
            input = open(KER_SAVE_FILE, 'rb')
            ker_cache = pickle.load(input)
            input.close()
        except:
            ker_cache = {}

        ker_cache[ker_cache_key] = events_nbr

        try:
            print("Save KER results in cache...")
            output = open(KER_SAVE_FILE, "wb")
            pickle.dump(ker_cache, output, pickle.HIGHEST_PROTOCOL)
            output.close()
        except Exception as e:
            print("Pickle failed to save KER results.")
            print(e)

    return events_nbr


#Transition probability p at curve-crossing point R_x,k
def transition_proba_at_one_curve_crossing_pt(R, E, v, b, deltaE_ic, deltaF):

    #print("R "+str(R)+" E "+str(E)+" v "+str(v)+" b "+str(b)+" deltaE_ic "+str(deltaE_ic)+" deltaF "+str(deltaF))

    #Radial relative collision of H_2^+ and H^-
    v_r = v*np.sqrt(1+1/(R*E)-(b/R)**2)
    #print("b "+str(b)+" v_r "+str(v_r))
    #Transition probability p at curve-crossing point R_x,k
    p = np.exp(-math.pi*deltaE_ic**2/(2*v_r*deltaF))
    return p

def proba_for_pop_of_given_exit_channel(R, E, v, b, deltaE_ic, deltaF):
    p = transition_proba_at_one_curve_crossing_pt(R, E, v, b, deltaE_ic, deltaF)

    #if p*(1-p) > 0.01:
    #    print("R "+str(R)+" E "+str(E)+" v "+str(v)+" b "+str(b)+" deltaE_ic "+str(deltaE_ic)+" deltaF "+str(deltaF))
    #    print("p "+str(p))
    #P = 2*np.prod(p[:p.size-1])*(1-p[-1])
    return 2*p*(1-p)
    #return P

#
# Following "Mutual neutralization in slow H_2^+ - H^- collisions", C L Liu,
# H G Wang and R K Janev
#
#def landau_zener(molecular_state, energies, E_A, franck_condon_matrix_el):

    # #noramlization constant of the asymptotic radial wavefunction of active
    # #electron in initial H^-
    # A_i = 1.12
    # #Electron binding energy of final state H_2 in H_2^+ + H_^- -> H_2 + H
    # #E_A = molecular_state.get_electron_binding_energy()
    #
    # L = molecular_state.get_electron_angular_momentum()
    # S = molecular_state.get_total_spin()
    #
    # #Factor resulting from coupling of initial and final state angular and spin
    # #momenta
    # D = math.sqrt((2*L+1)*(2*S+1))
    # E_A = E_A/au_to_ev
    # gamma = np.sqrt(2*E_A)
    #
    #
    #
    # R = au_to_ev/energies
    #
    # #R = 30.493
    #
    # # normalization constant of asymptotic radial wavefunction of active
    # #electron in final H_2
    # if (1.0/gamma-L) < 0:
    #     #print("Warning (1.0/gamma-L) < 0 in landau_zener, A_c set to 0.")
    #     A_c = 0
    # else:
    #     A_c = gamma*(2*gamma)**(1.0/gamma)/np.sqrt(
    #     sp.special.gamma(1.0/gamma+L+1)*sp.special.gamma(1.0/gamma-L))
    #
    # #Calcul de H_ic
    # H_12 = 0.5*A_i*A_c*D*R**(1.0/gamma-1)*np.exp(-gamma*R)*franck_condon_matrix_el


    # if math.isnan(H_12):
    #     print("H_12 "+str(H_12)+" au")
    #     print("A_c "+str(A_c))
    #     print("R "+str(R)+" a0")
    #     print("E_A "+str(E_A*au_to_ev)+" eV")
    #     print("gamma "+str(gamma))

def landau_zener(H_12, R):
    deltaE_ic = 2*H_12
    #Difference of slopes of ionic and covalent potential energy curves
    deltaF = R**-2

    #reduced mass, m_D = 2
    D_mass_SI = 2*m_p
    D_mass = D_mass_SI/m_e
    #D_mass = 2
    mu = 2*D_mass/3

    #Collision velocity
    E = 0.005/(au_to_ev)
    v = math.sqrt(2*E/mu)
    #E = 0.5*mu*v**2

    #b is the impact parameter
    b_upper_bound = R*np.sqrt(1+1/(R*E))


    #cross_sec = np.zeros(R.size)

    #for n in range(0, R.size):

    to_integrate = lambda b: proba_for_pop_of_given_exit_channel(R, E,
    v, b, deltaE_ic, deltaF)*b
    #print(sp.integrate.quad(to_integrate, 0, b_upper_bound[n-1])[0])
    cross_sec = 2*math.pi*sp.integrate.quad(to_integrate, 0, b_upper_bound)[0]

    #return proba_for_pop_of_given_exit_channel(R, E,
    #v, 15, deltaE_ic, deltaF)

    #Initial collision energy
    #E_0 = 0





    #return p*(1-p)
    #print(H_12*au_to_ev)

    #print("H_12 "+str(H_12*au_to_ev)+" eV")
    #print(0.5*A_i*A_c*D)
    #print(0.606*10**-3/au_to_ev/(R**(1/gamma-1))*math.exp(gamma*R))
    #print("gamma "+str(gamma))

    #print("H_12 "+str(H_12))
    return cross_sec

# E_A in the article
def electron_binding_energy_of_final_state_matrix(numerov_res_i, numerov_res_f):

    (r_bohr_i, V_i, eigen_values_i, eigen_vectors_i) = (numerov_res_i.r_bohr,
    numerov_res_i.V, numerov_res_i.eigen_values, numerov_res_i.eigen_vectors)
    (r_bohr_f, V_f, eigen_values_f, eigen_vectors_f) = (numerov_res_f.r_bohr,
    numerov_res_f.V, numerov_res_f.eigen_values, numerov_res_f.eigen_vectors)

    dr_i = (r_bohr_i[1]-r_bohr_i[0])*bohr_to_meter
    dr_f = (r_bohr_f[1]-r_bohr_f[0])*bohr_to_meter


    E_As = np.zeros((len(eigen_vectors_i), len(eigen_vectors_f)))

    for i in range(0,len(eigen_vectors_i)):
        for f in range(0,len(eigen_vectors_f)):

            psi_i = eigen_vectors_i[i]
            psi_f = eigen_vectors_f[f]
            #Compute <Psi | R | Psi > = <R>
            R_i = wave_fun_scalar_prod(psi_i*r_bohr_i, psi_i, dr_i)
            R_f = wave_fun_scalar_prod(psi_f*r_bohr_f, psi_f, dr_f)
            #print(R_f)
            V_i_R = V_i(R_i)
            V_f_R = V_f(R_f)
            E_As[i,f] = V_i_R-V_f_R

    return E_As

def ker_to_file(energies, ker, out_path):
    of = open(out_path, "w")
    of.write("energy(eV) events_nbr\n")
    for i in range(0, energies.size):
        of.write(str(energies[i])+" "+str(ker[i])+"\n")
    of.close()

#
# def pop_from_coef(energies, numerov_params_list,
# alpha1, alpha2, alpha3=0, alpha4=0, alpha5=0):
#
#     ker = ker_vec_fit(energies, a, b, c, alpha1, alpha2) #alpha3, alpha4, alpha5)
#     alphas = [alpha1, alpha2, alpha3, alpha4, alpha5]
#     ker_fun =interp1d(energies, ker,kind=0,fill_value=0, bounds_error = False)
#
#     I, abserr = sp.integrate.quad(ker_fun, energies[0], energies[-1],limit=20000)
#     pop = []
#
#     for i in range(0, len(numerov_params_list)):
#         ker_component = gaussian(a, b, c, energies)* \
#         comp_ker_vector(D2P_NUMEROV_PARAMS, numerov_params_list[i],
#         D2_plus_vib_level_distrib, energies)*alphas[i]
#         ker_comp_fun =interp1d(energies, ker_component, kind=0, fill_value=0,
#         bounds_error = False)
#         I_i, abserr_i= sp.integrate.quad(ker_comp_fun, energies[0], energies[-1],limit=20000)
#         pop.append(I_i/I)
#     return pop
#

#In D2, energy when electron is on orbital n - energy when e- in ground state
def energy_diff_with_ground_state(n):
    return 0.5*(1-1/n**2)*27.21
