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

#Some constants
NUMEROV_SAVE_PKL_DIR = "cache_pkl"
KER_SAVE_FILE = "cache_pkl/computed_ker.pkl"
Q_R_PATH = "data/q_wrt_r.txt"

POTENTIALS_DIR = "data"
D2P_POTENTIAL_FILE = "pot_d2+.txt"
D2DISSOC_POTENTIAL_FILE = "pot_d2_b.txt"
D2STAR_GK1SG_POTENTIAL_FILE = "gk1sg.dat"
D2STAR_2_3SG_POTENTIAL_FILE = "2_3Sigma_g_truncat"
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


EXP_KER_PATH = "data/ker_d2_d.txt"

#Definition of physical constants
h = 6.626*10**-34 #Planck Constant [J s]
hbar = h/(2*math.pi)
m_p = 1.672*10**-27 #Proton mass [kg]
reduced_mass_d2 = m_p #approximation of D_2 reduced mass [kg]
reduced_mass_he2 = 2*m_p
electron_charge = 1.60217662*10**-19 #coulomb
bohr_to_meter = 5.291772*10**-11
ionisation_potential_d2 = 15.466 #eV
au_to_ev = 27.211396

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

    def __init__(self, r_bohr, V, eigen_values, eigen_vectors):
        self.r_bohr = r_bohr
        self.V = V
        self.eigen_values = eigen_values
        self.eigen_vectors = eigen_vectors

    def plot(self, plot_ratio=1):
        plot_wave_fun(self.eigen_vectors, self.eigen_values, self.r_bohr, self.V, plot_ratio)

    def write_eigen_vector(self, path):
        of = open(path, "w")
        for i in range(0, self.r_bohr.size):
            of.write(str(self.r_bohr[i])+" ")
            for j in range(0, len(self.eigen_vectors)):
                of.write(str(self.eigen_vectors[j][i])+" ")
            of.write("\n")
        of.close()


refine = 2
#pot_file_path, is_bound_potential, E_max, reduced_mass,
#r_end_for_unbound_potential=10, refine=1, pot_in_au, auto_E_max
D2P_NUMEROV_PARAMS = NumerovParams(D2P_POTENTIAL_FILE, True, E_max_bound_states_D2p,
reduced_mass_d2, 10, refine, False, True)
D2B_NUMEROV_PARAMS = NumerovParams(D2DISSOC_POTENTIAL_FILE, False, E_max_bound_states_D2b,
reduced_mass_d2, 10, refine, False, False)
D2STAR_GK1SG_NUMEROV_PARAMS = NumerovParams(D2STAR_GK1SG_POTENTIAL_FILE, True,
E_max_bound_states_D2STAR_GK1SG, reduced_mass_d2, 10, refine, True, True)
D2STAR_2_3SG_NUMEROV_PARAMS = NumerovParams(D2STAR_2_3SG_POTENTIAL_FILE, True,
E_max_bound_states_D2STAR_2_3SG, reduced_mass_d2, 10, refine, True, True)
D2STAR_3_3SG_NUMEROV_PARAMS = NumerovParams(D2STAR_3_3SG_POTENTIAL_FILE, True,
E_max_bound_states_D2STAR_3_3SG, reduced_mass_d2, 10, refine, True, True)
D2STAR_1_SU_B_NUMEROV_PARAMS = NumerovParams(D2STAR_1_SU_B_POTENTIAL_FILE, True,
E_max_bound_states_D2STAR_1_SU_B, reduced_mass_d2, 10, refine, True, True)
D2STAR_1_SU_BP_NUMEROV_PARAMS = NumerovParams(D2STAR_1_SU_BP_POTENTIAL_FILE, True,
0, reduced_mass_d2, 10, refine, True, True)
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

q_r_data = np.loadtxt(Q_R_PATH)
r = q_r_data[:,0]
q = q_r_data[:,1]
q_r = interp1d(r,q,kind="linear",fill_value="extrapolate")

#q_r = lambda x: 1

def gaussian(a, b, c, x):
    return a*np.exp(-((x-b)/c)**2/2)

# <psi1 | psi2> = int_{-\infty}^{\infty} psi1(r)^* psi2(r) dr
def wave_fun_scalar_prod(psi1, psi2, dr):
    return np.dot(psi1,psi2)*dr-dr/2.0*psi1[0]*psi2[0]-dr/2.0*psi1[-1]*psi2[-1]

def normalization_of_continuum_wave_fun(psi, energy, r):
    mu = reduced_mass_d2
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
    #print("ormalization_of_continuum_wave_fun "+str(math.sqrt((A**2+B**2)*math.pi*k)))
    return math.sqrt((A**2+B**2)*math.pi*k)
    #return np.sqrt(k)
    #return 1

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

def numerov(NumerovParams):

    (pot_file_path, is_bound_potential, E_max, reduced_mass,
    r_end_for_unbound_potential, refine, pot_in_au, auto_E_max) = \
    NumerovParams.as_tuple()

    print("Starting Numerov with potential file: "+pot_file_path)

    numerov_save_path = NumerovParams.to_string()+".pkl"

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
        E_min = V(r_inf)
        r_min_of_V = r_inf

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

    n = int(round((V_E_intersect_right-V_E_intersect_left)/dr+5))
    #print("n "+str(n)+" V_E_intersect_right "+str(V_E_intersect_right/bohr_to_meter)+" V_E_intersect_left "+str(V_E_intersect_left/bohr_to_meter)+" dr "+str(dr/bohr_to_meter))

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
    H = (KE + sp.sparse.diags(V(r), offsets=0)).todense()

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

    c = 0
    for ev_fat in eigen_vectors_temp:
        #ev = np.real(ev) #Required using eig for sparce matrix
        ev = ev_fat[0]
        if is_bound_potential:
            eigen_vectors.append(ev/wave_fun_norm(ev, dr))
        else:
            eigen_vectors.append(normalize_continuum_wave_fun(ev, eigen_values[c]-E_min, r))
        c = c+1

    eigen_vectors = np.asarray(eigen_vectors)
    eigen_values = eigen_values/electron_charge#-shift

    print("Eigenvalues:")
    print(eigen_values)

    r_bohr = r/bohr_to_meter
    V = interp1d(r_init,V_init,kind="linear",fill_value="extrapolate")
    res = NumerovResult(r_bohr, V, eigen_values, eigen_vectors)

    #Save Numerov result in cache
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

    delta_E = eigen_values_free[1]-eigen_values_free[0]

    if E < eigen_values_free[0]-delta_E:
       return (0, np.zeros(eigen_vectors_free[0].size))

    # if E < eigen_values_free[0]:
    #     eigen_vec = eigen_vectors_free[0]
    #     r_meter = r_bohr*bohr_to_meter
    #     eigen_vec = eigen_vec*normalization_of_continuum_wave_fun(eigen_vec, eigen_values_free[0], r_meter)
    #     eigen_vec = eigen_vec/normalization_of_continuum_wave_fun(eigen_vec, E, r_meter)
    #     #print("in final dissoc state np.linalg.norm(eigen_vec) "+str(np.linalg.norm(eigen_vec)))
    #     #print(np.linalg.norm(eigen_vec))
    #     return (E, eigen_vec)

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
def ker_dissoc(numerov_res_bound, numerov_res_free, bound_vib_level_distrib, E,
bounded_state_IP, gamma=1.547):

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
    (E_final_state, final_free_state) = final_dissoc_state(eigen_values_free,
    eigen_vectors_free, E, r_bohr_free)
    final_free_statef = interp1d(r_bohr_free, final_free_state,
    kind=0,fill_value=0, bounds_error = False)
    final_free_state = final_free_statef(r_bohr_bound)

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
        trans_proba_squared = wave_fun_scalar_prod(e_vec_b*r_bohr_bound, final_free_state, dr_bound)**2

        #sigma = 0.05/math.sqrt(2*math.log(2))

        #divide by cosh: remove from probability the fact that you can form a
        #bounded D2 molecule 1/cosh((E(eV)-E_res)/1.547)^2
        E_res = e_val_b-bounded_state_IP
        p_E = p_E+proba_v*trans_proba_squared/math.cosh((E-E_res)/gamma)**2

        #*math.exp(-(E-E_final_state)**2/(2*sigma**2))/0.106447

        # print("E "+str(E))
        # print("E final state "+str(E))
        # print(math.exp(-(E-E_final_state)**2/(2*sigma**2))/0.106447)

        # f = lambda x: V_free(x)-E
        # r_star_bohr = sp.optimize.brentq(f, 0, 10)
        #
        # dr = (r_bohr_free[1]-r_bohr_free[0])
        # V_free_deriv = np.abs((V_free(r_star_bohr+dr)-V_free(r_star_bohr))/dr)
        #
        # e_vec_b_f = interp1d(r_bohr_bound, e_vec_b, kind=0,fill_value="extrapolate")
        # p_E = p_E + proba_v*r_star_bohr*e_vec_b_f(r_star_bohr)**2/V_free_deriv
        #p_E = p_E + proba_v*e_vec_b_f(r_star_bohr)**2/V_free_deriv

        #if E < 0.02:
        #    print("proba_v "+str(proba_v))
        #    print("trans_proba_squared "+str(trans_proba_squared))
        #    print("math.cosh((E-E_res)/gamma) "+str(math.cosh((E-E_res)/gamma)))

    return p_E

#def bound_vib_level_distrib(mol_energy):
#    return 1/eigen_values_bound.size

def under_plot_wave_fun(eigen_vectors, eigen_values, r_bohr, V, plot_ratio=1):

    plt.plot(r_bohr, V(r_bohr))

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

def plot_wave_fun(eigen_vectors, eigen_values, r_bohr, V_eV, plot_ratio=1):
    under_plot_wave_fun(eigen_vectors, eigen_values, r_bohr, V_eV, plot_ratio=1)
    plt.show()

def plot_bound_and_free(eigen_vectors_bound, eigen_values_bound, r_bohr_bound, V_bound,
eigen_vectors_free, eigen_values_free, r_bohr_free, V_free, plot_ratio_bound=1, plot_ratio_free=1,
    V_free_shift=0):
    V_bound_ev = lambda r: V_bound(r)
    under_plot_wave_fun(eigen_vectors_bound, eigen_values_bound, r_bohr_bound, V_bound_ev, plot_ratio_bound)
    V_free_shifted = lambda r: V_free(r)+ V_free_shift
    under_plot_wave_fun(eigen_vectors_free, eigen_values_free, r_bohr_free, V_free_shifted, plot_ratio_free)
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

    def proba_of_E(E):
        i = np.abs(eigen_values_bound-E).argmin()
        if i >= proba_of_levels.size:
            return 0
        else:
            return proba_of_levels[i]
    #proba_of_E = lambda E: proba_of_levels[np.abs(eigen_values_bound-E).argmin()]
    return proba_of_E


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

################### END Compute KER D2_+, D2_b #############################


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


ker_exp_f_path = "data/d2_ker_distribution.txt"
ker_exp = np.loadtxt(ker_exp_f_path, skiprows=1)
#Drop values below 0.2eV and above 7ev
ker_exp = ker_exp[ker_exp[:,0] >= 0.5]
ker_exp = ker_exp[ker_exp[:,0] <= 7]

numerov_res_D2P = numerov(D2P_NUMEROV_PARAMS)
numerov_res_D2B = numerov(D2B_NUMEROV_PARAMS)
numerov_res_i = numerov_res_D2P
numerov_res_f = numerov_res_D2B
numerov_res_f.write_eigen_vector("data/free_evec.txt")
#numerov_res_f.plot()

# plot_bound_and_free(numerov_res_i.eigen_vectors, numerov_res_i.eigen_values,
# numerov_res_i.r_bohr, numerov_res_i.V,
# numerov_res_f.eigen_vectors, numerov_res_f.eigen_values,
# numerov_res_f.r_bohr, numerov_res_f.V, plot_ratio_bound=1, plot_ratio_free=1,
#     V_free_shift=0)

def ker_to_fit(energies, alpha, gamma=1.54):

    p_e = np.zeros(energies.size)
    for i in range(0, p_e.size):
        p_e[i] = alpha*ker_dissoc(numerov_res_D2P, numerov_res_D2B,
        D2_plus_vib_level_distrib(numerov_res_D2P.eigen_values), energies[i],
        ionisation_potential_d2, gamma)
    return p_e

energies = ker_exp[:,0]
proba_exp = ker_exp[:,1]

#
res = sp.optimize.curve_fit(ker_to_fit, energies, proba_exp, p0=(1,1))
#res = sp.optimize.curve_fit(ker_to_fit, energies, proba_exp, p0=(35*10**8))#, 3))
#print(res)
#res = [(0.000146889116013, 8.00268053247)]
#(alpha, gamma) = (222521.324352, -4455915.30647)#res[0]
(alpha, gamma) = res[0]
#alpha = res[0]
#
print("minimum error for alpha "+str(alpha)+" and gamma "+str(gamma))
#print(res[1])
#
energies_theo = np.linspace(0.0, 10, 200)
# #energies_theo = np.linspace(0, 0.5, 2000)
proba_theo = []
for e in energies_theo:
    p_e = ker_dissoc(numerov_res_D2P, numerov_res_D2B,
    D2_plus_vib_level_distrib(numerov_res_D2P.eigen_values), e,
    ionisation_potential_d2, gamma)
    proba_theo.append(p_e)
proba_theo = alpha*np.array(proba_theo)
#
plt.plot(energies, proba_exp)
plt.plot(energies_theo, proba_theo)
plt.show()


#q(r)


########################## BOUND state D_2^* potential ########################

E_max = 1-.663091017*27.211

#eigen_values_bound = eigen_values_bound-27.211

#(r_bohr_bound_exc, V_bound_exc, eigen_values_bound_exc, eigen_vectors_bound_exc) = numerov(
#numerov_res_D2STAR_GK1SG = numerov(
#D2STAR_GK1SG_POTENTIAL_FILE, True, E_max, reduced_mass_d2, refine=5, pot_in_au=True)
#eigen_values_bound_exc = eigen_values_bound_exc-.663091017*27.211

#plot_wave_fun(eigen_vectors_bound_exc, eigen_values_bound_exc, r_bohr_bound_exc, V_bound_exc)


def comp_franck_condon_matrix(numerov_res_i, numerov_res_f):

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


    # print("*********")
    # print(np.sum(franck_condon_matrix, axis=0))
    # fig, axis = plt.subplots()
    # heatmap = axis.pcolor(franck_condon_matrix)
    # plt.colorbar(heatmap)
    # plt.show()

    return franck_condon_matrix

def energy_diff_matrix(numerov_res_i, numerov_res_f):

    (r_bohr_i, V_i, eigen_values_i, eigen_vectors_i) = (numerov_res_i.r_bohr,
    numerov_res_i.V, numerov_res_i.eigen_values, numerov_res_i.eigen_vectors)
    (r_bohr_f, V_f, eigen_values_f, eigen_vectors_f) = (numerov_res_f.r_bohr,
    numerov_res_f.V, numerov_res_f.eigen_values, numerov_res_f.eigen_vectors)

    Ei_minus_Ef = np.zeros((len(eigen_vectors_i), len(eigen_vectors_f)))

    for i in range(0,len(eigen_vectors_i)):
        for j in range(0,len(eigen_vectors_f)):
            Ei_minus_Ef[i,j] = eigen_values_i[i]-eigen_values_f[j]-0.754

    return Ei_minus_Ef

#Compute ker but for case such as D^- + D_2^+ -> D_2^* + D
#ker is different w.r.t to prev case
def ker(E, bound_vib_level_distrib, numerov_res_i, numerov_res_f,
franck_condon_matrix, Ei_minus_Ef):

    (r_bohr_i, V_i, eigen_values_i, eigen_vectors_i) = (numerov_res_i.r_bohr,
    numerov_res_i.V, numerov_res_i.eigen_values, numerov_res_i.eigen_vectors)
    (r_bohr_f, V_f, eigen_values_f, eigen_vectors_f) = (numerov_res_f.r_bohr,
    numerov_res_f.V, numerov_res_f.eigen_values, numerov_res_f.eigen_vectors)

    ker_of_E = 0

    #states_comb_list = []
    #val_list = []

    for i in range(0,len(eigen_vectors_i)):
        for j in range(0,len(eigen_vectors_f)):
            proba_v = bound_vib_level_distrib(eigen_values_i[i])
            #Sigma of Gaussian such as width at half-height is 0.05eV
            sigma = 0.05/math.sqrt(2*math.log(2))

            #states_comb_list.append((i, j))
            #val_list.append(proba_v*franck_condon_matrix[i, j]* \
            #math.exp(-(E-Ei_minus_Ef[i,j])**2/(2*sigma**2))/0.106447)

            ker_of_E = ker_of_E + proba_v*franck_condon_matrix[i, j]* \
            math.exp(-(E-Ei_minus_Ef[i,j])**2/(2*sigma**2))/0.106447

    # val_list = np.array(val_list)
    # max_index = np.argmax(val_list)
    # print("Max contrib for "+str(states_comb_list[max_index]))
    # print("proba_v "+str(proba_v))
    # print("franck_condon_matrix[i, j] "+str(franck_condon_matrix[i, j]))
    # print("Gaussian "+str(math.exp(-(E-Ei_minus_Ef[i,j])**2/(2*sigma**2))/0.106447))
    # print("E "+str(E))
    # print("Ei_minus_Ef[i,j] "+str(Ei_minus_Ef[i,j]))

    return ker_of_E


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

####### Now compare ker result with experimental ker #####################

def comp_ker_vector(numerov_params_i, numerov_params_f, vib_level_distrib_i,
energies):

    ker_cache_key = numerov_params_i.to_string()+" "+numerov_params_f.to_string() \
    +" "+vib_level_distrib_i.__name__+" "+str(energies)

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

    #numerov_res_f.plot()

    # plot_bound_and_free(numerov_res_i.eigen_vectors, numerov_res_i.eigen_values,
    # numerov_res_i.r_bohr, numerov_res_i.V,
    # numerov_res_f.eigen_vectors, numerov_res_f.eigen_values,
    # numerov_res_f.r_bohr, numerov_res_f.V, plot_ratio_bound=1, plot_ratio_free=1,
    #     V_free_shift=0)



    vib_level_distrib_i = vib_level_distrib_i(numerov_res_i.eigen_values)

    franck_condon_matrix = comp_franck_condon_matrix(numerov_res_i, numerov_res_f)
    Ei_minus_Ef_matrix = energy_diff_matrix(numerov_res_i, numerov_res_f)

    events_nbr = np.zeros(energies.size)

    for i in range(0,energies.size):
        events_nbr[i] = ker(energies[i], vib_level_distrib_i, numerov_res_i,
        numerov_res_f, franck_condon_matrix, Ei_minus_Ef_matrix)

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



# ker_exp = np.loadtxt(EXP_KER_PATH)
# energies = ker_exp[:,0]
# events_nbr_exp = ker_exp[:,1]
#
# # end_states_no_q_r = [(D2STAR_GK1SG_NUMEROV_PARAMS, 10**2*3, "GK1SG"),
# # (D2STAR_2_3SG_NUMEROV_PARAMS, 150*3/4, "2_3SG"),
# # (D2STAR_3_3SG_NUMEROV_PARAMS, 140, "3_3SG"),
# # (D2STAR_1_SU_B_NUMEROV_PARAMS, 125, "1_SU_B"),
# # (D2STAR_1_SU_BP_NUMEROV_PARAMS, 450, "1_SU_BP"),
# # (D2STAR_1_SU_BPP_NUMEROV_PARAMS, 100, "1_SU_BPP"),
# # (D2STAR_2_1SG_NUMEROV_PARAMS, 90, "2_1SG"),
# # (D2STAR_3_1SG_NUMEROV_PARAMS, 300, "3_1SG"),
# # (D2STAR_4_1SG_NUMEROV_PARAMS, 125, "4_1SG"),
# # (D2STAR_2_3SU_NUMEROV_PARAMS, 125*1.5, "2_3SU"),
# # (D2STAR_3_3SU_NUMEROV_PARAMS, 10, "3_3SU"),
# # (D2STAR_4_3SU_NUMEROV_PARAMS, 125, "4_3SU")]
#
# end_states = [(D2STAR_GK1SG_NUMEROV_PARAMS, 10**2*4*2.7, "GK1SG"),
# (D2STAR_2_3SG_NUMEROV_PARAMS, 150*4/5, "2_3SG"),
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
# (D2STAR_1PI_U_D_NUMEROV_PARAMS,50, "1PI_U_D")]
# #
# # end_states_q_r_quad = [(D2STAR_GK1SG_NUMEROV_PARAMS, 10**2*4*5/6, "GK1SG"),
# # (D2STAR_2_3SG_NUMEROV_PARAMS, 150*8/15, "2_3SG"),
# # (D2STAR_3_3SG_NUMEROV_PARAMS, 20, "3_3SG"),
# # (D2STAR_1_SU_B_NUMEROV_PARAMS, 125, "1_SU_B"),
# # (D2STAR_1_SU_BP_NUMEROV_PARAMS, 233, "1_SU_BP"),
# # (D2STAR_1_SU_BPP_NUMEROV_PARAMS, 10, "1_SU_BPP"),
# # (D2STAR_2_1SG_NUMEROV_PARAMS, 90, "2_1SG"),
# # #(D2STAR_3_1SG_NUMEROV_PARAMS, 300*1.5*2.7, "3_1SG"),
# # (D2STAR_4_1SG_NUMEROV_PARAMS, 70, "4_1SG"),
# # (D2STAR_2_3SU_NUMEROV_PARAMS, 125*3/4, "2_3SU"),
# # (D2STAR_3_3SU_NUMEROV_PARAMS, 10, "3_3SU"),
# # (D2STAR_4_3SU_NUMEROV_PARAMS, 10, "4_3SU"),
# # (D2STAR_1PI_U_C_NUMEROV_PARAMS,50, "1PI_U_C"),
# # (D2STAR_1PI_U_D_NUMEROV_PARAMS,20, "1PI_U_D")]
# #
# #
# def ker_vec_fit(energies, a, b, c, alpha1, alpha2):#, alpha3, alpha4, alpha5):
#
#     return gaussian(a, b, c, energies)*( \
#     comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_GK1SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha1+ \
#     comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_1_SU_BP_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha2)# + \
#     #comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_3_3SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha3 + \
#     #comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_2_3SU_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha4 + \
#     #comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_4_3SU_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha5)
#     #comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_2_3SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha4)
#     # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_2_3SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha3 + \
#     # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_1_SU_B_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha4 + \
#     # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_2_1SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha5 + \
#     # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_4_1SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha6 + \
#     # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_2_3SU_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha7 + \
#     # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_3_3SU_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha8 + \
#     # comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_3_3SG_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)*alpha9
#     #)
#
# energies_fit_idx = np.logical_and(energies > 0.5, energies < 2)
# # energies_fit_idx = np.logical_and(energies > 0.5, energies < 2)
# #
# energies_fit = energies[energies_fit_idx]
# events_nbr_exp_fit = events_nbr_exp[energies_fit_idx]
#
# #Bound for width at mid height of gaussian 1eV
#
# b_bot_bound = 1.15
# b_up_bound = 1.5
# c_up_bound = 1/math.sqrt(2*math.log(2))
#
# res = sp.optimize.curve_fit(ker_vec_fit, energies_fit,
# events_nbr_exp_fit, p0=(1, 1.3, 0.5, 1, 1),
# bounds = ((0,b_bot_bound,0,0,0), (math.inf,b_up_bound,c_up_bound,math.inf,math.inf))
# )
#
# print(res)
# (a, b, c, alpha1, alpha2) = res[0]
#
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
# numerov_params_list = [D2STAR_GK1SG_NUMEROV_PARAMS, D2STAR_1_SU_BP_NUMEROV_PARAMS,
# D2STAR_3_3SG_NUMEROV_PARAMS, D2STAR_2_3SU_NUMEROV_PARAMS, D2STAR_4_3SU_NUMEROV_PARAMS]
#
# pop = pop_from_coef(energies, numerov_params_list, alpha1, alpha2)
# #alpha3, alpha4, alpha5)
#
# print("GK1SG "+str(pop[0]))
# print("1_SU_BP "+str(pop[1]))
# print("3_3SG "+str(pop[2]))
# print("2_3SU "+str(pop[3]))
# print("4_3SU "+str(pop[4]))





#
# plt.plot(energies, events_nbr_exp)
# plt.plot(energies, ker_vec_fit(energies, a, b, c, alpha1, alpha2))#, alpha3, alpha4, alpha5))
# plt.plot(energies, alpha1*gaussian(a, b, c,energies))
# plt.show()

# energy_shift = 0
# plt.plot(energies, events_nbr_exp)
# for (numerov_params, scale_coef, label) in end_states_q_r_quad:
#     events_nbr = comp_ker_vector(D2P_NUMEROV_PARAMS,
#     numerov_params, D2_plus_vib_level_distrib,
#     energies)*scale_coef
#     plt.plot(energies+energy_shift, events_nbr, label=label)
# plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)
#
# plt.show()

#essayer largeur 1, 2, 3 eV entre 0.7 et 2.7 1/e^2 largeur à mi hauter +/- 1eV



#################### DEBUG ZONE ###################
# ker_fit = comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_1_SU_BP_NUMEROV_PARAMS, D2_plus_vib_level_distrib,
# energies_fit)
#
# max_ker_idx = np.argmax(ker_fit)
# max_ker = ker_fit[max_ker_idx]
# max_ker_energy = energies_fit[max_ker_idx]
#
#
# print(ker)
# temp = comp_ker_vector(D2P_NUMEROV_PARAMS, D2STAR_1_SU_BP_NUMEROV_PARAMS, D2_plus_vib_level_distrib, energies)
#
# plt.plot(energies, events_nbr_exp)
# plt.plot(energies, temp*450)
# plt.scatter(max_ker_energy, max_ker*450)
# plt.show()
