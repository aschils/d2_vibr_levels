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

V = lambda r: np.absolute(r)

E_max = 20

st = E_max  #sp.find_roots(V, E_max)

ds = 1/math.sqrt(2*E_max)

n = round(2*(st/ds + 4*math.pi))
i = np.arange(1,n+1)
s = -ds*(n+1)/2.0+ds*i

ones = np.ones(n)
ones_short = np.ones(n-1)
B = (sp.sparse.diags(ones_short, offsets=-1)+10*sp.sparse.diags(ones, offsets=0)+sp.sparse.diags(ones_short, offsets=1))/12.0
A = (sp.sparse.diags(ones_short, offsets=-1)-2*sp.sparse.diags(ones, offsets=0)+sp.sparse.diags(ones_short, offsets=1))/ds**2
KE = -sp.linalg.inv(B.todense()).dot(A.todense())/2.0

H = KE + sp.sparse.diags(V(s), offsets=0)
w,v = np.linalg.eig(H)

w = np.sort(w)

print(np.sort(w))

#print(v)
