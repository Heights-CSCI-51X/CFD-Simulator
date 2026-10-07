import numpy as np
from matplotlib import pyplot as plt

#initialazation of variables

#number of "cells"
nx = 50
#distance -- this is needed later for math
dx = 1/(nx - 1)
#this is the heat capacity of air, the simualtor medium
GAMMA = 1.4

#density
rho = np.where(np.linespace(0, 1, nx) < .5, 1.0, 0.1)
#velocity
u = np.zeroes()
#pressure
p = np.where(np.linespace(0, 1, nx) < .5, 1.0, 0.1)

#energy
E = p / (GAMMA - 1) + 1/2 * rho * u ** 2

#the stuff we want to advect
U = np.zeroes((3, nx))

#actually set values to state vector
U[0, :] = rho
U[1, :] = u * rho
U[2, :] = E

F = np.zeroes((3, nx))

def get_F():
    F[0, :] = U[0, :]
    F[1, :] = U[0, :] * U[1, :] * U[1, :] + p
    F[2, :] = U[1, :] * (U[2, :] + p) 

def weno_Z5():
    stencil1 = 0
    stencil2 = 0
    stencil3 = 0

def lr():
    for i in range(1, nx - 1):
        grad = 

