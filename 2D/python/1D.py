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
#the stuff we wamt to advect
U = np.zeroes((3, nx))

U[0, :] = rho
U[1, :] = rho * u
U[2, :] = E