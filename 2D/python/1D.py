#!usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import math

#initialazation of variables

#number of "cells"
nx = 50
#number of time steps
nt = 100
#dt, this will be adjusted so the simulation does as little math as possible, but still doesn't crash
dt = .01
#distance -- this is needed later for math
dx = 1/(nx - 1)
#this is the heat capacity of air, the simualtor medium
GAMMA = 1.4
#this is a factor to help smooth the time step
CFL = .7

#density
rho = np.where(np.linspace(0, 1, nx) < .5, 1.0, 0.1)
#velocity
u = np.zeros(nx)
#pressure
p = np.where(np.linspace(0, 1, nx) < .5, 1.0, 0.1)

#energy
E = p / (GAMMA - 1) + 1/2 * rho * u ** 2

#the stuff we want to advect
U = np.zeros((3, nx))

#actually set values to state vector
U[0, :] = rho
U[1, :] = u * rho
U[2, :] = E

F = np.zeros((3, nx))

def get_F():
    F[0, :] = U[0, :]
    F[1, :] = U[0, :] * U[1, :] * U[1, :] + p
    F[2, :] = U[1, :] * (U[2, :] + p) 

def weno_Z5():
    stencil1 = 0
    stencil2 = 0
    stencil3 = 0

def calc_dt():
    u = U[1, :]
    scaled_p = p * GAMMA
    a = np.sqrt(scaled_p / U[0, :])
    determ = np.abs(u) + a
    a_final = max(determ)
    dt = CFL * dx / a_final

def calc_a(i):
    a_i = math.sqrt(GAMMA * p[i] / U[0, i])
    a_iplus1 = math.sqrt(GAMMA * p[i + 1] / U[0, i + 1])
    a_return = max((U[1, i] / U[0, i] + a_i), (U[1, i + 1] / U[0, i + 1] + a_iplus1))
    return a_return

#Lax-Friedrichs method instead of weno
def lf():
    u = U.copy()
    for i in range(1, nx - 1):
        a_left = calc_a(i - 1)
        flux_left = 1/2 * (F[:, i - 1] + F[:, i]) + 1/2 * a_left * (U[:, i] - U[:, i - 1])
        a_right = calc_a(i)
        flux_right = 1/2 * (F[:, i] + F[:, i + 1]) + 1/2 * a_right * (U[:, i + 1] - U[:, i])
        u[:, i] = 1/2 * (U[i + 1] + U[i - 1]) - dt * (flux_right - flux_left) / dx
    U = u
         

def main():
    fig, ax = plt.subplots()
    grid = np.ones((nx, 3))
    max_momentum = np.max(np.abs(U[1, :]))
    max_energy = np.max(np.abs(U[2, :]))
    norm_momentum = U[1, :] / max_momentum
    norm_energy = U[2, :] / max_energy
    grid[:, 0] = .1
    grid[:, 1] = (max_momentum + 1) * 0.5
    grid[:, 2] = (max_energy + 1) * 0.5
    im = ax.imshow(grid)
    plt.ion()
    for i in range(nt):
        calc_dt()
        get_F()
        lf()
        im.set_data(grid)
        plt.pause(dt)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()