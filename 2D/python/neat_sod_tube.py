#!usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt

#globals

#this is the heat ratio of air, the medium we will be working with
GAMMA = 1.4

#this is a small number we will need to prevent division by zero
EPSILON = .00000001

#we use this to calculate how little work the simulator can do without crashing
CFL = .5

#This is a simulation of a one dimensional tunnel sending a wave down itself

class Simulator():
    def __init__(self):
        #number of cells
        self.nx = 100
        #number of time steps
        self.nt = 1000
        #cell distance
        self.dx = 1/(self.nx - 1)
        #length of timestep, we will calculate this later
        self.dt = None
        #this is the state vector, u
        self.u_state = np.zeros((3, self.nx))
        #this is the flux vector, the amount of what is stored in the u that is moving out of the cell
        self.f = np.zeros((3, self.nx))
        #rho, the density of the fluid at a given point
        rho = np.where(np.linspace(0, 1, self.nx) < .5, 2.0, 0.1)
        #u, the velocity of the fluid at a given point
        u = np.zeros(self.nx)
        #p, the pressure of the fluid at a given point
        self.p = np.where(np.linspace(0, 1, self.nx) < .5, 2.0, 0.1)
        #e, the total energy in a fluid cell
        e = self.p / (GAMMA - 1) + 1/2 * rho * u ** 2
        #now we set the values of u_state
        self.u_state[:, 0] = rho
        self.u_state[:, 1] = u * rho
        self.u_state[:, 2] = e
        #now we set the values of flux
        self.flux = self.calculate_flux()

    def calculate_flux(self):
        self.f[:, 0] = self.u_state[:, 0]
        self.f[:, 1] = self.u_state[:, 0] * self.u_state[:, 1] * self.u_state[:, 1] + self.p
        self.f[:, 2] = self.u_state[:, 1] * (self.u_state[:, 2] + self.p)

    def calculate_dt(self):
        u = self.u_state[:, 1]
        a = np.sqrt(self.p * GAMMA / self.u_state[:, 0])
        determ = np.abs(u) + a
        a_final = max(determ)
        self.dt = CFL * self.dx / a_final

    def calculate_a(self, i):
        return max(
            (self.u_state[i, 1] / self.u_state[i, 0] + np.sqrt(GAMMA * self.p / self.u_state[i, 0]))
            (self.u_state[i + 1, 1] / self.u_state[i + 1, 0] + np.sqrt(GAMMA * self.p / self.u_state[i + 1, 0]))
        )

def main():
    pass

if __name__ == "__main__":
    main()