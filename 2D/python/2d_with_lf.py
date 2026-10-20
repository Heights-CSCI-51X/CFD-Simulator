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

#This is a simulation of a two dimensional wave traveling.

class Simulator():
    def __init__(self):
        #number of cells
        self.nx = 10
        self.ny = 10
        #number of time steps
        self.nt = 100
        #cell distance
        self.dx = 1/(self.nx - 1)
        self.dy = 1/(self.ny - 1)
        #length of timestep, we will calculate this later
        self.dt = None
        #this is the state vector, u
        self.u_state = np.zeros((self.nx, self.ny, 4))
        #rho, the density of the fluid at a given point
        rho_1d = np.where(np.linspace(0, 1, self.nx) < 0.5, 2.0, 0.1)
        rho = np.tile(rho_1d[:, np.newaxis], (1, self.ny))
        #u and v, the velocity of the fluid at a given point
        u = np.zeros((self.nx, self.ny))
        v = np.zeros((self.nx, self.ny))
        #p, the pressure of the fluid at a given point
        p_1d = np.where(np.linspace(0, 1, self.nx) < .5, 2.0, 0.1)
        self.p = np.tile(p_1d[:, np.newaxis], (1, self.ny))
        #e, the total energy in a fluid cell
        e = self.p / (GAMMA - 1) + 1/2 * rho *  (u ** 2 + v ** 2)
        #now we set the values of u_state
        self.u_state[:, :, 0] = rho
        self.u_state[:, :, 1] = u * rho
        self.u_state[:, :, 2] = v * rho
        self.u_state[:, :, 3] = e
        #these are the flux vector, the amount of what is stored in the state that is moving in and out of the cell
        self.f_flux = np.zeros((self.nx, self.ny, 4))
        self.g_flux = np.zeros((self.nx, self.ny, 4))
        #add a vector list to keep track of max speeds
        self.max_speed = np.zeros((self.nx, self.ny))

    #this calculates the flux vector, how fast u_state is moving on average in a cell
    def update_flux(self):
        rho = self.u_state[:, :, 0]
        u = self.u_state[:, :, 1] / rho
        v = self.u_state[:, :, 2] / rho
        E = self.u_state[:, :, 3]

        self.f_flux[:, :, 0] = rho * u
        self.f_flux[:, :, 1] = rho * u**2 + self.p
        self.f_flux[:, :, 2] = rho * u * v
        self.f_flux[:, :, 3] = (E + self.p) * u

        self.g_flux[:, :, 0] = rho * v
        self.g_flux[:, :, 1] = rho * u * v
        self.g_flux[:, :, 2] = rho * v**2 + self.p
        self.g_flux[:, :, 3] = (E + self.p) * v

    #this is a function which dynamically calculates the time step to prevent crashes using CFL
    def calculate_dt(self):
        u = self.u_state[:, 1] / self.u_state[:, 0]
        a = np.sqrt(self.p * GAMMA / self.u_state[:, 0])
        determ = np.abs(u) + a
        a_final = np.max(determ)
        self.dt = CFL * self.dx / a_final

    #this calculates the speed of sound (aka max wave speed) in a certain cell
    def calculate_max_speed(self):
        u = self.u_state[:, :, 1] / self.u_state[:, :, 0]
        a = np.sqrt(GAMMA * self.p / self.u_state[:, :, 0])
        self.max_speed = np.abs(u) + a

    #this recalculates p after state is updated
    def calculate_p(self):
        self.p = (GAMMA - 1) * (self.u_state[:, 2] - 1/2 * (self.u_state[:, 1] ** 2) / self.u_state[:, 0])

    #this advects all the conserved qualities (state)
    # def lf(self):
    #     u = self.u_state.copy()
    #     for i in range(1, self.nx - 1):
    #         #this gets the local top wave speed for the left cell
    #         a_left = self.calculate_a(i - 1)
    #         #this does the same for the right
    #         a_right = self.calculate_a(i)
    #         #this gets the flux on the left side with a little artificial dissapation
    #         flux_left = 1/2 * (self.f[i - 1, :] + self.f[i, :]) - 1/2 * a_left * (self.u_state[i, :] - self.u_state[i - 1, :])
    #         #this does the same for the right
    #         flux_right = 1/2 * (self.f[i, :] + self.f[i + 1, :]) - 1/2 * a_right * (self.u_state[i + 1, :] - self.u_state[i, :])
    #         #averages them and then updates state
    #         u[i, :] = self.u_state[i, :] - (self.dt / self.dx) * (flux_right - flux_left)
    #     #updates the actual state array
    #     self.u_state = u

    def optimized_lf(self):
        u_copy = self.u_state.copy()
        #this will return the max speed across two cells: i & i + 1
        a_interface = np.maximum(self.max_speed[:-1], self.max_speed[1:])
        #now we calculate the flux using supa-efficient numpy!
        flux = 1/2 * (self.f[:-1, :] + self.f[1:, :]) - 1/2 * a_interface[:, np.newaxis] * (self.u_state[1:, :] - self.u_state[:-1, :])
        u_copy[1:-1, :] = self.u_state[1:-1, :] - (self.dt / self.dx) * (flux[1:, :] - flux[:-1, :])
        self.u_state = u_copy

def main():
    #this is a quick demo to make sure stuff works
    simulator = Simulator()
    simulator.calculate_dt()
    simulator.calculate_flux()
    test = simulator.calculate_a(4)
    print(simulator.u_state)
    # for i in range(4):
    #     simulator.optimized_lf()
    #     print(simulator.u_state)

if __name__ == "__main__":
    main()