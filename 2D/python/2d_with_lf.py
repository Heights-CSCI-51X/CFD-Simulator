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
        #add a vector list to keep track of max speeds for x direction
        self.max_speed_x = np.zeros((self.nx, self.ny))
        #the same for the y
        self.max_speed_y = np.zeros((self.nx, self.ny))
        #this is the value a for any i, j concerning the x direction
        self.ax = np.zeros((self.nx, self.ny))
        #the same for the y
        self.ay = np.zeros((self.nx, self.ny))

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

    def update_max_wave_speeds(self):
        rho = self.u_state[:, :, 0]
        u = self.u_state[:, :, 1] / rho
        v = self.u_state[:, :, 2] / rho
        
        c = np.sqrt(GAMMA * self.p / rho)
        
        self.max_speed_x = np.abs(u) + c
        self.max_speed_y = np.abs(v) + c

    def update_a(self): 
        x_left   = self.max_speed_x[:-2, 1:-1]
        x_center = self.max_speed_x[1:-1, 1:-1]
        x_right  = self.max_speed_x[2:, 1:-1]
        self.ax  = np.maximum.reduce([x_left, x_center, x_right])
        
        y_bottom = self.max_speed_y[1:-1, :-2]
        y_center = self.max_speed_y[1:-1, 1:-1]
        y_top    = self.max_speed_y[1:-1, 2:]
        self.ay  = np.maximum.reduce([y_bottom, y_center, y_top])

    def update_dt(self):
        a_x_max = np.max(self.max_speed_x[1:-1, 1:-1])
        a_y_max = np.max(self.max_speed_y[1:-1, 1:-1])
        
        self.dt = CFL / ((a_x_max / self.dx) + (a_y_max / self.dy))


    #this recalculates p after state is updated
    def update_p(self):
        self.p = (GAMMA - 1) * (self.u_state[:, 2] - 1/2 * (self.u_state[:, 1] ** 2) / self.u_state[:, 0])

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
    simulator.update_flux()
    simulator.update_max_speed()
    simulator.update_a()
    simulator.update_p()

if __name__ == "__main__":
    main()