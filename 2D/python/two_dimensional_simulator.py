#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import sys
import math

#globals

#this is the heat ratio of air, the medium we will be working with
GAMMA = 1.4

#this is a small number we will need to prevent division by zero
EPSILON = .000000000000000000000001

#we use this to calculate how little work the simulator can do without crashing
CFL = .3

#This is a simulation of a two dimensional wave traveling.

class Simulator():
    def __init__(self):
        #number of cells
        self.nx = 1000
        self.ny = 1000
        #number of time steps
        self.nt = 10000000
        #cell distance
        self.dx = 1/(self.nx - 1)
        self.dy = 1/(self.ny - 1)
        #length of timestep, we will calculate this later
        self.dt = None
        #this holds the values of our rendering array
        self.grid = np.ones((self.nx, self.ny, 3))
        #this is the state vector, u
        self.u_state = np.zeros((self.nx, self.ny, 4))
        #let us define the intervals were our unnatrually square wave will lie
        r_start, r_end = self.ny // 3, 2 * (self.ny // 3)
        c_start, c_end = self.nx // 3, 2 * (self.nx // 3)
        #rho, the density of the fluid at a given point
        rho = np.zeros((self.nx, self.ny)) + EPSILON
        #p, the pressure of the fluid at a given point
        self.p = np.zeros((self.nx, self.ny)) + EPSILON
        #u and v, the velocity of the fluid at a given point
        u = np.zeros((self.nx, self.ny)) + EPSILON
        v = np.zeros((self.nx, self.ny)) + EPSILON
        #let us set the velocity to these strange values
        rho[r_start:r_end, c_start:c_end] = .1
        u[r_start:r_end, c_start:c_end] = .1
        v[r_start:r_end, c_start:c_end] = .1
        self.p[r_start:r_end, c_start:c_end] = .1
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
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.im = ax.imshow(self.grid)
        def on_close(event):
            sys.exit(0)
        self.fig.canvas.mpl_connect('close_event', on_close)

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

    def update_max_speed(self):
        rho = self.u_state[:, :, 0]
        u = self.u_state[:, :, 1] / rho
        v = self.u_state[:, :, 2] / rho
        
        c = np.sqrt(GAMMA * np.abs(self.p) / np.abs(rho))
        
        self.max_speed_x = np.abs(u) + c
        self.max_speed_y = np.abs(v) + c

    def update_a(self): 
        self.ax = np.maximum(self.max_speed_x[:-1, :], self.max_speed_x[1:, :])
        self.ay = np.maximum(self.max_speed_y[:, :-1], self.max_speed_y[:, 1:])

    def update_dt(self):
        a_x_max = np.max(self.max_speed_x[1:-1, 1:-1])
        a_y_max = np.max(self.max_speed_y[1:-1, 1:-1])

        dt = CFL / ((a_x_max / self.dx) + (a_y_max / self.dy) + EPSILON)
        
        self.dt = min(dt, 1)


    #this recalculates p after state is updated
    def update_p(self):
        self.p = (GAMMA - 1) * (self.u_state[:, :, 3] - 1/2 * (self.u_state[:, :, 1] ** 2 + self.u_state[:, :, 2] ** 2) / self.u_state[:, :, 0])

    def optimized_lf(self):
        self.update_p()
        self.update_max_speed() 
        self.update_dt()
        self.update_flux()
        self.update_a()

        #now we calculate the flux using supa-efficient numpy!
        flux_at_the_walls_x = 1/2 * (self.f_flux[:-1, :, :] + self.f_flux[1:, :, :]) - 1/2 * self.ax[:, :, np.newaxis] * (self.u_state[1:, :, :] - self.u_state[:-1, :, :])
        flux_at_the_walls_y = 1/2 * (self.g_flux[:, :-1, :] + self.g_flux[:, 1:, :]) - 1/2 * self.ay[:, :, np.newaxis] * (self.u_state[:, 1:, :] - self.u_state[:, :-1, :])

        #this is the average of the four cells adjacent to a given cell
        average = 1/4 * (self.u_state[2:, 1:-1, :] + self.u_state[:-2, 1:-1, :] + self.u_state[1:-1, 2:, :] + self.u_state[1:-1, :-2, :])

        #this is the artificial dissapation to prevent crashes
        artificial_dissipation = (self.dt / self.dx) * (flux_at_the_walls_x[1:, 1:-1, :] - flux_at_the_walls_x[:-1, 1:-1, :]) + (self.dt / self.dy) * (flux_at_the_walls_y[1:-1, 1:, :] - flux_at_the_walls_y[1:-1, :-1, :])

        #this adjusts the boundary (which we cant compute normally)
        self.u_state[0, :, :] = self.u_state[1, :, :]
        self.u_state[-1, :, :] = self.u_state[-2, :, :]
        self.u_state[:, 0, :] = self.u_state[:, 1, :]
        self.u_state[:, -1, :] = self.u_state[:, -2, :]

        #this adjusts everything else
        self.u_state[1:-1, 1:-1, :] = average - artificial_dissipation

    def render(self):
        x = self.u_state[:, :, 1]
        y = self.u_state[:, :, 2]
        p = self.p

        normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x))
        normalized_y = (y - np.min(y)) / (np.max(y) - np.min(y))
        normalized_p = (p - np.min(p)) / (np.max(p) - np.min(p))

        self.grid[..., 0] = normalized_p
        self.grid[..., 1] = normalized_x
        self.grid[..., 2] = normalized_y

        self.im.set_data(self.grid)
        plt.draw()
        plt.pause(self.dt)

def main():
    #this is a quick demo to make sure stuff works
    simulator = Simulator()
    simulator.optimized_lf()

    plt.ion()

    for i in range(simulator.nt):
        simulator.optimized_lf()
        simulator.render()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()