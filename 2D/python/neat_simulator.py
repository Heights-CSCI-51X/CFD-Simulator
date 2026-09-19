#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from random import random

class Fluid_Simulator():
    def __init__(self, rows, columns, actual_width, dt):
        self.rows = rows
        self.columns = columns
        self.width = actual_width
        self.height = actual_width * rows / columns
        self.grid = np.ones((rows, columns, 3))
        self.u = np.zeros((rows, columns))
        self.v = np.zeros((rows, columns))
        self.dt = dt
        self.dx = actual_width / columns
        self.dy = self.height / rows
        self.fig = None
        self.ax = None
        self.im = None
        self.plot_extent = [0, self.width, 0, self.height]

    def advection(self):
        u = np.copy(self.u)
        v = np.copy(self.v)
        
        for i in range(1, self.rows - 1):
            for j in range(1, self.columns - 1):

                if u[i, j] > 0:
                    du_dx = (u[i, j] - u[i, j-1]) / self.dx
                else:
                    du_dx = (u[i+1, j] - u[i, j]) / self.dx
                    
                if v[i, j] > 0:
                    du_dy = (u[i, j] - u[i-1, j]) / self.dy
                else:
                    du_dy = (u[i+1, j] - u[i, j]) / self.dy
                    
                self.u[i, j] = u[i, j] - self.dt * (u[i, j] * du_dx + v[i, j] * du_dy)

                if u[i, j] > 0:
                    dv_dx = (v[i, j] - v[i, j-1]) / self.dx
                else:
                    dv_dx = (v[i, j-1] - v[i, j]) / self.dx
                    
                if v[i, j] > 0:
                    dv_dy = (v[i, j] - v[i-1, j]) / self.dy
                else:
                    dv_dy = (v[i-1, j] - v[i, j]) / self.dy
                    
                self.v[i, j] = v[i, j] - self.dt * (u[i, j] * dv_dx + v[i, j] * dv_dy)
    
    def set_grid_color(self, color=[0, 0, 0]):
        for i in range(self.rows):
            for j in range(self.columns):
                self.grid[i, j] = color

    def init(self):
        for i in range(self.rows):
            for j in range(self.columns):
                self.u[i, j] = .2
                self.v[i, j] = .2
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.im = ax.imshow(self.grid, extent=self.plot_extent)

    def render(self):
        clipped_u = np.clip(self.u, 0, 1)
        clipped_v = np.clip(self.v, 0, 1)
        
        for i in range(self.rows):
            for j in range(self.columns):
                self.grid[i, j] = [0, clipped_u[i, j] * .5 , clipped_v[i, j] * .5]
                
        self.im.set_data(self.grid)
        plt.pause(self.dt)

def main():
    simulator = Fluid_Simulator(100, 100, 2, .001)
    simulator.set_grid_color()
    simulator.init()

    r_start, r_end = simulator.rows // 3, 2 * (simulator.rows // 3)
    c_start, c_end = simulator.columns // 3, 2 * (simulator.columns // 3)

    simulator.u[r_start:r_end, c_start:c_end] = 1
    simulator.v[r_start:r_end, c_start:c_end] = 1

    plt.ion()

    for i in range(1000):
        simulator.advection()
        simulator.render()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()