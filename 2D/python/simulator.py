#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from random import random
import math
import sys

class Fluid_Simulator():
    def __init__(self, rows, columns, actual_width, dt):
        self.rows = rows
        self.columns = columns
        self.width = actual_width
        self.height = actual_width * rows / columns
        self.grid = np.ones((rows, columns, 3))
        self.velocity = np.zeros((rows, columns, 2))
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
        u = self.velocity[:, :, 0].copy()
        v = self.velocity[:, :, 1].copy()

        for i in range(self.rows):
            for j in range(self.columns):
                previous_x = i - self.dt * u[i, j] / self.dx
                previous_y = j - self.dt * v[i, j] / self.dy
                previous_x = max(0, min(self.rows - 1.001, previous_x))
                previous_y = max(0, min(self.columns - 1.001, previous_y))
                x0 = math.floor(previous_x)
                x1 = x0 + 1
                y0 = math.floor(previous_y)
                y1 = y0 + 1
                a = previous_x - x0
                b = previous_y - y0
                velocity_x = (1 - a) * (1 - b)* u[x0, y0] + a * (1 - b) * u[x1, y0] + (1 - a) * b * u[x0, y1] + a * b * u[x1, y1]
                velocity_y = (1 - a) * (1 - b)* v[x0, y0] + a * (1 - b) * v[x1, y0] + (1 - a) * b * v[x0, y1] + a * b * v[x1, y1]
                self.velocity[i, j] = [velocity_x, velocity_y]
                
    def init(self):
        self.velocity[:, :, :] = .1
        fig, ax = plt.subplots()

        def on_close(event):
            sys.exit(0)

        fig.canvas.mpl_connect('close_event', on_close)

        self.fig = fig
        self.ax = ax
        self.im = ax.imshow(self.grid, extent=self.plot_extent)

    def render(self):
        max_vel = np.max(np.abs(self.velocity))

        norm_velocity = self.velocity / max_vel

        self.grid[..., 0] = .1
        self.grid[..., 1] = (norm_velocity[:, :, 0] + 1) * 0.5
        self.grid[..., 2] = (norm_velocity[:, :, 1] + 1) * 0.5

        self.im.set_data(self.grid)
        plt.pause(self.dt)


def main():
    simulator = Fluid_Simulator(100, 100, 2, .05)
    simulator.init()

    r_start, r_end = simulator.rows // 3, 2 * (simulator.rows // 3)
    c_start, c_end = simulator.columns // 3, 2 * (simulator.columns // 3)

    simulator.velocity[r_start:r_end, c_start:c_end, :] = .7

    plt.ion()

    for i in range(500):
        simulator.advection()
        simulator.render()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()