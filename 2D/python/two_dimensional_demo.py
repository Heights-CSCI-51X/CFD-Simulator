#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from two_dimensional_simulator import Simulator
import sys

EPSILON = .0000000000000000001

def render(simulator):
    x = simulator.u_state[:, :, 1]
    y = simulator.u_state[:, :, 2]
    p = simulator.p

    normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x) + EPSILON)
    normalized_y = (y - np.min(y)) / (np.max(y) - np.min(y) + EPSILON)
    normalized_p = (p - np.min(p)) / (np.max(p) - np.min(p) + EPSILON)

    simulator.grid[..., 0] = normalized_p
    simulator.grid[..., 1] = normalized_x
    simulator.grid[..., 2] = normalized_y

    simulator.im.set_data(simulator.grid)
    plt.draw()
    plt.pause(simulator.dt)

def main():

    my_simulator = Simulator()
    fig, ax = plt.subplots()

    grid = np.ones((my_simulator.nx, my_simulator.ny, 3))

    im = ax.imshow(grid)

    def on_close(event):
        sys.exit(0)
    fig.canvas.mpl_connect('close_event', on_close)

    

