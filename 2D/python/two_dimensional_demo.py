#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from two_dimensional_simulator import Simulator
import sys
import random

EPSILON = .0000000000000000001

def random_array_gen(nx, ny, min, max):
    return np.random.uniform(min, max, (nx, ny))

def render(simulator, grid, initial_max_e, initial_min_e):
    e = simulator.u_state[:, :, 3]

    normalized_e = (e - np.min(e)) / (initial_max_e - initial_min_e + EPSILON)
    normalized_e = np.clip(normalized_e, 0, 1)

    grid[..., 2] = normalized_e # normalized_y

    plt.draw()
    plt.pause(simulator.dt)

def main():

    my_simulator = Simulator(75, 75)

    my_simulator.nt = 1000000000

    my_simulator.update()

    initial_max_e = np.max(my_simulator.u_state[:, :, 3])

    initial_min_e = np.min(my_simulator.u_state[:, :, 3])

    fig, ax = plt.subplots()

    grid = np.zeros((my_simulator.nx, my_simulator.ny, 3))

    im = ax.imshow(grid)

    def on_click(event):
        if event.xdata is not None and event.ydata is not None:

            click_x = int(round(event.xdata))
            click_y = int(round(event.ydata))

            radius = 5
            high_energy_value = initial_max_e * .1

            y, x = np.ogrid[:my_simulator.nx, :my_simulator.ny]
            dist_from_center = np.sqrt((x - click_x)**2 + (y - click_y)**2)

            mask = dist_from_center <= radius

            my_simulator.u_state[mask, 3] = high_energy_value
    
    fig.canvas.mpl_connect('button_press_event', on_click)

    def on_close(event):
        sys.exit(0)

    fig.canvas.mpl_connect('close_event', on_close)

    plt.ion()
    
    for i in range(my_simulator.nt):
        my_simulator.optimized_lf()
        im.set_data(grid)
        render(my_simulator, grid, initial_max_e, initial_min_e)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()