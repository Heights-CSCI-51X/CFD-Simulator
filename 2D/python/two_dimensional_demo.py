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
    x = simulator.u_state[:, :, 1]
    y = simulator.u_state[:, :, 2]
    p = simulator.p
    e = simulator.u_state[:, :, 3]

    normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x) + EPSILON)
    normalized_y = (y - np.min(y)) / (np.max(y) - np.min(y) + EPSILON)
    normalized_p = (p - np.min(p)) / (np.max(p) - np.min(p) + EPSILON)
    normalized_e = (e - np.min(e)) / (initial_max_e - initial_min_e + EPSILON)

    normalized_e = np.clip(normalized_e, 0, 1)

    grid[..., 0] = 0 # normalized_p
    grid[..., 1] = 0 # normalized_x
    grid[..., 2] = normalized_e # normalized_y

    plt.draw()
    plt.pause(simulator.dt / 6)

def main():

    my_simulator = Simulator(50, 50)

    print(random_array_gen(4, 4, .01, .011))

    # my_simulator.u_state[:, :, 1] = random_array_gen(my_simulator.nx, my_simulator.ny, .01, 0.011)

    my_simulator.update()

    initial_max_e = np.max(my_simulator.u_state[:, :, 3])

    initial_min_e = np.min(my_simulator.u_state[:, :, 3])

    # my_simulator = Simulator(8, 8)

    fig, ax = plt.subplots()

    grid = np.ones((my_simulator.nx, my_simulator.ny, 3))

    im = ax.imshow(grid)

    def on_close(event):
        sys.exit(0)

    fig.canvas.mpl_connect('close_event', on_close)

    plt.ion()
    
    for i in range(my_simulator.nt):
        my_simulator.optimized_lf()
        im.set_data(grid)
        render(my_simulator, grid, initial_max_e, initial_min_e)
        # print(np.max(my_simulator.u_state[:, :, 3]))
            

    plt.ioff()
    plt.show()
    print("Done!")

if __name__ == "__main__":
    main()