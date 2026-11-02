#!/usr/bin/env python3
import hegel
from hegel import generators as gs
import numpy as np
from two_dimensional_simulator import Simulator
from matplotlib import pyplot as plt

@hegel.composite
def random_array_gen(tc, nx, ny):
    flat_data = tc.draw(
        gs.lists(
            gs.floats(min_value=0.01, max_value=0.5), 
            min_size=nx * ny, 
            max_size=nx * ny
        )
    )
    return np.array(flat_data).reshape(nx, ny)

@hegel.test
def test_if_simulator_settles(tc):
    nx = ny = tc.draw(gs.integers(min_value=8, max_value=16))
    # ny = tc.draw(gs.integers(min_value=8, max_value=32))
    my_simulator = Simulator(nx, ny)
    my_simulator.u_state[:, :, 0] = tc.draw(random_array_gen(my_simulator.nx, my_simulator.ny))
    
    max_history = []

    test_steps = 2500 
    
    for i in range(test_steps):
        my_simulator.optimized_lf()
        
        current_max = np.max(my_simulator.u_state[:, :, 0])
        
        if np.isnan(current_max) or current_max > 10.0: 
            assert False, f"Simulator diverged at step {i} with max value {current_max}"
            
        if i >= (test_steps - 100):
            max_history.append(current_max)

    settling_variance = np.var(max_history)

    assert settling_variance < 1e-2, f"Variance over last 100 steps: {settling_variance}"