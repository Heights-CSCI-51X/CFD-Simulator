#!/usr/bin/env python3
import hegel
from hegel import generators as gs
import numpy as np
from two_dimensional_simulator import Simulator

@hegel.composite
def random_array_gen(tc, nx, ny):
    flat_data = tc.draw(
        gs.lists(
            gs.floats(min_value=0.0, max_value=1.0), 
            min_size=nx * ny, 
            max_size=nx * ny
        )
    )
    return np.array(flat_data).reshape(nx, ny)

@hegel.test
def test_if_simulator_settles(tc):
    my_simulator = Simulator()
    matrix = tc.draw(random_array_gen(my_simulator.nx, my_simulator.ny))

    assert matrix.shape == (my_simulator.nx, my_simulator.ny)
    assert np.all(matrix >= 0) and np.all(matrix <= 1)

