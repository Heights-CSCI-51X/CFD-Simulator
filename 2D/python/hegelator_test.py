#!/usr/bin/env python3
import hegel
from hegel import generators as gs


@hegel.test
def test_integers_always_below_50(tc):
    n = tc.draw(gs.integers())
    assert n < 50