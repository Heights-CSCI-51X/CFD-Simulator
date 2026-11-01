import hegel
from hegel import generators as gs


@hegel.test
def test_integer_self_equality(tc):
    n = tc.draw(gs.integers())
    assert n == n