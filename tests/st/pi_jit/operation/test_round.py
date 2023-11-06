import pytest
from mindspore import jit, context
from ..share.utils import match_array


@jit(mode="PIJit")
def fallback_round(x, n=None):
    return round(x, n)


@jit
def jit_fallback_round(x, n=None):
    return round(x, n)


test_data = [
    (10.678, None, 0),
    (10.678, 0, 0),
    (10.678, 1, 5),
    (10.678, -1, 0),
    (17.678, -1, 0)
]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func, ms_func', [(fallback_round, jit_fallback_round)])
@pytest.mark.parametrize('x, n, error', test_data)
def test_round_operations(func, ms_func, x, n, error):
    """
    Feature: ALL TO ALL
    Description: test cases for round in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, n)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(x, n)
    match_array(res, ms_res, error=error, err_msg=str(ms_res))
