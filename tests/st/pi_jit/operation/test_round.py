import pytest
import numpy as onp
from mindspore import Tensor, jit


def match_array(actual, expected, error=0, err_msg=''):

    if isinstance(actual, int):
        actual = onp.asarray(actual)

    if isinstance(actual, Tensor):
        actual = actual.asnumpy()

    if isinstance(expected, (int, tuple)):
        expected = onp.asarray(expected)

    if isinstance(expected, Tensor):
        expected = expected.asnumpy()

    if error > 0:
        onp.testing.assert_almost_equal(
            actual, expected, decimal=error, err_msg=err_msg)
    else:
        onp.testing.assert_equal(actual, expected, err_msg=err_msg)


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
    res = func(x, n)
    ms_res = ms_func(x, n)
    match_array(res, ms_res, error=error, err_msg=str(ms_res))
