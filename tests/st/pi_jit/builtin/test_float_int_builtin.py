import pytest
import numpy as onp
from mindspore import jit


def match_array(actual, expected, error=0, err_msg=''):
    if isinstance(actual, (int, float, tuple)):
        actual = onp.asarray(actual)
    if isinstance(expected, (int, float, tuple)):
        expected = onp.asarray(expected)

    if error > 0:
        onp.testing.assert_almost_equal(
            actual, expected, decimal=error, err_msg=err_msg)
    else:
        onp.testing.assert_equal(actual, expected, err_msg=err_msg)


def convert_numbers_to_float_and_int(val1, val2):
    return float(val1), float(val2), int(val1), int(val2)


@jit(mode="PIJit")
def fallback_float_and_int():
    return convert_numbers_to_float_and_int(5, 5.0)


@jit(mode="PIJit")
def fallback_float_and_int_empty():
    return float(), int()


@jit
def ms_fallback_float_and_int():
    return convert_numbers_to_float_and_int(5, 5.0)


@jit
def ms_fallback_float_and_int_empty():
    return float(), int()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [fallback_float_and_int])
@pytest.mark.parametrize('ms_func', [ms_fallback_float_and_int])
def test_int_float_conversion_with_args(func, ms_func):
    """
    Feature: Conversion of int and float
    Description: Test cases for argument support in PYNATIVE mode
    Expectation: Results match between GraphJit and JIT functions
    """
    res = func()
    ms_res = ms_func()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [fallback_float_and_int_empty])
@pytest.mark.parametrize('ms_func', [ms_fallback_float_and_int_empty])
def test_int_float_conversion_no_args(func, ms_func):
    """
    Feature: Conversion of int and float without arguments
    Description: Test cases for argument support in PYNATIVE mode
    Expectation: Results match between GraphJit and JIT functions
    """
    res = func()
    ms_res = ms_func()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
