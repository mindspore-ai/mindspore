import pytest
import numpy as onp
from mindspore import numpy as np
from mindspore import Tensor, jit


def match_array(actual, expected, error=0, err_msg=''):
    if isinstance(actual, (int, bool)):
        actual = onp.asarray(actual)
    if isinstance(expected, (int, bool)):
        expected = onp.asarray(expected)
    if error > 0:
        onp.testing.assert_almost_equal(
            actual, expected, decimal=error, err_msg=err_msg)
    else:
        onp.testing.assert_equal(actual, expected, err_msg=err_msg)


@jit(mode="PIJit")
def in_contains(a, b):
    return a in b


@jit
def jit_in_contains(a, b):
    return a in b


@jit(mode="PIJit")
def not_in_contains(a, b):
    return a not in b


@jit
def jit_not_in_contains(a, b):
    return a not in b


@jit(mode="PIJit")
def in_contains_string():
    a = "1"
    b = "123"
    return a in b


@jit
def jit_in_contains_string():
    a = "1"
    b = "123"
    return a in b


@jit(mode="PIJit")
def not_in_contains_string():
    a = "1"
    b = "123"
    return a not in b


@jit
def jit_not_in_contains_string():
    a = "1"
    b = "123"
    return a not in b


def common_test_case(func, ms_func, a, b, error=0, type_check='array'):
    if type_check == 'string':
        res = func()
        ms_res = ms_func()
    else:
        res = func(a, b)
        ms_res = ms_func(a, b)
    match_array(res, ms_res, error=error, err_msg=str(ms_res))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func, ms_func', [(in_contains, jit_in_contains), (not_in_contains, jit_not_in_contains)])
@pytest.mark.parametrize('a', [1])
@pytest.mark.parametrize('b', [[1, 2, 3], {1: 1, 2: 2}, (1, 2, 3)])
def test_in_not_in(func, ms_func, a, b):
    """
    Feature: Test 'in' and 'not in' operators with PIJit and with PSJit
    Description: Validate the behavior of 'in' and 'not in' operators for different types of data structures.
    Expectation: Both should return the same results.
    """
    common_test_case(func, ms_func, a, b)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func, ms_func', [(in_contains, jit_in_contains), (not_in_contains, jit_not_in_contains)])
@pytest.mark.parametrize('a', [Tensor(np.ones((2, 3)).astype(np.float32))])
@pytest.mark.parametrize('b', [Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype='float32'))])
def test_tensor_in_list(func, ms_func, a, b):
    """
    Feature: Test 'in' and 'not in' operators with Tensors and lists
    Description: Validate the behavior of 'in' and 'not in' operators when a Tensor is in a list.
    Expectation: Both PIJit and PSJit functions should return the same results.
    """
    res = func(a, [a, b])
    ms_res = ms_func(a, [a, b])
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func, ms_func', [(in_contains_string, jit_in_contains_string),
                                           (not_in_contains_string, jit_not_in_contains_string)])
def test_string_in_not_in(func, ms_func):
    """
    Feature: Test 'in' and 'not in' operators with strings
    Description: Validate the behavior of 'in' and 'not in' operators when the operands are strings.
    Expectation: Both PIJit and PSJit functions should return the same results.
    """
    common_test_case(func, ms_func, None, None, type_check='string')
