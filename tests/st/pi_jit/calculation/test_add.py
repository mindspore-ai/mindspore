import pytest
import numpy as onp
from mindspore import numpy as np
from mindspore import Tensor, jit


def to_numpy_array(data):
    if isinstance(data, (int, tuple)):
        return onp.asarray(data)
    if isinstance(data, Tensor):
        return data.asnumpy()
    return data


def match_array(actual, expected, error=0, err_msg=''):
    actual = to_numpy_array(actual)
    expected = to_numpy_array(expected)
    if error > 0:
        onp.testing.assert_almost_equal(
            actual, expected, decimal=error, err_msg=err_msg)
    else:
        onp.testing.assert_equal(actual, expected, err_msg=err_msg)


@jit(mode="PIJit")
def add(a, b):
    return a + b


@jit
def jit_add(a, b):
    return a + b


TEST_CASES = [
    (1, 100, int),
    (1.0, 100.0, float),
    (2.0, Tensor(np.ones((2, 3)).astype(np.float32)),
     "float-Tensor"),
    (Tensor(np.ones((2, 3)).astype(np.float32)), Tensor(
        np.ones((2, 3)).astype(np.float32)), "Tensor-Tensor"),
    ((1, 2, 3), (4, 5, 6), tuple),
    ("Hello", "World", str)
]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [add])
@pytest.mark.parametrize('ms_func', [jit_add])
@pytest.mark.parametrize('test_data', TEST_CASES)
def test_add(func, ms_func, test_data):
    """
    Feature:
        Addition Operation Across Different Data Types

    Description:
        Evaluate the addition operation for various data types (e.g., integers, floats, strings)
        using specified functions.

    Expectation:
        Computed result should match the expected result for each data type.
        No errors should occur during execution.
    """
    a, b, data_type = test_data
    res = func(a, b)
    if data_type == str:
        ms_res = a + b
    else:
        ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
