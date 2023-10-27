import pytest
import numpy as onp
from mindspore import numpy as np
from mindspore import ops
from mindspore import Tensor, jit


def match_array(actual, expected, error=0, err_msg=''):
    """Assert two arrays are equal with a certain error."""
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
def div(a, b):
    """Divide two numbers using GraphJit."""
    return a / b


@jit
def jit_div(a, b):
    """Divide two numbers using jit."""
    return a / b


TEST_CASES = [
    (11, 10, 5),
    (10.0, 11.0, 5),
    (20, 11.0, 5),
    (2.0, Tensor(np.ones((2, 3)).astype(np.float32)), 0),
    (Tensor(ops.fill(np.float32, (2, 3), 8)), 2.0, 0),
    (Tensor(ops.fill(np.float32, (2, 3), 8)), Tensor(
        np.ones((2, 3)).astype(np.float32)), 0),
    ((1.0, 2.0, 3.0), Tensor(np.ones((2, 3)).astype(np.float32)), 0),
    (Tensor(np.ones((2, 3)).astype(np.float32)), (1.0, 2.0, 3.0), 0),
    ([1.0, 2.0, 3.0], Tensor(np.ones((2, 3)).astype(np.float32)), 0),
    (Tensor(np.ones((2, 3)).astype(np.float32)), [1.0, 2.0, 3.0], 0)
]


class TestDiv:
    funcs = [div]
    ms_funcs = [jit_div]

    @staticmethod
    def generic_div_test(func, ms_func, a, b, error=0):
        """
        Feature: ALL TO ALL
        Description: test cases for div in PYNATIVE mode
        Expectation: the result match
        """
        res = func(a, b)
        ms_res = ms_func(a, b)
        match_array(res, ms_res, error=error, err_msg=str(ms_res))

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize('func', [f for f in funcs])
    @pytest.mark.parametrize('ms_func', [mf for mf in ms_funcs])
    @pytest.mark.parametrize('test_data', TEST_CASES)
    def test_div_cases(self, func, ms_func, test_data):
        a, b, error = test_data
        self.generic_div_test(func, ms_func, a, b, error)
