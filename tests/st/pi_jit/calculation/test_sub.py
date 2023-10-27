import pytest
import numpy as onp
from mindspore import numpy as np
from mindspore import ops, Tensor, jit


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
def sub(a, b):
    return a - b


@jit
def jit_sub(a, b):
    return a - b


combinations = [
    (11, 10),
    (10.0, 11.0),
    (20, 11.0),
    (2.0, Tensor(np.ones((2, 3)).astype(np.float32))),
    (Tensor(ops.fill(np.float32, (2, 3), 8)), 2.0),
    (Tensor(ops.fill(np.float32, (2, 3), 8)),
     Tensor(np.ones((2, 3)).astype(np.float32))),
    ((1.0, 2.0, 3.0), Tensor(np.ones((2, 3)).astype(np.float32))),
    (Tensor(np.ones((2, 3)).astype(np.float32)), (1.0, 2.0, 3.0)),
    ([1.0, 2.0, 3.0], Tensor(np.ones((2, 3)).astype(np.float32))),
    (Tensor(np.ones((2, 3)).astype(np.float32)), [1.0, 2.0, 3.0])
]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [sub])
@pytest.mark.parametrize('ms_func', [jit_sub])
@pytest.mark.parametrize('a, b', combinations)
def test_subtraction_operations(func, ms_func, a, b):
    """
    Feature: ALL TO ALL
    Description: test cases for sub in PYNATIVE mode
    Expectation: the result match
    """
    res = func(a, b)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
