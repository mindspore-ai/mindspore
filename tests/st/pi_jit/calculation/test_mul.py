import pytest
from mindspore import numpy as np
from mindspore import ops
from mindspore import Tensor, jit, context
from ..share.utils import match_array


@jit(mode="PIJit")
def mul(a, b):
    return a * b


@jit
def jit_mul(a, b):
    return a * b


standard_test_cases = [
    ([11], [10]),
    ([10.0], [11.0]),
    ([20], [11.0]),
    ([2.0], [Tensor(np.ones((2, 3)).astype(np.float32))]),
    ([Tensor(ops.fill(np.float32, (2, 3), 8))], [2.0]),
    ([Tensor(ops.fill(np.float32, (2, 3), 8))],
     [Tensor(np.ones((2, 3)).astype(np.float32))]),
    ([(1.0, 2.0, 3.0)], [Tensor(np.ones((2, 3)).astype(np.float32))]),
    ([Tensor(np.ones((2, 3)).astype(np.float32))], [(1.0, 2.0, 3.0)]),
    ([[1.0, 2.0, 3.0]], [Tensor(np.ones((2, 3)).astype(np.float32))]),
    ([Tensor(np.ones((2, 3)).astype(np.float32))], [[1.0, 2.0, 3.0]]),
    ([(1.0, 2.0, 3.0)], [3]),
    ([2], [(1.0, 2.0, 3.0)]),
    ([[1.0, 2.0, 3.0]], [3]),
    ([2], [[1.0, 2.0, 3.0]]),
]

special_test_cases = [
    (["Hello"], [3]),
    ([3], ["Hello"])
]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a, b', standard_test_cases)
def test_standard_mul_operations(func, ms_func, a, b):
    """
    Feature: ALL TO ALL (Except special cases)
    Description: test cases for mul in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a[0], b[0])
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a[0], b[0])
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a, b', special_test_cases)
def test_special_mul_operations(func, ms_func, a, b):
    """
    Feature: Special Cases
    Description: test cases for mul in PYNATIVE mode
    Expectation: an exception is raised
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a[0], b[0])
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = a[0] * b[0]
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
