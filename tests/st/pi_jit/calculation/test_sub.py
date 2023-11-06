import pytest
from mindspore import numpy as np
from mindspore import ops, Tensor, jit, context
from ..share.utils import match_array


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
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
