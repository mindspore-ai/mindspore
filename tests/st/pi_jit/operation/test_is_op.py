import pytest
from mindspore import numpy as np
from mindspore import Tensor, jit, context
from ..share.utils import match_array
from tests.mark_utils import arg_mark


@jit(mode="PIJit")
def add(a, b):
    if a is b:
        return a
    return a + b


def jit_add(a, b):
    if a is b:
        return a
    return a + b


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func, ms_func', [(add, jit_add)])
@pytest.mark.parametrize('a, b', [(1, 1), (1, 2)])
def test_int_add(func, ms_func, a, b):
    """
    Feature: ALL TO ALL
    Description: Test cases for integer addition in PYNATIVE mode.
    Expectation: The results match.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.skip
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func, ms_func', [(add, jit_add)])
@pytest.mark.parametrize('a, b',
                         [(Tensor(np.ones((2, 3)).astype(np.float32)), Tensor(np.ones((2, 3)).astype(np.float32)))])
def test_tensor_add(func, ms_func, a, b):
    """
    Feature: ALL TO ALL
    Description: Test cases for tensor addition in PYNATIVE mode.
    Expectation: The results match.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
