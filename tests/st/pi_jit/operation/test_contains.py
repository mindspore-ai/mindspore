import pytest
import numpy as np
from mindspore import Tensor, jit, context
from ..share.utils import match_array, assert_executed_by_graph_mode, pi_jit_with_config
from tests.mark_utils import arg_mark


jit_cfg = {'compile_with_try': False}


@pi_jit_with_config(jit_config=jit_cfg)
def pijit_in(a, b):
    return a in b


def pynative_in(a, b):
    return a in b


@pi_jit_with_config(jit_config=jit_cfg)
def pijit_not_in(a, b):
    return a not in b


def pynative_not_in(a, b):
    return a not in b


def common_test_case(func, ms_func, a, b, error=0, type_check='array'):
    context.set_context(mode=context.PYNATIVE_MODE)
    if type_check == 'string':
        ms_res = ms_func(a, b)
        res = func(a, b)
    else:
        ms_res = ms_func(a, b)
        res = func(a, b)
    match_array(res, ms_res, error=error, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func, ms_func', [(pijit_in, pynative_in), (pijit_not_in, pynative_not_in)])
@pytest.mark.parametrize('a', [1, 0])
@pytest.mark.parametrize('b', [[1, 2, 3], {1: 1, 2: 2}, (1, 2, 3)])
def test_in_not_in(func, ms_func, a, b):
    """
    Feature: Test 'in' and 'not in' operators with PIJit and with pynative
    Description: Validate the behavior of 'in' and 'not in' operators for different types of data structures.
    Expectation: Both should return the same results.
    """
    common_test_case(func, ms_func, a, b)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func, ms_func', [(pijit_in, pynative_in), (pijit_not_in, pynative_not_in)])
@pytest.mark.parametrize('a', [Tensor(np.ones((2, 3)).astype(np.float32))])
@pytest.mark.parametrize('b', [Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype='float32'))])
def test_tensor_in_list(func, ms_func, a, b):
    """
    Feature: Test 'in' and 'not in' operators with Tensors and lists
    Description: Validate the behavior of 'in' and 'not in' operators when a Tensor is in a list.
    Expectation: Both PIJit and PSJit functions should return the same results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, [a, b])
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, [a, b])
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func, ms_func', [(pijit_in, pynative_in), (pijit_not_in, pynative_not_in)])
@pytest.mark.parametrize('a', ['123', '012'])
@pytest.mark.parametrize('b', ['123-456', '456-789'])
def test_string_in_not_in(func, ms_func, a, b):
    """
    Feature: Test 'in' and 'not in' operators with strings
    Description: Validate the behavior of 'in' and 'not in' operators when the operands are strings.
    Expectation: Both PIJit and PSJit functions should return the same results.
    """
    common_test_case(func, ms_func, a, b, type_check='string')
