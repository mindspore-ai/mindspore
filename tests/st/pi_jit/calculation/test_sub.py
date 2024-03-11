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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [sub])
@pytest.mark.parametrize('ms_func', [jit_sub])
@pytest.mark.parametrize('a', [11])
@pytest.mark.parametrize('b', [10])
def test_subtraction_int(func, ms_func, a, b):
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

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [sub])
@pytest.mark.parametrize('ms_func', [jit_sub])
@pytest.mark.parametrize('a', [10.0])
@pytest.mark.parametrize('b', [11.0])
def test_subtraction_float(func, ms_func, a, b):
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

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [sub])
@pytest.mark.parametrize('ms_func', [jit_sub])
@pytest.mark.parametrize('a', [20])
@pytest.mark.parametrize('b', [11.0])
def test_subtraction_int_float(func, ms_func, a, b):
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

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [sub])
@pytest.mark.parametrize('ms_func', [jit_sub])
@pytest.mark.parametrize('a', [2.0])
@pytest.mark.parametrize('b', [Tensor(np.ones((2, 3)).astype(np.float32))])
def test_subtraction_float_tensor(func, ms_func, a, b):
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

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [sub])
@pytest.mark.parametrize('ms_func', [jit_sub])
@pytest.mark.parametrize('a', [Tensor(ops.fill(np.float32, (2, 3), 8))])
@pytest.mark.parametrize('b', [2.0])
def test_subtraction_tensor_float(func, ms_func, a, b):
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

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [sub])
@pytest.mark.parametrize('ms_func', [jit_sub])
@pytest.mark.parametrize('a', [Tensor(ops.fill(np.float32, (2, 3), 8))])
@pytest.mark.parametrize('b', [Tensor(np.ones((2, 3)).astype(np.float32))])
def test_subtraction_tensor_tensor(func, ms_func, a, b):
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

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [sub])
@pytest.mark.parametrize('ms_func', [jit_sub])
@pytest.mark.parametrize('a', [(1.0, 2.0, 3.0)])
@pytest.mark.parametrize('b', [Tensor(np.ones((2, 3)).astype(np.float32))])
def test_subtraction_tuple_tensor(func, ms_func, a, b):
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

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [sub])
@pytest.mark.parametrize('ms_func', [jit_sub])
@pytest.mark.parametrize('a', [Tensor(np.ones((2, 3)).astype(np.float32))])
@pytest.mark.parametrize('b', [(1.0, 2.0, 3.0)])
def test_subtraction_tensor_tuple(func, ms_func, a, b):
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

@pytest.mark.skip(reason="GetDevicePtr() error")
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [sub])
@pytest.mark.parametrize('ms_func', [jit_sub])
@pytest.mark.parametrize('a', [[1.0, 2.0, 3.0]])
@pytest.mark.parametrize('b', [Tensor(np.ones((2, 3)).astype(np.float32))])
def test_subtraction_list_tensor(func, ms_func, a, b):
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

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [sub])
@pytest.mark.parametrize('ms_func', [jit_sub])
@pytest.mark.parametrize('a', [Tensor(np.ones((2, 3)).astype(np.float32))])
@pytest.mark.parametrize('b', [[1.0, 2.0, 3.0]])
@pytest.mark.skip(reason="GetDevicePtr is NULL")
def test_subtraction_tensor_list(func, ms_func, a, b):
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
