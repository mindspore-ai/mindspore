import pytest
from mindspore import numpy as np
from mindspore import Tensor, jit, context
import mindspore.ops as ops
from .share.utils import match_array
from tests.mark_utils import arg_mark


def func_impl(a, b):
    x = a[0][0] + b[0][0]
    a = a + x if a[0][0] > b[0][0] else a - x
    return ops.MatMul()(a, b)


@jit(mode="PIJit")
def func_with_anotion(a, b):
    return func_impl(a, b)


@jit
def ms_func_with_anotion(a, b):
    return func_impl(a, b)


def grad_func_impl(x, y, z, func_to_grad):
    x = x * z
    gradient_function = ops.GradOperation()(func_to_grad)
    return gradient_function(x, y)


def grad_func(x, y, z):
    return grad_func_impl(x, y, z, func_with_anotion)


def ms_grad_func(x, y, z):
    return grad_func_impl(x, y, z, ms_func_with_anotion)


tensor_params = [
    (Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1], [2.1, 1.2, 3.3]], dtype=np.float32),
     Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3],
             [2.1, 1.2, 3.3]], dtype=np.float32),
     Tensor(np.array([1.0], np.float32)))
]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func, ms_func', [(func_with_anotion, ms_func_with_anotion)])
@pytest.mark.parametrize('x, y', [(tensor_params[0][0], tensor_params[0][1])])
def test_forward(func, ms_func, x, y):
    """
    Feature: Forward Function Testing
    Description: Test the forward function with different inputs and modes (PYNATIVE_MODE, GRAPH_MODE).
    Expectation: The results of MindSpore and custom functions should match for each mode.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func, ms_func', [(func_with_anotion, ms_func_with_anotion)])
@pytest.mark.parametrize('x, y', [(tensor_params[0][0], tensor_params[0][1])])
def test_forward2(func, ms_func, x, y):
    """
    Feature: Forward Function Testing
    Description: Test the forward function with different inputs and modes (PYNATIVE_MODE, GRAPH_MODE).
    Expectation: The results of MindSpore and custom functions should match for each mode.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms_res = ms_func(x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func, ms_func', [(grad_func, ms_grad_func)])
@pytest.mark.parametrize('x, y, z', tensor_params)
def test_backward(func, ms_func, x, y, z):
    """
    Feature: Backward Function Testing
    Description: Test the backward function with different inputs and modes (PYNATIVE_MODE, GRAPH_MODE).
    Expectation: The results of MindSpore and custom functions should match for each mode.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func, ms_func', [(grad_func, ms_grad_func)])
@pytest.mark.parametrize('x, y, z', tensor_params)
def test_backward2(func, ms_func, x, y, z):
    """
    Feature: Backward Function Testing
    Description: Test the backward function with different inputs and modes (PYNATIVE_MODE, GRAPH_MODE).
    Expectation: The results of MindSpore and custom functions should match for each mode.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms_res = ms_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
