import pytest
from mindspore import jit, context, Tensor
from mindspore.common import dtype as mstype
from .share.utils import match_value, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config

# set feature expand graph output on
config = { "expand_graph_output": True, }

def create_nested_tuple(x, y):
    ret = (x, y)
    for index in range(22):
        ret = (ret, x * index, y * index)
    return ret

def run_call_result(x, y, t):
    return (x * 100, y * 100, t)

def run_nested_tuple(x, y, t):
    res = run_call_result(x, y, t)
    return x, y, (x * 10, y * 10), ((x * 20, y * 20), (x * 30, y * 30), res)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_call_result])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
def test_call_result(func, x, y):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_first_input with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    t = create_nested_tuple(x, y)
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(x, y, t)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(x, y, t)
    match_value(ms_res, res)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_nested_tuple])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
def test_nested_tuple(func, x, y):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_first_input with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    t = create_nested_tuple(x, y)
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(x, y, t)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(x, y, t)
    match_value(ms_res, res)
