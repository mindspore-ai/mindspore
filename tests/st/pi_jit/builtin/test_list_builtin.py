import pytest
import numpy as onp
from mindspore import Tensor, jit, context
from ..share.utils import match_array
from tests.mark_utils import arg_mark


@jit(mode="PIJit")
def fallback_list_with_input_tuple(a):
    res = list(a)
    return res


@jit(mode="PIJit")
def fallback_list_with_input_dict(a):
    res = list(a)
    return res


@jit(mode="PIJit")
def fallback_list_with_input_numpy_array(a):
    res = list(a)
    return res


@jit(mode="PIJit")
def fallback_list_with_input_numpy_tensor(a, b):
    res = list(a)
    res2 = list(b)
    res3 = list(())
    return res, res2, res3


@jit
def ms_fallback_list_with_input_tuple(a):
    res = list(a)
    return res


@jit
def ms_fallback_list_with_input_dict(a):
    res = list(a)
    return res


@jit
def ms_fallback_list_with_input_numpy_array():
    a = onp.array([1, 2, 3])
    res = list(a)
    return res


@jit
def ms_fallback_list_with_input_numpy_tensor(a, b):
    res = list(a)
    res2 = list(b)
    res3 = list(())
    return res, res2, res3


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [fallback_list_with_input_tuple])
@pytest.mark.parametrize('ms_func', [ms_fallback_list_with_input_tuple])
@pytest.mark.parametrize('a', [(1, 2, 3)])
def test_list_with_input_tuple(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    1. Test list() in PYNATIVE mode
    2. give the input data: tuple'''
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [fallback_list_with_input_dict])
@pytest.mark.parametrize('ms_func', [ms_fallback_list_with_input_dict])
@pytest.mark.parametrize('a', [{'a': 1, 'b': 2, 'c': 3}])
def test_list_with_input_dict(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    1. Test list() in PYNATIVE mode
    2. give the input data: dict'''
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.skip(reason="pynative mode mix graph mode, results has an random error in pynative")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [fallback_list_with_input_numpy_array])
@pytest.mark.parametrize('ms_func', [ms_fallback_list_with_input_numpy_array])
@pytest.mark.parametrize('a', [onp.array([1, 2, 3])])
def test_list_with_input_array(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for builtin list function support in PYNATIVE mode
    Expectation: the result match
    1. Test list() in PYNATIVE mode
    2. give the input data: numpy array'''
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.skip
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [fallback_list_with_input_numpy_tensor])
@pytest.mark.parametrize('ms_func', [ms_fallback_list_with_input_numpy_tensor])
@pytest.mark.parametrize('a', [Tensor([1, 2])])
@pytest.mark.parametrize('b', [[Tensor([1, 2]), Tensor([2, 3])]])
def test_list_with_input_tensor(func, ms_func, a, b):
    """
    Feature: ALL TO ALL
    Description: test cases for builtin list function support in PYNATIVE mode
    Expectation: the result match
    1. Test list() in PYNATIVE mode
    2. give the input data: tensor and (); output tuple
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res[0], ms_res[0], error=0, err_msg=str(ms_res))
    match_array(res[1], ms_res[1], error=0, err_msg=str(ms_res))
    match_array(res[2], ms_res[2], error=0, err_msg=str(ms_res))
