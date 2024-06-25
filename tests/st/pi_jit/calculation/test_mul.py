import pytest
from mindspore import numpy as np
from mindspore import ops
from mindspore import Tensor, jit, context
from ..share.utils import match_array
from tests.mark_utils import arg_mark


@jit(mode="PIJit")
def mul(a, b):
    return a * b


@jit
def jit_mul(a, b):
    return a * b

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a', [[11]])
@pytest.mark.parametrize('b', [[10]])
def test_standard_mul_list(func, ms_func, a, b):
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

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a', [[10.0]])
@pytest.mark.parametrize('b', [[11.0]])
def test_standard_mul_list1(func, ms_func, a, b):
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

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a', [[20]])
@pytest.mark.parametrize('b', [[11.0]])
def test_standard_mul_list2(func, ms_func, a, b):
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

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a', [[2.0]])
@pytest.mark.parametrize('b', [[Tensor(np.ones((2, 3)).astype(np.float32))]])
def test_standard_mul_list3(func, ms_func, a, b):
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

@pytest.mark.skip(reason="GetDevicePtr() error")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a', [[Tensor(ops.fill(np.float32, (2, 3), 8))]])
@pytest.mark.parametrize('b', [[2.0]])
def test_standard_mul_Tensor1(func, ms_func, a, b):
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

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a', [[Tensor(ops.fill(np.float32, (2, 3), 8))]])
@pytest.mark.parametrize('b', [[Tensor(np.ones((2, 3)).astype(np.float32))]])
def test_standard_mul_Tensor2(func, ms_func, a, b):
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

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a', [[(1.0, 2.0, 3.0)]])
@pytest.mark.parametrize('b', [[Tensor(np.ones((2, 3)).astype(np.float32))]])
def test_standard_mul_tuple1(func, ms_func, a, b):
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

@pytest.mark.skip(reason="GetDevicePtr() error")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a', [[Tensor(np.ones((2, 3)).astype(np.float32))]])
@pytest.mark.parametrize('b', [[(1.0, 2.0, 3.0)]])
def test_standard_mul_tuple2(func, ms_func, a, b):
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

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a', [[(1.0, 2.0, 3.0)]])
@pytest.mark.parametrize('b', [[3]])
def test_standard_mul_tuple_list(func, ms_func, a, b):
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

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a', [[2]])
@pytest.mark.parametrize('b', [[(1.0, 2.0, 3.0)]])
def test_standard_mul_list_tuple(func, ms_func, a, b):
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

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a', [["Hello"]])
@pytest.mark.parametrize('b', [[3]])
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

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [mul])
@pytest.mark.parametrize('ms_func', [jit_mul])
@pytest.mark.parametrize('a', [[3]])
@pytest.mark.parametrize('b', [["Hello"]])
def test_special_mul_operations2(func, ms_func, a, b):
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
