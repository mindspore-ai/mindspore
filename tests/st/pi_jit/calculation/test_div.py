import pytest
import numpy as onp
from mindspore import numpy as np
from mindspore import ops
from mindspore import Tensor, jit, context


def match_array(actual, expected, error=0, err_msg=''):
    """Assert two arrays are equal with a certain error."""
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
def div(a, b):
    """Divide two numbers using GraphJit."""
    return a / b


@jit
def jit_div(a, b):
    """Divide two numbers using jit."""
    return a / b


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [div])
@pytest.mark.parametrize('ms_func', [jit_div])
@pytest.mark.parametrize('test_data', [(11, 10)])
def test_div_int(func, ms_func, test_data):
    """
    Feature: ALL TO ALL
    Description: test cases for div in PYNATIVE mode
    Expectation: the result match
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=5, err_msg=str(ms_res))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [div])
@pytest.mark.parametrize('ms_func', [jit_div])
@pytest.mark.parametrize('test_data', [(11.0, 10.0)])
def test_div_float(func, ms_func, test_data):
    """
    Feature: ALL TO ALL
    Description: test cases for div in PYNATIVE mode
    Expectation: the result match
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=5, err_msg=str(ms_res))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [div])
@pytest.mark.parametrize('ms_func', [jit_div])
@pytest.mark.parametrize('test_data', [(20, 11.0)])
def test_div_int_float(func, ms_func, test_data):
    """
    Feature: ALL TO ALL
    Description: test cases for div in PYNATIVE mode
    Expectation: the result match
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=5, err_msg=str(ms_res))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [div])
@pytest.mark.parametrize('ms_func', [jit_div])
@pytest.mark.parametrize('test_data', [(2.0, Tensor(np.ones((2, 3)).astype(np.float32)))])
def test_div_float_tensor(func, ms_func, test_data):
    """
    Feature: ALL TO ALL
    Description: test cases for div in PYNATIVE mode
    Expectation: the result match
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [div])
@pytest.mark.parametrize('ms_func', [jit_div])
@pytest.mark.parametrize('test_data', [(Tensor(ops.fill(np.float32, (2, 3), 8)), 2.0)])
def test_div_tensor_float(func, ms_func, test_data):
    """
    Feature: ALL TO ALL
    Description: test cases for div in PYNATIVE mode
    Expectation: the result match
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [div])
@pytest.mark.parametrize('ms_func', [jit_div])
@pytest.mark.parametrize('test_data', [(Tensor(ops.fill(np.float32, (2, 3), 8)),
                                        Tensor(np.ones((2, 3)).astype(np.float32)))])
def test_div_tensor(func, ms_func, test_data):
    """
    Feature: ALL TO ALL
    Description: test cases for div in PYNATIVE mode
    Expectation: the result match
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [div])
@pytest.mark.parametrize('ms_func', [jit_div])
@pytest.mark.parametrize('test_data', [((1.0, 2.0, 3.0),
                                        Tensor(np.ones((2, 3)).astype(np.float32)))])
def test_div_tuple_tensor(func, ms_func, test_data):
    """
    Feature: ALL TO ALL
    Description: test cases for div in PYNATIVE mode
    Expectation: the result match
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [div])
@pytest.mark.parametrize('ms_func', [jit_div])
@pytest.mark.parametrize('test_data', [([1.0, 2.0, 3.0],
                                        Tensor(np.ones((2, 3)).astype(np.float32)))])
def test_div_list_tensor(func, ms_func, test_data):
    """
    Feature: ALL TO ALL
    Description: test cases for div in PYNATIVE mode
    Expectation: the result match
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [div])
@pytest.mark.parametrize('ms_func', [jit_div])
@pytest.mark.parametrize('test_data', [(Tensor(np.ones((2, 3)).astype(np.float32)),
                                        (1.0, 2.0, 3.0))])
def test_div_tensor_tuple(func, ms_func, test_data):
    """
    Feature: ALL TO ALL
    Description: test cases for div in PYNATIVE mode
    Expectation: the result match
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [div])
@pytest.mark.parametrize('ms_func', [jit_div])
@pytest.mark.parametrize('test_data', [(Tensor(np.ones((2, 3)).astype(np.float32)),
                                        [1.0, 2.0, 3.0])])
def test_div_tensor_list(func, ms_func, test_data):
    """
    Feature: ALL TO ALL
    Description: test cases for div in PYNATIVE mode
    Expectation: the result match
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
