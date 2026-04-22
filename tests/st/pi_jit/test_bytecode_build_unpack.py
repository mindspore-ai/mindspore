import pytest
import sys
from mindspore import jit, context
from mindspore._c_expression import get_code_extra
from .share.utils import match_array
from tests.mark_utils import arg_mark

@pytest.fixture(autouse=True)
def skip_if_python_version_too_high():
    if sys.version_info >= (3, 9) or sys.version_info < (3, 8):
        pytest.skip("Only tests on Python 3.8.")

def build_list_unpack(x, y, z):
    return [*x, *y, *z]

def build_set_unpack(x, y, z):
    return {*x, *y, *z}

def build_tuple_unpack(x, y, z):
    return (*x, *y, *z)

def build_tuple(*x):
    return x

def build_tuple_unpack_with_call(x, y, z):
    return build_tuple(*x, *y, *z)

def build_map_keys_values(**x):
    keys = tuple(x.keys())
    values = tuple(x.values())
    return keys, values

def build_map_unpack(x, y, z):
    d = {**x, **y, **z}
    return build_map_keys_values(**d)

def build_map_unpack_with_call(x, y, z):
    return build_map_keys_values(**x, **y, **z)

def build_mix_unpack_with_call(a, b, c, d):
    return build_tuple(*a, *b), build_map_keys_values(**c, **d)

def check_func_compile_state(func):
    jcr = get_code_extra(func.__wrapped__)
    assert jcr is not None
    assert jcr['break_count_'] == 0
    assert jcr['compile_count_'] == 1
    assert jcr['stat'] == 'GRAPH_CALLABLE'

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [build_list_unpack])
@pytest.mark.parametrize('x', [(1, 2, 3)])
@pytest.mark.parametrize('y', [(5, 6)])
@pytest.mark.parametrize('z', [(7, 8, 9, 10)])
def test_build_list_unpack(func, x, y, z):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_LIST_UNPACK in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [build_set_unpack])
@pytest.mark.parametrize('x', [(1, 2, 3)])
@pytest.mark.parametrize('y', [(5, 6)])
@pytest.mark.parametrize('z', [(7, 8, 9, 10)])
def test_build_set_unpack(func, x, y, z):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_SET_UNPACK in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = set(wrapped_func(x, y, z))
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [build_tuple_unpack])
@pytest.mark.parametrize('x', [(1, 2, 3)])
@pytest.mark.parametrize('y', [(5, 6)])
@pytest.mark.parametrize('z', [(7, 8, 9, 10)])
def test_build_tuple_unpack(func, x, y, z):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_TUPLE_UNPACK in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [build_tuple_unpack_with_call])
@pytest.mark.parametrize('x', [(1, 2, 3)])
@pytest.mark.parametrize('y', [(5, 6)])
@pytest.mark.parametrize('z', [(7, 8, 9, 10)])
def test_build_tuple_unpack_with_call(func, x, y, z):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_TUPLE_UNPACK_WITH_CALL in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [build_map_unpack])
@pytest.mark.parametrize('x', [{"a":1, "b":2, "c":3}])
@pytest.mark.parametrize('y', [{"d":5, "e":6}])
@pytest.mark.parametrize('z', [{"f":7, "g":8, "h":9, "i":10}])
def test_build_map_unpack(func, x, y, z):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_MAP_UNPACK in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [build_map_unpack_with_call])
@pytest.mark.parametrize('x', [{"a":1, "b":2, "c":3}])
@pytest.mark.parametrize('y', [{"d":5, "e":6}])
@pytest.mark.parametrize('z', [{"f":7, "g":8, "h":9, "i":10}])
def test_build_map_unpack_with_call(func, x, y, z):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_MAP_UNPACK_WITH_CALL in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [build_mix_unpack_with_call])
@pytest.mark.parametrize('a', [[1, 2, 3]])
@pytest.mark.parametrize('b', [[4, 5, 6]])
@pytest.mark.parametrize('c', [{"a":7, "b":8}])
@pytest.mark.parametrize('d', [{"c":9, "d":10, "e":100}])
def test_build_mix_unpack_with_call(func, a, b, c, d):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_TUPLE_UNPACK_WITH_CALL & BUILD_MAP_UNPACK_WITH_CALL in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b, c, d)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(a, b, c, d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    check_func_compile_state(wrapped_func)
