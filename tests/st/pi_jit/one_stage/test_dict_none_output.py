import dis
import pytest
from mindspore import jit, context, Tensor
from mindspore._c_expression import get_code_extra
from ..share.utils import match_array
from tests.mark_utils import arg_mark

def create_tuple_with_none(x):
    return (x * x, None)

def create_nested_tuple_with_none(x):
    return (x * x, (x * 100, None))

def create_tuple_with_dict(x):
    return (x * x, {'x' : x, 'none': None})

def create_tuple_with_recursive(x):
    t = (x, x * x, x *100)
    return (t, {'x' : x, 'none': None, 't': t})

def create_dict_with_none(x):
    return {'x' : x, 'none': None}

def create_dict_with_tuple(x):
    return {'x' : x, 'tuple': (x * x, None)}

def create_dict_with_nested_tuple(x):
    return {'x' : x, 'tuple': (x * x, (x * 100, None))}

def create_nested_dict_with_tuple(x):
    return {'x' : x, 'dict': {'x' : x, 'tuple': (x * x, None)}}

def create_tuple(x):
    return (x, x, x)

def create_dict(x):
    return {'x' : x, '2x' : x * 2}

def create_nested_dict_tuple_with_call(x):
    return create_dict(create_tuple(create_dict(create_tuple(x))))

def check_func_compile_state(func):
    jcr = get_code_extra(func.__wrapped__)
    assert jcr is not None
    assert jcr['break_count_'] == 0
    assert jcr['compile_count_'] == 1
    assert jcr['stat'] == 'GRAPH_CALLABLE'

def result_compare(actual, expected):
    if actual is None and expected is None:
        return True

    if actual is Tensor and expected is Tensor:
        match_array(actual, expected, error=0, err_msg=str(expected))
        return True

    if isinstance(actual, tuple) and isinstance(expected, tuple):
        for index in range(len(actual)):
            if not result_compare(actual[index], expected[index]):
                return False
        return True

    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in actual.keys():
            if not result_compare(actual[key], expected[key]):
                return False
        return True

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [create_tuple_with_none])
@pytest.mark.parametrize('x', [Tensor([1, 2, 3, 4, 5, 6])])
def test_tuple_with_none(func, x):
    """
    Feature: ALL TO ALL
    Description: test cases for return tuple contain None in graph
    Expectation: the result correct, no cracked graph
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x,)
    result_compare(res, ms_res)
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [create_nested_tuple_with_none])
@pytest.mark.parametrize('x', [Tensor([1, 2, 3, 4, 5, 6])])
def test_nested_tuple_with_none(func, x):
    """
    Feature: ALL TO ALL
    Description: test cases for return nested tuple contain None in graph
    Expectation: the result correct, no cracked graph
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x,)
    result_compare(res, ms_res)
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [create_tuple_with_dict])
@pytest.mark.parametrize('x', [Tensor([1, 2, 3, 4, 5, 6])])
def test_tuple_with_dict(func, x):
    """
    Feature: ALL TO ALL
    Description: test cases for return tuple contain None in graph
    Expectation: the result correct, no cracked graph
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x,)
    result_compare(res, ms_res)
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [create_tuple_with_recursive])
@pytest.mark.parametrize('x', [Tensor([1, 2, 3, 4, 5, 6])])
def test_tuple_with_recursive(func, x):
    """
    Feature: ALL TO ALL
    Description: test cases for return tuple contain None in graph
    Expectation: the result correct, no cracked graph
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x,)
    result_compare(res, ms_res)
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [create_dict])
@pytest.mark.parametrize('x', [Tensor([1, 2, 3, 4, 5, 6])])
def test_dict(func, x):
    """
    Feature: ALL TO ALL
    Description: test cases for return tuple contain None in graph
    Expectation: the result correct, no cracked graph
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x,)
    result_compare(res, ms_res)
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [create_dict_with_none])
@pytest.mark.parametrize('x', [Tensor([1, 2, 3, 4, 5, 6])])
def test_dict_with_none(func, x):
    """
    Feature: ALL TO ALL
    Description: test cases for return tuple contain None in graph
    Expectation: the result correct, no cracked graph
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x,)
    result_compare(res, ms_res)
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [create_dict_with_tuple])
@pytest.mark.parametrize('x', [Tensor([1, 2, 3, 4, 5, 6])])
def test_dict_with_tuple(func, x):
    """
    Feature: ALL TO ALL
    Description: test cases for return tuple contain None in graph
    Expectation: the result correct, no cracked graph
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x,)
    result_compare(res, ms_res)
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [create_dict_with_nested_tuple])
@pytest.mark.parametrize('x', [Tensor([1, 2, 3, 4, 5, 6])])
def test_dict_with_nested_tuple(func, x):
    """
    Feature: ALL TO ALL
    Description: test cases for return tuple contain None in graph
    Expectation: the result correct, no cracked graph
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x,)
    result_compare(res, ms_res)
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [create_nested_dict_with_tuple])
@pytest.mark.parametrize('x', [Tensor([1, 2, 3, 4, 5, 6])])
def test_nested_dict_with_tuple(func, x):
    """
    Feature: ALL TO ALL
    Description: test cases for return tuple contain None in graph
    Expectation: the result correct, no cracked graph
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x,)
    result_compare(res, ms_res)
    check_func_compile_state(wrapped_func)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [create_nested_dict_tuple_with_call])
@pytest.mark.parametrize('x', [Tensor([1, 2, 3, 4, 5, 6])])
def test_nested_dict_tuple_with_call(func, x):
    """
    Feature: ALL TO ALL
    Description: test cases for return tuple contain None in graph
    Expectation: the result correct, no cracked graph
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(x,)
    result_compare(res, ms_res)

    jcr = get_code_extra(func)
    ops = [i.opname for i in dis.get_instructions(jcr['code']['compiled_code_'])]
    assert "UNPACK_SEQUENCE" not in ops
    check_func_compile_state(wrapped_func)
