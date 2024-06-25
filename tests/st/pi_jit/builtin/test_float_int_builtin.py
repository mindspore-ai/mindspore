import pytest
from mindspore import jit, context
from ..share.utils import match_array
from tests.mark_utils import arg_mark


def convert_numbers_to_float_and_int(val1, val2):
    return float(val1), float(val2), int(val1), int(val2)


@jit(mode="PIJit")
def fallback_float_and_int():
    return convert_numbers_to_float_and_int(5, 5.0)


@jit(mode="PIJit")
def fallback_float_and_int_empty():
    return float(), int()


@jit
def ms_fallback_float_and_int():
    return convert_numbers_to_float_and_int(5, 5.0)


@jit
def ms_fallback_float_and_int_empty():
    return float(), int()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [fallback_float_and_int])
@pytest.mark.parametrize('ms_func', [ms_fallback_float_and_int])
def test_int_float_conversion_with_args(func, ms_func):
    """
    Feature: Conversion of int and float
    Description: Test cases for argument support in PYNATIVE mode
    Expectation: Results match between GraphJit and JIT functions
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func()
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [fallback_float_and_int_empty])
@pytest.mark.parametrize('ms_func', [ms_fallback_float_and_int_empty])
def test_int_float_conversion_no_args(func, ms_func):
    """
    Feature: Conversion of int and float without arguments
    Description: Test cases for argument support in PYNATIVE mode
    Expectation: Results match between GraphJit and JIT functions
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func()
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
