import pytest
from .test_cross_file_inline_func import inlinef
from mindspore._c_expression import jit_mode_pi_enable, jit_mode_pi_disable
from mindspore import jit, context
from tests.mark_utils import arg_mark

conf = {
    "print_after_all": False,
    "interpret_captured_code": True,
    "allowed_inline_modules": ["mindspore", "test_cross_file_inline_func"],
    "MAX_INLINE_DEPTH": 10,
}

g = "yyyyy"


@jit(mode="PIJit", jit_config=conf)
def cross_inline_make_func_test(a=True):
    inner = inlinef()
    iinner = inlinef()()
    if a:
        return iinner(), g, inner()
    return g


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cross_file_inline_make_func():
    """
    Feature: Cross File Inline Function Testing
    Description: Test the cross_inline_make_func_test function from another file.
    Expectation: The results of step 1 and step 2 should match for both variables and function calls.
    """
    global g
    jit_mode_pi_enable()
    context.set_context(mode=context.PYNATIVE_MODE)
    xxxx1, yyyy1, func1 = cross_inline_make_func_test()
    jit_mode_pi_disable()
    context.set_context(mode=context.GRAPH_MODE)
    xxxx2, yyyy2, func2 = cross_inline_make_func_test()
    jit_mode_pi_enable()
    assert xxxx1 == xxxx2 and func1() == func2() and yyyy1 == yyyy2
