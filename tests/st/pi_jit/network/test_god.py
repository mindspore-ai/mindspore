import sys
import pytest
from mindspore import JitConfig, jit,context
from mindspore import numpy as np
from tests.mark_utils import arg_mark

from tests.st.pi_jit.share.utils import assert_no_graph_break, assert_equal

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_tensor_is_contiguous():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    def fn(x):
        if x.is_contiguous():
            x = x.contiguous()
        return x
    x = np.randn((2,4))
    jit_fn = jit(fn, capture_mode="bytecode")
    result1 = jit_fn(x)
    reuslt2 = fn(x)
    assert_no_graph_break(jit_fn)
    assert_equal(result1,reuslt2)


condition = not (sys.version_info.major == 3 and sys.version_info.minor in [9,10])
@pytest.mark.skipif(condition, reason="Only support python 3.9, 3.10")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_bytecode_LIST_EXTEND():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    def fn(x,y):
        a = [x,x]
        a += y
        return a
    x = np.randn((2,4))
    y = [x,x]
    jit_fn = jit(fn, capture_mode="bytecode")
    result1 = jit_fn(x,y)
    reuslt2 = fn(x,y)
    assert_no_graph_break(jit_fn)
    assert_equal(result1,reuslt2)