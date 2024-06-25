import pytest
import mindspore.ops.operations as P
from mindspore import nn
from mindspore import jit, context
from ..share.utils import match_array
import numpy as np
from ..share.ops.primitive.round_ops import RoundFactory
from tests.mark_utils import arg_mark


@jit(mode="PIJit")
def fallback_round(x, n=None):
    return round(x, n)


@jit
def jit_fallback_round(x, n=None):
    return round(x, n)


test_data = [
    (10.678, None, 0),
    (10.678, 0, 0),
    (10.678, 1, 5),
    (10.678, -1, 0),
    (17.678, -1, 0)
]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func, ms_func', [(fallback_round, jit_fallback_round)])
@pytest.mark.parametrize('x, n, error', test_data)
def test_round_operations(func, ms_func, x, n, error):
    """
    Feature: ALL TO ALL
    Description: test cases for round in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, n)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(x, n)
    match_array(res, ms_res, error=error, err_msg=str(ms_res))


class VmapRound(nn.Cell):
    def __init__(self):
        super().__init__()
        self.round = P.Round()

    def construct(self, x):
        return self.round(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_round_input_512x12():
    """
    Feature: ALL TO ALL
    Description: test operator round with input shape 512x12, and data_type float16
    Expectation: the result match
    """
    fact = RoundFactory(input_shape=(512, 12), dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_round_input_512():
    """
    Feature: ALL TO ALL
    Description: test operator round with input shape 512, and data_type float16
    Expectation: the result match
    """
    fact = RoundFactory(input_shape=(512,), dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_round_input_64x128x1():
    """
    Feature: ALL TO ALL
    Description: test operator round with input shape 64x128x1, and data_type float16
    Expectation: the result match
    """
    fact = RoundFactory(input_shape=(64, 128, 1), dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_round_input_64x128x1x512():
    """
    Feature: ALL TO ALL
    Description: test operator round with input shape 64x128x1x512, and data_type float16
    Expectation: the result match
    """
    fact = RoundFactory(input_shape=(64, 128, 1, 512), dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_round_input_2048():
    """
    Feature: ALL TO ALL
    Description: test operator round with input shape 2048, and data_type float32
    Expectation: the result match
    """
    fact = RoundFactory(input_shape=(2048,), dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_round_input_16x1024():
    """
    Feature: ALL TO ALL
    Description: test operator round with input shape(16, 1024), and data_type float32
    Expectation: the result match
    """
    fact = RoundFactory(input_shape=(16, 1024), dtype=np.int32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_round_input_20x48_fp64():
    """
    Feature: ALL TO ALL
    Description: test operator round with input shape 20x48, and data_type float64
    Expectation: the result match
    """
    fact = RoundFactory(input_shape=(20, 48), dtype=np.float64)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_round_input_5x12x4_fp64():
    """
    Feature: ALL TO ALL
    Description: test operator round with input shape 5x12x4, and data_type float64
    Expectation: the result match
    """
    fact = RoundFactory(input_shape=(5, 12, 4), dtype=np.float64)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_round_input_32x16x128x8_int64():
    """
    Feature: ALL TO ALL
    Description: test operator round with input shape 32x16x128x8, and data_type int64
    Expectation: the result match
    """
    fact = RoundFactory(input_shape=(32, 16, 128, 8), dtype=np.int64)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_round_input_32x4x28x8x6_int64():
    """
    Feature: ALL TO ALL
    Description: test operator round with input shape 32x4x28x8x6, and data_type int64
    Expectation: the result match
    """
    fact = RoundFactory(input_shape=(32, 4, 28, 8, 6), dtype=np.int64)
    fact.forward_cmp()
    fact.grad_cmp()
