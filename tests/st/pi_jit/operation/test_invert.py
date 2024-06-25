from ..share.ops.primitive.invert_ops import InvertFactory
import numpy as np
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_256():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(1,), dtype=int16.
    Expectation: expect correct result.
    """
    input_shape = (256,)
    fact = InvertFactory(input_shape, dtype=np.int16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_256x256():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(256, 256), dtype=uint16.
    Expectation: expect correct result.
    """
    input_shape = (256, 256)
    fact = InvertFactory(input_shape, dtype=np.uint16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_128x8x1():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(128,8,1), dtype=int16.
    Expectation: expect correct result.
    """
    input_shape = (128, 8, 1)
    fact = InvertFactory(input_shape, dtype=np.int16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_32x16x8x4():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(32, 16, 8, 4), dtype=int16.
    Expectation: expect correct result.
    """
    input_shape = (32, 26, 8, 4)
    fact = InvertFactory(input_shape, dtype=np.uint16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_32x8x16x8x32():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(32, 8, 16, 8, 32), dtype=int16.
    Expectation: expect correct result.
    """
    input_shape = (32, 8, 16, 8, 32)
    fact = InvertFactory(input_shape, dtype=np.int16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_8x8x16x32x8x16():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(8,8,16,32,8,16), dtype=int16.
    Expectation: expect correct result.
    """
    input_shape = (8, 8, 16, 32, 8, 16)
    fact = InvertFactory(input_shape, dtype=np.int16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_8x2x4x128x1x16x7():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(32, 8, 16, 8, 32), dtype=int16.
    Expectation: expect correct result.
    """
    input_shape = (8, 2, 4, 128, 1, 16, 7)
    fact = InvertFactory(input_shape, dtype=np.int16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_8x2x4x128x1x16x7_int8():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(32, 8, 16, 8, 32), dtype=int8.
    Expectation: expect correct result.
    """
    input_shape = (8, 2, 4, 128, 1, 16, 7)
    fact = InvertFactory(input_shape, dtype=np.int8)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_8x8x16x32x8x16_uint8():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(32, 8, 16, 8, 32), dtype=int8.
    Expectation: expect correct result.
    """
    input_shape = (8, 8, 16, 32, 8, 16)
    fact = InvertFactory(input_shape, dtype=np.uint8)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_8x2x4x128x1x16x7_int64():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(32, 8, 16, 8, 32), dtype=int8.
    Expectation: expect correct result.
    """
    input_shape = (8, 2, 4, 128, 1, 16, 7)
    fact = InvertFactory(input_shape, dtype=np.int64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_8x8x16x32x8x16_uint64():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(8,8,16,32,8,16), dtype=uint64.
    Expectation: expect correct result.
    """
    input_shape = (8, 8, 16, 32, 8, 16)
    fact = InvertFactory(input_shape, dtype=np.uint64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_2x2_int32():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(2, 2), dtype=int32.
    Expectation: expect correct result.
    """
    input_shape = (2, 2)
    fact = InvertFactory(input_shape, dtype=np.int32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_invert_input_2x2_uint32():
    """
    Feature: Ops.
    Description: test operator Invert, input_shape=(2, 2), dtype=uint32.
    Expectation: expect correct result.
    """
    input_shape = (2, 2)
    fact = InvertFactory(input_shape, dtype=np.uint32)
    fact.forward_cmp()
