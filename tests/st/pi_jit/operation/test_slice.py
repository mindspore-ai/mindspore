import numpy as np
import pytest
from mindspore import Tensor
from mindspore.common import dtype as mstype
from ..share.ops.primitive.slice_ops import SliceFactory
from ..share.ops.primitive.slice_ops import SliceMock
from ..dynamic_shape_operations.slice import DynamicShapeSliceFactory
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_n1024x1270_0x0_4x4():
    """
    Feature: ALL TO ALL
    Description: test sin with dynamic shape input, dtype=float16
    Expectation: the result match
    """
    for n in (128,):
        input_shape = (n * 1024, 1270)
        begin = (0, 0)
        size = (4, 4)
        fact = SliceFactory(input_shape, begin, size, dtype=np.float32)
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_8x32x6_0x28x0_8x4x6():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape=(8, 32, 6), begin=(0, 28, 0), size=(8, 4, 6)
    Expectation: the result match
    """
    input_shape = (8, 32, 6)
    begin = (0, 28, 0)
    size = (8, 4, 6)
    fact = SliceFactory(input_shape, begin, size)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_2d():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape=(8, 87), begin=(0, 56), size=(8, 27)
    Expectation: the result match
    """
    input_shape = (8, 87)
    begin = (0, 56)
    size = (8, 27)
    fact = SliceFactory(input_shape, begin, size)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_3d():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape=(8, 87, 4), begin=(0, 56, 0), size=(8, 27, 4)
    Expectation: the result match
    """
    input_shape = (8, 87, 4)
    begin = (0, 56, 0)
    size = (8, 27, 4)
    fact = SliceFactory(input_shape, begin, size)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_0d_fp64():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，the type of input is float64, shape 0d
    Expectation: the result match
    """
    input_x = Tensor(np.random.randn(), dtype=mstype.float64)
    begin = ()
    size = ()
    fact = SliceMock(inputs=[input_x, begin, size])
    with pytest.raises(ValueError):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_0d_dtype_complex64():
    """
    Feature: ALL TO ALL
    Description: test slice with input shape from 0d, type complex64
    Expectation: the result match
    """
    x_real = np.random.randn()
    x_imag = np.random.randn()
    x = Tensor((x_real + 1j * x_imag), dtype=mstype.complex64)
    begin = ()
    size = ()
    fact = SliceMock(inputs=[x, begin, size])
    with pytest.raises(ValueError):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_5d_dtype_complex64_begin_size_shape_larger_than_input_x():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，test slice with input shape from 5d, type complex128, real type float16
    Expectation: the result match
    """
    x_real = np.random.randn(12, 32, 18, 24, 8).astype(np.float16)
    x_imag = np.random.randn(12, 32, 18, 24, 8).astype(np.float64)
    x = Tensor((x_real + 1j * x_imag), dtype=mstype.complex64)
    begin = (0, 12, 6, 12, 5)
    size = (8, 9, 6, 12, 5)
    fact = SliceMock(inputs=[x, begin, size])
    with pytest.raises(ValueError):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_begin_bool():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape(8,87,4), begin=bool, size=(8, 57, 4)
    Expectation: the result match
    """
    input_shape = (8, 87, 4)
    begin = [True, False, True]
    size = (8, 57, 4)
    fact = SliceFactory(input_shape, begin, size)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_begin_int():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape(8,87,4), begin=int, size=(8, 57, 4)
    Expectation: the result match
    """
    input_shape = (8, 87, 4)
    begin = 0
    size = (8, 57, 4)
    fact = SliceFactory(input_shape, begin, size)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_begin_list():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape(8,87,4), begin=list, size=(8, 57, 4)
    Expectation: the result match
    """
    input_shape = (8, 87, 4)
    begin = [1, 0.1]
    size = (8, 57, 4)
    fact = SliceFactory(input_shape, begin, size)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_size_bool():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape(8, 87, 4), size=bool, begin=(8, 57, 4)
    Expectation: the result match
    """
    input_shape = (8, 87, 4)
    size = True
    begin = (8, 57, 4)
    fact = SliceFactory(input_shape, begin, size)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_size_list():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape(8, 87, 4), size=list, begin=(8, 57, 4)
    Expectation: the result match
    """
    input_shape = (8, 87, 4)
    size = [1, 0.1]
    begin = (8, 57, 4)
    fact = SliceFactory(input_shape, begin, size)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_size_int():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape(8, 87, 4), size=int, begin=(8, 57, 4)
    Expectation: the result match
    """
    input_shape = (8, 87, 4)
    size = 2
    begin = (8, 57, 4)
    fact = SliceFactory(input_shape, begin, size)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_4d_dtype_float16():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape=4d, float64
    Expectation: the result match
    """
    input_shape = (1, 2, 3, 4)
    begin = (0, 0, 0, 0)
    size = (1, 1, 1, 4)
    fact = SliceFactory(input_shape, begin, size, dtype=np.float16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_5d_dtype_float32():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape=5d dtype=fp32
    Expectation: the result match
    """
    input_shape = (1, 2, 3, 4, 5)
    begin = (0, 0, 0, 0, 0)
    size = (1, 1, 1, 1, 5)
    fact = SliceFactory(input_shape, begin, size, dtype=np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_shape_2x32x112x112x48_begin_1x2x3x4x5_size_1x10x1x20x40_fp32():
    """
    Feature: ALL TO ALL
    Description: slice算子测试， input shape [2, 32, 112, 112, 48]
                 begin=[1, 2, 3, 4, 5]
                 size=[1, 10, 1, 20, 40]
                 type=float32
    Expectation: the result match
    """
    input_shape = (2, 32, 112, 112, 48)
    begin = (1, 2, 3, 4, 5)
    size = (1, 10, 1, 20, 40)
    fact = SliceFactory(input_shape, begin, size, dtype=np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_5d_dtype_fp16():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape=5d dtype=fp16
    Expectation: the result match
    """
    input_shape = (2, 32, 112, 112, 48)
    begin = (1, 2, 3, 4, 5)
    size = (1, 10, 1, 20, 40)
    fact = SliceFactory(input_shape, begin, size, dtype=np.float16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_5d_dtype_fp64():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape=5d dtype=fp64
    Expectation: the result match
    """
    input_shape = (2, 32, 112, 112, 48)
    begin = (1, 2, 3, 4, 5)
    size = (1, 10, 1, 20, 40)
    fact = SliceFactory(input_shape, begin, size, dtype=np.float64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_5d_dtype_int64():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape=5d dtype=int64
    Expectation: the result match
    """
    input_shape = (2, 32, 112, 112, 48)
    begin = (1, 2, 3, 4, 5)
    size = (1, 10, 1, 20, 40)
    fact = SliceFactory(input_shape, begin, size, dtype=np.int64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_5d_dtype_int32():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape=5d dtype=int32
    Expectation: the result match
    """
    input_shape = (2, 32, 112, 112, 48)
    begin = (1, 2, 3, 4, 5)
    size = (1, 10, 1, 20, 40)
    fact = SliceFactory(input_shape, begin, size, dtype=np.int32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_5d_dtype_int16():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape=5d dtype=int16
    Expectation: the result match
    """
    input_shape = (2, 32, 112, 112, 48)
    begin = (1, 2, 3, 4, 5)
    size = (1, 10, 1, 20, 40)
    fact = SliceFactory(input_shape, begin, size, dtype=np.int16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_6d_dtype_int64():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape=5d dtype=int64
    Expectation: the result match
    """
    input_shape = (1, 2, 3, 4, 5, 6)
    begin = (0, 0, 0, 0, 0, 3)
    size = (1, 1, 1, 1, 2, 2)
    fact = SliceFactory(input_shape, begin, size, dtype=np.int64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_input_dtype_int32():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，input_shape=int32
    Expectation: the result match
    """
    input_shape = (56, 45)
    begin = (10, 9)
    size = (3, 6)
    fact = SliceFactory(input_shape, begin, size, dtype=np.int32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_slice_size_greater_than_input():
    """
    Feature: ALL TO ALL
    Description: slice算子测试，size greater than input
    Expectation: the result match
    """
    input_shape = (56, 45)
    begin = (8, 9)
    size = (1, 1, 1)
    fact = SliceFactory(input_shape, begin, size, dtype=np.float32)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_slice_input_3d():
    """
    Feature: ALL TO ALL
    Description: slice算子动态shape测试，input_shape 3d
    Expectation: the result match
    """
    input_shape = (8, 32, 6)
    begin = (0, 28, 0)
    size = (8, 4, 6)
    fact = SliceFactory(input_shape, begin, size)
    fact.forward_cmp()
    fact.forward_dynamic_shape_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_slice_input_5d():
    """
    Feature: ALL TO ALL
    Description: slice算子动态shape测试，input_shape 5d
    Expectation: the result match
    """
    input_shape = (12, 32, 18, 24, 8)
    begin = (0, 12, 6, 12, 2)
    size = (8, 9, 6, 12, 5)
    fact = SliceFactory(input_shape, begin, size)
    fact.forward_cmp()
    fact.forward_dynamic_shape_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_slice_input_7d():
    """
    Feature: ALL TO ALL
    Description: slice算子动态shape测试，input_shape 7d
    Expectation: the result match
    """
    input_shape = (1, 2, 3, 4, 5, 6, 7)
    begin = (0, 0, 1, 2, 0, 3, 1)
    size = (1, 1, 1, 1, 2, 2, 5)
    fact = SliceFactory(input_shape, begin, size)
    fact.forward_cmp()
    fact.forward_dynamic_shape_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_slice_2d_fp32():
    """
    Feature: ALL TO ALL
    Description: slice算子动态shape测试，input_shape 2d with float32
    Expectation: the result match
    """
    input_shape = (8, 512)
    begin = (0,)
    size = (128,)
    axis = np.array([0])
    fact = DynamicShapeSliceFactory(input_shape, begin, size, axis, dtype=np.float32)
    fact.forward_cmp()
