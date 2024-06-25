import pytest
import numpy as np
from mindspore.common.tensor import Tensor
from ..share.ops.primitive.sin_ops import SinMock
from ..dynamic_shape_operations.sin import SinDynamicShapeFactory
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_sin_32x1024x1269():
    """
    Feature: ALL TO ALL
    Description: sin算子测试，input:[32 * 1024, 1269], np.float32
    Expectation: the result match
    """
    input_x = Tensor(np.random.randn(32 * 1024, 1269).astype(np.float32))
    fact = SinMock(inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_sin_x1269():
    """
    Feature: ALL TO ALL
    Description: sin算子测试，input:[1269], np.float32
    Expectation: the result match
    """
    input_x = Tensor(np.random.randn(1269).astype(np.float32))
    fact = SinMock(inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_sin_2x4x8():
    """
    Feature: ALL TO ALL
    Description: sin算子测试，input:[2, 4, 8], np.float16
    Expectation: the result match
    """
    input_x = Tensor(np.random.randn(2, 4, 8).astype(np.float16))
    fact = SinMock(inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_sin_2x4x8x16():
    """
    Feature: ALL TO ALL
    Description: sin算子测试，input:[2, 4, 8, 16], np.float32
    Expectation: the result match
    """
    input_x = Tensor(np.random.randn(2, 4, 8, 16).astype(np.float32))
    fact = SinMock(inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_sin_2x4x8x16_fp64():
    """
    Feature: ALL TO ALL
    Description: sin算子测试，input:[2, 4, 8, 16], np.float64
    Expectation: the result match
    """
    input_x = Tensor(np.random.randn(2, 4, 8, 16).astype(np.float64))
    fact = SinMock(inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_sin_1x2x4x8x16():
    """
    Feature: ALL TO ALL
    Description: sin算子测试，input:[1, 2, 4, 8, 16], np.float16
    Expectation: the result match
    """
    input_x = Tensor(np.random.randn(1, 2, 4, 8, 16).astype(np.float16))
    fact = SinMock(inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_sin_2x4x8x16x1x16():
    """
    Feature: ALL TO ALL
    Description: sin算子测试，input:[2, 4, 8, 16, 1, 16], np.float32
    Expectation: the result match
    """
    input_x = Tensor(np.random.randn(2, 4, 8, 16, 1, 16).astype(np.float32))
    fact = SinMock(inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_sin_input_4d_cp64():
    """
    Feature: ALL TO ALL
    Description: test Sin with 4D input, dtype=complex64
    Expectation: the result match
    """
    input_x_real = np.random.rand(2, 3, 5, 7).astype(np.float32)
    input_x_imag = np.random.rand(2, 3, 5, 7).astype(np.float32)
    input_x = Tensor((input_x_real + 1j * input_x_imag).astype(np.complex64))
    fact = SinMock(inputs=[input_x])
    fact.loss = 2e-6
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_sin_input_5d_cp128():
    """
    Feature: ALL TO ALL
    Description: test Sin with 5D input, dtype=complex128
    Expectation: the result match
    """
    input_x_real = np.random.rand(8, 4, 3, 12, 7).astype(np.float64)
    input_x_imag = np.random.rand(8, 4, 3, 12, 7).astype(np.float64)
    input_x = Tensor((input_x_real + 1j * input_x_imag).astype(np.complex128))
    fact = SinMock(inputs=[input_x])
    fact.loss = 2e-10
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_sin_input_2d_cp128():
    """
    Feature: ALL TO ALL
    Description: test Sin with 2D input, dtype=complex128
    Expectation: the result match
    """
    input_x_real = np.random.rand(38, 65).astype(np.float64)
    input_x_imag = np.random.rand(38, 65).astype(np.float64)
    input_x = Tensor((input_x_real + 1j * input_x_imag).astype(np.complex128))
    fact = SinMock(inputs=[input_x])
    fact.loss = 2e-10
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_sin_input_7d_cp64():
    """
    Feature: ALL TO ALL
    Description: test Sin with 7D input, dtype=complex64
    Expectation: the result match
    """
    input_x_real = np.random.rand(9, 6, 4, 2, 9, 8, 12).astype(np.float32)
    input_x_imag = np.random.rand(9, 6, 4, 2, 9, 8, 12).astype(np.float32)
    input_x = Tensor((input_x_real + 1j * input_x_imag).astype(np.complex64))
    fact = SinMock(inputs=[input_x])
    fact.loss = 2e-6
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_sin_input_type_not_support():
    """
    Feature: ALL TO ALL
    Description: sin算子测试，input:int32,int8,uint8
    Expectation: the result match
    """
    input_x1 = Tensor(np.random.randn(2, 4, 8).astype(np.int32))
    fact1 = SinMock(inputs=[input_x1])

    input_x2 = Tensor(np.random.randn(2, 4, 8).astype(np.int8))
    fact2 = SinMock(inputs=[input_x2])

    input_x3 = Tensor(np.random.randn(2, 4, 8).astype(np.uint8))
    fact3 = SinMock(inputs=[input_x3])

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact1.forward_cmp()
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact2.forward_cmp()
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact3.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_sin_input_float32():
    """
    Feature: ALL TO ALL
    Description: Sin算子正反向dynamic shape测试,input_shape=(3, 16, 32), dtype=np.float32
    Expectation: the result match
    """
    input_x = Tensor(np.random.randn(3, 16, 32).astype(np.float32))
    fact = SinMock(inputs=[input_x])
    fact.forward_dynamic_shape_cmp()
    fact.grad_dynamic_shape_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_sin_float32():
    """
    Feature: ALL TO ALL
    Description: test sin with dynamic shape input, dtype=float32
    Expectation: the result match
    """
    input_x = Tensor(np.random.rand(2, 10, 5, 10).astype(np.float32))
    indices = Tensor(np.random.choice(3, 2, replace=False).astype(np.int32))
    fact = SinDynamicShapeFactory([input_x, indices], dtype=np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_sin_float16():
    """
    Feature: ALL TO ALL
    Description: test sin with dynamic shape input, dtype=float16
    Expectation: the result match
    """
    input_x = Tensor(np.random.rand(1, 1, 2, 4, 10).astype(np.float16))
    indices = Tensor(np.random.choice(3, 1, replace=False).astype(np.int32))
    fact = SinDynamicShapeFactory([input_x, indices], dtype=np.float16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_sin():
    """
    Feature: ALL TO ALL
    Description: test tensor API sin
    Expectation: the result match
    """
    input_x = Tensor(np.random.random((8, 3, 6)).astype(np.float32))
    fact = SinMock(inputs=[input_x])
    fact.forward_tensor_cmp()
