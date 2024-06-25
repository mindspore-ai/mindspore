import numpy as np
from ..share.ops.primitive.abs_ops import AbsFactory
from ..dynamic_shape_operations.abs import AbsDynamicShapeMock
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_abs_input_1():
    """
    Description:
        1.abs算子正反向测试，input_shape=(1,), dtype:fp32

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1,)
    fact = AbsFactory(input_shape, dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_abs_input_1x1():
    """
    Description:
        1.abs算子正反向测试，input_shape=(1,1), dtype=uint8.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 1)
    dtype = np.uint8
    fact = AbsFactory(input_shape, dtype=dtype)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_abs_input_256x256x256():
    """
    Description:
        1.abs算子正反向测试，input_shape=(256,256,256), dtype=fp32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (256, 256, 256)
    dtype = np.float32
    fact = AbsFactory(input_shape, dtype=dtype)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_abs_input_1x1x1x1():
    """
    Description:
        1.abs算子正反向测试，input_shape=(1,1,1,1), dtype=fp32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 1, 1, 1)
    dtype = np.float32
    fact = AbsFactory(input_shape, dtype=dtype)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_abs_input_32x2x16x8():
    """
    Description:
        1.abs算子正反向测试，input_shape=(32, 2, 16, 8), dtype=fp32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (32, 2, 16, 8)
    dtype = np.float32
    fact = AbsFactory(input_shape, dtype=dtype)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_abs_input_1x1x1x1x1():
    """
    Description:
        1.abs算子正反向测试，input_shape=(1, 1, 1, 1, 1), dtype=fp32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 1, 1, 1, 1)
    dtype = np.float32
    fact = AbsFactory(input_shape, dtype=dtype)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_abs_input_32x8x16x8x32():
    """
    Description:
        1.abs算子正反向测试，input_shape=(32,8,16,8,32), dtype=fp32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (32, 8, 16, 8, 32)
    dtype = np.float32
    fact = AbsFactory(input_shape, dtype=dtype)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_abs_input_32x8_dtype_fp16():
    """
    Description:
        1.abs算子正反向测试，input_shape=(32,8), dtype=fp16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (32, 8)
    dtype = np.float16
    fact = AbsFactory(input_shape, dtype=dtype)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_abs_4d_float32():
    """
    Description:
        1.test abs with dynamic shape input, dtype=float32, 4d.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_np = np.random.randn(8, 8, 8, 8).astype(np.float32)
    indices_np = np.unique(np.random.randint(0, 3, size=6).astype(np.int32))
    fact = AbsDynamicShapeMock(input_np, indices_np)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_abs_3d_float32():
    """
    Description:
        1.test abs with dynamic shape input, dtype=float32, 3d.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_np = np.random.randn(128, 128, 32).astype(np.float32)
    indices_np = np.unique(np.random.randint(0, 1, size=5).astype(np.int32))
    fact = AbsDynamicShapeMock(input_np, indices_np)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_abs_6d_float16():
    """
    Description:
        1.test abs with dynamic shape input, dtype=float32, 6d.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_np = np.random.randn(3, 6, 6, 6, 4, 4).astype(np.float32)
    indices_np = np.unique(np.random.randint(1, 3, size=2).astype(np.int32))
    fact = AbsDynamicShapeMock(input_np, indices_np)
    fact.forward_cmp()
