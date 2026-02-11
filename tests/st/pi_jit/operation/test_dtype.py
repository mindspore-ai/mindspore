from ..share.ops.primitive.dtype_ops import DTypeFactory
from ..share.ops.primitive.dtype_ops import DType
import numpy as np
from mindspore import jit, context, Tensor
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1_dtype_fp32():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1), dtype:fp32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1, 1)
    fact = DTypeFactory(input_shape, dtype=np.float32)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=np.float32)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1_dtype_bool():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1), dtype:bool.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1, 1)
    dtype = np.bool_
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1_dtype_fp32():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1), dtype=fp32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1)
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12_dtype_fp32():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12), dtype=fp32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12)
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_12_dtype_fp32():
    """
    Description:
        1. DType算子正向测试，input_shape=(12,), dtype=fp32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (12,)
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_5x1x2x5x1x2x8_dtype_fp32():
    """
    Description:
        1. DType算子正向测试，input_shape=(5,1,2,5,1,2,8), dtype=fp32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (5, 1, 2, 5, 1, 2, 8)
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1x2x3_dtype_fp16():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1,2,3), dtype=fp16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1, 1, 2, 3)
    dtype = np.float16
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1x2_dtype_fp64():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1,2), dtype=fp64.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1, 1, 2)
    dtype = np.float64
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1_dtype_int8():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1), dtype=int8.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1, 1)
    dtype = np.int8
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_forward_input_1x12x1x1_dtype_int16():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1), dtype=int16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1, 1)
    dtype = np.int16
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1_dtype_int32():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1), dtype=int32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1, 1)
    dtype = np.int32
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1_dtype_int64():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1), dtype=int64.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1, 1)
    dtype = np.int64
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1_dtype_uint8():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1), dtype=uint8.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1, 1)
    dtype = np.uint8
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1_dtype_uint16():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1), dtype=uint16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1, 1)
    dtype = np.uint16
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1_dtype_uint32():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1), dtype=uint32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1, 1)
    dtype = np.uint32
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1_dtype_uint64():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1), dtype=uint64.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = (1, 12, 1, 1)
    dtype = np.uint64
    fact = DTypeFactory(input_shape, dtype=dtype)
    net = DType()
    input_np = np.random.randn(*input_shape).astype(dtype=dtype)
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_scalar():
    """
    Description:
        1. DType算子正向测试，input_dtype=np, dtype=fp64.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = np.float32(8.88)
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_int():
    """
    Description:
        1. DType算子正向测试，input_dtype=int.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = 888
    dtype = np.int64
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_float():
    """
    Description:
        1. DType算子正向测试，input_dtype=float.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = 8.88
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1_contains_none():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = np.random.randn(1, 12, 1, 1).astype(np.float32)
    input_np[0, 0, 0, 0] = None
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1_contains_nan():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = np.random.randn(1, 12, 1, 1).astype(np.float32)
    input_np[0, 0, 0, 0] = np.nan
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_1x12x1x1_contains_inf():
    """
    Description:
        1. DType算子正向测试，input_shape=(1,12,1,1).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = np.random.randn(1, 12, 1, 1).astype(np.float32)
    input_np[0, 0, 0, 0] = np.inf
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_tuple_int():
    """
    Description:
        1. DType算子正向测试，input_dtype=tuple, dtype=int.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = (666, 888, 999)
    dtype = np.int64
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_tuple_bool():
    """
    Description:
        1. DType算子正向测试，input_dtype=tuple, dtype=bool.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = (True, False, True)
    dtype = np.bool_
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_tuple_float():
    """
    Description:
        1. DType算子正向测试，input_dtype=tuple, dtype=float.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = (6.66, 8.88, 9.99)
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_tuple_int_float():
    """
    Description:
        1. DType算子正向测试，input_dtype=tuple, dtype=int & float.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = (6.66, 888, 999)
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_tuple_int_bool():
    """
    Description:
        1. DType算子正向测试，input_dtype=tuple, dtype=int & bool.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = (False, 666, 888, 999, True)
    dtype = np.int64
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_tuple_float_bool():
    """
    Description:
        1. DType算子正向测试，input_dtype=tuple, dtype=float & bool.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = (False, 6.66, 8.88, 9.99, True)
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_tuple_int_nan():
    """
    Description:
        1. DType算子正向测试，input_dtype=tuple, dtype=int & nan.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = (np.nan, 666, 888, 999)
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_tuple_int_inf():
    """
    Description:
        1. DType算子正向测试，input_dtype=tuple, dtype=int & inf.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = (np.inf, 666, 888, 999)
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_list_int():
    """
    Description:
        1. DType算子正向测试，input_dtype=list, dtype=int.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = [666, 888, 999]
    dtype = np.int64
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_list_bool():
    """
    Description:
        1. DType算子正向测试，input_dtype=list, dtype=bool.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = [True, False, True]
    dtype = np.bool_
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_list_float():
    """
    Description:
        1. DType算子正向测试，input_dtype=list, dtype=float.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = [6.66, 8.88, 9.99]
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_list_int_float():
    """
    Description:
        1. DType算子正向测试，input_dtype=list, dtype=int & float.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = [6.66, 888, 999]
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_list_int_bool():
    """
    Description:
        1. DType算子正向测试，input_dtype=list, dtype=int & bool.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = [False, 666, 888, 999, True]
    dtype = np.int64
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_list_float_bool():
    """
    Description:
        1. DType算子正向测试，input_dtype=list, dtype=float & bool.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = [False, 6.66, 8.88, 9.99, True]
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_list_int_nan():
    """
    Description:
        1. DType算子正向测试，input_dtype=list, dtype=int & nan.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = [np.nan, 666, 888, 999]
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_list_int_inf():
    """
    Description:
        1. DType算子正向测试，input_dtype=list, dtype=int & inf.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = [np.inf, 666, 888, 999]
    dtype = np.float32
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dtype_input_bool():
    """
    Description:
        1. DType算子正向测试，input_dtype=bool.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_shape = ()
    input_np = True
    dtype = np.bool_
    fact = DTypeFactory(input_shape, dtype=dtype, input_x=input_np)
    net = DType()
    jit(net.construct, capture_mode="bytecode")(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    fact.forward_cmp(net)
    fact.grad_cmp(net)
