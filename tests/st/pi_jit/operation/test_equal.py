from ..share.ops.primitive.equal_ops import EqualFactory
from ..share.ops.primitive.equal_ops import EqualMock
from ..share.ops.primitive.equal_ops import Equal
from ..share.utils import allclose_nparray
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import Tensor, jit, context
import mindspore.ops.operations as op
import numpy as np
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_input_245520():
    """
    Description:
        1. Equal算子正向测试，input_shape=(245520,).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(245520,), dtype=np.float16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_input_n():
    """
    Description:
        1. Equal算子正向测试，input_shape=(n,w)，n、m in (64, 96, 128).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    for n in (64, 96, 128):
        for w in (64, 96, 128):
            fact = EqualFactory(input_shape=(n, w), dtype=np.float32)
            fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_input():
    """
    Description:
        1. Equal算子测试，dtype不一致.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_1 = np.random.randn(1, 1).astype(np.float32)
    input_2 = np.random.randn(1, 2).astype(np.float16)
    fact = EqualFactory(input_shape=(1, 2))
    fact.left_input_np = input_1
    fact.right_input_np = input_2
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_normal_outshape_sameas_first_input():
    """
    Description:
        1. Equal算子测试，验证输出的shape与第一个输入相等.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(256, 1), dtype=np.float32)
    net = Equal()
    jit(net.construct, mode="PIJit")(fact.left_input, fact.right_input)
    context.set_context(mode=context.PYNATIVE_MODE)
    out = fact.forward_mindspore_impl(net)
    assert out.shape == (256, 1), out.dtype == ms.bool_


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_dtype_float16_0d():
    """
    Description:
        1. Equal算子正向测试，input_shape=0d，dtype=float16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_list = []
    x0 = Tensor(np.random.randn(70).astype(np.float16))
    input_list.append(x0)
    x1 = Tensor(np.random.randn(70).astype(np.float16))
    input_list.append(x1)
    fact = EqualMock(inputs=input_list)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_dtype_float64_1d():
    """
    Description:
        1. Equal算子正向测试，input_shape=1d，dtype=float64.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(64,), dtype=np.float64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_dtype_float16_2d():
    """
    Description:
        1. Equal算子正向测试，input_shape=2d，dtype=float16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(4, 8), dtype=np.float16)
    fact.forward_cmp()



@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_dtype_float32_3d():
    """
    Description:
        1. Equal算子正向测试，input_shape=3d，dtype=float32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(4, 8, 16), dtype=np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_dtype_int8_4d():
    """
    Description:
        1. Equal算子正向测试，input_shape=4d，dtype=int16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(4, 8, 16), dtype=np.int8)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_dtype_int16_4d():
    """
    Description:
        1. Equal算子正向测试，input_shape=4d，dtype=int16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(4, 8, 16, 8), dtype=np.int16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_dtype_int32_5d():
    """
    Description:
        1. Equal算子正向测试，input_shape=4d，dtype=int32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(4, 8, 16, 8, 4), dtype=np.int32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_dtype_int64_6d():
    """
    Description:
        1. Equal算子正向测试，input_shape=6d，dtype=int64.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(4, 8, 16, 8, 4, 9), dtype=np.int64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_dtype_uint8_7d():
    """
    Description:
        1. Equal算子正向测试，input_shape=7d，dtype=uint8.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(4, 8, 16, 8, 4, 2, 2), dtype=np.uint8)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_dtype_int64():
    """
    Description:
        1. Equal算子正向测试，input_shape=3d，dtype=int64.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(4, 8, 16), dtype=np.int64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_input_num():
    """
    Description:
        1. Equal算子正向测试，input num.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_1 = Tensor(np.array([1]), ms.float32)
    input_2 = 1.0
    net = op.Equal()
    jit(net, mode="PIJit")(input_1, input_2)
    context.set_context(mode=context.PYNATIVE_MODE)
    out = net(input_1, input_2)
    assert out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_input_uint32():
    """
    Description:
        1. Equal算子正向测试，input shape (4, 8) dtype np.uint32 for cpu.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(4, 8, 16), dtype=np.int64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_input_bool_for_gpu():
    """
    Description:
        1. Equal算子正向测试，input bool for gpu.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_1 = np.array([1]).astype(np.bool_)
    input_2 = True
    ps_net = op.Equal()
    jit(ps_net, mode="PIJit")(Tensor(input_1), input_2)
    context.set_context(mode=context.PYNATIVE_MODE)
    out1 = ps_net(Tensor(input_1), input_2)

    pi_net = op.Equal()
    jit(pi_net, mode="PIJit")(Tensor(input_1), input_2)
    context.set_context(mode=context.PYNATIVE_MODE)
    out2 = pi_net(Tensor(input_1), input_2)

    allclose_nparray(out2[0].numpy(), out1[0].asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_input_type_float64_for_gpu():
    """
    Description:
        1. Equal算子正向测试，input float64 for gpu.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(4, 8), dtype=np.float64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_forward_dtype_bool_for_gpu():
    """
    Description:
        1. Equal算子正向测试，input_shape=(4,8,16)，dtype bool for gpu.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = EqualFactory(input_shape=(4, 8), dtype=np.float64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_input_dtype_string():
    """
    Description:
        1. test Equal with input shape from 3d, dtype string.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_list = []
    x0 = Tensor(np.random.randn(8, 7, 1), dtype=mstype.string)
    input_list.append(x0)
    x1 = Tensor(np.random.randn(8, 7, 1), dtype=mstype.string)
    input_list.append(x1)
    fact = EqualMock(inputs=input_list)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_input_dtype_bool():
    """
    Description:
        1. test Equal with input shape from 3d, dtype bool.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_list = []
    x0 = Tensor(np.random.randn(7, 6, 13).astype(np.bool_))
    input_list.append(x0)
    x1 = Tensor(np.random.randn(7, 6, 13).astype(np.bool_))
    input_list.append(x1)
    fact = EqualMock(inputs=input_list)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_equal_input_dtype_bool2():
    """
    Description:
        1. test Equal with input shape from 3d, dtype complex64.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_list = []
    x0 = Tensor(np.random.randn(7, 6, 13).astype(np.bool_))
    input_list.append(x0)
    x1 = Tensor(np.random.randn(7, 6, 13).astype(np.bool_))
    input_list.append(x1)
    fact = EqualMock(inputs=input_list)
    fact.forward_cmp()
