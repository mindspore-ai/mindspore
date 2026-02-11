from ..share.ops.primitive.div_ops import DivFactory
from ..share.ops.primitive.div_ops import Div
from mindspore import jit, context
import numpy as np
import pytest 
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_245520_245520():
    """
    Description:
        1. div算子测试，input (245520, ), (245520, ).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((245520,), (245520,), dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_512_256():
    """
    Description:
        1. div算子测试，input (512, 256), (512, 256).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((512, 256), (512, 256))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_1024x81x4_1024x81x4():
    """
    Description:
        1. div算子测试，input (1024, 81, 4), (1024, 81, 4).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((1024, 81, 4), (1024, 81, 4))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_32x256x14x14_32x256x14x14():
    """
    Description:
        1. div算子测试，input (1024, 81, 4), (1024, 81, 4).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((32, 256, 14, 14), (32, 256, 14, 14))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_5d_7d():
    """
    Description:
        1. div算子测试，input 5d-7d.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((2, 4, 8, 16, 8), (2, 4, 8, 16, 8), dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()

    fact = DivFactory((2, 4, 8, 16, 8, 4), (2, 4, 8, 16, 8, 4), dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()

    fact = DivFactory((2, 4, 8, 16, 8, 4, 2), (2, 4, 8, 16, 8, 4, 2), dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_scalar_scalar():
    """
    Description:
        1. div算子测试，input scalar, scalar.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((1,), (1,))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_1_128x64():
    """
    Description:
        1. div算子测试，input (1), (128, 64), dtype=np.float16,反向的时候inputx的精度有误差.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((1,), (128, 64), dtype=np.float16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_256x256_256x1():
    """
    Description:
        1. div算子测试，input (256, 256), (256, 1).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((256, 256), (256, 1))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_32x32x7x7_32x32x1x1():
    """
    Description:
        1. div算子测试，input (32, 32, 7, 7), (32, 32, 1, 1).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((32, 32, 7, 7), (32, 32, 1, 1))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_4x4x1_3():
    """
    Description:
        1. div算子测试，input (4, 4, 1), (3).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((4, 4, 1), (3,))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_4x4x4x1_3():
    """
    Description:
        1. div算子测试，input (4, 4, 1), (3).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((4, 4, 4, 1), (3,))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_4x4x2_4x2():
    """
    Description:
        1. div算子测试，input (4, 4, 2), (4, 2).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((4, 4, 2), (4, 2))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_4x4x4x2_4x2():
    """
    Description:
        1. div算子测试，input (4, 4, 2), (4, 2).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((4, 4, 4, 2), (4, 2))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_32x12x128x128_1():
    """
    Description:
        1. div算子测试，input (4, 4, 2), (4, 2).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((32, 12, 128, 128), (1,))
    fact.loss = 0.005
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_forward_input_256x256_int32():
    """
    Description:
        1. div算子正向测试，input (8), (1).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((256, 256), (256, 256), dtype=np.int32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_forward_input_256x256_int64():
    """
    Description:
        1. div算子测试，input (1), (1024, 4096).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((256, 256), (256, 256), dtype=np.int64)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_input_1_1024x4096():
    """
    Description:
        1. div算子测试，input (1), (1024, 4096).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((1,), (1024, 4096))
    fact.loss = 0.005
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_forward_input_2x2_3x2():
    """
    Description:
        1. div算子异常测试，input (2, 2), (3, 2).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((2, 2), (3, 2))
    with pytest.raises(ValueError):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_abnormal_input_2x2_str_2x2_str32():
    """
    Description:
        1. div算子异常测试，input str(2, 2), str(2, 2).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    with pytest.raises(TypeError):
        fact = DivFactory((2, 2), (2, 2), dtype=np.str_)
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_normal_input_1_32x64():
    """
    Description:
        1. div算子测试，input (1), (32, 64).

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((1,), (32, 64))
    pi_net = Div()
    jit(pi_net.construct, capture_mode="bytecode")(fact.inputx_ms, fact.inputy_ms)
    context.set_context(mode=context.PYNATIVE_MODE)
    out = fact.forward_mindspore_impl(pi_net)
    assert out.shape == (32, 64), out.dtype == np.float32


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_div_abnormal_inputy_zero():
    """
    Description:
        1. div算子测试，inputy 0, 除数为0，输出结果为inf.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = DivFactory((5,), (5,))
    fact.inputy = np.array([0, 2, 0, 2, 0], dtype=np.float32)
    fact.forward_cmp()
