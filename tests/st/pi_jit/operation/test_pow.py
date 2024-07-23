import pytest
import numpy as np
from mindspore import context, jit
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops.operations as op
from mindspore.common.tensor import Tensor
from ..share.ops.primitive.pow_ops import PowFactory
from ..share.utils import allclose_nparray
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_25x51():
    """
    Feature: Ops.
    Description: pow算子测试， input 25x51.
    Expectation: expect correct result.
    """
    exp_np = 2.000000
    fact = PowFactory(input_shape=(25, 51), exp=exp_np, dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_nx512():
    """
    Feature: Ops.
    Description: pow算子测试，input (64, 96, 128)x512.
    Expectation: expect correct result.
    """
    for n in (64, 96, 128):
        exp_np = 2.000000
        fact = PowFactory(input_shape=(n, 512), exp=exp_np, dtype=np.float16)
        fact.forward_cmp()
        fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_256_512():
    """
    Feature: Ops.
    Description: pow算子测试，input (256, 512).
    Expectation: expect correct result.
    """
    exp_np = 2
    fact = PowFactory(input_shape=(256, 512), exp=exp_np, dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_512_256():
    """
    Feature: Ops.
    Description: pow算子测试，input (512, 256).
    Expectation: expect correct result.
    """
    exp_np = np.absolute(np.random.randn())
    fact = PowFactory(input_shape=(512, 256), exp=exp_np, dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_scalar_exp_scalar_invalid():
    """
    Feature: Ops.
    Description: pow算子测试，input=-1.35, exp=2.35.
    Expectation: expect correct result.
    """
    fact = PowFactory(input_shape=(1, 1), exp=2.35, dtype=np.float32)
    fact.input = -1.35
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_3x5x2x2_exp_tensor():
    """
    Feature: Ops.
    Description: pow算子测试，input (3, 5, 2, 2), exp ().
    Expectation: expect correct result.
    """
    exp_np = np.absolute(np.random.randn())
    fact = PowFactory(input_shape=(3, 5, 2, 2), exp=exp_np, dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_3x5x2x2x12_exp_2():
    """
    Feature: Ops.
    Description: pow算子测试，input (3, 5, 2, 2, 12).
    Expectation: expect correct result.
    """
    fact = PowFactory(input_shape=(3, 5, 2, 2, 12), exp=2.00000, dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_3x5x2x2x1x1_exp_bool():
    """
    Feature: Ops.
    Description: pow算子测试，input (3, 5, 2, 2, 12, 2).
    Expectation: expect correct result.
    """
    fact = PowFactory(input_shape=(3, 5, 2, 2, 12, 2), exp=True, dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_3x5x2x2x12x2x32_exp_tensor():
    """
    Feature: Ops.
    Description: pow算子测试，input (3, 5, 2, 2, 12, 2, 32).
    Expectation: expect correct result.
    """
    exp_np = np.absolute(np.random.randn(), dtype=np.float16)
    fact = PowFactory(input_shape=(3, 5, 2, 2, 12, 2, 32), exp=Tensor(exp_np), dtype=np.float16)
    fact.forward_cmp()
    fact.exp = exp_np.astype(np.float16)
    fact.exp = Tensor(exp_np.astype(np.float16))
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_exp_not_broadcastable():
    """
    Feature: Ops.
    Description: pow算子测试，input (2, 2), exp (3, 2).
    Expectation: expect correct result.
    """
    exp_np = np.random.randn(3, 2).astype(np.float32)
    fact = PowFactory(input_shape=(2, 2), exp=Tensor(exp_np, ms.float32))
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_1_scalar():
    """
    Feature: Ops.
    Description: pow算子测试，input (2, 2), exp (3, 2).
    Expectation: expect correct result.
    """
    exp_np = np.absolute(np.random.randn(), dtype=np.float16)
    fact = PowFactory(input_shape=(1,), exp=Tensor(exp_np), dtype=np.float32)
    fact.forward_cmp()
    fact.exp = exp_np.astype(np.float32)
    fact.exp = Tensor(fact.exp)
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_1_1():
    """
    Feature: Ops.
    Description: pow算子测试，input (2, 2), exp (3, 2).
    Expectation: expect correct result.
    """
    exp_np = np.abs(np.random.randn(1))
    fact = PowFactory(input_shape=(1,), exp=Tensor(exp_np, ms.int32))
    fact.forward_cmp()
    fact.exp = exp_np.astype(np.float32)
    fact.exp = Tensor(fact.exp)
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_scalar_negative_exp_scalar_positive_2():
    """
    Feature: Ops.
    Description: pow算子测试，input (1), exp 2.0.
    Expectation: expect correct result.
    """
    exp = 2.0
    fact = PowFactory(input_shape=(1,), exp=exp, dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_scalar_negative_exp_scalar_positive():
    """
    Feature: Ops.
    Description: pow算子测试，input (1), exp (2.5).
    Expectation: expect correct result.
    """
    exp = 2.5
    fact = PowFactory(input_shape=(1,), exp=exp, dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_exp_broadcastable_2d():
    """
    Feature: Ops.
    Description: pow算子测试，input (2, 2), exp (3, 2).
    Expectation: expect correct result.
    """
    exp_np = np.random.randn(1, 2).astype(np.float32)
    fact = PowFactory(input_shape=(2, 2), exp=Tensor(exp_np, ms.float32))
    fact.forward_cmp()
    fact.exp = exp_np.astype(np.float32)
    fact.exp = Tensor(fact.exp)
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_num_exp_tensor():
    """
    Feature: Ops.
    Description: pow算子测试，input 3.0, exp tensor.
    Expectation: expect correct result.
    """
    class Net(Cell):
        def __init__(self, input_np):
            super(Net, self).__init__()
            self.pow = op.Pow()
            self.input_np = input_np

        @jit(mode="PIJit")
        def construct(self, exp):
            return self.pow(input_np, exp)

    input_np = 3.0
    exp = Tensor(2, dtype=ms.float32)
    pow_net = Net(input_np)
    jit(pow_net.construct, mode="PSJit")(exp)
    context.set_context(mode=context.GRAPH_MODE)
    psjit_out = pow_net(exp)

    pow_net = Net(input_np)
    jit(pow_net.construct, mode="PIJit")(exp)
    context.set_context(mode=context.PYNATIVE_MODE)
    pijit_out = pow_net(exp)

    allclose_nparray(pijit_out.asnumpy(), psjit_out.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_float_exp_tensor():
    """
    Feature: Ops.
    Description: pow算子测试，input 3.0, exp tensor.
    Expectation: expect correct result.
    """
    class Net(Cell):
        def __init__(self, input_np):
            super(Net, self).__init__()
            self.pow = op.Pow()
            self.input_np = input_np

        def construct(self, exp):
            return self.pow(input_np, exp)

    input_np = True
    exp = Tensor(2, dtype=ms.float32)
    net = Net(input_np)
    jit(net.construct, mode="PSJit")
    context.set_context(mode=context.GRAPH_MODE)
    psjit_out = net(exp)

    net = Net(input_np)
    jit(net.construct, mode="PSJit")
    context.set_context(mode=context.GRAPH_MODE)
    pijit_out = net(exp)

    allclose_nparray(pijit_out.asnumpy(), psjit_out.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_bool_exp_tensor():
    """
    Feature: Ops.
    Description: pow算子测试，input bool, exp tensor.
    Expectation: expect correct result.
    """
    class Net(Cell):
        def __init__(self, input_np):
            super(Net, self).__init__()
            self.pow = op.Pow()
            self.input_np = input_np

        def construct(self, exp):
            return self.pow(input_np, exp)

    input_np = True
    exp = Tensor(2, dtype=ms.float32)
    net = Net(input_np)
    jit(net.construct, mode="PSJit")(exp)
    context.set_context(mode=context.GRAPH_MODE)
    psjit_out = net(exp)
    net = Net(input_np)
    jit(net.construct, mode="PIJit")(exp)
    context.set_context(mode=context.PYNATIVE_MODE)
    pijit_out = net(exp)

    allclose_nparray(pijit_out.asnumpy(), psjit_out.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_exp_tensor_bool():
    """
    Feature: Ops.
    Description: pow算子测试，input (2, 2), exp Tensor(True).
    Expectation: expect correct result.
    """
    exp = Tensor(True, ms.bool_)
    fact = PowFactory(input_shape=(2, 2), exp=exp)
    fact.forward_cmp()

    fact.exp = fact.exp.asnumpy().astype(np.float32)
    fact.exp = exp
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_pow_input_exp_bool():
    """
    Feature: Ops.
    Description: pow算子测试，input (2, 2), exp True.
    Expectation: expect correct result.
    """
    fact = PowFactory(input_shape=(2, 2), exp=False)
    fact.forward_cmp()
    fact.grad_cmp()
