from mindspore import context, jit
from mindspore.nn import Cell
from mindspore.common import dtype
from mindspore.common import Tensor
import mindspore.ops.functional as F
import numpy as np
from ..share.grad import GradOfFirstInput
from tests.mark_utils import arg_mark


class Net1(Cell):
    def __init__(self):
        super().__init__()
        self.a = Tensor([True], dtype.bool_)
        self.b = Tensor([False], dtype.bool_)
        self.flag = True

    def construct(self, x):
        out = x
        if self.a:
            out = out * x
        while self.b:
            out = out + x
        if self.a and self.b:
            out = 2 * out
        elif self.a or self.b:
            out = out - x
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_tensor_bool():
    """
    TEST_SUMMARY:
    Description: create a net use bool tensor as condition
    Expectation: result match
    """
    npx = np.random.rand(3, 4).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net1()
    jit(fn=Net1.construct, mode="PIJit", jit_config={"loop_unrolling":True})(pi_net, Tensor(npx))
    grad_net = F.grad(pi_net)
    pi_net(Tensor(npx))
    grad_net(Tensor(npx))


class Net2(Cell):
    def __init__(self):
        super().__init__()
        self.a = Tensor([True], dtype.bool_)

    def construct(self, x):
        out = x
        if self.a and x > 1:
            out = out + x
        else:
            out = out + 2 * x
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_tensor_bool_with_x():
    """
    TEST_SUMMARY:
    Description: create a net use bool tensor as condition
    Expectation: result match
    """
    x = Tensor([0], dtype.float32)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net2()
    jit(fn=Net2.construct, mode="PIJit")(pi_net, x)
    grad_net = GradOfFirstInput(pi_net, sens_param=False)
    pi_net(x)
    grad_net(x)
