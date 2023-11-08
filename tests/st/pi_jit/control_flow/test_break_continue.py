import numpy as np
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import context, jit
from mindspore.common.parameter import Parameter
import pytest

grad_all = C.GradOperation(get_all=True)

class Grad(nn.Cell):
    def __init__(self, net):
        super(Grad, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation(get_all=True)
    @jit(mode="PIJit")
    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_true_break():
    """
    Feature: PIJit
    Description: Test while true with control flow.
    Expectation: No exception.
    """
    class WhileTrueBreakNet(nn.Cell):
        def __init__(self, t):
            super(WhileTrueBreakNet, self).__init__()
            self.add = P.Add()
            self.mul = P.Mul()
            self.para = Parameter(Tensor(t, mstype.int32), name="a")

        @jit(mode="PIJit")
        def construct(self, x, y):
            out = self.mul(y, self.para)
            while True:
                if x == 5:
                    x = x - 3
                    continue
                if x == 2:
                    break
                out = self.add(out, out)
            return out

    context.set_context(mode=context.PYNATIVE_MODE)
    t = np.array([1]).astype(np.int32)
    y = Tensor([1], mstype.int32)
    x = Tensor([5], mstype.int32)
    net = WhileTrueBreakNet(t)
    grad_net = Grad(net)
    grad_out = grad_net(x, y)
    expect = (Tensor([0], mstype.int32), Tensor([1], mstype.int32))
    assert expect == grad_out
