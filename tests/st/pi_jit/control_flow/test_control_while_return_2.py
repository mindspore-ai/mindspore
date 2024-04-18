from mindspore.nn import Cell
from mindspore.common import dtype
from mindspore.common import Tensor
import mindspore.ops.operations as P
from mindspore import context, jit
from ..share.utils import allclose_nparray
import pytest


class CtrlWhile2ElifReturnInIf(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x):
        while x[2] < 4:
            x[2] -= 1
            if x[0] > 2:
                return x
            elif x[1] > 2:
                x[2] += 1
            elif x[2] > 2:
                x[1] += 1
            else:
                x = self.mul(x, x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_control_flow_while_2elif_return_in_if():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with return in if, use get_item
        2. run the net
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as torch
    '''
    x = [1, 2, 3]
    context.set_context(mode=context.GRAPH_MODE)
    jit(fn=CtrlWhile2ElifReturnInIf.construct, mode="PSJit")
    ps_net = CtrlWhile2ElifReturnInIf()
    ps_out = ps_net(Tensor(x, dtype.float32))

    context.set_context(mode=context.PYNATIVE_MODE)
    jit(fn=CtrlWhile2ElifReturnInIf.construct, mode="PIJit")
    pi_net = CtrlWhile2ElifReturnInIf()
    pi_out = pi_net(Tensor(x, dtype.float32))
    allclose_nparray(ps_out.asnumpy(), pi_out.asnumpy(), 0.001, 0.001)
