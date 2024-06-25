from mindspore.nn import Cell
from mindspore.common import dtype
from mindspore.common import Tensor
import mindspore.ops.operations as P
from mindspore import context, jit
from ..share.utils import allclose_nparray
from tests.mark_utils import arg_mark


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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    ps_net = CtrlWhile2ElifReturnInIf()
    jit(fn=CtrlWhile2ElifReturnInIf.construct, mode="PSJit")(ps_net, Tensor(x, dtype.float32))
    ps_out = ps_net(Tensor(x, dtype.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlWhile2ElifReturnInIf()
    jit(fn=CtrlWhile2ElifReturnInIf.construct, mode="PIJit")(pi_net, Tensor(x, dtype.float32))
    pi_out = pi_net(Tensor(x, dtype.float32))
    allclose_nparray(ps_out.asnumpy(), pi_out.asnumpy(), 0.001, 0.001)
