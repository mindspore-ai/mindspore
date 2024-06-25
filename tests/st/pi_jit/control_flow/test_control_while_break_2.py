from mindspore.nn import Cell
from mindspore.common import dtype
from mindspore.common import Tensor
import mindspore.ops.operations as P
from mindspore import context, jit
from ..share.utils import allclose_nparray
from tests.mark_utils import arg_mark


class CtrlWhile2ElifBreakInIf(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x):
        while x[2] < 4:
            x[2] -= 1
            if x[0] > 2:
                break
            elif x[1] > 2:
                x[2] += 1
            elif x[2] > 2:
                x[1] += 1
            else:
                x = self.mul(x, x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_break_in_if():
    '''
    Description: test control flow, 2elif in while, break in if
    use tensor get_item, set_item as condition, torch not supports grad
    graph mode set item change inputs, cause load mindir endless loop
    Expectation: no expectation
    '''
    x = [1, 2, 3]
    context.set_context(mode=context.GRAPH_MODE)
    jit(fn=CtrlWhile2ElifBreakInIf.construct, mode="PSJit")
    ps_net = CtrlWhile2ElifBreakInIf()
    ps_out = ps_net(Tensor(x, dtype.float32))
    pi_net = CtrlWhile2ElifBreakInIf()
    pi_out = pi_net(Tensor(x, dtype.float32))
    allclose_nparray(ps_out.asnumpy(), pi_out.asnumpy(), 0.001, 0.001)
