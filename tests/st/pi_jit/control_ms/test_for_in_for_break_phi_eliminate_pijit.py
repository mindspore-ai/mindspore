import pytest
from mindspore.nn import Cell
from mindspore.common import Tensor, dtype
import mindspore.ops.operations as P
from mindspore import Parameter
from mindspore import context, jit
import numpy as np
from tests.mark_utils import arg_mark


class IfInFor(Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(np.ones((1,), dtype=np.int32)), name="w1")
        self.shape = P.Shape()

    @jit(mode="PIJit")
    def construct(self, x, y):
        shape = self.shape(y)
        for _ in range(1):
            if shape[2] % 2 == 0:
                for m in range(0):
                    m -= 1
                    if m > 10:
                        m /= 5
                    x = x + m * self.param
                    if m < 0:
                        break
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_for_in_for_break_phi_node_eliminate():
    """
    Feature: Phi node eliminate.
    Description: For loop created some redundant  phi node, such as phi_range, which will cause some
        problems in infer process.
    Expectation: Compiling success.
    """

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([2])
    y = Tensor(np.ones((2, 2, 2)), dtype.int32)
    net = IfInFor()
    out = net(x, y)
    assert out == Tensor([2])
