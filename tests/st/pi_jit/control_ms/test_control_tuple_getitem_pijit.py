import pytest
from mindspore.nn import Cell
from mindspore.common import Tensor, dtype, Parameter
from mindspore import jit, context
import numpy as np
from tests.mark_utils import arg_mark


class Net(Cell):

    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor([(- 1)], dtype.float32), name='weight')
        self.b = Parameter(Tensor([(- 5)], dtype.float32), name='bias')

    @jit(mode="PIJit")
    def construct(self, x, y):
        if y == x:
            for a in range(2):
                x = x - y
                self.w = a * x
                if self.w < 0:
                    return x
        elif self.b >= x:
            for a in range(2):
                x = x - x
                y = y - 3
        return x + y


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tuple_getitem_err():
    """
    Feature: Control flow.
    Description: This test case failed before, add it to CI. Related issue: I5G160.
    Expectation: No exception raised.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = np.array([2], np.float32)
    y = np.array([1], np.float32)
    net = Net()
    out = net(Tensor(x), Tensor(y))
    assert out == Tensor([3], dtype.float32)
