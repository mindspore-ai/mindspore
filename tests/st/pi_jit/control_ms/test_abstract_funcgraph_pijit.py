from mindspore.nn import Cell
from mindspore.common import Tensor, dtype
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore import jit, context
import numpy as np
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_watch_get_func_graphs_from_abstract():
    """
    Feature: Get func_graph from abstract.
    Description: Watching the function of getting func graph from abstract.
    Expectation: Output correct.
    """

    class Net(Cell):

        def __init__(self):
            super().__init__()
            self.op = P.Add()

        @jit(mode="PIJit")
        def construct(self, x, y):
            for t in range(2):
                if y != x:
                    if x > 4:
                        x = y / x
                        y = 1 - x
                        y = y - y
                    elif x > 2:
                        y = x - 1
                    else:
                        y = 3 - y
                    y = t * x
                elif x != 3:
                    x = x - x
                if x == y:
                    continue
            return self.op(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    x = np.array([4], np.float32)
    y = np.array([1], np.float32)
    net = Net()
    grad_net = F.grad(net, grad_position=(0, 1))
    fgrad = grad_net(Tensor(x), Tensor(y))
    assert fgrad[0] == Tensor([2], dtype.float32)
    assert fgrad[1] == Tensor([0], dtype.float32)
