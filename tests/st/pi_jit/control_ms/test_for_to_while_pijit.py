import pytest
from mindspore import context
from mindspore import Tensor, nn
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore import jit
from tests.mark_utils import arg_mark


grad_all = C.GradOperation(get_all=True)


# Although we don't transform for to while any more, we keep this test case.
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_single_for_01():
    """
    Feature: Get single for from abstract.
    Description: Watching the single for func graph from abstract.
    Expectation: Output correct.
    """
    class SingleForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.mul = P.Mul()

        def construct(self, x, y, z):
            x = self.add(x, y)
            for _ in range(0, 3):
                z = self.add(z, x)
            y = self.mul(z, y)
            return y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        @jit(mode="PIJit")
        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([2], mstype.int32)
    y = Tensor([5], mstype.int32)
    z = Tensor([4], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_net = SingleForNet()
    net = GradNet(for_net)
    graph_forward_res = for_net(x, y, z)
    graph_backward_res = net(x, y, z)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    for_net = SingleForNet()
    net = GradNet(for_net)
    pynative_forward_res = for_net(x, y, z)
    pynative_backward_res = net(x, y, z)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res
