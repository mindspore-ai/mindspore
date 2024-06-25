import pytest
import numpy as np
from mindspore import Tensor, context, jit
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import GradOperation
from tests.mark_utils import arg_mark


class GradOperationNet(nn.Cell):
    def __init__(self, net, get_all=False, get_by_list=False):
        super().__init__()
        self.net = net
        self.grad_op = GradOperation(get_all=get_all, get_by_list=get_by_list)

    def construct(self, *args):
        gradient_function = self.grad_op(self.net)
        return gradient_function(*args)


class NoneNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.Add()

    @jit(mode="PIJit")
    def func(self, z):
        if z == 0:
            out = None
        else:
            out = None
        return out, z

    def construct(self, x, y, z):
        if self.func(z)[0] is None:
            out = self.add(x, y)
        else:
            out = None
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('x', [np.array([1.2, 1.3, 1.1]).astype(np.float32)])
@pytest.mark.parametrize('y', [np.array([0.01, 0.3, 1.1]).astype(np.float32)])
@pytest.mark.parametrize('z', [np.array([0]).astype(np.float32)])
@pytest.mark.parametrize('GradOperationNet1', [GradOperationNet])
@pytest.mark.parametrize('NoneNet1', [NoneNet])
def test_parser_fallback_none_control_flow(x, y, z, GradOperationNet1, NoneNet1):
    """
    Feature: Test control flow with None type in MindSpore.
    Description: This test verifies that control flow statements involving 'None' type are handled correctly.
                 We use GradOperationNet and NoneNet for the test.
    Expectation: The gradients should be computed correctly without any errors.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net_ms = NoneNet1()
    ms_grad = GradOperationNet1(net_ms)(*[Tensor(i) for i in [x, y, z]])
    assert np.allclose([1, 1, 1], ms_grad.asnumpy(), 1e-4, 1e-4)
