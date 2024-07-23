from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, x, other):
        return ops.subtract(x, other, alpha=2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_subtract(mode):
    """
    Feature: ops.subtract()
    Description: Verify the result of ops.subtract
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor([4, 5, 6], dtype=mstype.float32)
    y = Tensor([1, 2, 3], dtype=mstype.float32)
    output = net(x, y)
    expected = np.array([2, 1, 0], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
