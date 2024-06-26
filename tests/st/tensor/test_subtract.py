import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, x, other):
        return x.subtract(other, alpha=2)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_subtract(mode):
    """
    Feature: tensor.subtract()
    Description: Verify the result of tensor.subtract
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor([4, 5, 6], dtype=mstype.float32)
    y = Tensor([1, 2, 3], dtype=mstype.float32)
    output = net(x, y)
    expected = np.array([2, 1, 0], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
