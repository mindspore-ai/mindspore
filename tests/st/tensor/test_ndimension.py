import pytest
from tests.mark_utils import arg_mark
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, x):
        return x.ndimension()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ndimension(mode):
    """
    Feature: tensor.ndimension()
    Description: Verify the result of tensor.ndimension
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor([[1.5, 2.0], [3, 4.6], [0.3, 4.6]], dtype=mstype.float32)
    output = net(x)
    expected = 2
    assert output == expected
