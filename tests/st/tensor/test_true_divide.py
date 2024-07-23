import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, x, other):
        return x.true_divide(other)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_true_divide(mode):
    """
    Feature: tensor.true_divide()
    Description: Verify the result of tensor.true_divide
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor(np.array([1.0, 2.0, 3.0]), mstype.float32)
    y = Tensor(np.array([4.0, 5.0, 6.0]), mstype.float32)
    output = net(x, y)
    expected = np.array([0.25, 0.4, 0.5], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
