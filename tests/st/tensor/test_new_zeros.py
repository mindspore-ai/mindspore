import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, x, size, dtype):
        return x.new_zeros(size, dtype=dtype)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [None, mstype.int32])
def test_new_zeros(mode, dtype):
    """
    Feature: tensor.new_zeros()
    Description: Verify the result of tensor.new_zeros
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor(np.arange(4).reshape((2, 2)), dtype=mstype.float32)
    output = net(x, (3, 3), dtype)
    expected = np.zeros((3, 3))
    if dtype is None:
        assert output.dtype == mstype.float32
    else:
        assert output.dtype == dtype
    assert np.allclose(output.asnumpy(), expected)
