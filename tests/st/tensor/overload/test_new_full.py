import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, x, size, fill_value, dtype):
        return x.new_full(size, fill_value, dtype=dtype)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [None, mstype.int32])
def test_new_full(mode, dtype):
    """
    Feature: tensor.new_full()
    Description: Verify the result of tensor.new_full
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor(np.arange(4).reshape((2, 2)), dtype=mstype.float32)
    fill_value = 1
    output = net(x, (3, 3), fill_value, dtype)
    expected = np.ones((3, 3))
    if dtype is None:
        assert output.dtype == mstype.float32
    else:
        assert output.dtype == dtype
    assert np.allclose(output.asnumpy(), expected)
