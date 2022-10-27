import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, x):
        return x.triu(diagonal=1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_subtract(mode):
    """
    Feature: tensor.subtract()
    Description: Verify the result of tensor.subtract
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor(np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [10, 11, 12, 13],
                         [14, 15, 16, 17]]))
    output = net(x)
    expected = np.array([[0, 2, 3, 4],
                         [0, 0, 7, 8],
                         [0, 0, 0, 13],
                         [0, 0, 0, 0]])
    assert np.array_equal(output.asnumpy(), expected)
