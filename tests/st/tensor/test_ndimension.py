import pytest
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, x):
        return x.ndimension()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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
