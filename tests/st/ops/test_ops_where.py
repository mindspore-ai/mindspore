import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, condition, x, y):
        return ops.where(condition, x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_where(mode):
    """
    Feature: ops.where
    Description: Verify the result of ops.where
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor(np.arange(4).reshape((2, 2)), mstype.float32)
    y = Tensor(np.ones((2, 2)), mstype.float32)
    condition = x < 3
    output = net(condition, x, y)
    expected = np.array([[0, 1], [2, 1]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
