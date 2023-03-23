import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, x, other):
        return ops.true_divide(x, other)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_true_divide(mode):
    """
    Feature: ops.true_divide
    Description: Verify the result of ops.true_divide
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor(np.array([1.0, 2.0, 3.0]), mstype.float32)
    y = Tensor(np.array([4.0, 5.0, 6.0]), mstype.float32)
    output = net(x, y)
    expected = np.array([0.25, 0.4, 0.5], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
