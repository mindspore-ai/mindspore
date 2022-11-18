import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import ops


class NetNone(nn.Cell):
    def construct(self, x, other):
        return ops.divide(x, other)


class NetFloor(nn.Cell):
    def construct(self, x, other):
        return ops.divide(x, other, rounding_mode="floor")


class NetTrunc(nn.Cell):
    def construct(self, x, other):
        return ops.divide(x, other, rounding_mode="trunc")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_divide_none(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode)
    net = NetNone()
    x = Tensor(np.array([1.0, 5.0, 7.5]), mstype.float32)
    y = Tensor(np.array([4.0, 2.0, 3.0]), mstype.float32)
    output = net(x, y)
    expected = np.array([0.25, 2.5, 2.5], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_divide_floor(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode)
    net = NetFloor()
    x = Tensor(np.array([1.0, 5.0, 9.5]), mstype.float32)
    y = Tensor(np.array([4.0, 2.0, 3.0]), mstype.float32)
    output = net(x, y)
    expected = np.array([0.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_divide_trunc(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode)
    net = NetTrunc()
    x = Tensor(np.array([1.0, 5.0, 9.5]), mstype.float32)
    y = Tensor(np.array([4.0, 2.0, 3.0]), mstype.float32)
    output = net(x, y)
    expected = np.array([0.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
