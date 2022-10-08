import numpy as np
import pytest
from mindspore import nn
from mindspore.ops import operations as P
from mindspore import context
from mindspore import Tensor
from .test_grad_of_dynamic import TestDynamicGrad


class NetOneHot(nn.Cell):
    def __init__(self):
        super(NetOneHot, self).__init__()
        self.onehot = P.OneHot()
        self.depth = 3

    def construct(self, indices, on_value, off_value):
        return self.onehot(indices, self.depth, on_value, off_value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_one_hot_shape():
    """
    Feature: OneHot Grad DynamicShape.
    Description: Test case of dynamic shape for OneHot grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetOneHot())
    indices = Tensor(np.array([0, 1, 2]).astype(np.int32))
    on_value = Tensor(np.array([1.0]).astype(np.float32))
    off_value = Tensor(np.array([0.0]).astype(np.float32))
    test_dynamic.test_dynamic_grad_net([indices, on_value, off_value])


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_one_hot_rank():
    """
    Feature: OneHot Grad DynamicRank.
    Description: Test case of dynamic rank for OneHot grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetOneHot())
    indices = Tensor(np.array([0, 1, 2]).astype(np.int32))
    on_value = Tensor(np.array([1.0]).astype(np.float32))
    off_value = Tensor(np.array([0.0]).astype(np.float32))
    test_dynamic.test_dynamic_grad_net([indices, on_value, off_value], True)
