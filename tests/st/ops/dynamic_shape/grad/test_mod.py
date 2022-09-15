import numpy as np
import pytest
import mindspore as mp
from mindspore import nn, context, Tensor
from mindspore.ops.operations.math_ops import Mod
from .test_grad_of_dynamic import TestDynamicGrad


class NetMod(nn.Cell):
    def __init__(self):
        super(NetMod, self).__init__()
        self.mod = Mod()

    def construct(self, x1, x2):
        return self.mod(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_mod_shape():
    """
    Feature: Mod Grad DynamicShape.
    Description: Test case of dynamic shape for Mod grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetMod())
    x = Tensor(np.array([-4.0, 5.0, 6.0]), mp.float32)
    y = Tensor(np.array([3.0, 2.0, 3.0]), mp.float32)
    test_dynamic.test_dynamic_grad_net([x, y])


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_mod_rank():
    """
    Feature: Mod Grad DynamicRank.
    Description: Test case of dynamic rank for Mod grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetMod())
    x = Tensor(np.array([-4.0, 5.0, 6.0]), mp.float32)
    y = Tensor(np.array([3.0, 2.0, 3.0]), mp.float32)
    test_dynamic.test_dynamic_grad_net([x, y], True)
