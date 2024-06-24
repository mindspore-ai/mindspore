from tests.mark_utils import arg_mark
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
