from tests.mark_utils import arg_mark
import numpy as np

import pytest
from mindspore import nn
from mindspore.ops.operations import _grad_ops as G
from mindspore import Tensor
from mindspore import context
from .test_grad_of_dynamic import TestDynamicGrad


class NetPoolGrad(nn.Cell):
    def __init__(self):
        super(NetPoolGrad, self).__init__()
        self.maxpool_grad_fun = G.MaxPoolGrad(pad_mode="VALID",
                                              kernel_size=2,
                                              strides=2)

    def construct(self, x, a, d):
        return self.maxpool_grad_fun(x, a, d)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_mod_shape():
    """
    Feature: Mod Grad DynamicShape.
    Description: Test case of dynamic shape for Mod grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetPoolGrad())
    x = Tensor(np.array([[[
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35]
    ]]]).astype(np.float32))
    d = Tensor(np.array([[[
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3]
    ]]]).astype(np.float32))
    a = Tensor(np.array([[[
        [7, 9, 11],
        [19, 21, 23],
        [31, 33, 35]
    ]]]).astype(np.float32))
    test_dynamic.test_dynamic_grad_net([x, a, d])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_mod_rank():
    """
    Feature: Mod Grad DynamicRank.
    Description: Test case of dynamic rank for Mod grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetPoolGrad())
    x = Tensor(np.array([[[
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35]
    ]]]).astype(np.float32))
    d = Tensor(np.array([[[
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3]
    ]]]).astype(np.float32))
    a = Tensor(np.array([[[
        [7, 9, 11],
        [19, 21, 23],
        [31, 33, 35]
    ]]]).astype(np.float32))
    test_dynamic.test_dynamic_grad_net([x, a, d], True)
