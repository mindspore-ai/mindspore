# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


class NetAdaptiveMaxPool2DGrad(nn.Cell):
    def __init__(self):
        super(NetAdaptiveMaxPool2DGrad, self).__init__()
        self.adaptive_max_pool2d_grad_fun = G.AdaptiveMaxPool2DGrad()

    def construct(self, dy, x, argmax):
        return self.adaptive_max_pool2d_grad_fun(dy, x, argmax)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_adaptive_max_pool2d_grad_fp32():
    """
    Feature: test adaptivemaxpool2dgrad op.
    Description: test the ops.
    Expectation: expect correct shape result.
    """
    x = Tensor(np.array([[[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]]]).astype(np.float32))
    dy = Tensor(np.array([[[
        [0.7, 0.9],
        [0.19, 0.21]
    ]]]).astype(np.float32))
    index = Tensor(np.array([[[
        [1, 3],
        [8, 10]
    ]]]).astype(np.int64))
    expect_result = (np.array([[[
        [0., 0.7, 0., 0.9],
        [0., 0., 0., 0.],
        [0.19, 0., 0.21, 0.],
        [0., 0., 0., 0.],
    ]]]))

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    adaptive_max_pool2d_grad = NetAdaptiveMaxPool2DGrad()
    output = adaptive_max_pool2d_grad(dy, x, index)
    assert np.allclose(expect_result, output.asnumpy())

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    adaptive_max_pool2d_grad = NetAdaptiveMaxPool2DGrad()
    output = adaptive_max_pool2d_grad(dy, x, index)
    assert np.allclose(expect_result, output.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_adaptive_max_pool2d_grad_fp16():
    """
    Feature: test adaptivemaxpool2dgrad op.
    Description: test the ops.
    Expectation: expect correct shape result.
    """
    x = Tensor(np.array([[[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]]]).astype(np.float16))
    dy = Tensor(np.array([[[
        [0.7, 0.9],
        [0.19, 0.21]
    ]]]).astype(np.float16))
    index = Tensor(np.array([[[
        [4, 2],
        [12, 10]
    ]]]).astype(np.int64))
    expect_result = (np.array([[[
        [0., 0., 0.9, 0.],
        [0.7, 0., 0., 0.],
        [0., 0., 0.21, 0.],
        [0.19, 0., 0., 0.],
    ]]])).astype(np.float16)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    adaptive_max_pool2d_grad = NetAdaptiveMaxPool2DGrad()
    output = adaptive_max_pool2d_grad(dy, x, index)
    assert np.allclose(expect_result, output.asnumpy())

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    adaptive_max_pool2d_grad = NetAdaptiveMaxPool2DGrad()
    output = adaptive_max_pool2d_grad(dy, x, index)
    assert np.allclose(expect_result, output.asnumpy())
