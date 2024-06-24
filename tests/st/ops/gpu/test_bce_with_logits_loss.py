# Copyright 2021 Huawei Technologies Co., Ltd
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

import math
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, reduction):
        super(Net, self).__init__()
        self.loss = P.BCEWithLogitsLoss(reduction=reduction)

    def construct(self, predict, target, weight, pos_weight):
        return self.loss(predict, target, weight, pos_weight)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_reduction_none_testcases():
    # fp32 + both modes
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    loss = Net("none")
    predict = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]).astype(np.float32))
    target = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(np.float32))
    weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(np.float32))
    pos_weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(np.float32))
    output = loss(predict, target, weight, pos_weight)
    expected = np.array([[0.6111006, 0.5032824, 0.26318598],
                         [0.58439666, 0.55301523, -0.436814]]).astype(np.float32)
    np.testing.assert_almost_equal(expected, output.asnumpy())
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    loss = Net("none")
    predict = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]).astype(np.float32))
    target = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(np.float32))
    weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(np.float32))
    pos_weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(np.float32))
    output = loss(predict, target, weight, pos_weight)
    expected = np.array([[0.6111006, 0.5032824, 0.26318598],
                         [0.58439666, 0.55301523, -0.436814]])
    np.testing.assert_almost_equal(expected, output.asnumpy())
    # fp16
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    loss = Net("none")
    predict = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]).astype(np.float16))
    target = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(np.float16))
    weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(np.float16))
    pos_weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(np.float16))
    output = loss(predict, target, weight, pos_weight)
    expected = np.array([[0.611, 0.503, 0.2627],
                         [0.584, 0.5527, -0.437]]).astype(np.float16)
    np.testing.assert_almost_equal(expected, output.asnumpy(), decimal=3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_reduction_mean_testcases():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    loss = Net("mean")
    predict = Tensor(np.arange(6).reshape(2, 3).astype(np.float32))
    target = Tensor(np.arange(34, 40).reshape(2, 3).astype(np.float32))
    weight = Tensor(np.array([2, 3, 1]).astype(np.float32))
    pos_weight = Tensor(np.array([6, 3, 4]).astype(np.float32))
    output = loss(predict, target, weight, pos_weight)
    expected = -113.55404
    # assert scalar
    assert math.isclose(output.asnumpy().tolist(), expected, abs_tol=0.00001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_reduction_sum_testcases():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    loss = Net("sum")
    predict = Tensor(np.arange(6, 12).reshape(2, 3).astype(np.float32))
    target = Tensor(np.arange(6).reshape(2, 3).astype(np.float32))
    weight = Tensor(np.array([3, 3, 4]).astype(np.float32))
    pos_weight = Tensor(np.array([6, 3, 4]).astype(np.float32))
    output = loss(predict, target, weight, pos_weight)
    expected = -333.96677
    # assert scalar
    assert math.isclose(output.asnumpy().tolist(), expected, abs_tol=0.00001)
