# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.image_ops as P
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self, centered, normalized, uniform_noise, noise):
        super(Net, self).__init__()
        self.op = P.ExtractGlimpse(centered=centered, normalized=normalized,
                                   uniform_noise=uniform_noise, noise=noise)

    def construct(self, x, size, offsets):
        return self.op(x, size, offsets)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_graph1():
    '''
    Feature: ALL To ALL.
    Description: test cases for ExtractGlimpse with centered:False, normalized:False.
    Expectation: The result matches to tensorflow.
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x = Tensor([[[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]], [[6.0], [7.0], [8.0]]]], dtype=mindspore.float32)
    size_ = Tensor((2, 2), dtype=mindspore.int32)
    offsets = Tensor([[1, 1]], dtype=mindspore.float32)
    expect = np.array([[[[0.0], [1.0]], [[3.0], [4.0]]]]).astype(np.float32)
    net = Net(False, False, True, "uniform")
    output = net(x, size_, offsets)
    np.testing.assert_almost_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_graph2():
    '''
    Feature: ALL To ALL.
    Description: test cases for ExtractGlimpse with centered:True, normalized:True.
    Expectation: The result matches to tensorflow.
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x = Tensor([[[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]], [[6.0], [7.0], [8.0]]]], dtype=mindspore.float32)
    size_ = Tensor((2, 2), dtype=mindspore.int32)
    offsets = Tensor([[1, 1]], dtype=mindspore.float32)
    expect = np.array([[[[8.0], [0.0]], [[0.0], [0.0]]]]).astype(np.float32)
    net = Net(True, True, False, "zero")
    output = net(x, size_, offsets)
    np.testing.assert_almost_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_graph3():
    '''
    Feature: ALL To ALL.
    Description: test cases for ExtractGlimpse with centered:True, normalized:False.
    Expectation: The result matches to tensorflow.
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x = Tensor([[[[8.0], [7.0], [6.0]], [[5.0], [4.0], [3.0]], [[2.0], [1.0], [0.0]]]], dtype=mindspore.float32)
    size_ = Tensor((2, 2), dtype=mindspore.int32)
    offsets = Tensor([[0, 0]], dtype=mindspore.float32)
    expect = np.array([[[[8.0], [7.0]], [[5.0], [4.0]]]]).astype(np.float32)
    net = Net(True, False, False, "zero")
    output = net(x, size_, offsets)
    print(output)
    np.testing.assert_almost_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_graph4():
    '''
    Feature: ALL To ALL.
    Description: test cases for ExtractGlimpse with centered:False, normalized:True.
    Expectation: The result matches to tensorflow.
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x = Tensor([[[[8.0], [7.0], [6.0]], [[5.0], [4.0], [3.0]], [[2.0], [1.0], [0.0]]]], dtype=mindspore.float32)
    size_ = Tensor((2, 2), dtype=mindspore.int32)
    offsets = Tensor([[0, 0]], dtype=mindspore.float32)
    expect = np.array([[[[0.0], [0.0]], [[0.0], [8.0]]]]).astype(np.float32)
    net = Net(False, True, False, "zero")
    output = net(x, size_, offsets)
    np.testing.assert_almost_equal(output.asnumpy(), expect)
