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
import mindspore.nn as nn
import mindspore.context as context
import mindspore.ops.operations as ops
from mindspore.ops import composite as C
from mindspore.ops.operations import _inner_ops
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
grad = C.GradOperation(get_all=True)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.args = _inner_ops.DynamicBroadcastGradientArgs()

    def construct(self, s0, s1):
        return self.args(s0, s1)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net():
    """
    Feature: DynamicBroadcastGradientArgs op.
    Description: test cases for DynamicBroadcastGradientArgs op.
    Expectation: the result match expected array.
    """
    shape0 = (4, 2, 1)
    shape1 = (2, 7)
    net = Net()
    r0, r1 = net(shape0, shape1)
    r0_expected = [2]
    r1_expected = [0]

    assert np.array_equal(r0_expected, r0)
    assert np.array_equal(r1_expected, r1)


class NetWrap(nn.Cell):
    def __init__(self):
        super(NetWrap, self).__init__()
        self.shape = ops.Shape()
        self.broadcastto = _inner_ops.DynamicBroadcastTo()

    def construct(self, data, shape):
        shape = self.shape(shape)
        return self.broadcastto(data, shape)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, data, shape):
        gout = grad(self.network)(data, shape)
        return gout


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_broadcast_to_net():
    """
    Feature: Test DynamicBroadcastTo grad process. The input shape is not dynamic.
    Description: The input shape is not dynamic.
    Expectation: Assert that results is right.
    """
    data = Tensor(np.array([1, 2, 3]), mindspore.int64)
    shape = Tensor(np.zeros((2, 3)), mindspore.int64)
    grad_net = GradWrap(NetWrap())
    output = grad_net(data, shape)
    expected_0 = [2, 2, 2]
    expected_1 = [[0, 0, 0], [0, 0, 0]]
    assert np.array_equal(output[0].asnumpy(), expected_0)
    assert np.array_equal(output[1].asnumpy(), expected_1)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_broadcast_to_net_dyn():
    """
    Feature: Test DynamicBroadcastTo grad process. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results is right.
    """
    data = Tensor(np.array([1, 2, 3]), mindspore.int64)
    shape = Tensor(np.zeros((2, 3)), mindspore.int64)
    grad_net = GradWrap(NetWrap())

    data_dyn = [None for _ in data.shape]
    data_input_dyn = Tensor(shape=data_dyn, dtype=data.dtype)

    shape_dyn = [None for _ in shape.shape]
    shape_input_dyn = Tensor(shape=shape_dyn, dtype=shape.dtype)

    grad_net.set_inputs(data_input_dyn, shape_input_dyn)

    output = grad_net(data, shape)
    expected_0 = [2, 2, 2]
    expected_1 = [[0, 0, 0], [0, 0, 0]]
    assert np.array_equal(output[0].asnumpy(), expected_0)
    assert np.array_equal(output[1].asnumpy(), expected_1)
