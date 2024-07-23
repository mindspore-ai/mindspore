# Copyright 2020-2023 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Size()

    def construct(self, x):
        return self.ops(x)


class NetGrad(nn.Cell):
    def __init__(self, forward):
        super().__init__()
        self.forward = forward
        self.grad = C.GradOperation(get_all=True)

    def construct(self, x):
        return self.grad(self.forward)(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_size_1_dimension(mode):
    """
    Feature: test pynative mode and graph mode
    Description: Test 1-D Tensor
    Expectation: the result match to expected value
    """
    np_array = np.array([2, 3, 4]).astype(np.int32)
    input_x = Tensor(np_array)
    expect = 3
    net = Net()
    out = net(input_x)
    assert out == expect

    size_grad = NetGrad(net)
    actual_grad = size_grad(input_x)
    expect_grad = np.zeros(3).astype(np.int32)
    assert (actual_grad[0].asnumpy() == expect_grad).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_size_2_dimension(mode):
    """
    Feature: test pynative mode and graph mode
    Description: Test 2-D Tensor
    Expectation: the result match to expected value
    """
    np_array = np.array([[2, 2], [2, 2], [3, 3]]).astype(np.int32)
    input_x = Tensor(np_array)
    expect = 6
    net = Net()
    out = net(input_x)
    assert out == expect

    size_grad = NetGrad(net)
    actual_grad = size_grad(input_x)
    expect_grad = np.zeros((3, 2)).astype(np.int32)
    assert (actual_grad[0].asnumpy() == expect_grad).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_size_3_dimension(mode):
    """
    Feature: test pynative mode and graph mode
    Description: Test 3-D Tensor
    Expectation: the result match to expected value
    """
    np_array = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]]]).astype(np.int32)
    input_x = Tensor(np_array)
    expect = 12
    net = Net()
    out = net(input_x)
    assert out == expect

    size_grad = NetGrad(net)
    actual_grad = size_grad(input_x)
    expect_grad = np.zeros((3, 2, 2)).astype(np.int32)
    assert (actual_grad[0].asnumpy() == expect_grad).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_size_dynamic(mode):
    """
    Feature: test pynative mode and graph mode
    Description: Test dynamic shape
    Expectation: the result match to expected value
    """
    context.set_context(mode=mode)
    net = Net()
    input_x_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    net.set_inputs(input_x_dyn)
    input_x = Tensor(np.random.random(([3, 4])), dtype=ms.float32)
    output = net(input_x)
    expect = 12
    assert output == expect

    size_grad = NetGrad(net)
    actual_grad = size_grad(input_x)
    expect_grad = np.zeros((3, 4)).astype(np.int32)
    assert (actual_grad[0].asnumpy() == expect_grad).all()
