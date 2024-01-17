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
import numpy as np
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.resize = P.ResizeBilinearV2(False)

    def construct(self, x):
        return self.resize(x, (2, 4))


class NetCenter(nn.Cell):
    def __init__(self):
        super(NetCenter, self).__init__()
        self.resize = P.ResizeBilinearV2(False, True)

    def construct(self, x):
        return self.resize(x, (2, 4))


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network
        self.network.set_train()

    def construct(self, x, y):
        return self.grad(self.network)(x, y)


def net_float16():
    tensor = Tensor([[[[1, 2, 3, 4, 5], [2, 4, 6, 4, 9]]]], mindspore.float16)
    net = Net()
    output = net(tensor)
    return output


def test_net_grad():
    net = Grad(Net())
    x = Tensor([[[[1, 2, 3, 4, 5], [2, 4, 6, 4, 9]]]], mindspore.float16)
    y = net_float16()
    dy = Tensor([[[[1, 2, 3, 4], [2, 4, 6, 4]]]], mindspore.float16)
    dy = P.Cast()(dy, mindspore.float32)
    dx = net(x, dy)
    print("forward input: ", x)
    print("forward output: ", y)
    print("backward input: ", dy)
    print("backward output: ", dx)

    y_expect = np.array([[[[1.0, 2.25, 3.5, 4.75],
                           [2.0, 4.5, 5.0, 7.75]]]])
    dx_expect = np.array([[[[1.0, 1.5, 2.0, 2.5, 3.0],
                            [2.0, 3.0, 4.0, 4.0, 3.0]]]])
    assert np.array_equal(y_expect, y.asnumpy())
    assert np.array_equal(dx_expect, dx[0].asnumpy())


def net_center_float16():
    tensor = Tensor([[[[1, 2, 3, 4, 5], [2, 4, 6, 4, 9]]]], mindspore.float16)
    net = NetCenter()
    output = net(tensor)
    return output


def test_net_center_grad():
    """
    Feature: Test ResizeBilinear operator in args with align_corners=False and half_pixel_centers=True.
    Description: input with half_pixel_centers=True.
    Expectation: align_corners and half_pixel_centers are all True.
    """
    net = Grad(NetCenter())
    x = Tensor([[[[1, 2, 3, 4, 5], [2, 4, 6, 4, 9]]]], mindspore.float16)
    y = net_center_float16()
    grad = Tensor([[[[1, 2, 3, 4], [2, 4, 6, 4]]]], mindspore.float16)
    grad = P.Cast()(grad, mindspore.float32)
    dx = net(x, grad)
    print("forward input: ", x)
    print("forward output: ", y)
    print("backward input: ", grad)
    print("backward output: ", dx)

    y_expect = np.array([[[[1.0, 2.25, 3.5, 4.75],
                           [2.0, 4.5, 5.0, 7.75]]]])
    dx_expect = np.array([[[[1.0, 1.5, 2.0, 2.5, 3.0],
                            [2.0, 3.0, 4.0, 4.0, 3.0]]]])
    assert np.array_equal(y_expect, y.asnumpy())
    assert np.array_equal(dx_expect, dx[0].asnumpy())
