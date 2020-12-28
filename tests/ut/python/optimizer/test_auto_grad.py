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
import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C

grad_all = C.GradOperation(get_all=True)


class CropAndResizeNet(nn.Cell):
    def __init__(self, crop_size):
        super(CropAndResizeNet, self).__init__()
        self.crop_and_resize = P.CropAndResize()
        self.crop_size = crop_size

    def construct(self, x, boxes, box_indices):
        return self.crop_and_resize(x, boxes, box_indices, self.crop_size)

    def bprop(self, x, boxes, box_indices, out, dout):
        return x, boxes, box_indices


class TestUserDefinedBpropNet(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(TestUserDefinedBpropNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=1, has_bias=False,
                              weight_init='ones', pad_mode='same')
        self.crop = CropAndResizeNet((10, 10))
        self.boxes = Tensor(np.ones((128, 4)).astype(np.float32))
        self.box_indices = Tensor(np.ones((128,)).astype(np.int32))

    def construct(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.crop(x, self.boxes, self.box_indices)
        return x


class TestUserDefinedBpropGradNet(nn.Cell):
    def __init__(self, net):
        super(TestUserDefinedBpropGradNet, self).__init__()
        self.net = net

    def construct(self, x):
        return grad_all(self.net)(x)


def test_user_defined_bprop():
    context.set_context(mode=context.GRAPH_MODE)
    net = TestUserDefinedBpropNet(3, 10)
    grad_net = TestUserDefinedBpropGradNet(net)
    x = Tensor(np.ones((128, 3, 12, 12)).astype(np.float32))
    grad_net(x)


class SinNet(nn.Cell):
    def __init__(self):
        super(SinNet, self).__init__()
        self.sin = ops.Sin()

    def construct(self, x):
        out = self.sin(x)
        return out


class SinGrad(nn.Cell):
    def __init__(self, network):
        super(SinGrad, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network

    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout


class SinGradSec(nn.Cell):
    def __init__(self, network):
        super(SinGradSec, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network

    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout


def test_second_grad_with_j_primitive():
    context.set_context(mode=context.GRAPH_MODE)
    net = SinNet()
    first_grad = SinGrad(net)
    second_grad = SinGradSec(first_grad)
    x = Tensor(np.array([1.0], dtype=np.float32))
    second_grad(x)
