# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


def compile_net(net, x, y, b):
    net.set_auto_parallel()
    net.set_train()
    _executor.compile(net, x, y, b)


def test_rhombus1():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = P.MatMul()
            self.tadd1 = P.Add()
            self.tadd2 = P.Add()
            self.weight = Parameter(Tensor(np.ones([128, 128]).astype(np.float32) * 0.01), "w", requires_grad=True)

        def construct(self, x, y, z):
            mm_out = self.matmul(x, self.weight)
            ta1_out = self.tadd1(y, z)
            out = self.tadd2(ta1_out, mm_out)
            return out

    size = 16
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    y = Tensor(np.ones([128, 128]), dtype=ms.float32)
    b = Tensor(np.ones([128, 128]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    compile_net(net, x, y, b)


def test_rhombus2():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()
            self.tadd1 = P.Add()
            self.tadd2 = P.Add()
            self.tadd3 = P.Add()
            self.weight1 = Parameter(Tensor(np.ones([128, 128]).astype(np.float32) * 0.01), "w", requires_grad=True)
            self.weight2 = Parameter(Tensor(np.ones([128, 128]).astype(np.float32) * 0.01), "w", requires_grad=True)

        def construct(self, x, y, z):
            mm1_out = self.matmul1(x, self.weight1)
            ta1_out = self.tadd1(y, z)
            ta2_out = self.tadd2(mm1_out, ta1_out)
            mm2_out = self.matmul2(ta1_out, self.weight2)
            ta3_out = self.tadd3(ta2_out, mm2_out)
            return ta3_out

    size = 16
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    y = Tensor(np.ones([128, 128]), dtype=ms.float32)
    b = Tensor(np.ones([128, 128]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    compile_net(net, x, y, b)


def test_rhombus3():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul()
            self.tadd1 = P.Add()
            self.tadd2 = P.Add()
            self.tadd3 = P.Add()
            self.tadd4 = P.Add()
            self.weight1 = Parameter(Tensor(np.ones([128, 128]).astype(np.float32) * 0.01), "w", requires_grad=True)
            self.t = Tensor(np.ones([128, 128]).astype(np.float32) * 0.01)

        def construct(self, x, y, z):
            mm1_out = self.matmul1(x, self.weight1)
            ta1_out = self.tadd1(y, z)
            ta2_out = self.tadd2(mm1_out, ta1_out)
            ta3_out = self.tadd3(ta1_out, self.t)
            ta4_out = self.tadd4(ta2_out, ta3_out)
            return ta4_out

    size = 16
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    y = Tensor(np.ones([128, 128]), dtype=ms.float32)
    z = Tensor(np.ones([128, 128]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    compile_net(net, x, y, z)
