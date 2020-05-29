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
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b, sens):
        return C.grad_all_with_sens(self.network)(x, y, b, sens)


class GradWrap2(nn.Cell):
    def __init__(self, network):
        super(GradWrap2, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        loss = self.network(x, y, b)
        sens = P.Fill()(mstype.float32, P.Shape()(loss), 1.0)
        return C.grad_all_with_sens(self.network)(x, y, b, sens)


class GradWrap3(nn.Cell):
    def __init__(self, network):
        super(GradWrap3, self).__init__()
        self.network = network

    def construct(self, x, y, bias):
        return C.grad_all(self.network)(x, y, bias)

class GradWrap4(nn.Cell):
    def __init__(self, network):
        super(GradWrap4, self).__init__()
        self.network = network

    def construct(self, x, y):
        return C.grad_all(self.network)(x, y)

def compile_net(net, x, y, b):
    net.set_auto_parallel()
    _executor.compile(net, x, y, b)

def compile_net_no_bias(net, x, y):
    net.set_auto_parallel()
    _executor.compile(net, x, y)

def test_no_grad():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)

    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = Net(strategy1, strategy2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_grad_sens_parameter_type():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap(Net(strategy1, strategy2))

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    sens = Tensor(np.ones([128, 64]), dtype=ms.float32)
    # net(x, y, b, sens)
    net.set_auto_parallel()
    _executor.compile(net, x, y, b, sens)


def test_grad_sens_tensor_type():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)

    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap2(Net(strategy1, strategy2))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_grad_sens_scalar_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy0, strategy1):
            super().__init__()
            self.fc_nobias = P.MatMul(transpose_b=True).set_strategy(strategy0)
            self.reduce_sum = P.ReduceSum(keep_dims=False).set_strategy(strategy1)

        def construct(self, x, y):
            out = self.fc_nobias(x, y)
            out = self.reduce_sum(out, (0, 1))
            return out

    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy0 = ((4, 1), (4, 1))
    strategy1 = ((4, 1),)
    net = GradWrap4(Net(strategy0, strategy1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 32]), dtype=ms.float32)
    compile_net_no_bias(net, x, y)
