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
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


class NetWithLoss(nn.Cell):
    def __init__(self, network, strategy3):
        super(NetWithLoss, self).__init__()
        self.loss = P.SoftmaxCrossEntropyWithLogits().set_strategy(strategy3)
        self.network = network

    def construct(self, x, y, bias, label):
        predict = self.network(x, y, bias)
        return self.loss(predict, label)[0]


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, bias, label):
        return C.grad_all(self.network)(x, y, bias, label)


def test_linear():
    class Net(nn.Cell):
        def __init__(self, strategy0, strategy1, strategy2):
            super().__init__()
            self.fc_nobias = P.MatMul(transpose_b=True).set_strategy(strategy0)
            self.add = P.TensorAdd().set_strategy(strategy1)
            self.gelu = P.Gelu().set_strategy(strategy2)

        def construct(self, x, y, bias):
            out = self.fc_nobias(x, y)
            out = self.add(out, bias)
            out = self.gelu(out)
            return out

    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy0 = ((2, 4), (2, 4))
    strategy1 = ((2, 4), (4,))
    strategy2 = ((2, 8),)
    strategy3 = ((16, 1), (16, 1))
    net = GradWrap(NetWithLoss(Net(strategy0, strategy1, strategy2), strategy3))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 32]), dtype=ms.float32)
    bias = Tensor(np.ones([64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, bias, label)
