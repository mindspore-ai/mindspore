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
from mindspore import context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from tests.ut.python.ops.test_math_ops import VirtualLoss
import mindspore as ms
from mindspore.common.api import _executor
from mindspore.ops import composite as C


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, bias):
        return C.grad_all(self.network)(x, y, bias)


def compile(net, x, y, bias):
    net.set_auto_parallel()
    _executor.compile(net, x, y, bias)


def test_sum_as_loss():
    class Net(nn.Cell):
        def __init__(self, strategy0, strategy1):
            super().__init__()
            self.fc_nobias = P.MatMul(transpose_b=True).set_strategy(strategy0)
            self.reduce_sum = P.ReduceSum(keep_dims=False).set_strategy(strategy1)

        def construct(self, x, y, bias):
            out = self.fc_nobias(x, y)
            out = self.reduce_sum(out, (0, 1))
            return out

    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy0 = ((4, 1), (4, 1))
    strategy1 = ((4, 1),)
    net = GradWrap(Net(strategy0, strategy1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 32]), dtype=ms.float32)
    bias = Tensor(np.ones([64]), dtype=ms.float32)
    compile(net, x, y, bias)


def test_sum_as_loss2():
    class Net(nn.Cell):
        def __init__(self, strategy0, strategy1):
            super().__init__()
            self.fc_nobias = P.MatMul(transpose_b=True).set_strategy(strategy0)
            self.reduce_sum = P.ReduceSum(keep_dims=False).set_strategy(strategy1)

        def construct(self, x, y, bias):
            out = self.fc_nobias(x, y)
            out = self.reduce_sum(out, (0, 1))
            return out

    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy0 = ((4, 1), (4, 1))
    strategy1 = ((1, 1),)
    net = GradWrap(Net(strategy0, strategy1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 32]), dtype=ms.float32)
    bias = Tensor(np.ones([64]), dtype=ms.float32)
    compile(net, x, y, bias)
