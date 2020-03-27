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


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b, z):
        predict = self.network(x, y, b, z)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b, z):
        return C.grad_all(self.network)(x, y, b, z)


class Net1(nn.Cell):
    def __init__(self, strategy1, strategy2, strategy3):
        super().__init__()
        self.matmul1 = P.MatMul().set_strategy(strategy1)
        self.matmul2 = P.MatMul().set_strategy(strategy2)
        self.matmul3 = P.MatMul().set_strategy(strategy3)

    def construct(self, x, y, b):
        out1 = self.matmul1(x, b)
        out2 = self.matmul2(y, b)
        out = self.matmul3(out1, out2)
        return out


def test_two_matmul():
    class Net2(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3, strategy4):
            super().__init__()
            self.net1_out = Net1(strategy1, strategy2, strategy3)
            self.matmul = P.MatMul().set_strategy(strategy4)

        def construct(self, x, y, b, z):
            out = self.net1_out(x, y, b)
            out = self.matmul(out, z)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 1))
    strategy2 = ((4, 2), (2, 1))
    strategy3 = ((1, 8), (8, 1))
    strategy4 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net2(strategy1, strategy2, strategy3, strategy4).add_flags_recursive(fp16=True)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 32]), dtype=ms.float32)
    b = Tensor(np.ones([32, 64]), dtype=ms.float32)
    z = Tensor(np.ones([64, 64]), dtype=ms.float32)
    
    _executor.compile(net, x, y, b, z)
