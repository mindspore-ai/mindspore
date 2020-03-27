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
from mindspore import context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from tests.ut.python.ops.test_math_ops import VirtualLoss
from mindspore.common.api import _executor
from mindspore.ops import composite as C


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b, a):
        predict = self.network(x, y, b, a)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b, a):
        return C.grad_all(self.network)(x, y, b, a)


def test_two_matmul():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3, strategy4):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)
            self.matmul3 = P.MatMul().set_strategy(strategy3)
            self.matmul4 = P.MatMul().set_strategy(strategy4)

        def construct(self, x, y, b, a):
            out = self.matmul1(x, y)
            out1 = self.matmul2(out, b)
            out2 = self.matmul3(out, a)
            out3 = self.matmul4(out1, out2)
            return out3

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((1, 8), (8, 1))
    strategy3 = ((4, 1), (1, 2))
    strategy4 = ((4, 2), (2, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3, strategy4)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    b = Tensor(np.ones([128, 128]), dtype=ms.float32)
    a = Tensor(np.ones([128, 128]), dtype=ms.float32)
    
    _executor.compile(net, x, y, b, a)
