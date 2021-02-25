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

import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _executor
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P
import mindspore.nn as nn
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


# model_parallel test
def test_two_matmul_batchnorm_ex():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.BatchMatMul().shard(strategy1)
            self.norm = P.FusedBatchNormEx()
            self.gamma = Parameter(Tensor(np.ones([64]), dtype=ms.float32), name="gamma")
            self.beta = Parameter(Tensor(np.ones([64]), dtype=ms.float32), name="beta")
            self.mean = Parameter(Tensor(np.ones([64]), dtype=ms.float32), name="mean")
            self.var = Parameter(Tensor(np.ones([64]), dtype=ms.float32), name="var")
            self.matmul2 = P.BatchMatMul().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.norm(out, self.gamma, self.beta, self.mean, self.var)[0]
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8)
    strategy1 = ((1, 1, 4, 2), (1, 1, 2, 1))
    strategy2 = ((1, 1, 1, 8), (1, 1, 8, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    net.set_auto_parallel()
    x = Tensor(np.ones([64, 64, 128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y, b)
