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
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = P.SoftmaxCrossEntropyWithLogits()
        self.network = network

    def construct(self, x, b):
        predict = self.network(x)
        return self.loss(predict, b)[0]


def test_parameter_init():
    class Net(nn.Cell):
        def __init__(self, strategy1, weight):
            super().__init__()
            self.weight = Parameter(weight, "w1")
            self.matmul = P.MatMul(transpose_a=False, transpose_b=True).shard(strategy1)

        def construct(self, x):
            out = self.matmul(x, self.weight)
            return out

    context.set_auto_parallel_context(device_num=2, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((1, 1), (2, 1))
    context.set_context(mode=context.GRAPH_MODE)

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    weight = Tensor(np.ones([64, 32]), dtype=ms.float32)

    net = Net(strategy1, weight)
    net(x, )


if __name__ == '__main__':
    test_parameter_init()
