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
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_by_list = C.GradOperation(get_by_list=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network, strategy3):
        super(NetWithLoss, self).__init__()
        self.loss = P.SoftmaxCrossEntropyWithLogits().shard(strategy3)
        self.network = network

    def construct(self, x, b):
        predict = self.network(x)
        return self.loss(predict, b)[0]


class OneStepCell(nn.Cell):
    def __init__(self, network):
        super(OneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.network.trainable_params())

    def construct(self, data, label):
        weights = self.weights
        grads = grad_by_list(self.network, weights)(data, label)
        return grads


def test_two_weights_parameter():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, weight, weight2):
            super().__init__()
            self.weight = Parameter(weight, "w1", requires_grad=True)
            self.weight2 = Parameter(weight2, "w2", requires_grad=True)
            self.matmul = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x):
            out = self.matmul(x, self.weight)
            out = self.matmul2(out, self.weight2)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((4, 1), (1, 2))
    strategy2 = ((4, 2), (2, 1))
    strategy3 = ((8, 1), (8, 1))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    weight = Tensor(np.ones([32, 64]), dtype=ms.float32)
    weight2 = Tensor(np.ones([64, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    net = Net(strategy1, strategy2, weight, weight2)

    net_with_loss = NetWithLoss(net, strategy3)

    train_net = OneStepCell(net_with_loss)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    train_net.set_train()
    _cell_graph_executor.compile(train_net, x, b)
