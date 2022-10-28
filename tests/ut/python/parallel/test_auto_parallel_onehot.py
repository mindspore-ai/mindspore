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
from mindspore.common.api import _cell_graph_executor
from mindspore.common.parameter import Parameter
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.context import ParallelMode
from tests.dataset_mock import MindData
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

context.set_context(mode=context.GRAPH_MODE)


grad_all = C.GradOperation(get_all=True)


class Dataset(MindData):
    def __init__(self, predict, label, length=3):
        super(Dataset, self).__init__(size=length)
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0


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


def test_auto_parallel_arithmetic():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = P.MatMul()
            self.one_hot = P.OneHot()
            self.on_value = Tensor(1.0, ms.float32)
            self.off_value = Tensor(0.0, ms.float32)
            self.matmul2 = P.MatMul()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out1 = self.one_hot(b, 64, self.on_value, self.off_value)
            out2 = self.matmul2(out, out1)
            return out2

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.int32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b)


def test_auto_parallel_arithmetic_model():
    class NetOneHot(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = P.MatMul()
            self.one_hot = P.OneHot().shard(((1, 8), (), ()))
            self.on_value = Tensor(1.0, ms.float32)
            self.off_value = Tensor(0.0, ms.float32)
            self.matmul2 = P.MatMul()
            self.w = Parameter(Tensor(np.zeros([32, 64]).astype(np.float32)), "weight", requires_grad=True)

        def construct(self, x, b):
            out = self.matmul(x, self.w)
            out1 = self.one_hot(b, 64, self.on_value, self.off_value)
            out2 = self.matmul2(out, out1)
            return out2

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.AUTO_PARALLEL,
                                      dataset_strategy="data_parallel")
    net = NetOneHot()

    x = Tensor(np.ones([8, 32]), dtype=ms.float32)
    b = Tensor(np.ones([8]), dtype=ms.int32)
    dataset = Dataset(x, b, 2)

    opt = Momentum(net.trainable_params(), 0.1, 0.9)
    model = Model(net, optimizer=opt)

    model.train(2, dataset, dataset_sink_mode=False)
