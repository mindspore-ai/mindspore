# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore.train import Model
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


class DatasetLenet():
    def __init__(self, data, label, length=3):
        self.data = data
        self.label = label
        self.index = 1
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.data, self.label

    @staticmethod
    def get_dataset_size():
        return 32

    @staticmethod
    def get_repeat_count():
        return 1

    @staticmethod
    def get_batch_size():
        return 32

    def create_tuple_iterator(self, num_epochs=1, do_copy=True):
        return self

    def reset(self):
        self.index = 0


class MatMulCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = P.MatMul()
        self.relu = P.ReLU().shard(((2, 1),))
        self.weight = Parameter(initializer("ones", [64, 64]), name="param1")

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.relu(out)
        return out


class ConcatCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.concat = P.Concat().shard(((1, 8), (1, 8)))
        self.relu = P.ReLU()

    def construct(self, x, y):
        out = self.concat((y, x))
        out = self.relu(out)
        return out


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = P.MatMul().shard(((2, 4), (4, 1)))
        self.weight = Parameter(initializer("ones", [64, 64]), name="param")
        self.index = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.cell1 = MatMulCell()
        self.cell2 = ConcatCell()
        self.relu = P.ReLU().shard(((8, 1),))

    def construct(self, x, y):
        out = self.matmul(x, self.weight)
        while self.index < 3:
            out = self.cell1(out)
            self.index += 1
        out = self.cell2(out, x)
        out = self.relu(out)
        return out


def test_parallel_while():
    """
    Feature: test parallel while.
    Description: while + concat.
    Expectation: Successful graph compilation.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = Net()
    data = Tensor(np.ones([128, 64]), dtype=ms.float32)
    label = Tensor(np.ones([8, 8]), dtype=ms.float32)
    dataset = DatasetLenet(data, label, 3)
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=False)
