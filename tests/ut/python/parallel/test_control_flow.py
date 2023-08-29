# Copyright 2021 Huawei Technologies Co., Ltd
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

    def reset(self):
        self.index = 0

    def get_dataset_size(self):
        return 32

    def get_repeat_count(self):
        return 1

    def get_batch_size(self):
        return 32

    def create_tuple_iterator(self, num_epochs=1, do_copy=True):
        return self


class MatMulCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = P.MatMul()
        self.relu = P.ReLU()
        self.weight = Parameter(initializer("ones", [64, 64]), name="param1")

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.relu(out)
        return out


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.matmul = P.MatMul().shard(strategy1)
        self.weight = Parameter(initializer("ones", [64, 64]), name="param")
        self.cell1 = MatMulCell()
        self.cell2 = MatMulCell()
        self.cell3 = MatMulCell()
        self.cell4 = MatMulCell()
        self.relu = P.ReLU().shard(strategy2)
        self.reduce = P.ReduceSum()

    def construct(self, x, y):
        out = self.matmul(x, self.weight)
        if self.reduce(y) == 1.0:
            out = self.cell1(out)
        elif self.reduce(y) == 2.0:
            out = self.cell2(out)
        elif self.reduce(y) == 3.0:
            out = self.cell3(out)
        else:
            out = self.cell4(out)
        out = self.relu(out)
        out = out + x
        return out


def test_control_flow():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1),)
    net = Net(strategy1, strategy2)
    data = Tensor(np.ones([128, 64]), dtype=ms.float32)
    label = Tensor(np.ones([8, 8]), dtype=ms.float32)
    dataset = DatasetLenet(data, label, 3)
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=False)
