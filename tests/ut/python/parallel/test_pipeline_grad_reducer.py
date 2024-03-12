# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import context
from mindspore import Tensor, ops
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.nn.wrap.grad_reducer import PipelineGradReducer


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
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)

    def construct(self, x):
        out = self.matmul(x, self.param)
        out = self.matmul1(out, self.param1)
        return out, self.param


class MatMulCell2(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)

    def construct(self, x, param):
        out = self.matmul(x, param)
        out = self.matmul1(out, self.param1)
        return out


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None):
        super().__init__()
        self.cell1 = MatMulCell(strategy1, strategy2)
        self.cell1.pipeline_stage = 0
        self.cell2 = MatMulCell2(strategy1, strategy2)
        self.cell2.pipeline_stage = 1

    def construct(self, x, label):
        out, param = self.cell1(x)
        out = self.cell2(out, param)
        return out


def test_pipeline_functional_stage0():
    """
    Feature: pipeline parallel functional
    Description:  test pipeline parallel functional
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))

    net = nn.PipelineCell(Net(strategy1, strategy2), 4)

    def forward_fn(inputs, target):
        loss = net(inputs, target)
        return loss

    params = net.network.cell1.trainable_params()
    grad_fn = ops.value_and_grad(forward_fn, None, params)
    optimizer = nn.SGD(params, learning_rate=0.01)
    pp_grad_reducer = PipelineGradReducer(optimizer.parameters)

    @ms.jit
    def train_one_step(inputs, target):
        loss, grads = grad_fn(inputs, target)
        grads = pp_grad_reducer(grads)
        optimizer(grads)
        return loss, grads

    dataset = DatasetLenet(data, label, 3)
    for data, label in dataset:
        train_one_step(data, label)


def test_pipeline_functional_shard_stage0():
    """
    Feature: pipeline parallel functional
    Description:  test pipeline parallel functional with parameter shard
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", enable_parallel_optimizer=True)
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))

    net = nn.PipelineCell(Net(strategy1, strategy2), 4)

    def forward_fn(inputs, target):
        loss = net(inputs, target)
        return loss

    params = net.network.cell1.trainable_params()
    grad_fn = ops.value_and_grad(forward_fn, None, params)
    optimizer = nn.SGD(params, learning_rate=0.01)
    pp_grad_reducer = PipelineGradReducer(optimizer.parameters)

    @ms.jit
    def train_one_step(inputs, target):
        loss, grads = grad_fn(inputs, target)
        grads = pp_grad_reducer(grads)
        optimizer(grads)
        return loss, grads

    dataset = DatasetLenet(data, label, 3)
    for data, label in dataset:
        train_one_step(data, label)
