# Copyright 2023 Huawei Technologies Co., Ltd
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
# ============================================================================
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model
from mindspore.nn.wrap.cell_wrapper import GradAccumulationCell, MicroBatchInterleaved
from .test_pipeline_split import DatasetLenet


class MatMulCell(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None, dtype=ms.float32):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        if param is not None:
            self.param = param
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)
        self.cast = P.Cast()
        self.dtype = dtype

    def construct(self, x):
        out = self.matmul(self.cast(x, self.dtype), self.cast(self.param, self.dtype))
        out = self.matmul1(out, self.cast(self.param1, self.dtype))
        return out


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None, dtype=ms.float32):
        super().__init__()
        self.block = nn.CellList()
        for _ in range(2):
            cell = MatMulCell(strategy1, strategy2, param, dtype)
            self.block.append(cell)

    def construct(self, x, label):
        for i in range(2):
            x = self.block[i](x)
        return x


def test_grad_accumulation_base():
    '''
    Feature: grad_accumulation base
    Description: In grad_accumulation mode, expected success
    Expectation: success
    '''
    context.set_auto_parallel_context(device_num=32, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = GradAccumulationCell(Net(strategy1, strategy2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_grad_accumulation_predict():
    '''
    Feature: grad_accumulation + predict
    Description: In grad_accumulation mode, opt_shard is True, expected runtime error
    Expectation: raise error
    '''
    context.set_auto_parallel_context(device_num=32, global_rank=0, full_batch=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = GradAccumulationCell(Net(strategy1, strategy2), 2)
    model = Model(net)
    with pytest.raises(RuntimeError):
        model.predict(data, label)


def test_grad_accumulation_opt_shard():
    '''
    Feature: grad_accumulation + opt_shard
    Description: In grad_accumulation mode, opt_shard is True, expected success
    Expectation: success
    '''
    context.set_auto_parallel_context(device_num=32, global_rank=0, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = GradAccumulationCell(Net(strategy1, strategy2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_grad_accumulation_opt_shard_not_full():
    '''
    Feature: grad_accumulation + opt_shard_not_full
    Description: In grad_accumulation mode, opt_shard is True and do not fully split, expected success
    Expectation: success
    '''
    context.set_auto_parallel_context(device_num=32, global_rank=0, enable_parallel_optimizer=True,
                                      parallel_optimizer_config={'optimizer_weight_shard_size': 4})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = GradAccumulationCell(Net(strategy1, strategy2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_grad_accumulation_opt_shard_with_no_data_parallel():
    '''
    Feature: grad_accumulation + opt_shard
    Description: In grad_accumulation mode, if there is no data parallel, opt_shard is True, expected success
    Expectation: success
    '''
    context.set_auto_parallel_context(device_num=16, global_rank=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([128, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((1, 1), (1, 8))
    strategy2 = ((1, 1), (1, 4))
    net = GradAccumulationCell(Net(strategy1, strategy2), 8)
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_grad_accumulation_opt_shard_with_requires_grad_false():
    '''
    Feature: grad_accumulation + opt_shard
    Description: In grad_accumulation mode, if opt_shard is True and param's requiers_grad = False, expected success
    Expectation: success
    '''
    context.set_auto_parallel_context(device_num=32, global_rank=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([128, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = GradAccumulationCell(Net(strategy1, strategy2), 8)
    net.network.block[0].param1.requiers_grad = False
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_grad_accumulation_with_micro_batch_interleaved_stage0():
    """
    Feature: test GradAccumulation with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved and grad_accumulation in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    micro_batch_interleaved = 2
    net = GradAccumulationCell(MicroBatchInterleaved(Net(strategy1, strategy2), micro_batch_interleaved), 4)
    params = net.network.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
