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
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model
from mindspore.nn.wrap.cell_wrapper import PipelineCell


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
        for i in range(2):
            for j in range(2):
                cell = MatMulCell(strategy1, strategy2, param, dtype)
                cell.pipeline_segment = i
                cell.pipeline_stage = j
                self.block.append(cell)

    def construct(self, x):
        for i in range(4):
            x = self.block[i](x)
        return x


class FoldPipelineSplit2(nn.Cell):
    def __init__(self, strategy1, strategy2, dtype=ms.float32):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.cell = Net(strategy1, strategy2, self.param, dtype)
        self.cell.block[0].matmul.add_prim_attr("parameter_start", 0)

    def construct(self, x, label):
        x = self.cell(x)
        return x


def test_fold_pipeline_split_shared_parameter_stage0():
    """
    Feature: test fold PipelineSplit with shared parameter.
    Description: net with shared parameter in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2, pipeline_segments=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(FoldPipelineSplit2(strategy1, strategy2), 4)
    params = net.network.cell.block[0].trainable_params()
    params = params + net.network.cell.block[2].trainable_params()
    params = list(set(params))
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_fold_pipeline_split_shared_parameter_stage1():
    """
    Feature: test fold PipelineSplit with shared parameter.
    Description: net with shared parameter in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2, pipeline_segments=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(FoldPipelineSplit2(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    params = params + net.network.cell.block[3].trainable_params()
    params = list(set(params))
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_fold_pipeline_split_shared_parameter_stage0_predict():
    """
    Feature: test fold PipelineSplit with shared parameter.
    Description: net with shared parameter in semi auto parallel to predict.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2, pipeline_segments=2,
                                      full_batch=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = FoldPipelineSplit2(strategy1, strategy2)
    model = Model(net)
    model.predict(data, label)


def test_fold_pipeline_split_shared_parameter_stage1_predict():
    """
    Feature: test fold PipelineSplit with shared parameter.
    Description: net with shared parameter in semi auto parallel to predict.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2, pipeline_segments=2,
                                      full_batch=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = FoldPipelineSplit2(strategy1, strategy2)
    model = Model(net)
    model.predict(data, label)


def test_fold_pipeline_split_shared_parameter_stage0_opt_shard():
    """
    Feature: test fold PipelineSplit with shared parameter and opt_shard.
    Description: net with shared parameter and opt_shard in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2, pipeline_segments=2,
                                      enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((1, 1), (1, 16))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(FoldPipelineSplit2(strategy1, strategy2), 4)
    params = net.network.cell.block[0].trainable_params()
    params = params + net.network.cell.block[2].trainable_params()
    params = list(set(params))
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_fold_pipeline_split_shared_parameter_stage1_opt_shard():
    """
    Feature: test fold PipelineSplit with shared parameter and opt_shard.
    Description: net with shared parameter and opt_shard in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2, pipeline_segments=2,
                                      enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(FoldPipelineSplit2(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    params = params + net.network.cell.block[3].trainable_params()
    params = list(set(params))
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
