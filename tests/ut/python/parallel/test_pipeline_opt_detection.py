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
# ============================================================================
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train.model import Model
from mindspore.nn.wrap.cell_wrapper import PipelineCell, MicroBatchInterleaved

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
            cell = MatMulCell(strategy1, strategy2, param, dtype)
            cell.pipeline_stage = i
            self.block.append(cell)

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
        return x


class PipelineSplit(nn.Cell):
    def __init__(self, strategy1, strategy2, dtype=ms.float32):
        super().__init__()
        self.cell = Net(strategy1, strategy2, dtype=dtype)

    def construct(self, x, label):
        x = self.cell(x)
        return x


class PipelineSplitSharedParam(nn.Cell):
    def __init__(self, strategy1, strategy2, dtype=ms.float32):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.cell = Net(strategy1, strategy2, self.param, dtype)

    def construct(self, x, label):
        x = self.cell(x)
        return x


def test_pipeline_split_stage0():
    """
    Feature:pipeline stage0 + opt detection
    Description:pipeline opt detection
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage1():
    """
    Feature:pipeline stage1 + opt detection
    Description:pipeline opt detection
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 4)
    optimizer = nn.Lamb(params, learning_rate=0.001)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_stage0():
    """
    Feature:pipeline stage0 + opt detection + shared parameter
    Description:pipeline opt detection
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplitSharedParam(strategy1, strategy2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 6)
    optimizer = nn.Lamb(params, learning_rate=0.03)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_stage1():
    """
    Feature:pipeline stage1 + opt detection + shared parameter
    Description:pipeline opt detection
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplitSharedParam(strategy1, strategy2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 7)
    optimizer = nn.Lamb(params, learning_rate=0.04)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage0_opt_shard():
    """
    Feature:pipeline stage0 + opt detection + opt shard
    Description:pipeline opt detection
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 6)
    optimizer = nn.Lamb(params, learning_rate=0.02)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage1_opt_shard():
    """
    Feature:pipeline stage1 + opt detection + opt shard
    Description:pipeline opt detection
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 8)
    optimizer = nn.Lamb(params, learning_rate=0.04)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_stage0_opt_shard():
    """
    Feature:pipeline stage0 + opt detection + opt shard + shared parameter
    Description:pipeline opt detection
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplitSharedParam(strategy1, strategy2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 2)
    optimizer = nn.Lamb(params, learning_rate=0.06)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_stage1_opt_shard():
    """
    Feature:pipeline stage1 + opt detection + opt shard + shared parameter
    Description:pipeline opt detection
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplitSharedParam(strategy1, strategy2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 9)
    optimizer = nn.Lamb(params, learning_rate=0.06)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_with_micro_batch_interleaved_stage0():
    """
    Feature: test PipelineSplit with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    micro_batch_interleaved = 2
    net = PipelineCell(MicroBatchInterleaved(PipelineSplit(strategy1, strategy2), micro_batch_interleaved), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.07)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_with_micro_batch_interleaved_stage1():
    """
    Feature: test PipelineSplit with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    micro_batch_interleaved = 2
    net = PipelineCell(MicroBatchInterleaved(PipelineSplit(strategy1, strategy2), micro_batch_interleaved), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.08)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_with_micro_batch_interleaved_stage0_opt_shard():
    """
    Feature: test PipelineSplitSharedParameter with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    micro_batch_interleaved = 2
    net = PipelineCell(MicroBatchInterleaved(PipelineSplitSharedParam(strategy1, strategy2),
                                             micro_batch_interleaved), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 5)
    optimizer = nn.Lamb(params, learning_rate=0.06)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_with_micro_batch_interleaved_stage1_opt_shard():
    """
    Feature: test PipelineSplitSharedParameter with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    micro_batch_interleaved = 2
    net = PipelineCell(MicroBatchInterleaved(PipelineSplitSharedParam(strategy1, strategy2),
                                             micro_batch_interleaved), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 4)
    optimizer = nn.Lamb(params, learning_rate=0.02)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def run_pipeline_split_function(pipeline_net, micro_batch_interleaved=1):
    """
    Feature: test PipelineSplitSharedParameter with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)

    net = PipelineCell(MicroBatchInterleaved(pipeline_net, micro_batch_interleaved), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
