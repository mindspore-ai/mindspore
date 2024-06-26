# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
from mindspore.nn.wrap.cell_wrapper import PipelineCell, MicroBatchInterleaved
import mindspore.common.lazy_inline as lazy_inline
from mindspore.parallel.shard import Layout


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


class LazyInlineNet(nn.Cell):
    @lazy_inline
    def __init__(self, stra1, stra2, param=None):
        super().__init__()
        self.cell1 = MatMulCell(stra1, stra2)
        self.cell1.pipeline_stage = 0
        self.cell2 = MatMulCell(stra1, stra2)
        self.cell2.pipeline_stage = 1
        self.cell3 = MatMulCell(stra1, stra2)
        self.cell3.pipeline_stage = 0
        self.cell4 = MatMulCell2(stra1, stra2)
        self.cell4.pipeline_stage = 1

    def construct(self, x, label):
        out, param = self.cell1(x)
        out, param = self.cell2(out)
        out, param = self.cell3(out)
        out = self.cell4(out, param)
        return out

class LazyInlineNetForFineGrain(nn.Cell):
    @lazy_inline
    def __init__(self, stra1, stra2, param=None):
        super().__init__()
        self.cell1 = MatMulCell(stra1, stra2)
        self.cell1.pipeline_stage = 0
        self.cell2 = MatMulCell(stra1, stra2)
        self.cell2.pipeline_stage = 1
        self.cell3 = MatMulCell(stra1, stra2)
        self.cell3.pipeline_stage = 0
        self.cell4 = MatMulCell2(stra1, ((8, 1), (1, 1)))
        self.cell4.pipeline_stage = 1

    def construct(self, x, label):
        out, param = self.cell1(x)
        out, param = self.cell2(out)
        out, param = self.cell3(out)
        out = self.cell4(out, param)
        return out

def test_pipeline_interleave_gpipe_stage0():
    """
    Feature: Pipeline Interleave
    Description: Pipeline Interleave with gpipe scheduler
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(pipeline_config={"pipeline_interleave": True, "pipeline_scheduler": "gpipe"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(LazyInlineNet(stra1, stra2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_interleave_gpipe_stage1():
    """
    Feature: Pipeline Interleave
    Description: Pipeline Interleave with gpipe scheduler
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(pipeline_config={"pipeline_interleave": True, "pipeline_scheduler": "gpipe"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(LazyInlineNet(stra1, stra2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_interleave_gpipe_batch_interleave_stage0():
    """
    Feature: Pipeline Interleave
    Description: Pipeline Interleave with gpipe scheduler and Batch-dim Interleave
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(pipeline_config={"pipeline_interleave": True, "pipeline_scheduler": "gpipe"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(MicroBatchInterleaved(LazyInlineNet(stra1, stra2), 2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_interleave_gpipe_batch_interleave_stage1():
    """
    Feature: Pipeline Interleave
    Description: Pipeline Interleave with gpipe scheduler and Batch-dim Interleave
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(pipeline_config={"pipeline_interleave": True, "pipeline_scheduler": "gpipe"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(MicroBatchInterleaved(LazyInlineNet(stra1, stra2), 2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_interleave_gpipe_fine_grain_interleave_stage0():
    """
    Feature: Pipeline Interleave
    Description: Pipeline Interleave with gpipe scheduler and Seq-dim Interleave
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(pipeline_config={"pipeline_interleave": True, "pipeline_scheduler": "gpipe"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    layout1 = Layout((16, 1, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout1 = (layout1(("dp", "interleaved_parallel"), "mp"), layout1("mp", "None"))
    layout2 = Layout((8, 2, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout2 = (layout2(("dp", "interleaved_parallel"), "None"), layout2("None", "None"))
    net = PipelineCell(LazyInlineNetForFineGrain(matmul_layout1, matmul_layout2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)

def test_pipeline_interleave_gpipe_fine_grain_interleave_stage1():
    """
    Feature: Pipeline Interleave
    Description: Pipeline Interleave with gpipe scheduler and Seq-dim Interleave
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(pipeline_config={"pipeline_interleave": True, "pipeline_scheduler": "gpipe"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    layout1 = Layout((16, 1, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout1 = (layout1(("dp", "interleaved_parallel"), "mp"), layout1("mp", "None"))
    layout2 = Layout((8, 2, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout2 = (layout2(("dp", "interleaved_parallel"), "None"), layout2("None", "None"))
    net = PipelineCell(LazyInlineNetForFineGrain(matmul_layout1, matmul_layout2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_interleave_1f1b_fine_grain_interleave_stage0():
    """
    Feature: Pipeline Interleave
    Description: Pipeline Interleave with gpipe scheduler and Seq-dim Interleave
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(pipeline_config={"pipeline_interleave": True, "pipeline_scheduler": "1f1b"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    layout1 = Layout((16, 1, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout1 = (layout1(("dp", "interleaved_parallel"), "mp"), layout1("mp", "None"))
    layout2 = Layout((8, 2, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout2 = (layout2(("dp", "interleaved_parallel"), "None"), layout2("None", "None"))
    net = PipelineCell(LazyInlineNetForFineGrain(matmul_layout1, matmul_layout2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)

def test_pipeline_interleave_1f1b_fine_grain_interleave_stage1():
    """
    Feature: Pipeline Interleave
    Description: Pipeline Interleave with gpipe scheduler and Seq-dim Interleave
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(pipeline_config={"pipeline_interleave": True, "pipeline_scheduler": "1f1b"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    layout1 = Layout((16, 1, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout1 = (layout1(("dp", "interleaved_parallel"), "mp"), layout1("mp", "None"))
    layout2 = Layout((8, 2, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout2 = (layout2(("dp", "interleaved_parallel"), "None"), layout2("None", "None"))
    net = PipelineCell(LazyInlineNetForFineGrain(matmul_layout1, matmul_layout2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)
