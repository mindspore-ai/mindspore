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
import mindspore.common.lazy_inline as lazy_inline
from .test_pipeline_interleave import DatasetLenet


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)

    def construct(self, x, y):
        out1 = self.matmul(x, self.param)
        out2 = self.matmul1(y, self.param1)
        return out1, out2


class MainNet(nn.Cell):
    @lazy_inline
    def __init__(self, stra1, stra2):
        super().__init__()
        self.cell1 = Net(stra1, stra2)
        self.cell1.pipeline_stage = 0
        self.cell2 = Net(stra1, stra2)
        self.cell2.pipeline_stage = 1
        self.cell3 = Net(stra1, stra2)
        self.cell3.pipeline_stage = 0
        self.cell4 = Net(stra1, stra2)
        self.cell4.pipeline_stage = 1
        self.add = P.Add()

    def construct(self, x, y):
        out1, out2 = self.cell1(x, y)
        out1, out2 = self.cell2(out1, out2)
        out1, out2 = self.cell3(out1, out2)
        out1, out2 = self.cell4(out1, out2)
        out = self.add(out1, out2)
        return out


def test_pipeline_interleave_stage0():
    """
    Feature: Pipeline Interleave
    Description: Pipeline Interleave with 1f1b scheduler stage0
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(pipeline_config={"pipeline_interleave": True, "pipeline_scheduler": "1f1b"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([32, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(MainNet(stra1, stra2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_interleave_stage1():
    """
    Feature: Pipeline Interleave
    Description: Pipeline Interleave with 1f1b scheduler stage1
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(pipeline_config={"pipeline_interleave": True, "pipeline_scheduler": "1f1b"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([32, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(MainNet(stra1, stra2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)
