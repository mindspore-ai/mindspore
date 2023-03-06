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
from mindspore import Tensor, Parameter
from mindspore import context, Model
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.nn.wrap.cell_wrapper import PipelineCell
from tests.ut.python.ops.test_math_ops import VirtualLoss
from parallel.utils.utils import ParallelValidator
from .test_pipeline_split import DatasetLenet


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


def test_opt_parallel_without_grad():
    """
    Feature: Test optimizer parallel with parameter's requires_grad=False.
    Description: Need insert AllGather.
    Expectation: Successful graph compilation.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.fc1 = P.MatMul().shard(((4, 1), (1, 2)))
            self.fc2 = P.MatMul().shard(((2, 2), (2, 1)))
            self.p1 = Parameter(Tensor(np.ones([1024, 1024]).astype(np.float32)), name="weight1", requires_grad=False)
            self.p2 = Parameter(Tensor(np.ones([1024, 64]).astype(np.float32)), name="weight2")

        def construct(self, x, y):
            x = self.fc1(x, self.p1)
            x = self.fc2(x, self.p2)
            return x - y
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, enable_parallel_optimizer=True)
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 1024]), dtype=ms.float32)
    y = Tensor(np.ones([128, 64]), dtype=ms.float32)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, x, y)
    validator = ParallelValidator(net, phase)
    expect_layout = ([4, 2], [-1, 0], [1024, 512], 0, True, '4-5226697808808137312')
    assert validator.check_parameter_layout("network.network.p1", expect_layout)


def test_opt_parallel_without_grad_pipeline():
    """
    Feature: Test optimizer parallel + pipeline with parameter's requires_grad=False.
    Description: Need insert AllGather.
    Expectation: Successful graph compilation.
    """
    class MatMulNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.fc1 = P.MatMul().shard(((4, 1), (1, 2)))
            self.fc2 = P.MatMul().shard(((2, 2), (2, 1)))
            self.p1 = Parameter(Tensor(np.ones([1024, 1024]).astype(np.float32)), name="weight1", requires_grad=False)
            self.p2 = Parameter(Tensor(np.ones([1024, 1024]).astype(np.float32)), name="weight2")

        def construct(self, x):
            x = self.fc1(x, self.p1)
            x = self.fc2(x, self.p2)
            return x

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.block = nn.CellList()
            for i in range(2):
                cell = MatMulNet()
                cell.pipeline_stage = i
                self.block.append(cell)

        def construct(self, x, y):
            for i in range(2):
                x = self.block[i](x)
            return x
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=16, global_rank=0, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", pipeline_stages=2)
    net = PipelineCell(Net(), 4)
    x = Tensor(np.ones([128, 1024]), dtype=ms.float32)
    y = Tensor(np.ones([128, 128]), dtype=ms.float32)
    dataset = DatasetLenet(x, y, 3)
    optimizer = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    assert net.network.block[0].p1.shape == (256, 512)
