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
""" test pipeline with recompute """
import numpy as np

import mindspore.nn as nn
from mindspore.common import dtype as mstype
import mindspore.ops.operations as P
from mindspore import Tensor, Parameter
from mindspore.nn.optim import Momentum
from mindspore import context, lazy_inline
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _TrainGradAccuStepCell
from parallel.utils.utils import compile_net, ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(nn.Cell):
    """Net definition"""

    def __init__(self):
        super(Net, self).__init__()
        self.w = Parameter(Tensor(np.ones([768, 768]).astype(np.float32)), name="w")
        self.matmul1 = P.MatMul().shard(((2, 1), (1, 2)))

        self.w2 = Parameter(Tensor(np.ones([768, 768]).astype(np.float32)), name="w2")
        self.matmul2 = P.MatMul().shard(((1, 1), (1, 1)))

        self.layer_norm = nn.LayerNorm(normalized_shape=(768,))
        self.layer_norm.layer_norm.shard(((1, 1, 1), (1,), (1,)))

        self.cast = P.Cast()

    def construct(self, x):
        x = self.cast(x, mstype.float16)
        x = self.matmul1(x, self.cast(self.w, mstype.float16))
        x = P.Reshape()(x, (-1, 1, 768))
        x = self.cast(x, mstype.float32)
        x = self.layer_norm(x)
        x = P.Reshape()(x, (-1, 768))
        x = self.cast(x, mstype.float16)
        x = self.matmul2(x, self.cast(self.w2, mstype.float16))
        return x

class Mlp(nn.Cell):
    def __init__(self):
        super(Mlp, self).__init__()
        self.w = Parameter(Tensor(np.ones([768, 768]).astype(np.float32)), name="w")
        self.matmul1 = P.MatMul().shard(((2, 1), (1, 2)))

    def construct(self, x):
        x = P.Reshape()(x, (-1, 768))
        x = self.matmul1(x, P.Cast()(self.w, mstype.float16))
        return x

class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Dense(768, 768)
        self.fc2 = nn.Dense(768, 768)

    def construct(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class FullNet(nn.Cell):
    """Net definition"""
    @lazy_inline
    def __init__(self):
        super(FullNet, self).__init__()
        self.net1 = Net()
        self.net2 = Net2()
        self.net3 = Mlp()

        self.w = Parameter(Tensor(np.ones([768, 768]).astype(np.float32)), name="w")
        self.matmul1 = P.MatMul().shard(((2, 1), (1, 2)))

        self.net1.pipeline_stage = 0
        self.net1.recompute()
        self.net2.pipeline_stage = 1

    def construct(self, x, y):
        x = P.ReLU()(x)
        x = self.net1(x)

        x = self.matmul1(x, P.Cast()(self.w, mstype.float16))
        x = self.net3(x)
        x = P.Cast()(x, mstype.float32)
        x = self.net2(x)
        return x


def test_two_net_with_different_stages():
    """
    Feature: test mirror insert when there is a subgraph
    Description: test mirror insert
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", pipeline_stages=2,
                                      device_num=8, enable_parallel_optimizer=False)
    inputs = Tensor(np.ones([32, 768]).astype(np.float32))
    label = Tensor(np.zeros([32, 768]).astype(np.float32))
    net = FullNet()
    net = PipelineCell(net, 4)

    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = _TrainGradAccuStepCell(net, optimizer)
    train_network.set_train()
    phase = compile_net(train_network, inputs, label)
    validator = ParallelValidator(train_network, phase)
    expect_output = ['Net_construct', 'ValueNode', 'ValueNode', 'ValueNode', 0, 0, 0, 0, 0]
    assert validator.check_node_inputs_fuzzy_match("StridedSlice-1", expect_output, graph_id=1)
