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
# ============================================================================
""" test_shared_param_and_mix_precision  """
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P, functional as F
from mindspore import context


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net1(nn.Cell):
    """Net definition"""
    def __init__(self, strategy1, strategy2):
        super(Net1, self).__init__()
        self.fc1 = P.MatMul().shard(strategy1)
        self.fc2 = P.MatMul().shard(strategy2)
        self.p1 = Parameter(Tensor(np.ones([48, 64]).astype(np.float32)), name="weight1")
        self.p2 = Parameter(Tensor(np.ones([64, 48]).astype(np.float32)), name="weight2")

    def construct(self, x, y):
        x = self.fc1(x, self.p1)
        x = self.fc2(x, self.p2)
        x = self.fc1(x, self.p1)
        return x - y


class Net2(nn.Cell):
    """Net definition"""
    def __init__(self, strategy1, strategy2):
        super(Net2, self).__init__()
        self.fc1 = P.MatMul().shard(strategy1)
        self.fc2 = P.MatMul().shard(strategy2)
        self.p1 = Parameter(Tensor(np.ones([48, 64]).astype(np.float32)), name="weight1")
        self.p2 = Parameter(Tensor(np.ones([64, 48]).astype(np.float32)), name="weight2")

    def construct(self, x, y):
        x = self.fc1(F.cast(x, mstype.float16), F.cast(self.p1, mstype.float16))
        x = self.fc2(x, F.cast(self.p2, mstype.float16))
        x = self.fc1(F.cast(x, mstype.float32), self.p1)
        return x - y


def auto_parallel_compile_net(mode, dev_num, net, strategy1=None, strategy2=None, enable_parallel_optimizer=False,
                              gradient_fp32_sync=True):
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode=mode, device_num=dev_num,
                                      enable_parallel_optimizer=enable_parallel_optimizer,
                                      gradient_fp32_sync=gradient_fp32_sync)
    inputs = Tensor(np.ones([32, 48]).astype(np.float32))
    label = Tensor(np.zeros([32, 64]).astype(np.float32))
    net = net(strategy1, strategy2)
    net = _VirtualDatasetCell(net)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = TrainOneStepCell(net, optimizer).set_comm_fusion(4)
    train_network.set_train()
    _cell_graph_executor.compile(train_network, inputs, label, phase="train")
    context.reset_auto_parallel_context()
    return train_network


def test_auto_parallel_momentum_1():
    auto_parallel_compile_net("auto_parallel", 8, Net1)


def test_auto_parallel_momentum_2():
    # data parallel case
    auto_parallel_compile_net("semi_auto_parallel", 8, Net1, ((8, 1), (1, 1)), ((8, 1), (1, 1)))


def test_auto_parallel_momentum_3():
    # parallel optimizer and mix precision case
    auto_parallel_compile_net("semi_auto_parallel", 8, Net2, ((8, 1), (1, 1)), ((8, 1), (1, 1)))


def test_auto_parallel_momentum_4():
    # parallel optimizer and mix precision case
    auto_parallel_compile_net("semi_auto_parallel", 8, Net2, ((8, 1), (1, 1)), ((8, 1), (1, 1)), True, False)


def test_auto_parallel_momentum_5():
    # test not fully use parallel optimizer with mix precision case
    context.set_auto_parallel_context(optimizer_weight_shard_size=2)
    auto_parallel_compile_net("semi_auto_parallel", 8, Net2, ((8, 1), (1, 1)), ((8, 1), (1, 1)), True)
