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

import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum, SyncBatchNorm
from mindspore.ops import operations as P


class Net(Cell):
    def __init__(self, conv2d_weight, out_channel, kernel_size, pad_mode, stride):
        super().__init__()
        self.conv2d = P.Conv2D(out_channel=out_channel, kernel_size=kernel_size,
                               pad_mode=pad_mode, stride=stride)
        self.conv2d_weight = Parameter(conv2d_weight, "w1")
        self.bn1 = SyncBatchNorm(num_features=8, process_groups=[[0, 1], [2, 3]])
        self.bn2 = SyncBatchNorm(num_features=8, process_groups=[[0, 1, 2, 3]])
        self.bn3 = SyncBatchNorm(num_features=8)
        self.bn4 = SyncBatchNorm(num_features=8, process_groups=[[0, 1], [2, 3]])
        self.bn5 = SyncBatchNorm(num_features=8, process_groups=[[0], [1], [2], [3]])

    def construct(self, x, b):
        out = self.conv2d(x, self.conv2d_weight)
        out = self.bn1(out)
        out = self.bn2(out)
        out = self.bn3(out)
        out = self.bn4(out)
        out = self.bn5(out)
        return out


_x = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
_w1 = Tensor(np.ones([8, 16, 2, 2]), dtype=ms.float32)
_b = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_syncbatchnorm():
    """
    Feature: test syncbatchnorm
    Description: create group
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="data_parallel", device_num=4, global_rank=0)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1)
    compile_net(net)
    assert net.bn1.group_name == "2_174882033225436b1440b7de44686450"
    assert net.bn2.group_name == "4_937e3b535d29ac4571b6fecb60df6169"
    assert net.bn3.group_name == "hccl_world_group"
    assert net.bn4.group_name == "2_174882033225436b1440b7de44686450"
    assert net.bn5.is_global is False
