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
import pytest

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Parameter, Tensor
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Momentum, TrainOneStepCell
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class NetSplitWithSize(nn.Cell):
    def __init__(self, axis=0, size_splits=None, strategy=None):
        super(NetSplitWithSize, self).__init__()
        self.size_splits = size_splits
        self.axis = axis
        self.split = ms.ops.auto_generate.SplitWithSize()
        if strategy is not None:
            self.split = self.split.shard(strategy)
        self.weight = Parameter(w, "w")
        self.mul = P.Mul()

    def construct(self, x):
        x = self.mul(x, self.weight)
        return self.split(x, self.size_splits, self.axis)[0]


class NetSplitTensor(nn.Cell):
    def __init__(self, axis=0, out_nums=1, strategy=None):
        super(NetSplitTensor, self).__init__()
        self.out_nums = out_nums
        self.axis = axis

        self.split = ms.ops.auto_generate.SplitTensor()
        if strategy is not None:
            self.split = self.split.shard(strategy)
        self.weight = Parameter(w, "w")
        self.mul = P.Mul()

    def construct(self, x):
        x = self.mul(x, self.weight)
        return self.split(x, self.out_nums, self.axis)[0]


target_shape = [8, 8, 8]
_x = Tensor(np.ones(target_shape), dtype=ms.float32)
w = Tensor(np.ones(target_shape), dtype=ms.float32)
strategy_ok = ((1, 4, 2),)
strategy_fail = ((2, 4, 1),)


def compile_net(net):
    context.set_context(mode=context.GRAPH_MODE)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x)
    context.reset_auto_parallel_context()


def test_splitwith_size_shard_auto():
    """
    Feature: test SplitWithSize auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = NetSplitWithSize(axis=1, size_splits=[6, 2])
    compile_net(net)


def test_splitwith_size_shard_success():
    """
    Feature: test SplitWithSize model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = NetSplitWithSize(axis=0, size_splits=[6, 2], strategy=strategy_ok)
    compile_net(net)


def test_splitwith_size_shard_fail():
    """
    Feature: test SplitWithSize parallel with invalid strategy
    Description: model parallel
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = NetSplitWithSize(axis=0, size_splits=[6, 2], strategy=strategy_fail)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_splitwith_size_strategy_skip_redistribution():
    """
    Feature: test SplitWithSize parallel skip_redistribution
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    # [6, 2] / strategy_fail[0] ->  [3, 1]
    net = NetSplitWithSize(0, [6, 2], strategy_fail)
    net.split.add_prim_attr("skip_redistribution", True)
    compile_net(net)
    context.reset_auto_parallel_context()


def test_split_tensor_shard_auto():
    """
    Feature: test SplitTensor auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = NetSplitTensor(axis=1, out_nums=2)
    compile_net(net)


def test_split_tensor_shard_success():
    """
    Feature: test SplitTensor model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = NetSplitTensor(axis=0, out_nums=2, strategy=strategy_ok)
    compile_net(net)


def test_split_tensor_shard_fail():
    """
    Feature: test SplitTensor parallel with invalid strategy
    Description: model parallel
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = NetSplitTensor(axis=0, out_nums=2, strategy=strategy_fail)
    with pytest.raises(RuntimeError):
        compile_net(net)
