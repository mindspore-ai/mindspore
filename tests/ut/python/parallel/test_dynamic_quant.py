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


class NetDynamicQuant(nn.Cell):
    def __init__(self, strategy=None):
        super(NetDynamicQuant, self).__init__()
        self.dynamic_quant = ms.ops.auto_generate.DynamicQuantExt()
        if strategy is not None:
            self.dynamic_quant = self.dynamic_quant.shard(strategy)
        self.weight = Parameter(w, "w")
        self.mul = P.Mul()

    def construct(self, x, smooth_scales):
        x = self.mul(x, self.weight)
        return self.dynamic_quant(x, smooth_scales)[0]


target_shape = [8, 8, 8]
_x = Tensor(np.ones(target_shape), dtype=ms.float32)
w = Tensor(np.ones(target_shape), dtype=ms.float32)
_smooth_scales = Tensor(np.ones(target_shape[-1:]), dtype=ms.float32)
strategy_ok = ((2, 4, 1), (1,))
strategy_fail = ((1, 4, 2), (1,))


def compile_net(net):
    context.set_context(mode=context.GRAPH_MODE)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _smooth_scales)
    context.reset_auto_parallel_context()


def test_dynamic_quant_shard_auto():
    """
    Feature: test DynamicQuant auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = NetDynamicQuant()
    compile_net(net)


def test_dynamic_quant_shard_success():
    """
    Feature: test DynamicQuant model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = NetDynamicQuant(strategy=strategy_ok)
    compile_net(net)


def test_dynamic_quant_shard_fail():
    """
    Feature: test DynamicQuant parallel with invalid strategy
    Description: model parallel
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = NetDynamicQuant(strategy=strategy_fail)
    with pytest.raises(RuntimeError):
        compile_net(net)
