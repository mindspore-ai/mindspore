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
import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from mindspore.common.api import _cell_graph_executor
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, weight, paddings, strategy1=None, strategy2=None):
        super().__init__()
        self.add = P.Add().shard(strategy1)
        self.pad = P.PadV3().shard(strategy2)
        self.weight = Parameter(weight, "w1")
        self.paddings = Tensor(paddings)
        self.value = Tensor(0.5)

    def construct(self, x, b):
        out = self.add(x, self.weight)
        out = self.pad(out, self.paddings, self.value)
        return out


_x = Tensor(np.ones([32, 16, 8]), dtype=ms.float32)
_w = Tensor(np.ones([32, 16, 8]), dtype=ms.float32)
_b = Tensor(np.ones([32, 16, 12]), dtype=ms.float32)


def compile_net_train(net, _x1, _b1):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x1, _b1)
    context.reset_auto_parallel_context()


def test_pad_v3_parallel_train():
    """
    Feature: test pad v3 parallel train
    Description: shard two dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 4, 1), (2, 4, 1))
    strategy2 = ((1, 4, 1), (1,), ())
    net = Net(_w, paddings=[2, 2], strategy1=strategy1, strategy2=strategy2)
    compile_net_train(net, _x, _b)


def test_pad_v3_parallel():
    """
    Feature: test pad v3 parallel
    Description: shard two dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 4, 1), (2, 4, 1))
    strategy2 = ((2, 4, 1), (1,), ())
    net = Net(_w, paddings=[2, 2], strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('PadV3-0', ['Add-0'])


def test_pad_v3_parallel_1():
    """
    Feature: test pad v3 parallel
    Description: shard two dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1, 4), (2, 1, 4))
    strategy2 = ((2, 1, 4), (1,), ())
    net = Net(_w, paddings=[0, 0, 1, 2], strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('PadV3-0', ['Add-0'])


def test_pad_v3_parallel_2():
    """
    Feature: test pad v3 parallel
    Description: shard two dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2, 4), (1, 2, 4))
    strategy2 = ((1, 2, 4), (1,), ())
    net = Net(_w, paddings=[0, 0, 0, 0, 1, 2], strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('PadV3-0', ['Add-0'])


def test_pad_v3_parallel_3():
    """
    Feature: test pad v3 parallel
    Description: shard two dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 4, 1), (2, 4, 1))
    strategy2 = ((2, 4, 1), (1,), ())
    net = Net(_w, paddings=[1, 2, 0, 0, 0, 0], strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('PadV3-0', ['Add-0'])


def test_pad_v3_wrong_strategy():
    """
    Feature: test pad v3
    Description:
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((1, 4, 2), (1,), ())
    net = Net(_w, paddings=[2, 2, 0, 0, 0, 0], strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x, _b)


def test_pad_v3_wrong_strategy_2():
    """
    Feature: test pad v3
    Description:
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 4, 1), (1,), ())
    net = Net(_w, paddings=[0, 0, 1, 2], strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x, _b)


def test_pad_v3_shard_last_dim():
    """
    Feature: test pad v3
    Description: shard last dim
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 2), (1,), ())
    net = Net(_w, paddings=[2, 2], strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x, _b)


def test_pad_v3_auto_rank0():
    """
    Feature: test pad v3 auto parallel
    Description: sharding propagation
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy1 = ((2, 4, 1), (2, 4, 1))
    net = Net(_w, paddings=[2, 2], strategy1=strategy1)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('PadV3-0', ['Add-0'])
