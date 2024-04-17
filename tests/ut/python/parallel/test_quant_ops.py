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
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.operations._inner_ops import Quant
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class FakeQuantPerChannelNet(Cell):
    def __init__(self, weight, min_max, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.add = P.Add().shard(strategy1)
        context.set_context(device_target="GPU")  # skip ut import error
        self.quant = P.FakeQuantPerChannel(channel_axis=0).shard(strategy2)
        context.set_context(device_target="Ascend")
        self.weight = Parameter(weight, "w1")
        self.min = min_max
        self.max = min_max
        self.relu = P.ReLU().shard(strategy3)

    def construct(self, x, b):
        out = self.add(x, self.weight)
        out = self.quant(out, self.min, self.max)
        out = self.relu(out)
        return out


class FakeQuantPerLayerNet(Cell):
    def __init__(self, weight, min_max, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.add = P.Add().shard(strategy1)
        context.set_context(device_target="GPU")  # skip ut import error
        self.quant = P.FakeQuantPerLayer().shard(strategy2)
        context.set_context(device_target="Ascend")
        self.weight = Parameter(weight, "w1")
        self.min = min_max
        self.max = min_max
        self.relu = P.ReLU().shard(strategy3)

    def construct(self, x, b):
        out = self.add(x, self.weight)
        out = self.quant(out, self.min, self.max)
        out = self.relu(out)
        return out


_x = Tensor(np.ones([32, 16, 8]), dtype=ms.float32)
_w = Tensor(np.ones([32, 16, 8]), dtype=ms.float32)
_b = Tensor(np.ones([32, 16, 12]), dtype=ms.float32)


def test_fake_quant_per_channel():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=64, global_rank=0)
    strategy1 = ((2, 4, 8), (2, 4, 8))
    strategy2 = ((2, 4, 8), (2,), (2,))
    strategy3 = ((2, 4, 8),)
    min_max = Tensor(np.ones([32]), dtype=ms.float32)
    net = FakeQuantPerChannelNet(_w, min_max=min_max, strategy1=strategy1, strategy2=strategy2, strategy3=strategy3)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('FakeQuantPerChannel-0', ['Add-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['FakeQuantPerChannel-0'])


def test_fake_quant_per_layer():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=64, global_rank=0)
    strategy1 = ((2, 4, 8), (2, 4, 8))
    strategy2 = ((2, 4, 8), (1,), (1,))
    strategy3 = ((2, 4, 8),)
    min_max = Tensor([1], dtype=ms.float32)
    net = FakeQuantPerLayerNet(_w, min_max=min_max, strategy1=strategy1, strategy2=strategy2, strategy3=strategy3)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('FakeQuantPerLayer-0', ['Add-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['FakeQuantPerLayer-0'])


class AscendQuantNet(Cell):
    def __init__(self, scale, offset, sqrt_mode, strategy, round_mode):
        super().__init__()
        self.quant = Quant(scale, offset, sqrt_mode, round_mode).shard(strategy)

    def construct(self, data):
        out = self.quant(data)
        return out


def test_ascend_quant_1D():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((4,),)

    net = AscendQuantNet(2.0, 1.0, False, strategy, "Round")

    data = Parameter(Tensor(np.ones([4096]), dtype=ms.float32), "data")
    net.set_inputs(data)

    phase = compile_net(net, data)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("data", [1024])


def test_ascend_quant_2D():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((4, 1),)

    net = AscendQuantNet(2.0, 1.0, False, strategy, "Round")

    data = Parameter(Tensor(np.ones([4, 1024]), dtype=ms.float32), "data")
    net.set_inputs(data)

    phase = compile_net(net, data)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("data", [1, 1024])


def test_ascend_quant_3D():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((4, 1, 1),)

    net = AscendQuantNet(2.0, 1.0, False, strategy, "Round")

    data = Parameter(Tensor(np.ones([4, 1024, 1024]), dtype=ms.float32), "data")
    net.set_inputs(data)

    phase = compile_net(net, data)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("data", [1, 1024, 1024])


def test_ascend_quant_4D():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((2, 1, 1, 2),)

    net = AscendQuantNet(2.0, 1.0, False, strategy, "Round")

    data = Parameter(Tensor(np.ones([4, 128, 128, 128]), dtype=ms.float32), "data")
    net.set_inputs(data)

    phase = compile_net(net, data)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("data", [2, 128, 128, 64])
