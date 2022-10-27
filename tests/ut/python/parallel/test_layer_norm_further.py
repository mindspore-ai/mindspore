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
import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.common.initializer import initializer
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, begin_norm_axis, begin_params_axis, mul_weight, normalized_shape,
                 strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.mul = P.Mul().shard(strategy1)
        self.layer_norm = P.LayerNorm(
            self.begin_norm_axis, self.begin_params_axis).shard(strategy2)
        self.relu = P.ReLU().shard(strategy3)
        self.mul_weight = Parameter(mul_weight, "w1")
        self.normalized_shape = normalized_shape
        self.gamma = Parameter(initializer(
            'ones', self.normalized_shape), name="gamma")
        self.beta = Parameter(initializer(
            'zeros', self.normalized_shape), name="beta")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out, _, _ = self.layer_norm(out, self.gamma, self.beta)
        out = self.relu(out)
        return out


class Net2(Cell):
    def __init__(self, begin_norm_axis, begin_params_axis, mul_weight, normalized_shape,
                 strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.mul = P.Mul().shard(strategy1)
        self.layer_norm = P.LayerNorm(
            self.begin_norm_axis, self.begin_params_axis).shard(strategy2)
        self.relu = P.ReLU().shard(strategy3)
        self.mul_weight = Parameter(mul_weight, "w1")
        self.normalized_shape = normalized_shape
        self.gamma = Parameter(initializer(
            'ones', self.normalized_shape), name="gamma")
        self.beta = Parameter(initializer(
            'zeros', self.normalized_shape), name="beta")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        _, out, _ = self.layer_norm(out, self.gamma, self.beta)
        out = self.relu(out)
        return out


class Net3(Cell):
    def __init__(self, begin_norm_axis, begin_params_axis, mul_weight, normalized_shape,
                 strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.mul = P.Mul().shard(strategy1)
        self.layer_norm = P.LayerNorm(
            self.begin_norm_axis, self.begin_params_axis).shard(strategy2)
        self.relu = P.ReLU().shard(strategy3)
        self.mul_weight = Parameter(mul_weight, "w1")
        self.normalized_shape = normalized_shape
        self.gamma = Parameter(initializer(
            'ones', self.normalized_shape), name="gamma")
        self.beta = Parameter(initializer(
            'zeros', self.normalized_shape), name="beta")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        _, _, out = self.layer_norm(out, self.gamma, self.beta)
        out = self.relu(out)
        return out


class Net4(Cell):
    def __init__(self, begin_norm_axis, begin_params_axis, mul_weight, normalized_shape,
                 strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.mul = P.Mul().shard(strategy1)
        self.layer_norm = P.LayerNorm(
            self.begin_norm_axis, self.begin_params_axis).shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")
        self.normalized_shape = normalized_shape
        self.gamma = Parameter(initializer(
            'ones', self.normalized_shape), name="gamma")
        self.beta = Parameter(initializer(
            'zeros', self.normalized_shape), name="beta")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        _, _, out = self.layer_norm(out, self.gamma, self.beta)
        return out


class Net5(Cell):
    def __init__(self, begin_norm_axis, begin_params_axis, mul_weight, normalized_shape,
                 strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.mul = P.Mul().shard(strategy1)
        self.layer_norm = P.LayerNorm(
            self.begin_norm_axis, self.begin_params_axis).shard(strategy2)
        self.relu = P.ReLU().shard(strategy3)
        self.mul_weight = Parameter(mul_weight, "w1")
        self.normalized_shape = normalized_shape
        self.gamma = Parameter(initializer(
            'ones', self.normalized_shape), name="gamma")
        self.beta = Parameter(initializer(
            'zeros', self.normalized_shape), name="beta")

    def construct(self, x, b):
        out, _, _ = self.layer_norm(x, self.gamma, self.beta)
        out = self.relu(out)
        return out


_x = Tensor(np.ones([16, 64, 32, 16]), dtype=ms.float32)
_w = Tensor(np.ones([16, 64, 32, 16]), dtype=ms.float32)
_b = Tensor(np.ones([16, 64, 32, 16]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(),
                         learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_layer_norm_data_parallel():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1, 1, 1), (16, 1, 1, 1))
    strategy2 = ((16, 1, 1, 1), (1, 1, 1), (1, 1, 1))
    strategy3 = ((16, 1, 1, 1),)
    net = Net(1, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_data_parallel2():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1, 1, 1), (16, 1, 1, 1))
    strategy2 = ((16, 1, 1, 1), (1, 1, 1), (1, 1, 1))
    strategy3 = ((16, 1, 1, 1),)
    net = Net2(1, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_data_parallel3():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1, 1, 1), (16, 1, 1, 1))
    strategy2 = ((16, 1, 1, 1), (1, 1, 1), (1, 1, 1))
    strategy3 = ((16, 1, 1, 1),)
    net = Net3(1, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_data_parallel4():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1, 1, 1), (16, 1, 1, 1))
    strategy2 = ((16, 1, 1, 1), (1, 1, 1), (1, 1, 1))
    strategy3 = ((16, 1, 1, 1),)
    net = Net4(1, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_data_parallel5():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1, 1, 1), (16, 1, 1, 1))
    strategy2 = ((16, 1, 1, 1), (1, 1, 1), (1, 1, 1))
    strategy3 = ((16, 1, 1, 1),)
    net = Net5(1, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_model_parallel():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16, 1, 1), (1, 16, 1, 1))
    strategy2 = ((1, 16, 1, 1), (16, 1, 1), (16, 1, 1))
    strategy3 = ((1, 16, 1, 1),)
    net = Net(2, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_model_parallel2():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16, 1, 1), (1, 16, 1, 1))
    strategy2 = ((1, 16, 1, 1), (16, 1, 1), (16, 1, 1))
    strategy3 = ((1, 16, 1, 1),)
    net = Net2(2, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_model_parallel3():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16, 1, 1), (1, 16, 1, 1))
    strategy2 = ((1, 16, 1, 1), (16, 1, 1), (16, 1, 1))
    strategy3 = ((1, 16, 1, 1),)
    net = Net3(2, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_model_parallel4():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16, 1, 1), (1, 16, 1, 1))
    strategy2 = ((1, 16, 1, 1), (16, 1, 1), (16, 1, 1))
    strategy3 = ((1, 16, 1, 1),)
    net = Net4(2, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_model_parallel5():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16, 1, 1), (1, 16, 1, 1))
    strategy2 = ((1, 16, 1, 1), (16, 1, 1), (16, 1, 1))
    strategy3 = ((1, 16, 1, 1),)
    net = Net5(2, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_hybrid_parallel():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 4, 2, 1), (2, 4, 2, 1))
    strategy2 = ((2, 4, 2, 1), (4, 2, 1), (4, 2, 1))
    strategy3 = ((2, 4, 2, 1),)
    net = Net(3, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_hybrid_parallel2():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 4, 2, 1), (2, 4, 2, 1))
    strategy2 = ((2, 4, 2, 1), (4, 2, 1), (4, 2, 1))
    strategy3 = ((2, 4, 2, 1),)
    net = Net2(3, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_hybrid_parallel3():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 4, 2, 1), (2, 4, 2, 1))
    strategy2 = ((2, 4, 2, 1), (4, 2, 1), (4, 2, 1))
    strategy3 = ((2, 4, 2, 1),)
    net = Net3(3, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_hybrid_parallel4():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 4, 2, 1), (2, 4, 2, 1))
    strategy2 = ((2, 4, 2, 1), (4, 2, 1), (4, 2, 1))
    strategy3 = ((2, 4, 2, 1),)
    net = Net4(3, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_hybrid_parallel5():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 4, 2, 1), (2, 4, 2, 1))
    strategy2 = ((2, 4, 2, 1), (4, 2, 1), (4, 2, 1))
    strategy3 = ((2, 4, 2, 1),)
    net = Net5(3, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_repeat_calc():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((2, 2, 1, 1), (2, 1, 1), (2, 1, 1))
    strategy3 = ((2, 2, 4, 1),)
    net = Net(3, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_repeat_calc2():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((2, 2, 1, 1), (2, 1, 1), (2, 1, 1))
    strategy3 = ((2, 2, 4, 1),)
    net = Net2(3, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_repeat_calc3():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((2, 2, 1, 1), (2, 1, 1), (2, 1, 1))
    strategy3 = ((2, 2, 4, 1),)
    net = Net3(3, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_repeat_calc4():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((2, 2, 1, 1), (2, 1, 1), (2, 1, 1))
    strategy3 = ((2, 2, 4, 1),)
    net = Net4(3, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_repeat_calc5():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((2, 2, 1, 1), (2, 1, 1), (2, 1, 1))
    strategy3 = ((2, 2, 4, 1),)
    net = Net5(3, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    compile_net(net)


def test_layer_norm_wrong_strategy1():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((1, 2, 1, 2), (2, 1, 2), (2, 1, 2))
    strategy3 = ((2, 2, 4, 1),)
    net = Net(1, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_layer_norm_wrong_strategy2():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((1, 2, 1, 2), (2, 1, 2), (2, 1, 2))
    strategy3 = ((2, 2, 4, 1),)
    net = Net(2, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_layer_norm_wrong_strategy3():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((1, 2, 1, 2), (2, 1, 2), (2, 1, 2))
    strategy3 = ((2, 2, 4, 1),)
    net = Net(3, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_layer_norm_wrong_strategy4():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((1, 2, 1, 1), (2, 1, 1), (2, 1, 1))
    strategy3 = ((2, 2, 4, 1),)
    net = Net2(2, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_layer_norm_wrong_strategy5():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((1, 2, 1, 2), (2, 1, 2), (2, 1, 2))
    strategy3 = ((2, 2, 1, 4),)
    net = Net3(3, 1, _w, [64, 32, 16], strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)
