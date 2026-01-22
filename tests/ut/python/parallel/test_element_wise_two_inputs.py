# Copyright 2025 Huawei Technologies Co., Ltd
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

import re
import numpy as np
import pytest

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import context, Tensor, Parameter
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.auto_generate.gen_ops_prim import Polar, IsClose, RemainderTensorTensor, FmodTensor, InplaceCopy
from mindspore.parallel.shard import Layout


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, strategy):
        super().__init__()
        w = Tensor(np.ones([4, 8, 4, 8]), dtype=ms.float32)
        w1 = Tensor(np.ones([4, 8, 1, 8]), dtype=ms.int32)
        self.hypot_w = Parameter(w, "w1")
        self.igamma_w = Parameter(w, "w2")
        self.igammac_w = Parameter(w, "w3")
        self.next_after_w = Parameter(w, "w4")
        self.zeta_w = Parameter(w, "w5")
        self.left_shift_w = Parameter(w1, "w6")
        self.right_shift_w = Parameter(w1, "w7")
        self.hypot = P.Hypot().shard(strategy)
        self.left_shift = P.LeftShift()
        self.right_shift = P.RightShift()
        self.next_after = P.NextAfter()
        self.zeta = P.Zeta()
        self.cast = P.Cast()
        self.gcd = P.Gcd()
        self.gcd_weight = Parameter(w1, "w8")
        self.polar = Polar()
        self.polar_w = Parameter(w, "w9")
        self.isclose = IsClose()
        self.isclose_w = Parameter(w1, "w10")
        self.remaindertensortensor_op = RemainderTensorTensor()
        self.remaindertensortensor_w = Parameter(w, "w11")

    def construct(self, x):
        out = self.hypot(x, self.hypot_w)
        out = ops.igamma(out, self.igamma_w)
        out = ops.igammac(out, self.igammac_w)
        out = self.next_after(out, self.next_after_w)
        out = self.zeta(out, self.zeta_w)
        out = self.polar(out, self.polar_w)
        out = self.remaindertensortensor_op(out, self.remaindertensortensor_w)
        out = self.cast(out, ms.int32)
        out = self.left_shift(out, self.left_shift_w)
        out = self.right_shift(out, self.right_shift_w)
        out = self.gcd(out, self.gcd_weight)
        return out


_x = Tensor(np.ones([4, 8, 4, 8]), dtype=ms.float32)


def test_element_wise_two_inputs_ops():
    """
    Features: test sharding propagation for element wise ops with two inputs
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=64, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((1, 2, 4, 8), (1, 2, 4, 8))
    net = Net(strategy=strategy)
    _cell_graph_executor.compile(net, _x, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("Igamma", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 4, 8]]
        elif re.search("Igammac", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 4, 8]]
        elif re.search("NextAfter", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 4, 8]]
        elif re.search("Zeta", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 4, 8]]
        elif re.search("Polar", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 4, 8]]
        elif re.search("RemainderTensorTensor", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 4, 8]]
        elif re.search("LeftShift", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 1, 8]]
        elif re.search("RightShift", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 1, 8]]
        elif re.search("Gcd", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 1, 8]]

class PolarNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.polar_op = Polar().shard(strategy)
        else:
            self.polar_op = Polar()

    def construct(self, abs_tensor, angle_tensor):
        return self.polar_op(abs_tensor, angle_tensor)

def compile_graph(net, abs_tensor, angle_tensor, device_num=8, parallel_mode="semi_auto_parallel"):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, abs_tensor, angle_tensor)
    return phase

def test_polar_with_positive_values():
    """
    Feature: distribute operator polar with positive values.
    Description: basic.
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    stra = ((1, 4), (1, 4))
    net = PolarNet(stra)
    abs_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    angle_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, abs_tensor, angle_tensor)

def test_polar_with_different_strategies():
    """
    Feature: distribute operator polar.
    Description: the strategies are different.
    Expectation: raise RuntimeError.
    """
    context.set_context(save_graphs=True)
    stra = ((1, 4), (2, 2))
    net = PolarNet(stra)
    abs_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    angle_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)

    with pytest.raises(RuntimeError):
        compile_graph(net, abs_tensor, angle_tensor)

def test_polar_with_different_input_shapes():
    """
    Feature: distribute operator polar.
    Description: the strategies are different.
    Expectation: raise RuntimeError.
    """
    context.set_context(save_graphs=True)
    stra = ((1, 4), (1, 4))
    net = PolarNet(stra)
    abs_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    angle_tensor = Tensor(np.ones([256, 128]), dtype=ms.float32)

    with pytest.raises(ValueError):  # Assuming the implementation raises an error for negative radius
        compile_graph(net, abs_tensor, angle_tensor)

def test_polar_layout_extend():
    """
    Feature: test polar layout extend
    Description: layout extend
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    int_layout = (layout("dp", "cp"), layout("dp", "cp"),)
    net = PolarNet(int_layout)
    abs_tensor = Tensor(np.ones([16, 16]), dtype=ms.float32)
    angle_tensor = Tensor(np.ones([16, 16]), dtype=ms.float32)
    compile_graph(net, abs_tensor, angle_tensor)

class IsCloseNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.isclose_op = IsClose().shard(strategy)
        else:
            self.isclose_op = IsClose()

    def construct(self, input_tensor, other):
        return self.isclose_op(input_tensor, other)

def test_isclose_with_default_values():
    """
    Feature: distribute operator isclose with positive values.
    Description: basic.
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    stra = ((1, 4), (1, 4))
    net = IsCloseNet(stra)
    input_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    other_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, input_tensor, other_tensor)

def test_isclose_shard_strategy_error():
    """
    Feature: test parallel error strategy.
    Description: Invalid shard strategy.
    Expectation: raise RuntimeError.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4), (2, 4))
    net = IsCloseNet(strategy=strategy)
    input_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    other_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    with pytest.raises(RuntimeError):
        compile_graph(net, input_tensor, other_tensor)

def test_isclose_auto_parallel():
    """
    Features: test isclose auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_context(save_graphs=True)
    net = IsCloseNet()
    input_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    other_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, input_tensor, other_tensor, device_num=8, parallel_mode="auto_parallel")

class RemainderTTNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.remainderTT_op = RemainderTensorTensor().shard(strategy)
        else:
            self.remainderTT_op = RemainderTensorTensor()

    def construct(self, input_tensor, other):
        return self.remainderTT_op(input_tensor, other)

def test_remaindertensortensor_with_default_values():
    """
    Feature: distribute operator RemainderTensorTensor with default values.
    Description: basic.
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    stra = ((1, 4), (1, 4))
    net = RemainderTTNet(stra)
    input_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    other_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, input_tensor, other_tensor)

def test_remaindertensortensor_shard_strategy_error():
    """
    Feature: test parallel error strategy.
    Description: Invalid shard strategy.
    Expectation: raise RuntimeError.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4), (2, 4))
    net = RemainderTTNet(strategy=strategy)
    input_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    other_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    with pytest.raises(RuntimeError):
        compile_graph(net, input_tensor, other_tensor)

def test_remaindertensortensor_auto_parallel():
    """
    Features: test RemainderTensorTensor auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_context(save_graphs=True)
    net = RemainderTTNet()
    input_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    other_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, input_tensor, other_tensor, device_num=8, parallel_mode="semi_auto_parallel")



class FmodTensorNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.fmodT_op = FmodTensor().shard(strategy)
        else:
            self.fmodT_op = FmodTensor()

    def construct(self, input_tensor, other):
        return self.fmodT_op(input_tensor, other)

def test_FmodTensor_with_default_values():
    """
    Feature: distribute operator FmodTensor with default values.
    Description: basic.
    Expectation: compile done without error.
    """
    stra = ((4, 1), (4, 1))
    net = FmodTensorNet(stra)
    input_tensor = Tensor(np.ones([128, 128]) * 8, dtype=ms.float32)
    other_tensor = Tensor(np.ones([128, 128]) * 3, dtype=ms.float32)
    compile_graph(net, input_tensor, other_tensor)

def test_FmodTensor_other_shape_broadcast_succ():
    """
    Feature: test FmodTensor parallel shape.
    Description: Valid other shape.
    Expectation: compile done without error.
    """
    stra = ((4, 1), (4, 1))
    net = FmodTensorNet(strategy=stra)
    input_tensor = Tensor(np.ones([128, 128]) * 8, dtype=ms.float32)
    other_tensor = Tensor(np.ones([128, 1]) * 3, dtype=ms.float32)
    compile_graph(net, input_tensor, other_tensor)

def test_FmodTensor_other_shape_broadcast_error():
    """
    Feature: test FmodTensor parallel shape error.
    Description: Invalid other shape.
    Expectation: raise ValueError.
    """
    stra = ((4, 1), (4, 1))
    net = FmodTensorNet(strategy=stra)
    input_tensor = Tensor(np.ones([128, 128]) * 8, dtype=ms.float32)
    other_tensor = Tensor(np.ones([128, 2]) * 3, dtype=ms.float32)
    with pytest.raises(ValueError):
        compile_graph(net, input_tensor, other_tensor)

def test_FmodTensor_auto_parallel():
    """
    Features: test FmodTensor auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    net = FmodTensorNet()
    input_tensor = Tensor(np.ones([128, 128]) * 8, dtype=ms.float32)
    other_tensor = Tensor(np.ones([128, 128]) * 3, dtype=ms.float32)
    compile_graph(net, input_tensor, other_tensor, device_num=8, parallel_mode="semi_auto_parallel")

def test_FmodTensor_auto_parallel_dynamic_shape():
    """
    Features: test FmodTensor auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    net = FmodTensorNet()
    input_dyn = Tensor(shape=[128, 128], dtype=ms.float32)
    other_tensor = Tensor(np.ones([128, 128]) * 3, dtype=ms.float32)
    net.set_inputs(input_dyn, other_tensor)
    compile_graph(net, input_dyn, other_tensor, device_num=8, parallel_mode="semi_auto_parallel")

class CopyNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.copy = InplaceCopy().shard(strategy)
        else:
            self.copy = InplaceCopy()

    def construct(self, output_data, input_data):
        return self.copy(output_data, input_data)

def test_inplace_copy_with_default_values():
    """
    Feature: distribute operator inplace_copy with default values.
    Description: basic.
    Expectation: compile done without error.
    """
    stra = ((4, 1), (4, 1))
    net = CopyNet(strategy=stra)
    input_tensor = Tensor(np.ones([128, 128]) * 8, dtype=ms.float32)
    output_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, output_tensor, input_tensor)

def test_inplace_copy_auto_parallel():
    """
    Features: test inplace_copy auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    net = CopyNet()
    input_tensor = Tensor(np.ones([128, 128]) * 8, dtype=ms.float32)
    output_tensor = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, output_tensor, input_tensor, device_num=8, parallel_mode="semi_auto_parallel")

def test_inplace_copy_other_shape_broadcast_succ():
    """
    Feature: test inplace_copy parallel shape.
    Description: Valid other shape.
    Expectation: compile done without error.
    """
    stra = ((4, 1), (4, 1))
    net = CopyNet(strategy=stra)
    input_tensor = Tensor(np.ones([128, 128]) * 8, dtype=ms.float32)
    output_tensor = Tensor(np.ones([128, 1]), dtype=ms.float32)
    compile_graph(net, output_tensor, input_tensor)
