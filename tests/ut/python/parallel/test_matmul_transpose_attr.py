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
from mindspore.nn import Cell
from mindspore.ops import operations as P
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class MatMulNet(Cell):
    def __init__(self, weight, a=False, b=False, strategy0=None, strategy1=None, strategy2=None):
        super().__init__()
        self.square = P.Square().shard(strategy0)
        self.matmul = P.MatMul(transpose_a=a, transpose_b=b).shard(strategy1)
        self.transpose_a = a
        self.transpose = P.Transpose().shard(strategy0)
        self.relu = P.ReLU().shard(strategy2)
        self.weight = Parameter(weight, "w1")

    def construct(self, x):
        out = self.square(x)
        if self.transpose_a:
            out = self.transpose(out, (1, 0))
        out = self.matmul(out, self.weight)
        out = self.relu(out)
        return out


class BatchMatMulNet(Cell):
    def __init__(self, weight, a=False, b=False, strategy0=None, strategy1=None, strategy2=None, t=(0, 1, 3, 2)):
        super().__init__()
        self.square = P.Square().shard(strategy0)
        self.matmul = P.BatchMatMul(transpose_a=a, transpose_b=b).shard(strategy1)
        self.transpose_a = a
        self.transpose = P.Transpose().shard(strategy0)
        self.relu = P.ReLU().shard(strategy2)
        self.weight = Parameter(weight, "w1")
        self.t = t

    def construct(self, x):
        out = self.square(x)
        if self.transpose_a:
            out = self.transpose(out, self.t)
        out = self.matmul(out, self.weight)
        out = self.relu(out)
        return out


def test_matmul_transpose_a():
    """
    Feature: test matmul
    Description: transpose a
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy0 = ((4, 2),)
    strategy1 = ((2, 4), (2, 1))
    strategy2 = ((4, 1),)
    x = Tensor(np.ones([16, 8]), dtype=ms.float32)
    w = Tensor(np.ones([8, 32]), dtype=ms.float32)
    net = MatMulNet(w, a=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('MatMul-0', ['Transpose-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['AllReduce-0'])


def test_matmul_transpose_a_data_parallel():
    """
    Feature: test matmul
    Description: transpose a, data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy0 = ((8, 1),)
    strategy1 = ((1, 8), (1, 1))
    strategy2 = ((8, 1),)
    x = Tensor(np.ones([16, 8]), dtype=ms.float32)
    w = Tensor(np.ones([8, 32]), dtype=ms.float32)
    net = MatMulNet(w, a=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('MatMul-0', ['Transpose-0'])


def test_matmul_transpose_a_repeat_calculation():
    """
    Feature: test matmul
    Description: transpose a, repeat calculation
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy0 = ((2, 2),)
    strategy1 = ((2, 2), (2, 1))
    strategy2 = ((2, 1),)
    x = Tensor(np.ones([16, 8]), dtype=ms.float32)
    w = Tensor(np.ones([8, 32]), dtype=ms.float32)
    net = MatMulNet(w, a=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('MatMul-0', ['Transpose-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['AllReduce-0'])


def test_matmul_transpose_a_b():
    """
    Feature: test matmul
    Description: transpose a and b
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy0 = ((4, 2),)
    strategy1 = ((2, 4), (1, 2))
    strategy2 = ((4, 1),)
    x = Tensor(np.ones([16, 8]), dtype=ms.float32)
    w = Tensor(np.ones([32, 8]), dtype=ms.float32)
    net = MatMulNet(w, a=True, b=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('MatMul-0', ['Transpose-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['AllReduce-0'])


def test_matmul_transpose_a_b_repeat_calculation():
    """
    Feature: test matmul
    Description: transpose a and b, repeat calculation
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy0 = ((2, 2),)
    strategy1 = ((2, 2), (1, 2))
    strategy2 = ((2, 1),)
    x = Tensor(np.ones([16, 8]), dtype=ms.float32)
    w = Tensor(np.ones([32, 8]), dtype=ms.float32)
    net = MatMulNet(w, a=True, b=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('MatMul-0', ['Transpose-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['AllReduce-0'])


def test_matmul_transpose_a_auto_parallel():
    """
    Feature: test matmul
    Description: transpose a, auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    x = Tensor(np.ones([16, 8]), dtype=ms.float32)
    w = Tensor(np.ones([8, 32]), dtype=ms.float32)
    net = MatMulNet(w, a=True)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('MatMul-0', ['Transpose-0'])


def test_matmul_transpose_a_b_auto_parallel():
    """
    Feature: test matmul
    Description: transpose a and b, auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    x = Tensor(np.ones([16, 8]), dtype=ms.float32)
    w = Tensor(np.ones([32, 8]), dtype=ms.float32)
    net = MatMulNet(w, a=True, b=True)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('MatMul-0', ['Transpose-0'])


def test_matmul_transpose_a_error_strategy():
    """
    Feature: test matmul
    Description: transpose a, error strategy
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy0 = ((2, 2),)
    strategy1 = ((2, 2), (1, 2))
    strategy2 = ((2, 1),)
    x = Tensor(np.ones([16, 8]), dtype=ms.float32)
    w = Tensor(np.ones([8, 32]), dtype=ms.float32)
    net = MatMulNet(w, a=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, x)


def test_matmul_transpose_a_b_error_strategy():
    """
    Feature: test matmul
    Description: transpose a and b, error strategy
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy0 = ((2, 2),)
    strategy1 = ((2, 4), (2, 1))
    strategy2 = ((2, 1),)
    x = Tensor(np.ones([16, 8]), dtype=ms.float32)
    w = Tensor(np.ones([32, 8]), dtype=ms.float32)
    net = MatMulNet(w, a=True, b=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, x)


def test_batch_matmul_transpose_a():
    """
    Feature: test batch matmul
    Description: transpose a
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=64, global_rank=0)
    strategy0 = ((2, 2, 4, 4),)
    strategy1 = ((2, 2, 4, 4), (4, 1))
    strategy2 = ((2, 2, 4, 1),)
    x = Tensor(np.ones([2, 4, 8, 16]), dtype=ms.float32)
    w = Tensor(np.ones([16, 32]), dtype=ms.float32)
    net = BatchMatMulNet(w, a=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('BatchMatMul-0', ['Transpose-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['AllReduce-0'])


def test_batch_matmul_transpose_a_b():
    """
    Feature: test batch matmul
    Description: transpose a and b
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=64, global_rank=0)
    strategy0 = ((2, 2, 4, 4),)
    strategy1 = ((2, 2, 4, 4), (1, 4))
    strategy2 = ((2, 2, 4, 1),)
    x = Tensor(np.ones([2, 4, 8, 16]), dtype=ms.float32)
    w = Tensor(np.ones([32, 16]), dtype=ms.float32)
    net = BatchMatMulNet(w, a=True, b=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('BatchMatMul-0', ['Transpose-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['AllReduce-0'])


def test_batch_matmul_transpose_a_broadcast():
    """
    Feature: test batch matmul
    Description: transpose a, broadcast
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=64, global_rank=0)
    strategy0 = ((2, 2, 4, 4),)
    strategy1 = ((2, 2, 4, 4), (2, 1, 4, 1))
    strategy2 = ((2, 2, 4, 1),)
    x = Tensor(np.ones([2, 4, 8, 16]), dtype=ms.float32)
    w = Tensor(np.ones([2, 1, 16, 32]), dtype=ms.float32)
    net = BatchMatMulNet(w, a=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('BatchMatMul-0', ['Transpose-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['AllReduce-0'])


def test_batch_matmul_transpose_a_b_broadcast():
    """
    Feature: test batch matmul
    Description: transpose a and b, broadcast
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=64, global_rank=0)
    strategy0 = ((2, 2, 4, 4),)
    strategy1 = ((2, 2, 4, 4), (2, 1, 1, 4))
    strategy2 = ((2, 2, 4, 1),)
    x = Tensor(np.ones([2, 4, 8, 16]), dtype=ms.float32)
    w = Tensor(np.ones([2, 1, 32, 16]), dtype=ms.float32)
    net = BatchMatMulNet(w, a=True, b=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('BatchMatMul-0', ['Transpose-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['AllReduce-0'])


def test_batch_matmul_transpose_a_b_broadcast_3x4():
    """
    Feature: test batch matmul
    Description: transpose a and b, broadcast, input dim is 3, weight dim is 4
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=64, global_rank=0)
    strategy0 = ((2, 4, 4),)
    strategy1 = ((2, 4, 4), (1, 1, 1, 4))
    strategy2 = ((1, 2, 4, 1),)
    x = Tensor(np.ones([4, 8, 16]), dtype=ms.float32)
    w = Tensor(np.ones([2, 1, 32, 16]), dtype=ms.float32)
    net = BatchMatMulNet(w, a=True, b=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2, t=(0, 2, 1))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('BatchMatMul-0', ['Transpose-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['AllReduce-0'])


def test_batch_matmul_transpose_a_auto():
    """
    Feature: test batch matmul
    Description: transpose a, auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    x = Tensor(np.ones([8, 4, 8, 16]), dtype=ms.float32)
    w = Tensor(np.ones([8, 1, 16, 32]), dtype=ms.float32)
    net = BatchMatMulNet(w, a=True)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('BatchMatMul-0', ['Transpose-0'])


def test_batch_matmul_transpose_a_b_auto():
    """
    Feature: test batch matmul
    Description: transpose a and b, auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    x = Tensor(np.ones([8, 4, 8, 16]), dtype=ms.float32)
    w = Tensor(np.ones([8, 1, 32, 16]), dtype=ms.float32)
    net = BatchMatMulNet(w, a=True, b=True)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Transpose-0', ['Square-0'])
    assert validator.check_node_inputs_has('BatchMatMul-0', ['Transpose-0'])


def test_batch_matmul_transpose_a_error_strategy():
    """
    Feature: test batch matmul
    Description: transpose a, error strategy
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy0 = ((2, 2, 1, 1),)
    strategy1 = ((2, 2, 2, 1), (2, 2, 1, 2))
    strategy2 = ((2, 2, 1, 1),)
    x = Tensor(np.ones([2, 4, 8, 16]), dtype=ms.float32)
    w = Tensor(np.ones([2, 4, 16, 32]), dtype=ms.float32)
    net = BatchMatMulNet(w, a=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, x)


def test_batch_matmul_transpose_a_b_error_strategy():
    """
    Feature: test batch matmul
    Description: transpose a and b, error strategy
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy0 = ((2, 2, 1, 1),)
    strategy1 = ((2, 2, 1, 2), (2, 2, 1, 2))
    strategy2 = ((2, 2, 1, 1),)
    x = Tensor(np.ones([2, 4, 8, 16]), dtype=ms.float32)
    w = Tensor(np.ones([2, 4, 32, 16]), dtype=ms.float32)
    net = BatchMatMulNet(w, a=True, b=True, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, x)
