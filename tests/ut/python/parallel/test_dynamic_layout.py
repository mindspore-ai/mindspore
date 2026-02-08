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
import pytest
import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter, Symbol
from mindspore.ops import operations as P
from mindspore.parallel.shard import Layout
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from parallel.utils.utils import ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


def compile_net(net, _x1, _b1):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net.set_inputs(_x1, _b1)
    phase, _ = _cell_graph_executor.compile(train_net, _x1, _b1)
    context.reset_auto_parallel_context()
    return phase, train_net


def test_layout():
    """
    Feature: config layout for dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    class DynamicMulNet(Cell):
        def __init__(self, strategy1):
            super().__init__()
            self.mul = P.Mul().shard(strategy1)
            layout = Layout((8, 1, 1), ("dp", "mp", "xp"))
            layout1 = (layout("dp", "mp", "xp"),)

            self.gelu = P.GeLU().shard(layout1)
            self.w = Parameter(Tensor(np.ones([1]), dtype=ms.float32), "w2")

        def construct(self, x, y):
            out = self.mul(x, self.w)
            out = self.gelu(out)
            return out

    strategy1 = ((8, 1, 1), (1,))
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True, full_batch=True)

    net = DynamicMulNet(strategy1)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(shape=[16, None, 1], dtype=ms.float32)
    y = Tensor(shape=[None, None, None], dtype=ms.float32)

    net.set_inputs(x, y)

    phase, _ = compile_net(net, x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('GeLU-0', ['Mul-0'])


def test_layout_error():
    """
    Feature: config layout for dynamic shape
    Description: symbol-divisor can not be divisible by shard strategy
    Expectation: compile failed
    """
    class DynamicMulNet(Cell):
        def __init__(self, strategy1):
            super().__init__()
            self.mul = P.Mul().shard(strategy1)
            layout = Layout((2, 4, 1), ("dp", "mp", "xp"))
            layout1 = (layout("dp", "mp", "xp"),)

            self.gelu = P.GeLU().shard(layout1)
            self.w = Parameter(Tensor(np.ones([1]), dtype=ms.float32), "w2")

        def construct(self, x, y):
            out = self.mul(x, self.w)
            out = self.gelu(out)
            return out

    strategy1 = ((8, 1, 1), (1,))
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True, full_batch=True)

    net = DynamicMulNet(strategy1)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(shape=[16, None, 1], dtype=ms.float32)
    y = Tensor(shape=[None, None, None], dtype=ms.float32)

    net.set_inputs(x, y)

    with pytest.raises(RuntimeError):
        compile_net(net, x, y)


def test_layout_multi_map():
    """
    Feature: config layout for dynamic shape
    Description: multi map
    Expectation: compile failed
    """
    class DynamicMulNet(Cell):
        def __init__(self, strategy1):
            super().__init__()
            self.mul = P.Mul().shard(strategy1)
            layout = Layout((2, 4, 1), ("dp", "mp", "xp"))
            layout1 = (layout(("dp", "mp"), "None", "xp"),)

            self.gelu = P.GeLU().shard(layout1)
            self.w = Parameter(Tensor(np.ones([1]), dtype=ms.float32), "w2")

        def construct(self, x, y):
            out = self.mul(x, self.w)
            out = self.gelu(out)
            return out

    strategy1 = ((8, 1, 1), (1,))
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True, full_batch=True)

    net = DynamicMulNet(strategy1)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(shape=[16, None, 1], dtype=ms.float32)
    y = Tensor(shape=[None, None, None], dtype=ms.float32)

    net.set_inputs(x, y)

    with pytest.raises(RuntimeError):
        compile_net(net, x, y)


def test_layout_interleaved_parallel():
    """
    Feature: config layout for dynamic shape
    Description: interleaved parallel
    Expectation: compile failed
    """
    class DynamicMulNet(Cell):
        def __init__(self, strategy1):
            super().__init__()
            self.mul = P.Mul().shard(strategy1)
            layout = Layout((2, 4, 2), ("dp", "mp", "interleaved_parallel"))
            layout1 = (layout(("dp", "interleaved_parallel"), "None", "None"),)

            self.gelu = P.GeLU().shard(layout1)
            self.w = Parameter(Tensor(np.ones([1]), dtype=ms.float32), "w2")

        def construct(self, x, y):
            out = self.mul(x, self.w)
            out = self.gelu(out)
            return out

    strategy1 = ((8, 1, 1), (1,))
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True, full_batch=True)

    net = DynamicMulNet(strategy1)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(shape=[16, None, 1], dtype=ms.float32)
    y = Tensor(shape=[None, None, None], dtype=ms.float32)

    net.set_inputs(x, y)

    with pytest.raises(RuntimeError):
        compile_net(net, x, y)


def test_layout_for_mul_dynamic_to_static():
    """
    Feature: config layout for dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    class DynamicMulNet(Cell):
        def __init__(self,):
            super().__init__()
            layout = Layout((8, 1, 1), ("dp", "mp", "xp"))
            layout0 = (layout("dp", "mp", "xp"), layout("dp", "mp", "xp"))
            layout1 = (layout("dp", "mp", "xp"),)
            self.mul = P.Mul().shard(layout0)
            self.gelu = P.GeLU().shard(layout1)
            self.w = Parameter(Tensor(np.ones([16, 16, 1]), dtype=ms.float32), "w2")

        def construct(self, x, y):
            out = self.mul(x, self.w)
            out = self.gelu(out)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, full_batch=True)

    net = DynamicMulNet()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(shape=[16, None, 1], dtype=ms.float32)
    y = Tensor(shape=[None, None, None], dtype=ms.float32)
    net.set_inputs(x, y)

    phase, _ = compile_net(net, x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('GeLU-0', ['Mul-0'])


def test_layout_redistribution():
    """
    Feature: config layout for dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    class DynamicMulNet(Cell):
        def __init__(self, strategy1):
            super().__init__()
            self.mul = P.Mul().shard(strategy1)
            layout = Layout((8, 1, 1), ("dp", "mp", "xp"))
            layout1 = (layout("None", "None", "None"),)

            self.gelu = P.GeLU().shard(layout1)
            self.w = Parameter(Tensor(np.ones([8]), dtype=ms.float32), "w2")

        def construct(self, x, y):
            out = self.mul(x, self.w)
            out = self.gelu(out)
            return out

    strategy1 = ((8, 1, 1), (1,))
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True, full_batch=True)
    context.set_context(save_graphs=True)
    net = DynamicMulNet(strategy1)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(shape=[16, None, 8], dtype=ms.float32)
    y = Tensor(shape=[None, None, None], dtype=ms.float32)

    net.set_inputs(x, y)

    phase, _ = compile_net(net, x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('_VirtualDiv-0', ['AllGather-0'])


def test_layout_for_some_ops():
    """
    Feature: config layout for dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    class DynamicMulNet(Cell):
        def __init__(self,):
            super().__init__()
            layout = Layout((2, 4), ("dp", "mp"))
            layout0 = (layout("dp", "None", "None", "mp"), layout("dp", "None", "None", "mp"))
            layout1 = (layout("dp", "None", "None", "mp"), layout("mp",))
            layout2 = (layout("dp", "None", "None", "mp"),)
            layout3 = (layout("dp", "None", "None", "None"),)
            layout4 = (layout("dp", "None"), layout("None", "mp"))
            layout5 = (layout("dp", "mp"), layout("mp",))
            layout6 = (layout("dp", "None"), layout("None",), layout("None",))
            self.mul = P.Mul().shard(layout0)
            self.add = P.Add().shard(layout1)
            self.sub = P.Sub().shard(layout1)
            self.div = P.Div().shard(layout1)

            self.gelu = P.GeLU().shard(layout2)
            self.relu = P.ReLU().shard(layout2)
            self.softmax = P.Softmax().shard(layout3)
            self.sigmoid = P.Sigmoid().shard(layout3)
            self.transpose = P.Transpose().shard(layout3)
            self.reshape = P.Reshape()
            self.matmul = P.MatMul().shard(layout4)
            self.bias_add = P.BiasAdd().shard(layout5)
            self.layernorm = P.LayerNorm().shard(layout6)
            self.w1 = Parameter(Tensor(np.ones([8]), dtype=ms.float32), "w1")
            self.w2 = Parameter(Tensor(np.ones([8, 64]), dtype=ms.float32), "w2")
            self.bias = Parameter(Tensor(np.ones([64]), dtype=ms.float32), "bias")
            self.gamma = Parameter(Tensor(np.ones([64]), dtype=ms.float32), "gamma")
            self.beta = Parameter(Tensor(np.ones([64]), dtype=ms.float32), "beta")

        def construct(self, x, y):
            out = self.mul(x, y)
            out = self.add(out, self.w1)
            out = self.sub(out, self.w1)
            out = self.div(out, self.w1)
            out = self.gelu(out)
            out = self.relu(out)
            out = self.softmax(out)
            out = self.sigmoid(out)
            out = self.transpose(out, (0, 2, 1, 3))
            out = self.reshape(out, (-1, 8))
            out = self.matmul(out, self.w2)
            out = self.bias_add(out, self.bias)
            out, _, _ = self.layernorm(out, self.gamma, self.beta)
            out = self.reshape(out, (16, -1, 32, 8))
            out = self.relu(out)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True, full_batch=True)
    context.set_context(save_graphs=True)
    net = DynamicMulNet()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(shape=[16, None, 32, 8], dtype=ms.float32)
    y = Tensor(shape=[16, None, 32, 8], dtype=ms.float32)

    net.set_inputs(x, y)

    phase, _ = compile_net(net, x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('GeLU-0', ['Div-0'])
    assert validator.check_node_inputs_has('Transpose-0', ['Sigmoid-0'])
    assert validator.check_node_inputs_has('BiasAdd-0', ['MatMul-0'])


def test_layout_for_bias_add():
    """
    Feature: config layout for bias add
    Description: no redistribution
    Expectation: compile success
    """
    class DynamicBiasAddNet(Cell):
        def __init__(self,):
            super().__init__()
            layout = Layout((2, 2, 2), ("dp", "mp", "xp"))
            layout0 = (layout("dp", "mp", "xp"), layout("xp",))
            layout1 = (layout("dp", "mp", "xp"),)
            self.bias_add = P.BiasAdd().shard(layout0)
            self.gelu = P.GeLU().shard(layout1)
            self.w = Parameter(Tensor(np.ones([32]), dtype=ms.float32), "w2")

        def construct(self, x, y):
            out = self.bias_add(x, self.w)
            out = self.gelu(out)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True, full_batch=True)

    net = DynamicBiasAddNet()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    s = Symbol(divisor=2)
    x = Tensor(shape=[s, s, s], dtype=ms.float32)
    y = Tensor(shape=[None, None, None], dtype=ms.float32)

    net.set_inputs(x, y)

    phase, _ = compile_net(net, x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('GeLU-0', ['BiasAdd-0'])
