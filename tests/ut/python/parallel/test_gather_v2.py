# Copyright 2019 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss
from parallel.utils.utils import ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


class Net(nn.Cell):
    def __init__(self, axis=0, strategy1=None, strategy2=None, shape=None, target="", gather_out_strategy=None):
        super().__init__()
        if shape is None:
            shape = [64, 64]
        self.gatherv2 = P.Gather().shard(strategy1, gather_out_strategy).set_device(target)
        self.mul = P.Mul().shard(strategy2)
        self.index = Tensor(np.ones(shape), dtype=ms.int32)
        self.axis = axis

    def construct(self, x, y):
        out = self.gatherv2(x, self.index, self.axis)
        out = self.mul(out, y)
        return out


class Net2(nn.Cell):
    def __init__(self, axis=0, strategy1=None, strategy2=None, shape=None, gather_out_strategy=None):
        super().__init__()
        if shape is None:
            shape = [64, 64]
        self.gatherv2 = P.Gather().shard(strategy1, gather_out_strategy)
        self.sub = P.Sub().shard(strategy2)
        self.index = Parameter(Tensor(np.ones(shape), dtype=ms.float32), "w")
        self.axis = axis

    def construct(self, x, y):
        out = self.gatherv2(self.index, x, self.axis)
        out = self.sub(out, y)
        return out


def compile_graph(net, device_num, parallel_mode, x, y, search_mode="dynamic_programming"):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode,
                                      search_mode=search_mode)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, x, y)
    return phase


def test_gatherv2_semi_auto0():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 8), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_semi_auto1():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_semi_auto2():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((2, 4), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_semi_auto3():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with strategy in semi auto parallel, gather axis is 1.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 8), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(1, strategy1, strategy2)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_semi_auto4():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with strategy in semi auto parallel, gather axis is 1.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(1, strategy1, strategy2)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_semi_auto5():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with strategy in semi auto parallel, gather axis is 1.
    Expectation: compile done without error.
    """
    strategy1 = ((2, 4), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(1, strategy1, strategy2)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_semi_auto6():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(0, None, strategy2)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 32]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_semi_auto7():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with strategy in semi auto parallel, gather axis is 1.
    Expectation: compile done without error.
    """
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(1, None, strategy2)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_semi_auto8():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((8,), (1, 1))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2)))
    x = Tensor(np.ones([64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_forward_all_reduce():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net using forward all_reduce in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((2, 4, 1), (2, 4, 1))
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, shape=[2, 64])))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([2, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_shard_batch_and_axis():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with batch and axis sharding strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((4, 1), (2, 1))
    strategy2 = ((2, 4, 1), (2, 4, 1))
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, shape=[2, 64])))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([2, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_split_axis_0_repeat_calc():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with repeat calculate strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 4, 1), (2, 4, 1))
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, shape=[2, 64])))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([2, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_auto0():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net without strategy in auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    net = GradWrap(NetWithLoss(Net(0)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 32]), dtype=ms.float32)
    compile_graph(net, 8, "auto_parallel", x, y)


def test_gatherv2_auto1():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net without strategy in auto parallel, gather axis is 1.
    Expectation: compile done without error.
    """
    net = GradWrap(NetWithLoss(Net(1)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "auto_parallel", x, y)


def test_gatherv2_out_strategy_allreduce():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: axis is 0, split axis with device num and out strategy use allreduce.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    out_strategy = ((1, 1, 1),)
    strategy2 = ((2, 4, 1), (2, 4, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, gather_out_strategy=out_strategy)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_out_strategy_allreduce_repeat_calc():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: axis is 0, split axis, split num small than device num and out strategy use allreduce.
    Expectation: compile done without error.
    """
    strategy1 = ((4, 1), (1, 1))
    out_strategy = ((1, 1, 1),)
    strategy2 = ((2, 4, 1), (2, 4, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, gather_out_strategy=out_strategy)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_out_strategy_reducescatter():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: axis is 0, split axis with device num and out strategy use reducescatter.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    out_strategy = ((8, 1, 1),)
    strategy2 = ((2, 4, 1), (2, 4, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, gather_out_strategy=out_strategy)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_out_strategy_reducescatter_repeat_calc():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: axis is 0, split axis, split num small than device num and out strategy use reducescatter.
    Expectation: compile done without error.
    """
    strategy1 = ((4, 1), (1, 1))
    out_strategy = ((4, 1, 1),)
    strategy2 = ((2, 4, 1), (2, 4, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, gather_out_strategy=out_strategy)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_shard_batch_and_axis_out_strategy_allreduce():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: axis is 0, split axis and batch, out strategy use allreduce.
    Expectation: compile done without error.
    """
    strategy1 = ((4, 1), (2, 1))
    out_strategy = ((2, 1, 1),)
    strategy2 = ((2, 4, 1), (2, 4, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, gather_out_strategy=out_strategy)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_shard_batch_and_axis_out_strategy_reducescatter():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: axis is 0, split axis and batch, out strategy use reducescatter.
    Expectation: compile done without error.
    """
    strategy1 = ((4, 1), (2, 1))
    out_strategy = ((8, 1, 1),)
    strategy2 = ((2, 4, 1), (2, 4, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, gather_out_strategy=out_strategy)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_target_cpu_reducescatter():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: axis is 0, split axis and batch, out strategy use reducescatter.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    out_strategy = ((8, 1, 1),)
    strategy2 = ((2, 4, 1), (2, 4, 1))
    context.set_auto_parallel_context(dataset_strategy="data_parallel")
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, target="CPU", gather_out_strategy=out_strategy)))
    x = Tensor(np.ones([64 // 8, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64 // 8, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_target_cpu_allreduce():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: axis is 0, split axis and batch, out strategy use allreduce.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    out_strategy = ((1, 1, 1),)
    strategy2 = ((2, 4, 1), (2, 4, 1))
    context.set_auto_parallel_context(dataset_strategy="data_parallel")
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, target="CPU", gather_out_strategy=out_strategy)))
    x = Tensor(np.ones([64 // 8, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64 // 8, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_bs_1_and_set_out_strategy_allreduce():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: bs is 1, and set out strategy.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    out_strategy = ((1, 1, 1),)
    strategy2 = ((1, 8, 1), (1, 8, 1))
    net = Net2(0, strategy1, strategy2, shape=(64, 64), gather_out_strategy=out_strategy)
    x = Tensor(np.ones([1, 64]), dtype=ms.int32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    phase = compile_graph(net, 8, "semi_auto_parallel", x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('AllReduce-0', ['Mul-0'])


def test_gatherv2_bs_1_and_set_out_strategy_reduce_scatter():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: bs is 1, and set out strategy with reduce scatter.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    out_strategy = ((8, 1, 1),)
    strategy2 = ((1, 8, 1), (1, 8, 1))
    net = Net2(0, strategy1, strategy2, shape=(64, 64), gather_out_strategy=out_strategy)
    x = Tensor(np.ones([1, 64]), dtype=ms.int32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    with pytest.raises(RuntimeError):
        compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv2_no_shard_axis_and_set_out_strategy():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: no shard axis
    Expectation: compile done without error.
    """
    strategy1 = ((1, 1), (8, 1))
    out_strategy = ((8, 1, 1),)
    strategy2 = ((8, 1, 1), (8, 1, 1))
    net = Net2(0, strategy1, strategy2, shape=(64, 64), gather_out_strategy=out_strategy)
    x = Tensor(np.ones([8, 64]), dtype=ms.int32)
    y = Tensor(np.ones([8, 64, 64]), dtype=ms.float32)
    phase = compile_graph(net, 8, "semi_auto_parallel", x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Sub-0', ['Gather-0'])


def test_gatherv2_dynamic_shape_and_set_out_strategy():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: no shard axis
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    out_strategy = ((1, 1, 1),)
    strategy2 = ((1, 1, 1), (1, 1))
    net = Net2(0, strategy1, strategy2, shape=(64, 64), gather_out_strategy=out_strategy)
    x = Tensor(shape=[None, 64], dtype=ms.int32)
    y = Tensor(np.ones([64, 64]), dtype=ms.float32)
    net.set_inputs(x, y)
    phase = compile_graph(net, 8, "semi_auto_parallel", x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('AllReduce-0', ['Mul-0'])


class Net3(nn.Cell):
    def __init__(self, axis=0, strategy1=None):
        super().__init__()
        self.gatherv2 = P.Gather().shard(strategy1)
        self.axis = axis

    def construct(self, x, y):
        out = self.gatherv2(x, y, self.axis)
        return out


def test_gather_dynamic_shape_constraint():
    """
    Feature: distribute operator gather in dynamic shape
    Description: shard axis, and param[axis] is dynamic
    Expectation: compile failed
    """
    context.set_auto_parallel_context(full_batch=False)
    strategy1 = ((8, 1), (1, 1))
    net = Net3(0, strategy1)
    x = Tensor(shape=[None, 64], dtype=ms.float32)
    y = Tensor(np.ones([64, 64]), dtype=ms.int32)
    with pytest.raises(RuntimeError):
        compile_graph(net, 8, "semi_auto_parallel", x, y)
