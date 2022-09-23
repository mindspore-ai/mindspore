# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


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
    def __init__(self, axis=0, strategy1=None, strategy2=None, strategy3=None,
                 shape=None, target="", gather_out_strategy=None):
        super().__init__()
        if shape is None:
            shape = [64, 64]
        self.gatherv2 = P.Gather().shard(
            strategy1, gather_out_strategy).set_device(target)
        self.mul = P.Mul().shard(strategy2)
        self.softmax = P.Softmax().shard(strategy3)
        self.index = Tensor(np.ones(shape), dtype=ms.int32)
        self.axis = axis

    def construct(self, x, y):
        out = self.gatherv2(x, self.index, self.axis)
        out = self.mul(out, y)
        out = self.softmax(out)
        return out


def compile_graph(net, device_num, parallel_mode, x, y):
    context.set_auto_parallel_context(
        device_num=device_num, global_rank=0, parallel_mode=parallel_mode)
    net.set_auto_parallel()
    net.set_train()
    _cell_graph_executor.compile(net, x, y)


def test_gather_and_softmax_1():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather and softmax net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 8), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    strategy3 = ((8, 1, 1),)
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_and_softmax_2():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather and softmax net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    strategy3 = ((1, 1, 1),)
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_and_softmax_3():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather and softmax net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((2, 4), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    strategy3 = ((8, 1, 1),)
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_and_softmax_4():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather and softmax net with strategy in semi auto parallel, gather axis is 1.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 8), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    strategy3 = ((8, 1, 1),)
    net = GradWrap(NetWithLoss(Net(1, strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_and_softmax_5():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with strategy in semi auto parallel, gather axis is 1.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    strategy3 = ((8, 1, 1),)
    net = GradWrap(NetWithLoss(Net(1, strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_and_softmax_6():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather and softmax net with strategy in semi auto parallel, gather axis is 1.
    Expectation: compile done without error.
    """
    strategy1 = ((2, 4), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    strategy3 = ((8, 1, 1),)
    net = GradWrap(NetWithLoss(Net(1, strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_and_softmax_7():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather and softmax net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy2 = ((4, 2, 1), (4, 2, 1))
    strategy3 = ((8, 1, 1),)
    net = GradWrap(NetWithLoss(Net(0, None, strategy2, strategy3)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 32]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv_and_softmax_8():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with strategy in semi auto parallel, gather axis is 1.
    Expectation: compile done without error.
    """
    strategy2 = ((4, 2, 1), (4, 2, 1))
    strategy3 = ((8, 1, 1),)
    net = GradWrap(NetWithLoss(Net(1, None, strategy2, strategy3)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gatherv_and_softmax_9():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((8,), (1, 1))
    strategy2 = ((4, 2), (4, 2))
    strategy3 = ((8, 1),)
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_softmax_forward_all_reduce():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net using forward all_reduce in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((2, 4, 1), (2, 4, 1))
    strategy3 = ((1, 1, 1),)
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    net = GradWrap(NetWithLoss(
        Net(0, strategy1, strategy2, strategy3, shape=[2, 64])))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([2, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_softmax_shard_batch_and_axis():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with batch and axis sharding strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((4, 1), (2, 1))
    strategy2 = ((2, 4, 1), (2, 4, 1))
    strategy3 = ((2, 1, 1),)
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    net = GradWrap(NetWithLoss(
        Net(0, strategy1, strategy2, strategy3, shape=[2, 64])))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([2, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_softmax_split_axis_0_repeat_calc():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net with repeat calculate strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 4, 1), (2, 4, 1))
    strategy3 = ((2, 1, 1),)
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    net = GradWrap(NetWithLoss(
        Net(0, strategy1, strategy2, strategy3, shape=[2, 64])))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([2, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_softmax_auto0():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net without strategy in auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    net = GradWrap(NetWithLoss(Net(0)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 32]), dtype=ms.float32)
    compile_graph(net, 8, "auto_parallel", x, y)


def test_gather_softmax_auto1():
    """
    Feature: distribute operator gather in auto parallel.
    Description: gather net without strategy in auto parallel, gather axis is 1.
    Expectation: compile done without error.
    """
    net = GradWrap(NetWithLoss(Net(1)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "auto_parallel", x, y)


def test_gather_softmax_out_strategy_allreduce():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: axis is 0, split axis with device num and out strategy use allreduce.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    out_strategy = ((1, 1, 1),)
    strategy2 = ((2, 4, 1), (2, 4, 1))
    strategy3 = ((8, 1, 1),)
    net = GradWrap(NetWithLoss(
        Net(0, strategy1, strategy2, strategy3, gather_out_strategy=out_strategy)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_softmax_out_strategy_allreduce_repeat_calc():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: axis is 0, split axis, split num small than device num and out strategy use allreduce.
    Expectation: compile done without error.
    """
    strategy1 = ((4, 1), (1, 1))
    out_strategy = ((1, 1, 1),)
    strategy2 = ((2, 4, 1), (2, 4, 1))
    strategy3 = ((8, 1, 1),)

    net = GradWrap(NetWithLoss(
        Net(0, strategy1, strategy2, strategy3, gather_out_strategy=out_strategy)))
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_softmax_out_strategy_reducescatter():
    """
    Feature: distribute operator gather in semi auto parallel.
    Description: axis is 0, split axis with device num and out strategy use reducescatter.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1, 1))
    out_strategy = ((8, 1, 1),)
    strategy2 = ((2, 4, 1), (2, 4, 1))
    strategy3 = ((8, 1, 1),)
    net = GradWrap(NetWithLoss(
        Net(0, strategy1, strategy2, strategy3, gather_out_strategy=out_strategy)))
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
    strategy3 = ((8, 1, 1),)
    net = GradWrap(NetWithLoss(
        Net(0, strategy1, strategy2, strategy3, gather_out_strategy=out_strategy)))
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
    strategy3 = ((8, 1, 1),)
    net = GradWrap(NetWithLoss(
        Net(0, strategy1, strategy2, strategy3, gather_out_strategy=out_strategy)))
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
    strategy3 = ((8, 1, 1),)
    net = GradWrap(NetWithLoss(
        Net(0, strategy1, strategy2, strategy3, gather_out_strategy=out_strategy)))
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
    strategy3 = ((8, 1, 1),)
    context.set_auto_parallel_context(dataset_strategy="data_parallel")
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, strategy3,
                                   target="CPU", gather_out_strategy=out_strategy)))
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
    strategy3 = ((8, 1, 1),)
    context.set_auto_parallel_context(dataset_strategy="data_parallel")
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, strategy3,
                                   target="CPU", gather_out_strategy=out_strategy)))
    x = Tensor(np.ones([64 // 8, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64 // 8, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)
