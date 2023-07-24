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
# ============================================================================
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss
from parallel.utils.utils import ParallelValidator, compile_net


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
    def __init__(self, shape, axis=0, strategy1=None, strategy2=None, batch_dims=0):
        super().__init__()
        self.gatherv2 = P.Gather(batch_dims=batch_dims).shard(strategy1)
        self.mul = P.Mul().shard(strategy2)
        self.relu = P.ReLU()
        if strategy2:
            self.relu.shard((strategy2[0],))
        self.index = None
        if shape:
            self.index = Tensor(np.ones(shape), dtype=ms.int32)
        else:
            self.index = Tensor(1, dtype=ms.int32)
        self.axis = axis

    def construct(self, x, y):
        out = self.gatherv2(x, self.index, self.axis)
        out = self.relu(out)
        out = self.mul(out, y)
        return out


def compile_graph(net, device_num, parallel_mode, x, y):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode)
    net.set_train()
    phase = compile_net(net, x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('ReLU-0', ['Gather-0'])


def test_gather_2x0():
    """
    Feature: distribute operator gather
    Description: gather net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 8), ())
    strategy2 = ((8,), (8,))
    net = GradWrap(NetWithLoss(Net(shape=(), axis=0, strategy1=strategy1, strategy2=strategy2)))
    x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_2x1():
    """
    Feature: distribute operator gather
    Description: gather net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 2), (4,))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(shape=(8,), axis=0, strategy1=strategy1, strategy2=strategy2)))
    x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([8, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_2x2():
    """
    Feature: distribute operator gather
    Description: gather net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 2), (1, 4))
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = GradWrap(NetWithLoss(Net(shape=(8, 16), axis=0, strategy1=strategy1, strategy2=strategy2)))
    x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([8, 16, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_2x3():
    """
    Feature: distribute operator gather
    Description: gather net with strategy in semi auto parallel, gather axis is 0.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 2), (2, 1, 2))
    strategy2 = ((2, 1, 2, 2), (2, 1, 2, 2))
    net = GradWrap(NetWithLoss(Net(shape=(8, 16, 4), axis=0, strategy1=strategy1, strategy2=strategy2)))
    x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([8, 16, 4, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_3x3_axis_1():
    """
    Feature: distribute operator gather
    Description: gather net with strategy in semi auto parallel, gather axis is 1.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 1, 2), (2, 1, 2))
    strategy2 = ((1, 2, 1, 2, 2), (1, 2, 1, 2, 2))
    net = GradWrap(NetWithLoss(Net(shape=(8, 16, 4), axis=1, strategy1=strategy1, strategy2=strategy2)))
    x = Tensor(np.ones([32, 64, 16]), dtype=ms.float32)
    y = Tensor(np.ones([32, 8, 16, 4, 16]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_2x2_batch_dims_1():
    """
    Feature: distribute operator gather
    Description: gather net with strategy in semi auto parallel, gather axis is 1, batch dims is 1.
    Expectation: compile done without error.
    """
    strategy1 = ((2, 1), (2, 4))
    strategy2 = ((2, 4), (2, 4))
    net = GradWrap(NetWithLoss(Net(shape=(8, 16), axis=1, strategy1=strategy1, strategy2=strategy2, batch_dims=1)))
    x = Tensor(np.ones([8, 32]), dtype=ms.float32)
    y = Tensor(np.ones([8, 16]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_2x2_batch_dims_1_gen_batch_strategy():
    """
    Feature: distribute operator gather
    Description: gather net without strategy in semi auto parallel, gather axis is 1, batch dims is 1.
    Expectation: compile done without error.
    """
    strategy1 = None
    strategy2 = None
    net = GradWrap(NetWithLoss(Net(shape=(8, 16), axis=1, strategy1=strategy1, strategy2=strategy2, batch_dims=1)))
    x = Tensor(np.ones([8, 32]), dtype=ms.float32)
    y = Tensor(np.ones([8, 16]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_2x3_batch_dims_1():
    """
    Feature: distribute operator gather
    Description: gather net with strategy in semi auto parallel, gather axis is 1, batch dims is 1.
    Expectation: compile done without error.
    """
    strategy1 = ((2, 1), (2, 2, 2))
    strategy2 = ((2, 2, 2), (2, 2, 2))
    net = GradWrap(NetWithLoss(Net(shape=(8, 16, 32), axis=1, strategy1=strategy1, strategy2=strategy2, batch_dims=1)))
    x = Tensor(np.ones([8, 64]), dtype=ms.float32)
    y = Tensor(np.ones([8, 16, 32]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_gather_4x4_batch_dims_2():
    """
    Feature: distribute operator gather
    Description: gather net with strategy in semi auto parallel, gather axis is 2, batch dims is 2.
    Expectation: compile done without error.
    """
    strategy1 = ((2, 2, 1, 2), (2, 2, 2, 2))
    strategy2 = ((2, 2, 2, 2, 2), (2, 2, 2, 2, 2))
    net = GradWrap(NetWithLoss(Net(shape=(4, 8, 16, 32), axis=2, strategy1=strategy1, strategy2=strategy2,
                                   batch_dims=2)))
    x = Tensor(np.ones([4, 8, 4, 64]), dtype=ms.float32)
    y = Tensor(np.ones([4, 8, 16, 32, 64]), dtype=ms.float32)
    compile_graph(net, 32, "semi_auto_parallel", x, y)
