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
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.ops.auto_generate import Embedding
from mindspore.graph.api import _cell_graph_executor
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
    def __init__(self, strategy1=None, strategy2=None, shape=None, embedding_out_strategy=None):
        super().__init__()
        if shape is None:
            shape = [64, 64]
        self.embedding = Embedding().shard(strategy1, embedding_out_strategy)
        self.mul = P.Mul().shard(strategy2)
        self.input = Tensor(np.ones(shape), dtype=ms.int32)

    def construct(self, x, y):
        out = self.embedding(self.input, x)
        out = self.mul(out, y)
        return out


def compile_graph(net, device_num, parallel_mode, x, y, search_mode="sharding_propagation"):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode,
                                      search_mode=search_mode)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, x, y)
    return phase


def test_embedding_semi_auto0():
    """
    Feature: distribute operator embedding in auto parallel.
    Description: embedding net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 1), (2, 4))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    x = Parameter(np.ones([64, 64]).astype(np.float32))
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_embedding_semi_auto1():
    """
    Feature: distribute operator embedding in auto parallel.
    Description: embedding net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 1), (1, 8))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    x = Parameter(np.ones([64, 64]).astype(np.float32))
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_embedding_semi_auto2():
    """
    Feature: distribute operator embedding in auto parallel.
    Description: embedding net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 1), (8, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    x = Parameter(np.ones([64, 64]).astype(np.float32))
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_embedding_semi_auto6():
    """
    Feature: distribute operator embedding in auto parallel.
    Description: embedding net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(None, strategy2)))
    x = Parameter(np.ones([64, 32]).astype(np.float32))
    y = Tensor(np.ones([64, 64, 32]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_embedding_auto0():
    """
    Feature: distribute operator embedding in auto parallel.
    Description: embedding net without strategy in auto parallel.
    Expectation: compile done without error.
    """
    net = GradWrap(NetWithLoss(Net()))
    x = Parameter(np.ones([64, 32]).astype(np.float32))
    y = Tensor(np.ones([64, 64, 32]), dtype=ms.float32)
    compile_graph(net, 8, "auto_parallel", x, y)


def test_embedding_semi_auto_parallel():
    """
    Feature: distribute operator embedding in semi auto parallel.
    Description: split axis, split num small than device num and out strategy use reducescatter.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 1), (2, 4))
    out_strategy = ((1, 1, 4),)
    strategy2 = ((2, 4, 1), (2, 4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, embedding_out_strategy=out_strategy)))
    x = Parameter(np.ones([64, 64]).astype(np.float32))
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)
