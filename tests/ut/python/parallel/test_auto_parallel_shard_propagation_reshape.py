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

import re
import numpy as np
from mindspore import Tensor, Parameter
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
import mindspore as ms
from mindspore.graph.api import _cell_graph_executor

def setup_function():
    ms.context.set_auto_parallel_context(dataset_strategy="full_batch")

class MyTestNet(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.TensorAdd().shard(((1, 1), (1, 1)))
        self.reshape0 = P.Reshape()
        self.reshape1 = P.Reshape()
        self.concat = P.Concat(0)
        self.reshape2 = P.Reshape()

    def construct(self, x, y, z):
        out1 = self.reshape0(x, [2, 2])
        out = self.add(y, z)
        out = self.reshape1(out, [4, 2])
        out = self.concat((out1, out))
        out = self.reshape2(out, [2, 6])
        return out

class MyTestNet2(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.reshape = P.Reshape()
        self.concat = P.Concat(0)
        self.add = P.TensorAdd()
        self.relu = P.ReLU().shard(strategy1)
        self.max = P.ReduceMax(keep_dims=True).shard(strategy2)
        self.sub = P.Sub().shard(strategy3)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out0 = self.add(b, Tensor(np.zeros([32000, 32]), dtype=ms.float32))
        out0 = self.reshape(out0, [32, 32000])
        out0 = self.concat((out0, x))
        out1 = self.add(x, self.mul_weight)
        out1 = self.relu(out1)
        out2 = self.max(out1)
        out1 = self.sub(out1, out2)
        return out1

_x = Tensor(np.ones([64, 32000]), dtype=ms.float32)
_w1 = Tensor(np.ones([64, 32000]), dtype=ms.float32)
_b = Tensor(np.ones([32000, 32]), dtype=ms.float32)

def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    ms.context.reset_auto_parallel_context()

def compile_net_and_get_strategies(net, x, y, z):
    net.set_train()
    _cell_graph_executor.compile(net, x, y, z, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    ms.context.reset_auto_parallel_context()
    return strategies

def test_reshape_virtualoutput():
    """
    Feature: Support None.
    Description: Support reshape-virtualoutput.
    Expectation: No exception/error.
    """
    ms.context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=1, global_rank=0,
                                         search_mode="sharding_propagation")
    x = Tensor(np.array([[[1, 1], [1, 1]]]))
    y = Tensor(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]))
    z = Tensor(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]))
    net_ms = MyTestNet()
    net_ms(x, y, z)

def test_reshape_tuple_concat():
    """
    Feature: Support None.
    Description: Support reshape-concat when inputs shapes of concat are different.
    Expectation: No exception/error.
    """
    ms.context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=1, global_rank=0,
                                         search_mode="sharding_propagation")
    x = Tensor(np.array([[[1, 1], [1, 1]]]))
    y = Tensor(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]))
    z = Tensor(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]))
    net_ms = MyTestNet()
    strategies = compile_net_and_get_strategies(net_ms, x, y, z)
    for (k, v) in strategies.items():
        if re.search('Reshape-op', k) is not None:
            assert v == [[1, 1]]
            break

def test_reshape_tuple_concat2():
    """
    Feature: Support None.
    Description: Support reshape-concat when inputs shapes of concat are different.
    Expectation: No exception/error.
    """
    ms.context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                         search_mode="sharding_propagation")
    strategy1 = None
    strategy2 = ((8, 1),)
    strategy3 = ((1, 8), (1, 1))
    net = MyTestNet2(_w1, strategy1, strategy2, strategy3)
    compile_net(net)

def test_reshape_tuple_concat3():
    """
    Feature: Support None.
    Description: Support reshape-concat when inputs shapes of concat are different.
    Expectation: No exception/error.
    """
    ms.context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                         search_mode="sharding_propagation")
    strategy1 = None
    strategy2 = ((4, 2),)
    strategy3 = ((2, 4), (1, 1))
    net = MyTestNet2(_w1, strategy1, strategy2, strategy3)
    compile_net(net)

def test_reshape_tuple_concat4():
    """
    Feature: Support None.
    Description: Support reshape-concat when inputs shapes of concat are different.
    Expectation: No exception/error.
    """
    ms.context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                         search_mode="sharding_propagation")
    strategy1 = None
    strategy2 = ((2, 4),)
    strategy3 = ((4, 2), (1, 1))
    net = MyTestNet2(_w1, strategy1, strategy2, strategy3)
    compile_net(net)

def test_reshape_tuple_concat5():
    """
    Feature: Support None.
    Description: Support reshape-concat when inputs shapes of concat are different.
    Expectation: No exception/error.
    """
    ms.context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                         search_mode="sharding_propagation")
    strategy1 = None
    strategy2 = ((1, 8),)
    strategy3 = ((8, 1), (1, 1))
    net = MyTestNet2(_w1, strategy1, strategy2, strategy3)
    compile_net(net)
