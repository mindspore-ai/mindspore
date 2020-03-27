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

import numpy as np
from mindspore import context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
import mindspore as ms
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.ops.operations.comm_ops import _VirtualDataset
context.set_context(mode=context.GRAPH_MODE)


class NetWithLoss(nn.Cell):
    def __init__(self, network, strategy3, strategy4, axis):
        super(NetWithLoss, self).__init__()
        self.virtual_dataset = _VirtualDataset()
        self.one_hot = P.OneHot(axis=axis).set_strategy(strategy3)
        self.on_value = Tensor(2.0, ms.float32)
        self.off_value = Tensor(1.0, ms.float32)
        self.loss = P.SoftmaxCrossEntropyWithLogits().set_strategy(strategy4)
        self.network = network

    def construct(self, x, y, b):
        b_virtual = self.virtual_dataset(b)
        predict = self.network(x, y)
        label = self.one_hot(b_virtual, 64, self.on_value, self.off_value)
        return self.loss(predict, label)[0]


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return C.grad_all(self.network)(x, y, b)


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.matmul = P.MatMul().set_strategy(strategy1)
        self.gelu = P.Gelu().set_strategy(strategy2)

    def construct(self, x, y):
        out = self.matmul(x, y)
        out = self.gelu(out)
        return out


def compile_graph(strategy1, strategy2, strategy3, strategy4, auto=False, onthot_axis=-1):
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2), strategy3, strategy4, axis=onthot_axis))
    if auto:
        context.set_auto_parallel_context(parallel_mode="auto_parallel")
    else:
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.int32)
    _executor.compile(net, x, y, b)


def test_onehot_model_parallel():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy1 = ((2, 4), (4, 2))
    strategy2 = ((2, 8), )
    strategy3 = ((1, 16), (), ())
    strategy4 = ((16, 1), (16, 1))
    compile_graph(strategy1, strategy2, strategy3, strategy4)


def test_onehot_batch_parallel():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy1 = ((2, 4), (4, 2))
    strategy2 = ((2, 8), )
    strategy3 = ((16, 1), (), ())
    strategy4 = ((16, 1), (16, 1))
    compile_graph(strategy1, strategy2, strategy3, strategy4)


def test_onehot_batch_parallel_invalid_strategy():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy1 = ((2, 4), (4, 2))
    strategy2 = ((2, 8), )
    strategy3 = ((16, ), (), ())
    strategy4 = ((16, 1), (16, 1))
    try:
        compile_graph(strategy1, strategy2, strategy3, strategy4)
    except:
        pass


def test_onehot_repeated_calculation():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy1 = ((2, 4), (4, 2))
    strategy2 = ((2, 8), )
    strategy3 = ((4, 1), (), ())
    strategy4 = ((16, 1), (16, 1))
    compile_graph(strategy1, strategy2, strategy3, strategy4)


def test_onehot_auto():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy1 = None
    strategy2 = None
    strategy3 = None
    strategy4 = None
    compile_graph(strategy1, strategy2, strategy3, strategy4, auto=True)


def test_onehot_model_parallel():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy1 = ((2, 4), (4, 2))
    strategy2 = ((2, 8), )
    strategy3 = ((1, 16), (), ())
    strategy4 = ((16, 1), (16, 1))
    compile_graph(strategy1, strategy2, strategy3, strategy4)


def test_onehot_batch_parallel_axis0():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy1 = ((2, 4), (4, 2))
    strategy2 = ((2, 8), )
    strategy3 = ((16, 1), (), ())
    strategy4 = ((16, 1), (16, 1))
    compile_graph(strategy1, strategy2, strategy3, strategy4, onthot_axis=0)


# auto parallel for onehot axis equal to 0 has not been supported yet
def test_onehot_batch_parallel_invalid_strategy_axis0():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy1 = ((2, 4), (4, 2))
    strategy2 = ((2, 8), )
    strategy3 = None
    strategy4 = ((16, 1), (16, 1))
    try:
        compile_graph(strategy1, strategy2, strategy3, strategy4, onthot_axis=0)
    except:
        pass


def test_onehot_repeated_calculation_axis0():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy1 = ((2, 4), (4, 2))
    strategy2 = ((2, 8), )
    strategy3 = ((4, 1), (), ())
    strategy4 = ((16, 1), (16, 1))
    compile_graph(strategy1, strategy2, strategy3, strategy4, onthot_axis=0)


def test_onehot_auto_axis0():
    context.set_auto_parallel_context(device_num=16, global_rank=14)
    strategy1 = None
    strategy2 = None
    strategy3 = None
    strategy4 = None
    compile_graph(strategy1, strategy2, strategy3, strategy4, auto=True, onthot_axis=0)
