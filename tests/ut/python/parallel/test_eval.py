# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out = self.neg(out)
        return out


class EvalNet(Cell):
    def __init__(self, network, strategy2=None):
        super().__init__()
        self.network = network
        self.relu = P.ReLU().shard(strategy2)

    def construct(self, x, b):
        out = self.network(x, b)
        out1 = self.relu(out)
        return out, out1


_x = Tensor(np.ones([64, 64]), dtype=ms.float32)
_w1 = Tensor(np.ones([64, 64]), dtype=ms.float32)
_b = Tensor(np.ones([64, 64]), dtype=ms.float32)

def compile_net(net, input_data, label, is_train=True):
    net.set_train(mode=is_train)
    phase = "train" if is_train else "eval"
    _cell_graph_executor.compile(net, input_data, label, phase=phase)

def test_train_and_eval():
    """
    Feature: test train and eval in semi auto parallel.
    Description: train and eval net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16)
    strategy1 = ((4, 4), (4, 4))
    strategy2 = ((4, 4),)
    net = Net(_w1, strategy1, strategy2)
    eval_net = EvalNet(net, strategy2=strategy2)
    compile_net(net, _x, _b, is_train=True)
    compile_net(eval_net, _x, _b, is_train=False)
    context.reset_auto_parallel_context()

def test_train_and_eval_auto():
    """
    Feature: test train and eval in semi auto parallel.
    Description: train and eval net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16)
    strategy1 = ((4, 4), (4, 4))
    strategy2 = ((4, 4),)
    net = Net(_w1, strategy1, strategy2)
    eval_net = EvalNet(net, strategy2=strategy2)
    compile_net(net, _x, _b, is_train=True)
    compile_net(eval_net, _x, _b, is_train=False)
    context.reset_auto_parallel_context()
