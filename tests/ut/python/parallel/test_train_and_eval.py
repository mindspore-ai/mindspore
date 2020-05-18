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
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from mindspore.common.api import _executor


class Net(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().set_strategy(strategy1)
        self.neg = P.Neg().set_strategy(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out = self.neg(out)
        return out


class EvalNet(Cell):
    def __init__(self, network, strategy2=None):
        super().__init__()
        self.network = network
        self.relu = P.ReLU().set_strategy(strategy2)

    def construct(self, x, b):
        out = self.network(x, b)
        out = self.relu(out)
        return out 


_x = Tensor(np.ones([8, 8]), dtype=ms.float32)
_w1 = Tensor(np.ones([8, 8]), dtype=ms.float32)
_b = Tensor(np.ones([8, 8]), dtype=ms.float32)


def test_train_and_eval():
    context.set_context(save_graphs=True, mode=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16)
    strategy1 = ((4, 4), (4, 4))
    strategy2 = ((4, 4), )
    net = Net(_w1, strategy1, strategy2)
    eval_net = EvalNet(net, strategy2=strategy2)
    net.set_train()
    net.set_auto_parallel()
    _executor.compile(net, _x,  _b, phase='train', auto_parallel_mode=True)

    eval_net.set_train(mode=False)
    eval_net.set_auto_parallel()
    _executor.compile(eval_net, _x,  _b, phase='eval', auto_parallel_mode=True)

    context.reset_auto_parallel_context()