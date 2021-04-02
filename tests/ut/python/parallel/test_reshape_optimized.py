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

import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


class Net(Cell):
    def __init__(self, mul_weight):
        super().__init__()
        self.reshape1 = P.Reshape()
        self.reshape2 = P.Reshape()
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.reshape1(self.mul_weight, (128, 64, 32))
        out = self.reshape2(out, (128, 64, 32))
        return out


_x = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_b = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)


def compile_net(net):
    context.set_context(save_graphs=False)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_reshape_optimized():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0)
    net = Net(_w1)
    compile_net(net)
