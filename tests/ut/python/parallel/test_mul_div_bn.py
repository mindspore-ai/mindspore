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
import mindspore.context as context
from mindspore.common.api import _executor
from mindspore import Tensor, Parameter
import mindspore.nn as nn
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


class TwoInputBpropOperator(Cell):
    def __init__(self):
        super().__init__()
        self.op = P.Mul()
        self.bp = P.Add()

    def construct(self, x, y):
        return self.op(x, y)

    def bprop(self, x, y, out, dout):
        return self.bp(5, x), self.bp(y, 8)


class ParallelFloorDivBpropNet(Cell):
    def __init__(self, mul_size, test_size, strategy=None, strategy2=None):
        super().__init__()
        mul_np = np.full(mul_size, 0.5, dtype=np.float32)
        floordiv_np = np.full(test_size, 0.1, dtype=np.float32)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        self.floordiv_weight = Parameter(Tensor(floordiv_np), name="floordiv_weight")
        self.mul = TwoInputBpropOperator()
        self.floor_div = P.FloorDiv()
        self.bn = nn.BatchNorm1d(num_features=96)
        if strategy is not None:
            self.mul.op.shard(strategy2)
            self.mul.bp.shard(strategy2)
            self.floor_div.shard(strategy)

    def construct(self, inputs, label):
        x = self.mul(inputs, self.mul_weight)
        x = self.floor_div(x, self.floordiv_weight)
        x = self.bn(x)
        return x


inputs_ = Tensor(np.random.randn(128, 96).astype(np.float32), dtype=ms.float32)
label_ = Tensor(np.random.randn(128, 96).astype(np.float32), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _executor.compile(train_net, inputs_, label_)
    context.reset_auto_parallel_context()


def test_net():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    strategy = ((4, 1), (4, 1))
    net = ParallelFloorDivBpropNet(mul_size=(128, 96), test_size=(128, 96), strategy=strategy, strategy2=strategy)
    compile_net(net)
