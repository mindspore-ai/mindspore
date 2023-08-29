# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, Parameter
import mindspore.nn as nn
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(nn.Cell):
    def __init__(self, wi, stra1=None, stra2=None, stra3=None):
        super(Net, self).__init__()
        self.wi = Parameter(wi, "wi")
        self.matmul = P.MatMul().shard(stra1)
        self.onehot = P.OneHot(axis=-1).shard(stra2)
        self.mul = P.Mul().shard(stra3)
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.cast = P.Cast()
        self.depth = 48

    def construct(self, x):
        output = self.matmul(x, self.wi)
        output = self.cast(output, ms.int32)
        output = self.onehot(output, self.depth, self.on_value, self.off_value)
        output = self.mul(output, output)
        return output


_x = Tensor(np.ones([16, 48]), dtype=ms.float32)
_wi = Tensor(np.ones([48, 16]), dtype=ms.float32)


def compile_net(net):
    context.set_context(mode=context.GRAPH_MODE)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x)
    context.reset_auto_parallel_context()


def test_onehot():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, enable_alltoall=True,
                                      global_rank=0)
    stra1 = ((8, 1), (1, 1))
    stra2 = ((8, 1, 1), (), ())
    stra3 = ((8, 1, 1), (8, 1, 1))
    net = Net(_wi, stra1=stra1, stra2=stra2, stra3=stra3)
    compile_net(net)
