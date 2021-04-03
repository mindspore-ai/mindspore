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
    def __init__(self, matmul_weight, strategy1=None):
        super().__init__()
        self.gatherv2 = P.Gather().shard(strategy1)
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.matmul = P.MatMul(transpose_b=False)
        self.index = Tensor(np.ones([64, 64]), dtype=ms.int32)
        self.matmul_weight = Parameter(matmul_weight, "w1")
        self.axis = 0

    def construct(self, x, b):
        out = self.gatherv2(x, self.index, self.axis)
        out = self.reshape(out, (64, -1))
        out = self.matmul(out, self.matmul_weight)
        return out


_w1 = Tensor(np.ones([4096, 32]), dtype=ms.float32)
_x = Tensor(np.ones([64, 64]), dtype=ms.float32)
_b = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)

def compile_net(net):
    context.set_context(save_graphs=False)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_reshape_skip_redistribution():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8), (1, 1))
    net = Net(_w1, strategy1)
    compile_net(net)
