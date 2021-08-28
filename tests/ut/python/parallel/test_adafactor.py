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
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell
from mindspore.nn.optim.adafactor import AdaFactor
from mindspore.ops import operations as P


class Net(Cell):
    def __init__(self, matmul_weight, add_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.matmul = P.MatMul().shard(strategy1)
        self.add = P.BiasAdd().shard(strategy2)
        self.mul_weight = Parameter(matmul_weight, "w1")
        self.bias = Parameter(add_weight, "bias")

    def construct(self, x, b):
        out = self.matmul(x, self.mul_weight)
        out = self.add(out, self.bias)
        return out


_x = Tensor(np.ones([64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([32, 32]), dtype=ms.float32)
_w2 = Tensor(np.ones([32]), dtype=ms.float32)
_b = Tensor(np.ones([64, 32]), dtype=ms.float32)


def compile_net(net):
    scale_parameter = False
    relative_step = True
    warmup_init = True
    compression = True
    optimizer = AdaFactor(net.trainable_params(), learning_rate=None, weight_decay=0.9,
                          scale_parameter=scale_parameter, relative_step=relative_step,
                          warmup_init=warmup_init, compression=compression)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_opt_data_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((16, 1), (1,))
    net = Net(_w1, _w2, strategy1, strategy2)
    compile_net(net)


def test_opt_model_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((4, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = Net(_w1, _w2, strategy1, strategy2)
    compile_net(net)
