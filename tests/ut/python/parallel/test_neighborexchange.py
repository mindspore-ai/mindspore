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
# ============================================================================
import numpy as np
import mindspore as ms
import mindspore.context as context
from mindspore import Tensor, Parameter
import mindspore.nn as nn
from mindspore.common.api import _executor
from mindspore.nn import TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from mindspore.ops.operations._inner_ops import NeighborExchange


class MatMulNet(nn.Cell):
    def __init__(self, weight1):
        super(MatMulNet, self).__init__()
        self.matmul = P.MatMul()
        self.mul = P.Mul()
        self.alltoallv = NeighborExchange(send_rank_ids=[0], recv_rank_ids=[1, 2], recv_shapes=([32, 32], [32, 64]),
                                          send_shapes=([32, 32], [32, 16]), recv_type=ms.float32)
        self.weight1 = Parameter(weight1, "w1")

    def construct(self, x1, x2):
        out = self.matmul(x1, x2)
        out = self.mul(out, self.weight1)
        out = self.alltoallv((out, x1))
        return out[0]


class MatMulNet2(nn.Cell):
    def __init__(self, weight1):
        super(MatMulNet2, self).__init__()
        self.matmul = P.MatMul()
        self.mul = P.Mul()
        self.alltoallv = NeighborExchange(send_rank_ids=[0], recv_rank_ids=[1, 2], recv_shapes=([32, 32], [32, 64]),
                                          send_shapes=([32, 32],), recv_type=ms.float32)
        self.weight1 = Parameter(weight1, "w1")

    def construct(self, x1, x2):
        out = self.matmul(x1, x2)
        out = self.mul(out, self.weight1)
        out = self.alltoallv((out,))
        return out[0]


_w1 = Tensor(np.ones([32, 32]), dtype=ms.float32)
_x1 = Tensor(np.ones([32, 16]), dtype=ms.float32)
_x2 = Tensor(np.ones([16, 32]), dtype=ms.float32)


def compile_net(net):
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _executor.compile(train_net, _x1, _x2)


def test_NeighborExchange_two_inputs():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = MatMulNet(_w1)
    compile_net(net)


def test_NeighborExchange_single_input():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = MatMulNet2(_w1)
    compile_net(net)
