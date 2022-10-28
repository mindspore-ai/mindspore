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
    def __init__(self, wi, wo, stra1=None, stra2=None, stra3=None, stra4=None,
                 stra5=None, stra6=None):
        super(Net, self).__init__()
        self.relu = P.ReLU().shard(stra1)
        self.transpose = P.Transpose().shard(stra2)
        self.wi = Parameter(wi, "wi")
        self.batch_mm = P.BatchMatMul().shard(stra3)
        self.wo = Parameter(wo, "wo")
        self.batch_mm2 = P.BatchMatMul().shard(stra4)
        self.transpose2 = P.Transpose().shard(stra5)
        self.relu2 = P.ReLU().shard(stra6)
        self.reshape = P.Reshape()
        self.reshape2 = P.Reshape()

    def construct(self, x):
        output = self.relu(x)
        trans_out = self.transpose(output, (2, 0, 3, 1))
        output = self.reshape(trans_out,
                              (trans_out.shape[0], trans_out.shape[1]*trans_out.shape[2], trans_out.shape[3]))
        output = self.batch_mm(output, self.wi)
        output = self.batch_mm2(output, self.wo)
        output = self.reshape2(output, trans_out.shape)
        output = self.transpose2(output, (1, 3, 0, 2))
        output = self.relu2(output)
        return output

_x = Tensor(np.ones([32, 16, 48, 128]), dtype=ms.float32)
_wi = Tensor(np.ones([48, 16, 64]), dtype=ms.float32)
_wo = Tensor(np.ones([48, 64, 16]), dtype=ms.float32)


def compile_net(net):
    context.set_context(mode=context.GRAPH_MODE)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x)
    context.reset_auto_parallel_context()


def test_batchmm():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, enable_alltoall=True,
                                      global_rank=0)
    stra1 = ((8, 1, 1, 1),)
    stra2 = ((8, 1, 1, 1),)
    stra3 = ((8, 1, 1), (8, 1, 1))
    stra4 = ((8, 1, 1), (8, 1, 1))
    stra5 = ((8, 1, 1, 1),)
    stra6 = ((8, 1, 1, 1),)
    net = Net(_wi, _wo, stra1=stra1, stra2=stra2, stra3=stra3, stra4=stra4, stra5=stra5, stra6=stra6)
    compile_net(net)


def test_batchmm2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", enable_alltoall=True,
                                      device_num=32, global_rank=0)
    stra1 = ((4, 1, 1, 1),)
    stra2 = ((4, 1, 1, 1),)
    stra3 = ((4, 1, 1), (4, 1, 8))
    stra4 = ((4, 1, 8), (4, 8, 1))
    stra5 = ((4, 1, 1, 1),)
    stra6 = ((4, 1, 1, 1),)
    net = Net(_wi, _wo, stra1=stra1, stra2=stra2, stra3=stra3, stra4=stra4, stra5=stra5, stra6=stra6)
    compile_net(net)
