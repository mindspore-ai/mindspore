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
from mindspore.common.api import _executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer

class Net(Cell):
    def __init__(self, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.gatherv2 = P.GatherV2().set_strategy(strategy1)
        self.gatherv2.add_prim_attr("manual_split", ((1, 0), (7, 1)))
        self.mul = P.Mul().set_strategy(strategy2)
        self.reshape = P.Reshape()
        self.matmul = P.MatMul().set_strategy(strategy3)
        self.matmul.add_prim_attr("forward_reduce_scatter", True)
        self.param = Parameter(initializer("ones", (8, 64), ms.float32), name="gatherv2_param")
        self.mul_weight = Parameter(initializer("ones", (2, 4, 64), ms.float32), name="mul_weight")
        self.matmul_weight = Parameter(initializer("ones", (256, 16), ms.float32), name="matmul_weight")

    def construct(self, x, b):
        out = self.gatherv2(self.param, x, 0)
        out = self.mul(out, self.mul_weight)
        out = self.reshape(out, (2, 256))
        out = self.matmul(out, self.matmul_weight)
        return out

_x = Tensor(np.ones([2, 4]), dtype=ms.int32)
_b = Tensor(np.ones([64, 8]), dtype=ms.float32)

def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    _executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()

def test_neg_data_parallel():
    context.set_context(save_graphs=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    strategy1 = ((2, 1), (1, 2))
    strategy2 = ((1, 2, 1), (1, 2, 1))
    strategy3 = ((1, 2), (2, 1))
    net = Net(strategy1, strategy2, strategy3)
    compile_net(net)
