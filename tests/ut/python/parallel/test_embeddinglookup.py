# Copyright 2019 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore.common.api import _executor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore import Tensor, context
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)

class Net(nn.Cell):
    def __init__(self, shape, offset, strategy1=None, strategy2=None, target="Device"):
        super().__init__()
        self.index = Tensor(np.ones(shape), dtype=ms.int32)
        self.offset = offset
        self.elu = P.EmbeddingLookup().shard(strategy1).add_prim_attr("primitive_target", target)
        self.mm = P.BatchMatMul().shard(strategy2)

    def construct(self, x, y):
        out = self.elu(x, self.index, self.offset)
        out = self.mm(out, y)
        return out


def test_embeddinglookup_reducescatter_false():
    shape = [8, 8]
    offset = 8
    net = NetWithLoss(Net(shape, offset))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([8, 32, 8]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_embeddinglookup_reducescatter_true():
    shape = [8, 8]
    offset = 8
    net = NetWithLoss(Net(shape, offset))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([8, 32, 8]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_embeddinglookup_reducescatter_false_grad():
    shape = [8, 8]
    offset = 8
    net = GradWrap(NetWithLoss(Net(shape, offset)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([8, 32, 8]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_embeddinglookup_reducescatter_true_grad():
    context.set_context(save_graphs=False)
    shape = [8, 8]
    offset = 8
    net = GradWrap(NetWithLoss(Net(shape, offset)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([8, 32, 8]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_embeddinglookup_semi_auto1():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    shape = [64, 32]
    offset = 0
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((4, 1, 2), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(shape, offset, strategy1, strategy2, "CPU")))

    net.set_auto_parallel()
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_embeddinglookup_semi_auto2():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    shape = [64, 32]
    offset = 0
    strategy1 = ((1, 8), (1, 1))
    strategy2 = ((4, 1, 2), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(shape, offset, strategy1, strategy2, "CPU")))

    net.set_auto_parallel()
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)
