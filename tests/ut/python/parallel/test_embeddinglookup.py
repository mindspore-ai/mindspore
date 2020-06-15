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
from mindspore import Tensor
from mindspore.common.api import _executor
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)

class Net(nn.Cell):
    def __init__(self, shape, offset, reduce_scatter_flag, split_num):
        super().__init__()
        self.index = Tensor(np.ones(shape), dtype=ms.int32)
        self.offset = offset
        self.reduce_scatter_flag = reduce_scatter_flag
        self.split_num = split_num
        self.elu = P.EmbeddingLookup()
        self.mm = P.BatchMatMul()

    def construct(self, x, y):
        out = self.elu(x, self.index, self.offset, self.reduce_scatter_flag, self.split_num)
        out = self.mm(out, y)
        return out


def test_embeddinglookup_reducescatter_false():
    shape = [8, 8]
    offset = 8
    reduce_scatter_flag = False
    split_num = 1
    net = NetWithLoss(Net(shape, offset, reduce_scatter_flag, split_num))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([8, 32, 8]), dtype=ms.float32)
    _executor.compile(net, x, y)


def test_embeddinglookup_reducescatter_true():
    shape = [64, 8]
    offset = 8
    reduce_scatter_flag = True
    split_num = 8
    net = NetWithLoss(Net(shape, offset, reduce_scatter_flag, split_num))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([8, 32, 8]), dtype=ms.float32)
    _executor.compile(net, x, y)
