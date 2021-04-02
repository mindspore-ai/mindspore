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

import re
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _executor
from mindspore.ops import operations as P
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._utils import _reset_op_id as reset_op_id
from tests.ut.python.ops.test_math_ops import VirtualLoss


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)


class Blockcell(nn.Cell):
    def __init__(self):
        super(Blockcell, self).__init__()
        self.bn = nn.BatchNorm1d(64, momentum=0.9)

    def construct(self, x):
        out = self.bn(x)
        return out


def get_block():
    return Blockcell()


def test_two_bn():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.block1 = get_block()
            self.block2 = get_block()
            self.relu = P.ReLU()
            self.add = P.Add()
            self.bias = Tensor(np.ones([64, 64]), dtype=ms.float32)

        def construct(self, x):
            out = self.block1(x)
            out = self.relu(out)
            out = self.add(out, self.bias)
            out = self.block2(out)
            return out

    context.set_context(save_graphs=False)
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net = NetWithLoss(Net())
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    net.set_auto_parallel()
    set_algo_parameters(elementwise_op_strategy_follow=True)
    reset_op_id()

    _executor.compile(net, x, phase='train')
    strategies = _executor._get_shard_strategy(net)
    assert len(strategies) == 4

    for (k, v) in strategies.items():
        if re.search('BatchNorm-op', k) is not None:
            assert v == [[8, 1], [1], [1], [1], [1]]
        elif re.search('TensorAdd-op', k) is not None:
            assert v == [[8, 1], [8, 1]]
        elif re.search('ReLU-op', k) is not None:
            assert v == [[8, 1]]
