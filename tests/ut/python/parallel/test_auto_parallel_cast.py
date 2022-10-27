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

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.parallel._utils import _reset_op_id as reset_op_id
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, z, w):
        predict = self.network(x, y, z, w)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, z, w):
        return grad_all(self.network)(x, y, z, w)

    # model_parallel test


def test_double_star_graph():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()
            self.matmul3 = P.MatMul()
            self.cast1 = P.Cast()
            self.cast2 = P.Cast()

        def construct(self, x, y, z, w):
            m1_result = self.matmul1(x, y)
            m2_result = self.matmul2(z, w)
            m3_result = self.matmul3(self.cast1(m2_result, mstype.float16), self.cast2(m1_result, mstype.float16))

            return m3_result

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)

    x = Tensor(np.ones([32, 8]), dtype=ms.float32)
    y = Tensor(np.ones([8, 16]), dtype=ms.float32)
    z = Tensor(np.ones([8, 16]), dtype=ms.float32)
    w = Tensor(np.ones([16, 32]), dtype=ms.float32)

    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    reset_op_id()

    net.set_train()
    _cell_graph_executor.compile(net, x, y, z, w, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    expected_strategies = {'Default/network-Net/MatMul-op2': [[1, 1], [1, 8]],
                           'Default/network-Net/MatMul-op5': [[1, 1], [1, 1]],
                           'Default/network-Net/MatMul-op0': [[1, 1], [1, 8]],
                           'Default/_VirtualDataset-op3': [[1, 1], [1, 1], [1, 1], [1, 1]]}
    assert strategies == expected_strategies
