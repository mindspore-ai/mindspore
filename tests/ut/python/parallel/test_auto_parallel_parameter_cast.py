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
from mindspore import context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from tests.ut.python.ops.test_math_ops import VirtualLoss
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore import Tensor, Parameter
from mindspore.parallel._utils import _reset_op_id as reset_op_id
from mindspore.parallel import set_algo_parameters


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
        return C.grad_all(self.network)(x, y, z, w)

    # model_parallel test


def test_common_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()
            self.matmul3 = P.MatMul()
            self.weight1 = Parameter(Tensor(np.ones([64, 64]).astype(np.float16) * 0.01), "w", requires_grad=True)
            self.cast1 = P.Cast()
            self.cast2 = P.Cast()

        def construct(self, x, y, z, w):
            m1_result = self.matmul1(x, self.cast1(self.weight1, mstype.float32))
            m2_result = self.matmul2(z, self.cast2(self.weight1, mstype.float32))
            m3_result = self.matmul3(m2_result, m1_result)

            return m3_result

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)

    set_algo_parameters(elementwise_op_strategy_follow=True)
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64]), dtype=ms.float32)
    z = Tensor(np.ones([64, 64]), dtype=ms.float32)
    w = Tensor(np.ones([64, 64]), dtype=ms.float32)

    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    reset_op_id()

    _executor.compile(net, x, y, z, w, phase='train')
    strategies = _executor._get_strategy(net)
    expected_strategies = {'Default/network-Net/MatMul-op1': [[8, 1], [1, 1]],
                           'Default/network-Net/MatMul-op3': [[8, 1], [1, 1]],
                           'Default/network-Net/Cast-op2': [[1, 1]],
                           'Default/network-Net/MatMul-op0': [[8, 1], [1, 1]],
                           'Default/network-Net/Cast-op4': [[1, 1]]}
    assert strategies == expected_strategies
