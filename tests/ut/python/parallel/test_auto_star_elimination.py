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
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.api import _cell_graph_executor
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


class CustomMatMul(nn.Cell):
    def __init__(self, transpose_a=False, transpose_b=False):
        super(CustomMatMul, self).__init__()
        self.fc = P.MatMul(transpose_a=transpose_a, transpose_b=transpose_b)

    def construct(self, x1, x2):
        out = self.fc(x1, x2)
        return out


class MarginCE(LossBase):
    def __init__(self):
        super(MarginCE, self).__init__()
        self.fc = CustomMatMul(transpose_b=True)
        self.fc1 = CustomMatMul(transpose_b=True)
        self.fc2 = CustomMatMul(transpose_b=True)
        self.fc3 = CustomMatMul(transpose_b=True)
        self.fc4 = CustomMatMul(transpose_b=True)
        self.param = Parameter(Tensor(np.ones([512, 512]), dtype=mstype.float32), name="param", requires_grad=False)
        self.param2 = Parameter(Tensor(np.ones([512, 512]), dtype=mstype.float32), name="param", requires_grad=False)

    def construct(self, feature, label):
        fc_out = self.fc(feature, label)

        fc1_out = self.fc1(self.param2, self.param)
        fc2_out = self.fc2(fc1_out, fc_out)
        fc3_out = self.fc3(fc1_out, fc_out)
        fc4_out = self.fc4(fc2_out, fc3_out)
        return fc4_out


def test_marin_loss():
    context.set_auto_parallel_context(device_num=4, global_rank=0)

    x = Tensor(np.ones([512, 512]), dtype=ms.float32)
    y = Tensor(np.ones([512, 512]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(MarginCE()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_train()
    _cell_graph_executor.compile(net, x, y)
