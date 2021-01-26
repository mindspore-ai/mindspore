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

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)


class CustomDense(nn.Cell):
    def __init__(self, row, column):
        super(CustomDense, self).__init__()
        self.weight = Parameter(Tensor(np.ones([row, column]).astype(np.float32) * 0.01), "w", requires_grad=True)
        self.bias = Parameter(Tensor(np.zeros([row, column]).astype(np.float32)), "b", requires_grad=True)
        self.matmul1 = P.MatMul()
        self.add2 = P.Add()
        self.activation3 = nn.ReLU()

    def construct(self, x):
        mat_output = self.matmul1(x, self.weight)
        add_output = self.add2(mat_output, self.bias)
        output = self.activation3(add_output)

        return output


class DenseMutMulNet(nn.Cell):
    def __init__(self):
        super(DenseMutMulNet, self).__init__()
        self.fc1 = CustomDense(4096, 4096)
        self.fc2 = CustomDense(4096, 4096)
        self.fc3 = CustomDense(4096, 4096)
        self.fc4 = CustomDense(4096, 4096)
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.transpose = P.Transpose()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()

    def construct(self, x):
        q = self.fc1(x)
        k = self.fc2(x)
        v = self.fc3(x)
        k = self.transpose(k, (1, 0))
        c = self.relu4(self.matmul1(q, k))
        s = self.relu5(self.matmul2(c, v))
        s = self.fc4(s)
        return s


class MultiTransformer(nn.Cell):
    def __init__(self, layer_nums=1):
        super(MultiTransformer, self).__init__()
        self.layer = self._make_layer(layer_nums)

    def _make_layer(self, layer_num):
        layers = []
        for _ in range(0, layer_num):
            layers.append(DenseMutMulNet())

        return nn.SequentialCell(layers)

    def construct(self, x):
        out = self.layer(x)
        return out


def test_dmnet_train_step():
    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)

    input_ = Tensor(np.ones([4096, 4096]).astype(np.float32) * 0.01)
    net = GradWrap(NetWithLoss(MultiTransformer()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    net.set_train()
    _executor.compile(net, input_)
