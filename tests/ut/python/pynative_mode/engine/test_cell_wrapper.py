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
""" test_cell_wrapper """
import numpy as np

import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import WithLossCell
from ...ut_filter import non_graph_engine


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([10]).astype(np.float32)), name="bias")
        self.matmul = P.MatMul()
        self.biasAdd = P.BiasAdd()
        self.softmax = P.Softmax()

    def construct(self, x):
        x = self.biasAdd(self.matmul(x, self.weight), self.bias)
        x = self.softmax(x)
        return x


@non_graph_engine
def test_loss_wrapper():
    """ test_loss_wrapper """
    input_data = Tensor(np.ones([1, 64]).astype(np.float32))
    input_label = Tensor(np.ones([1, 10]).astype(np.float32))
    net = Net()
    loss = SoftmaxCrossEntropyWithLogits()
    cost = WithLossCell(net, loss)
    cost(input_data, input_label)
