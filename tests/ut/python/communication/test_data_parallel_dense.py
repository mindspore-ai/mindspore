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

"""
@File   : test_data_parallel_dense.py
@Desc   : test data parallel dense
"""
import numpy as np
import mindspore.nn as nn
from mindspore.common.api import _executor
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn import Momentum
from mindspore.train.parallel_utils import ParallelMode
import mindspore.context as context


class DenseMMNet(nn.Cell):
    """DenseMMNet definition"""
    def __init__(self):
        super(DenseMMNet, self).__init__()
        self.fc1 = nn.Dense(128, 768, activation='relu')
        self.fc2 = nn.Dense(128, 768, activation='relu')
        self.fc3 = nn.Dense(128, 768, activation='relu')
        self.fc4 = nn.Dense(768, 768, activation='relu')
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


def test_data_parallel_dense():
    """test_data_parallel_dense"""
    context.set_context(mode=context.GRAPH_MODE)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, mirror_mean=True, device_num=8)
    inp = Tensor(np.ones([32, 128]).astype(np.float32) * 0.01)
    label = Tensor(np.zeros([32, 768]).astype(np.float32))
    net = DenseMMNet()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    net = WithLossCell(net, loss_fn)
    net = TrainOneStepCell(net, optimizer)

    _executor.compile(net, inp, label)
    context.reset_auto_parallel_context()
