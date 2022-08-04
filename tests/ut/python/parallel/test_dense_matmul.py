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
from mindspore import Tensor, context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P
from ....train_step_wrap import train_step_with_loss_warp


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class DenseMutMulNet(nn.Cell):
    def __init__(self):
        super(DenseMutMulNet, self).__init__()
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


def test_dmnet_train_step():
    context.reset_auto_parallel_context()
    input_ = Tensor(np.ones([32, 128]).astype(np.float32) * 0.01)
    label = Tensor(np.zeros([32, 768]).astype(np.float32))
    net = DenseMutMulNet()
    net = train_step_with_loss_warp(DenseMutMulNet())
    net.set_train()
    _cell_graph_executor.compile(net, input_, label)
