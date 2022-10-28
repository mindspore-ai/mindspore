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
import mindspore as ms
from mindspore import Tensor, context, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.context import _Context
from ....train_step_wrap import train_step_with_loss_warp


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class MatMulCell(nn.Cell):
    def __init__(self):
        super(MatMulCell, self).__init__()
        self.reshape = P.Reshape()
        self.matmul0 = P.MatMul(transpose_b=True)
        self.weight = Parameter(initializer("ones", [64, 128], ms.float32), name="weight")
        self.relu = P.ReLU().shard(((1, 8),))
    def construct(self, x):
        x = self.matmul0(x, self.weight)
        x = self.reshape(x, (32, 128))
        x = self.relu(x)
        return x

class DenseMutMulNet(nn.Cell):
    def __init__(self, mp_comm_recompute=True, recompute_slice_activation=False):
        super(DenseMutMulNet, self).__init__()
        self.fc1 = nn.Dense(128, 768, activation='relu')
        self.fc2 = nn.Dense(128, 768, activation='relu')
        self.fc3 = nn.Dense(128, 768, activation='relu')
        self.fc4 = nn.Dense(768, 768, activation='relu')
        self.fc1.matmul.shard(((1, 1), (8, 1)))
        self.fc2.matmul.shard(((1, 1), (8, 1)))
        self.fc3.matmul.shard(((1, 1), (8, 1)))
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.transpose = P.Transpose()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()
        self.matmul_cell = MatMulCell()
        self.fc1.recompute(mp_comm_recompute=mp_comm_recompute, recompute_slice_activation=recompute_slice_activation)
        self.fc2.recompute(mp_comm_recompute=mp_comm_recompute, recompute_slice_activation=recompute_slice_activation)
        self.fc3.recompute(mp_comm_recompute=mp_comm_recompute, recompute_slice_activation=recompute_slice_activation)
        self.matmul_cell.recompute(mp_comm_recompute=mp_comm_recompute,
                                   recompute_slice_activation=recompute_slice_activation)

    def construct(self, x):
        x = self.matmul_cell(x)
        q = self.fc1(x)
        k = self.fc2(x)
        v = self.fc3(x)
        k = self.transpose(k, (1, 0))
        c = self.relu4(self.matmul1(q, k))
        s = self.relu5(self.matmul2(c, v))
        s = self.fc4(s)
        return s

def compile_net(mp_comm_recompute, recompute_slice_activation):
    _Context().set_backend_policy("vm")
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8)
    input_ = Tensor(np.ones([64, 128]).astype(np.float32) * 0.01)
    label = Tensor(np.zeros([32, 768]).astype(np.float32))
    net = train_step_with_loss_warp(DenseMutMulNet(mp_comm_recompute=mp_comm_recompute,
                                                   recompute_slice_activation=recompute_slice_activation))
    net.set_train()
    _cell_graph_executor.compile(net, input_, label)
    _Context().set_backend_policy("ge")

def test_dmnet_train_step_mp_recompute():
    """
    Feature: test recompute interface.
    Description: test model parallel communication not recompute.
    Expectation: compile without error.
    """
    compile_net(False, False)

def test_dmnet_train_step_recompute_activation_slice():
    """
    Feature: test recompute interface.
    Description: test slicing recompute cell output.
    Expectation: compile without error.
    """
    compile_net(True, True)

def test_dmnet_train_step_mp_recompute_recompute_activation_slice():
    """
    Feature: test recompute interface.
    Description: test model parallel communication not recompute and slicing recompute cell output.
    Expectation: compile without error.
    """
    compile_net(False, True)
