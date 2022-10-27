# Copyright 2021 Huawei Technologies Co., Ltd
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

import re
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)

class DenseMutMulNet(nn.Cell):
    def __init__(self):
        super(DenseMutMulNet, self).__init__()
        self.fc1 = nn.Dense(128, 768)
        self.fc2 = nn.Dense(128, 768)
        self.fc3 = nn.Dense(128, 768)
        self.fc4 = nn.Dense(768, 768, has_bias=False)
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.transpose = P.Transpose()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()
        self.fc4.matmul.shard(((1, 1), (8, 1)))

    def construct(self, x):
        q = self.fc1(x)
        k = self.fc2(x)
        v = self.fc3(x)
        k = self.transpose(k, (1, 0))
        c = self.relu4(self.matmul1(q, k))
        s = self.relu5(self.matmul2(c, v))
        s = self.fc4(s)
        return s

class MulNegTwoOutputNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul().shard(((2, 4), (2, 4)))
        self.neg = P.Neg().shard(((2, 4),))
        self.mul_weight = Parameter(Tensor(np.ones([32, 128]), dtype=ms.float32), name="weight")

    def construct(self, x):
        out1 = self.mul(x, self.mul_weight)
        out2 = self.neg(out1)
        return out1, out2

class ReshapeMatMulNet(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.reshape = P.Reshape()
        self.matmul = P.MatMul().shard(strategy2)
        self.matmul_weight = Parameter(Tensor(np.ones([28, 64]), dtype=ms.float32), name="weight")
    # x (64, 4, 7)
    def construct(self, x):
        out = self.reshape(x, (64, 28))
        out = self.matmul(out, self.matmul_weight)
        return out

class MatMulReshapeNet(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.reshape = P.Reshape()
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul_weight = Parameter(Tensor(np.ones([28, 64]), dtype=ms.float32), name="weight")
    # x (128, 28)
    def construct(self, x):
        out = self.matmul(x, self.matmul_weight)
        out = self.reshape(out, (64, -1))
        return out

class ReshapeMulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reshape = P.Reshape()
        self.mul = P.Mul().shard(((1, 2, 4), (2, 4)))
        self.mul_weight = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="weight")

    def construct(self, x):
        weight = self.reshape(self.mul_weight, (1, 128, 96))
        out = self.mul(weight, self.mul_weight)
        return out

class ParallelMulNet(nn.Cell):
    def __init__(self, dense_in_channel=2048, dense_out_channel=250):
        super().__init__()
        weight_np = np.full((dense_out_channel, dense_in_channel), 0.01, dtype=np.float32)
        bias_np = np.full((dense_out_channel,), 0.01, dtype=np.float32)
        self.flat = nn.Flatten()
        self.dense = nn.Dense(in_channels=dense_in_channel,
                              out_channels=dense_out_channel,
                              weight_init=Tensor(weight_np),
                              bias_init=Tensor(bias_np),
                              has_bias=True)
        self.mul = P.Mul()
    def construct(self, inputs):
        x = self.flat(inputs)
        x = self.dense(x)
        x = self.mul(x, x)
        return x

def compile_graph(x, net):
    net.set_train(False)
    _cell_graph_executor.compile(net, x)
    strategies = _cell_graph_executor._get_shard_strategy(net)
    return strategies

def compile_graph_two_input(x, y, net):
    net.set_train(False)
    _cell_graph_executor.compile(net, x, y)
    strategies = _cell_graph_executor._get_shard_strategy(net)
    return strategies


def test_dense_relu_semi_auto():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel",
                                      dataset_strategy="data_parallel")
    net = DenseMutMulNet()
    x = Tensor(np.ones([32 // 8, 128]).astype(np.float32) * 0.01)
    strategies = compile_graph(x, net)
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            assert v[0][0] == 8

def test_dense_relu_semi_auto_full_batch():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel",
                                      dataset_strategy="full_batch")
    net = DenseMutMulNet()
    x = Tensor(np.ones([32, 128]).astype(np.float32) * 0.01)
    strategies = compile_graph(x, net)
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            assert v[0][0] == 1

def test_dense_relu_auto():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="auto_parallel",
                                      dataset_strategy="data_parallel")
    net = DenseMutMulNet()
    x = Tensor(np.ones([32 // 8, 128]).astype(np.float32) * 0.01)
    strategies = compile_graph(x, net)
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            assert v[0][0] == 8

def test_dense_relu_auto_full_batch():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="auto_parallel",
                                      dataset_strategy="full_batch")
    net = DenseMutMulNet()
    x = Tensor(np.ones([32, 128]).astype(np.float32) * 0.01)
    strategies = compile_graph(x, net)
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            assert v[0][0] == 1

def test_mul_neg_two_output_semi_auto():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel",
                                      dataset_strategy="data_parallel")
    net = MulNegTwoOutputNet()
    x = Tensor(np.ones([32 // 8, 128]).astype(np.float32) * 0.01)
    strategies = compile_graph(x, net)
    count = 0
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            count += 1
            assert v[0][0] == 8
    assert count == 2

def test_mul_neg_two_output_semi_auto_full_batch():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel",
                                      dataset_strategy="full_batch")
    net = MulNegTwoOutputNet()
    x = Tensor(np.ones([32, 128]).astype(np.float32) * 0.01)
    strategies = compile_graph(x, net)
    count = 0
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            count += 1
            assert v[0][0] == 1
    assert count == 2

def test_mul_neg_two_output_auto():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="auto_parallel",
                                      dataset_strategy="data_parallel")
    net = MulNegTwoOutputNet()
    x = Tensor(np.ones([32 // 8, 128]).astype(np.float32) * 0.01)
    strategies = compile_graph(x, net)
    count = 0
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            count += 1
            assert v[0][0] == 8
    assert count == 2

def test_mul_neg_two_output_full_batch():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="auto_parallel",
                                      dataset_strategy="full_batch")
    net = MulNegTwoOutputNet()
    x = Tensor(np.ones([32, 128]).astype(np.float32) * 0.01)
    strategies = compile_graph(x, net)
    count = 0
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            count += 1
            assert v[0][0] == 1
    assert count == 2

def test_reshape_matmul_semi_auto():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel",
                                      dataset_strategy="data_parallel")
    strategy1 = None
    strategy2 = ((1, 1), (1, 8))
    net = ReshapeMatMulNet(strategy1, strategy2)
    x = Tensor(np.ones([64 // 8, 4, 7]), ms.float32)
    strategies = compile_graph(x, net)
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            assert v[0][0] == 8

def test_reshape_matmul_auto():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="auto_parallel",
                                      dataset_strategy="data_parallel")
    strategy1 = None
    strategy2 = ((1, 1), (1, 8))
    net = ReshapeMatMulNet(strategy1, strategy2)
    x = Tensor(np.ones([64 // 8, 4, 7]), ms.float32)
    strategies = compile_graph(x, net)
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            assert v[0][0] == 8

def test_matmul_reshape_semi_auto():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel",
                                      dataset_strategy="data_parallel")
    strategy2 = None
    strategy1 = ((1, 1), (1, 8))
    net = MatMulReshapeNet(strategy1, strategy2)
    x = Tensor(np.ones([128 // 8, 28]), ms.float32)
    strategies = compile_graph(x, net)
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            assert v[0][0] == 8

def test_matmul_reshape_auto():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="auto_parallel",
                                      dataset_strategy="data_parallel")
    strategy2 = None
    strategy1 = ((1, 1), (1, 8))
    net = MatMulReshapeNet(strategy1, strategy2)
    x = Tensor(np.ones([128 // 8, 28]), ms.float32)
    strategies = compile_graph(x, net)
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            assert v[0][0] == 8

def test_reshape_mul_semi_auto():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel",
                                      dataset_strategy="full_batch")
    net = ReshapeMulNet()
    x = Tensor(np.ones([64, 4]), ms.float32)
    strategies = compile_graph(x, net)
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            assert v[0][0] == 1

def test_reshape_mul_auto():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="auto_parallel",
                                      dataset_strategy="full_batch")
    net = ReshapeMulNet()
    x = Tensor(np.ones([64, 4]), ms.float32)
    strategies = compile_graph(x, net)
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            assert v[0][0] == 1

def test_scalar_output_semi_auto():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel",
                                      dataset_strategy="data_parallel")
    net = ParallelMulNet()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
    eval_net = nn.WithEvalCell(net, loss_fn)
    x = Tensor(np.ones([4096 // 8, 1, 2, 1024]).astype(np.float32)*0.01)
    label = Tensor(np.ones([4096 // 8, 250]).astype(np.float32)*0.01)
    strategies = compile_graph_two_input(x, label, eval_net)
    count = 0
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            assert v[0][0] == 8
            count += 1
    assert count == 2

def test_scalar_output_auto():
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="auto_parallel",
                                      dataset_strategy="data_parallel")
    net = ParallelMulNet()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
    eval_net = nn.WithEvalCell(net, loss_fn)
    x = Tensor(np.ones([4096 // 8, 1, 2, 1024]).astype(np.float32)*0.01)
    label = Tensor(np.ones([4096 // 8, 250]).astype(np.float32)*0.01)
    strategies = compile_graph_two_input(x, label, eval_net)
    count = 0
    for (k, v) in strategies.items():
        if re.search('VirtualOutput-op', k) is not None:
            assert v[0][0] == 8
            count += 1
    assert count == 2
