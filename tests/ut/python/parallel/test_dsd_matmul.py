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

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.parallel import set_algo_parameters
from mindspore.ops.operations._inner_ops import DSDMatmul
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

context.set_context(mode=context.GRAPH_MODE)

grad_all = C.GradOperation(get_all=True)


#  input_w1, the shape is (batch_size, head, block_num, head_size // 16, block_size//16, 16, 16)
#  input_w1 cum_shape = batch_size * seq_len * embedding_size * (block_size // size_per_head)
#  = batch_size * seq_len * (embedding_size // 2)
#  input_w2, the shape is (batch_size, head, block_num, global_size // 16, head_size // 16, 16, 16)
#  input_w2 cum_shape = batch_size * seq_len * embedding_size * (global_size // size_per_head)
#  = batch_size * seq_len * embedding_size * 2
#  input_v, the shape is (batch_size * seq_len // 16, head * v_embedding // 16, 16, 16)
#  block_num = seq_len // block_size, block_size = 64, head * v_embedding = embedding_size, always.
#  output shape is (batch_size, head, v_embedding // 16, seq_len//16, 16, 16)


class Net(nn.Cell):
    def __init__(self, batch_size, num_heads, dp, mp, shard=True):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = 1024
        self.block_size = 64
        self.head_size = self.block_size
        self.block_num = self.seq_len // self.block_size
        self.global_size = 256
        self.v_embedding = 128
        self.embedding_size = num_heads * self.v_embedding
        self.dsd_matmul = DSDMatmul()
        self.reduce_sum = P.ReduceSum()
        self.dense1 = nn.Dense(self.embedding_size, self.embedding_size // 2, has_bias=False)
        self.dense2 = nn.Dense(self.embedding_size, self.embedding_size * 2, has_bias=False)
        self.dense3 = nn.Dense(self.embedding_size, self.embedding_size, has_bias=False)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.transpose1 = P.Transpose()
        self.add = P.Add()
        if shard:
            self.dsd_matmul.shard(((dp, mp, 1, 1, 1, 1, 1), (dp, mp, 1, 1, 1, 1, 1), (dp, mp, 1, 1)))
            self.dense1.matmul.shard(((dp, 1), (mp, 1)))
            self.dense2.matmul.shard(((dp, 1), (mp, 1)))
            self.dense2.matmul.shard(((dp, 1), (mp, 1)))
            self.transpose.shard(((dp, 1, mp, 1),))
            self.transpose1.shard(((dp, mp, 1, 1, 1, 1),))

    def construct(self, x):
        # x (batch_size * seq_len, embedding_size)
        q = self.dense1(x)
        # q (batch_size * seq_len, (embedding_size // 2))
        # (batch_size, head, block_num, head_size // 16, block_size//16, 16, 16)
        k = self.dense2(x)
        # k (batch_size * seq_len, (embedding_size * 2))
        # (batch_size, head, block_num, global_size // 16, head_size // 16, 16, 16)
        v = self.dense3(x)
        # v (batch_size * seq_len, embedding_size)
        q = self.reshape(q, (self.batch_size, self.num_heads, self.block_num, self.head_size // 16,
                             self.block_size // 16, 16, 16))
        k = self.reshape(k, (self.batch_size, self.num_heads, self.block_num, self.global_size // 16,
                             self.head_size // 16, 16, 16))
        v = self.transpose(self.reshape(v, (-1, 16, self.embedding_size // 16, 16)), (0, 2, 3, 1))
        dsd = self.dsd_matmul(q, k, v)
        # dsd (batch_size, head, v_embedding // 16, seq_len//16, 16, 16)
        dsd = self.transpose1(dsd, (0, 1, 3, 4, 2, 5))
        # dsd (batch_size, head, seq_len//16, 16, v_embedding_size//16, 16)
        dsd = self.reshape(dsd, (-1, self.seq_len, self.v_embedding * self.num_heads))
        result = self.reduce_sum(dsd, 2)
        return result


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.network = network
        self.loss = VirtualLoss()

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)


def compile_graph(batch_size, num_heads, dp, mp, auto=False, shard=True):
    if auto:
        context.set_auto_parallel_context(parallel_mode="auto_parallel")
    else:
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    x = Tensor(np.ones((batch_size * 1024, num_heads * 128)), ms.float32)
    net = GradWrap(NetWithLoss(Net(batch_size, num_heads, dp, mp, shard=shard)))
    net.set_train()
    _cell_graph_executor.compile(net, x)

def test_dsd_matmul_model_parallel_mix():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 2
    mp = 8
    compile_graph(batch_size, num_heads, dp, mp)

def test_dsd_matmul_model_parallel_dp():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 16
    mp = 1
    compile_graph(batch_size, num_heads, dp, mp)

def test_dsd_matmul_model_parallel_mp():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 1
    mp = 16
    compile_graph(batch_size, num_heads, dp, mp)

def test_dsd_matmul_model_parallel_mix_auto():
    set_algo_parameters(fully_use_devices=False)
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 2
    mp = 8
    compile_graph(batch_size, num_heads, dp, mp, auto=True)

def test_dsd_matmul_model_parallel_dp_auto():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 16
    mp = 1
    compile_graph(batch_size, num_heads, dp, mp, auto=True)

def test_dsd_matmul_model_parallel_mp_auto():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 1
    mp = 16
    compile_graph(batch_size, num_heads, dp, mp, auto=True)

def test_dsd_matmul_model_parallel_auto():
    set_algo_parameters(fully_use_devices=False)
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 1
    mp = 16
    compile_graph(batch_size, num_heads, dp, mp, auto=True, shard=False)
