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
from mindspore.ops.operations._inner_ops import MatmulDDS
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

context.set_context(mode=context.GRAPH_MODE)

grad_all = C.GradOperation(get_all=True)

# q: (num_heads * size_per_head // 16, bs * seq_len // 16, 16, 16)
# k: (num_heads * size_per_head // 16, bs * seq_len // 16, 16, 16)
# local_mask: (block_num * block_size // 16, bs * block_size // 16, 16, 16)
# global_mask: (bs * global_size // 16, seq_len // 16, 16, 16)
# local_prob: (bs, num_heads, block_num, block_size // 16, block_size // 16, 16, 16)
# global_prob: (bs, num_heads, block_num, global_size // 16, block_size // 16, 16, 16)
# x: (bs*seq_len, num_heads*size_per_head)
class Net(nn.Cell):
    def __init__(self, batch_size, num_heads, dp, mp, shard=True):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.size_per_head = 128
        self.seq_len = 1024
        self.block_size = 64
        self.block_num = self.seq_len // self.block_size
        self.global_size = 256
        self.embedding_size = num_heads * self.size_per_head
        self.cus_matmul = MatmulDDS(batch_size, num_heads)
        self.reduce_sum = P.ReduceSum()
        self.global_mask = Tensor(np.ones((batch_size * self.global_size // 16, self.seq_len // 16, 16, 16)))
        self.local_mask = Tensor(np.ones((self.block_num * self.block_size // 16,
                                          batch_size * self.block_size // 16, 16, 16)))
        self.dense1 = nn.Dense(self.embedding_size, self.embedding_size, has_bias=False)
        self.dense2 = nn.Dense(self.embedding_size, self.embedding_size, has_bias=False)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.add = P.Add()
        if shard:
            self.cus_matmul.shard(((mp, dp, 1, 1), (mp, dp, 1, 1), (1, dp, 1, 1), (dp, 1, 1, 1)))
            self.dense1.matmul.shard(((dp, 1), (mp, 1)))
            self.dense2.matmul.shard(((dp, 1), (mp, 1)))
            self.transpose.shard(((dp, 1, mp, 1),))


    def construct(self, x):
        q = self.dense1(x)
        k = self.dense2(x)
        q = self.transpose(self.reshape(q, (-1, 16, self.embedding_size // 16, 16)), (2, 0, 1, 3))
        k = self.transpose(self.reshape(k, (-1, 16, self.embedding_size // 16, 16)), (2, 0, 1, 3))
        local_prob, global_prob = self.cus_matmul(q, k, self.local_mask, self.global_mask)
        local_prob = self.reshape(local_prob, (self.batch_size, self.num_heads, -1))
        global_prob = self.reshape(global_prob, (self.batch_size, self.num_heads, -1))
        local_prob_reduce = self.reduce_sum(local_prob, 2)
        global_prob_reduce = self.reduce_sum(global_prob, 2)
        result = self.add(local_prob_reduce, global_prob_reduce)
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

def test_cus_matmul_dds_model_parallel_mix():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 2
    mp = 8
    compile_graph(batch_size, num_heads, dp, mp)

def test_cus_matmul_dds_model_parallel_dp():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 16
    mp = 1
    compile_graph(batch_size, num_heads, dp, mp)

def test_cus_matmul_dds_model_parallel_mp():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 1
    mp = 16
    compile_graph(batch_size, num_heads, dp, mp)

def test_cus_matmul_dds_model_parallel_mix_auto():
    set_algo_parameters(fully_use_devices=False)
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 2
    mp = 8
    compile_graph(batch_size, num_heads, dp, mp, auto=True)

def test_cus_matmul_dds_model_parallel_dp_auto():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 16
    mp = 1
    compile_graph(batch_size, num_heads, dp, mp, auto=True)

def test_cus_matmul_dds_model_parallel_mp_auto():
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 1
    mp = 16
    compile_graph(batch_size, num_heads, dp, mp, auto=True)

def test_cus_matmul_dds_model_parallel_auto():
    set_algo_parameters(fully_use_devices=False)
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 1
    mp = 16
    compile_graph(batch_size, num_heads, dp, mp, auto=True, shard=False)

def test_cus_matmul_dds_repeat_cal_auto():
    set_algo_parameters(fully_use_devices=False)
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 1
    mp = 2
    compile_graph(batch_size, num_heads, dp, mp, auto=True, shard=False)

def test_cus_matmul_dds_repeat1_cal_auto():
    set_algo_parameters(fully_use_devices=False)
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    batch_size = 128
    num_heads = 32
    dp = 2
    mp = 1
    compile_graph(batch_size, num_heads, dp, mp, auto=True, shard=False)
