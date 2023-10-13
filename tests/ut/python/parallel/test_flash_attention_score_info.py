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
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.context import set_auto_parallel_context
from mindspore.ops import composite as C
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


def generate_inputs(B, N, S, D):
    H = N * D
    query = Tensor(np.ones((B, S, H), dtype=np.float16))
    key = Tensor(np.ones((B, S, H), dtype=np.float16))
    value = Tensor(np.ones((B, S, H), dtype=np.float16))
    attn_mask = Tensor(np.ones((B, 1, S, S), dtype=np.uint8))
    return query, key, value, attn_mask


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

    def construct(self, *inputs):
        return grad_all(self.network)(*inputs)


def compile_net(net, *inputs):
    net.set_train()
    _cell_graph_executor.compile(net, *inputs)


class Net(nn.Cell):
    def __init__(self, head_num, keep_prob=0.9, dp=None, mp=None):
        super(Net, self).__init__()
        self.reshape = P.Reshape()
        self.drop_gen_mask = P.DropoutGenMask()
        self.keep_prob = Tensor(keep_prob, ms.float16)
        self.head_num = head_num
        self.fa_op = FlashAttentionScore(head_num=head_num, keep_prob=keep_prob)
        if dp is not None and mp is not None:
            stra = ((dp, 1, mp), (dp, 1, mp), (dp, 1, mp), (dp, 1, 1, 1))
            if keep_prob < 1.0:
                stra += ((dp, mp, 1, 1),)
            self.fa_op.shard(stra)

    def construct(self, query, key, value, attn_mask):
        bsz, seq_len, _ = query.shape
        if self.keep_prob < 1.0:
            drop_mask_bits = self.reshape(self.drop_gen_mask((bsz, self.head_num, seq_len, seq_len),
                                                             self.keep_prob),
                                          (bsz, self.head_num, seq_len, -1))
        else:
            drop_mask_bits = None
        return self.fa_op(query, key, value, attn_mask, drop_mask_bits, None, None)


@pytest.mark.parametrize('keep_prob', [0.9, 1.0])
def test_self_attention_standalone(keep_prob):
    """
    Features: test FlashAttentionScoreInfo
    Description: StandAlone
    Expectation: compile success
    """
    context.reset_auto_parallel_context()
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    B, N, S, D = 8, 16, 1024, 128
    inputs = generate_inputs(B, N, S, D)
    net = Net(N, keep_prob)
    compile_net(net, *inputs)


@pytest.mark.parametrize('keep_prob', [0.9, 1.0])
def test_flash_attention_semi_auto_parallel(keep_prob):
    """
    Features: test FlashAttentionScoreInfo
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    dp = 2
    mp = 4
    B, N, S, D = 8, 16, 1024, 128
    inputs = generate_inputs(B, N, S, D)
    net = Net(N, keep_prob, dp, mp)
    compile_net(net, *inputs)


@pytest.mark.parametrize('keep_prob', [0.9, 1.0])
def test_flash_attention_dp(keep_prob):
    """
    Features: test FlashAttentionScore under semi_auto_parallel
    Description: semi_auto_parallel without strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    B, N, S, D = 8, 16, 1024, 128
    inputs = generate_inputs(B, N, S, D)
    net = Net(N, keep_prob)
    compile_net(net, *inputs)


@pytest.mark.parametrize('keep_prob', [1.0])
def test_flash_attention_auto_parallel(keep_prob):
    """
    Features: test FlashAttentionScoreInfo
    Description: auto_parallel
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    B, N, S, D = 8, 16, 1024, 128
    inputs = generate_inputs(B, N, S, D)
    net = Net(N, keep_prob)
    compile_net(net, *inputs)
