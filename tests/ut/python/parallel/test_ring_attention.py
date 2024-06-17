# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


def generate_inputs(B, N, S, D, input_layout, use_mqa=False, with_real_shift=False, sparse_mode=0):
    N_Q = N
    N_KV = 1 if use_mqa else N
    if input_layout == "BSH":
        H_Q = N_Q * D
        H_KV = N_KV * D
        query = Tensor(np.ones((B, S, H_Q), dtype=np.float16))
        key = Tensor(np.ones((B, S, H_KV), dtype=np.float16))
        value = Tensor(np.ones((B, S, H_KV), dtype=np.float16))
    elif input_layout == "SBH":
        H_Q = N_Q * D
        H_KV = N_KV * D
        query = Tensor(np.ones((S, B, H_Q), dtype=np.float16))
        key = Tensor(np.ones((S, B, H_KV), dtype=np.float16))
        value = Tensor(np.ones((S, B, H_KV), dtype=np.float16))
    elif input_layout == "BNSD":
        query = Tensor(np.ones((B, N_Q, S, D), dtype=np.float16))
        key = Tensor(np.ones((B, N_KV, S, D), dtype=np.float16))
        value = Tensor(np.ones((B, N_KV, S, D), dtype=np.float16))
    elif input_layout == "BSND":
        query = Tensor(np.ones((B, S, N_Q, D), dtype=np.float16))
        key = Tensor(np.ones((B, S, N_KV, D), dtype=np.float16))
        value = Tensor(np.ones((B, S, N_KV, D), dtype=np.float16))
    else:
        raise ValueError(f"input_layout is invalid.")
    real_shift = Tensor(np.ones((B, N, S, S), dtype=np.float16)) if with_real_shift else None
    # attn_mask = Tensor(np.ones((S, S), dtype=np.uint8))
    attn_mask = None
    return query, key, value, real_shift, attn_mask


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)


def compile_net(net, *inputs):
    net.set_train()
    _cell_graph_executor.compile(net, *inputs)


class Net(nn.Cell):
    def __init__(self, head_num, keep_prob=1.0, input_layout="BSH", sparse_mode=0, use_mqa=False,
                 with_real_shift=False, dp=None, mp=None, sp=1, enable_ring_attention=True, enable_ud_mask=False):
        super(Net, self).__init__()
        self.reshape = P.Reshape()
        self.drop_gen_mask = P.DropoutGenMask()
        self.keep_prob = Tensor(keep_prob, ms.float16)
        compressed_mask_mode = [2, 3, 4]
        self.head_num = head_num
        self.input_layout = input_layout
        pre_tokens = 2147483647 if sparse_mode not in compressed_mask_mode else 512
        next_tokens = 2147483647 if sparse_mode not in compressed_mask_mode else 0
        self.fa_op = FlashAttentionScore(head_num=head_num,
                                         keep_prob=keep_prob,
                                         pre_tokens=pre_tokens,
                                         next_tokens=next_tokens,
                                         input_layout=input_layout,
                                         sparse_mode=sparse_mode)
        if dp is not None and mp is not None:
            kv_head_stra = 1 if use_mqa else mp
            if input_layout == "BSH":
                stra = ((dp, sp, mp), (dp, sp, kv_head_stra), (dp, sp, kv_head_stra))
            elif input_layout == "SBH":
                stra = ((sp, dp, mp), (sp, dp, kv_head_stra), (sp, dp, kv_head_stra))
            elif input_layout == "BNSD":
                stra = ((dp, mp, sp, 1), (dp, kv_head_stra, sp, 1), (dp, kv_head_stra, sp, 1))
            elif input_layout == "BSND":
                stra = ((dp, sp, mp, 1), (dp, sp, kv_head_stra, 1), (dp, sp, kv_head_stra, 1))
            else:
                raise ValueError(f"input_layout is invalid.")
            if enable_ud_mask:
                # if using user define mask
                stra += ((sp, 1),)
            if with_real_shift:
                stra += ((dp, mp, sp, 1),)
            if keep_prob < 1.0:
                stra += ((dp, mp, sp, 1),)
            self.fa_op.shard(stra)
        self.fa_op.add_prim_attr("enable_ring_attention", enable_ring_attention)

    def construct(self, query, key, value, real_shift, attn_mask):
        if self.input_layout == "BSH":
            bsz, seq_len, _ = query.shape
        elif self.input_layout == "SBH":
            seq_len, bsz, _ = query.shape
        elif self.input_layout == "BNSD":
            bsz, _, seq_len, _ = query.shape
        elif self.input_layout == "BSND":
            bsz, seq_len, _, _ = query.shape
        else:
            raise ValueError(f"input_layout is invalid.")
        if self.keep_prob < 1.0:
            drop_mask_bits = self.reshape(self.drop_gen_mask((bsz, self.head_num, seq_len, seq_len),
                                                             self.keep_prob),
                                          (bsz, self.head_num, seq_len, 128))
        else:
            drop_mask_bits = None
        return self.fa_op(query, key, value, real_shift, drop_mask_bits, None, attn_mask, None)


@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_ring_attention_semi_auto_parallel(input_layout):
    """
    Features: test Ring Attention
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    dp = 1
    mp = 1
    sp = 4
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, S, D,
                                                               input_layout)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_ring_attention=True)
    compile_net(net, query, key, value, real_shift, attn_mask)

@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_ring_attention_user_define_mask_semi_auto_parallel(input_layout):
    """
    Features: test Ring Attention with user define mask
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """

    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    dp = 1
    mp = 1
    sp = 4
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, S, D,
                                                               input_layout)
    np.random.seed(42)
    attn_mask = Tensor(np.random.uniform(0, 2, size=(S, S)).astype(np.uint8), dtype=ms.uint8)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_ring_attention=True, enable_ud_mask=True)
    compile_net(net, query, key, value, real_shift, attn_mask)
