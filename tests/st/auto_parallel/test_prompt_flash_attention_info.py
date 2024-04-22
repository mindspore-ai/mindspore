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
# ============================================================================
# pylint: disable=E1121
import numpy as np
import pytest

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations.nn_ops as P
from mindspore import Tensor, context
from mindspore.communication.management import init
from mindspore.ops import operations as OPS

context.set_context(mode=context.GRAPH_MODE)
init()


def generate_inputs(dims, optinal_inputs, input_layout='BSH', sparse_mode=0):
    B, N, S, D = dims
    has_atten_mask, has_actual_seq_lengths, has_actual_seq_lengths_kv, has_pse_shift, has_deq_scale1, \
        has_quant_scale1, has_deq_scale2, has_quant_scale2, has_quant_offset2 = optinal_inputs
    attn_mask = None
    pse_shift = None
    actual_seq_lengths = Tensor(np.ones((B,), dtype=np.int64)) if has_actual_seq_lengths else None
    actual_seq_lengths_kv = Tensor(np.ones((B,), dtype=np.int64)) if has_actual_seq_lengths_kv else None
    deq_scale1 = Tensor(1, dtype=mindspore.uint64) if has_deq_scale1 else None
    quant_scale1 = Tensor(1, dtype=mindspore.uint64) if has_quant_scale1 else None
    deq_scale2 = Tensor(1, dtype=mindspore.uint64) if has_deq_scale2 else None
    quant_scale2 = Tensor(1, dtype=mindspore.float32) if has_quant_scale2 else None
    quant_offset2 = Tensor(1, dtype=mindspore.float32) if has_quant_offset2 else None

    ret_inputs = None
    if input_layout == 'BSH':
        H = N * D
        query = Tensor(np.random.rand(B, S, H).astype(np.float16))
        key = Tensor(np.random.rand(B, S, H).astype(np.float16))
        value = Tensor(np.random.rand(B, S, H).astype(np.float16))
        if has_atten_mask:
            if sparse_mode == 0:
                expected_attn_mask = np.ones((S, S))
                expected_attn_mask[np.tril_indices(S)] = 0
                expected_attn_mask = np.concatenate([expected_attn_mask.reshape(1, 1, S, S)] * B)
            else:
                expected_attn_mask = np.ones((2048, 2048))
                expected_attn_mask[np.tril_indices(S)] = 0
                expected_attn_mask = expected_attn_mask.reshape(1, 1, 2048, 2048)
            attn_mask = Tensor(expected_attn_mask, dtype=mindspore.bool_)
        if has_pse_shift:
            pse_shift = Tensor(np.zeros((B, 1, S, S)), dtype=mindspore.float16)
        ret_inputs = (query, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift,
                      deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2)
    elif input_layout == 'BNSD':
        query = Tensor(np.random.rand(B, N, S, D).astype(np.float16))
        key = Tensor(np.random.rand(B, N, S, D).astype(np.float16))
        value = Tensor(np.random.rand(B, N, S, D).astype(np.float16))
        if has_atten_mask:
            if sparse_mode == 0:
                expected_attn_mask = np.ones((S, S))
                expected_attn_mask[np.tril_indices(S)] = 0
                expected_attn_mask = np.concatenate([expected_attn_mask.reshape(1, 1, S, S)] * B)
            else:
                expected_attn_mask = np.ones((2048, 2048))
                expected_attn_mask[np.tril_indices(S)] = 0
                expected_attn_mask = expected_attn_mask.reshape(1, 1, 2048, 2048)
            attn_mask = Tensor(expected_attn_mask, dtype=mindspore.bool_)
        if has_pse_shift:
            pse_shift = Tensor(np.zeros((B, 1, S, S)), dtype=mindspore.float16)
        ret_inputs = (query, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift,
                      deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2)
    else:
        print("unsupported input layout ", input_layout)
    return ret_inputs


def generate_strategy(dp, mp, optinal_inputs, input_layout='BSH', sparse_mode=0, sp=1):
    has_atten_mask, has_actual_seq_lengths, has_actual_seq_lengths_kv, has_pse_shift, has_deq_scale1, \
        has_quant_scale1, has_deq_scale2, has_quant_scale2, has_quant_offset2 = optinal_inputs
    if dp is None or mp is None:
        return ()
    if input_layout == 'BSH':
        stra = ((dp, sp, mp), (dp, 1, mp), (dp, 1, mp))
        if has_atten_mask:
            if sparse_mode in [2, 3, 4]:
                sp = 1
            stra += ((dp, 1, sp, 1),) if sparse_mode == 0 else ((1, 1, sp, 1),)
        if has_actual_seq_lengths:
            stra += ((dp,),)
        if has_actual_seq_lengths_kv:
            stra += ((dp,),)
        if has_pse_shift:
            stra += ((dp, 1, 1, 1),)
    if input_layout == 'BNSD':
        stra = ((dp, mp, sp, 1), (dp, mp, 1, 1), (dp, mp, 1, 1))
        if has_atten_mask:
            if sparse_mode in [2, 3, 4]:
                sp = 1
            stra += ((dp, 1, sp, 1),) if sparse_mode == 0 else ((1, 1, sp, 1),)
        if has_actual_seq_lengths:
            stra += ((dp,),)
        if has_actual_seq_lengths_kv:
            stra += ((dp,),)
        if has_pse_shift:
            stra += ((dp, 1, 1, 1),)
    for i in [has_deq_scale1, has_quant_scale1, has_deq_scale2, has_quant_scale2, has_quant_offset2]:
        if i:
            stra += ((),)
    return stra

class Net(nn.Cell):
    def __init__(self, num_heads, scale_value=1.0, pre_tokens=2147483547, next_tokens=0, input_layout='BSH',
                 num_key_value_heads=0, strategy=None, sparse_mode=0, set_atten_mask_as_constant=False):
        super(Net, self).__init__()
        self.fa_op = P.PromptFlashAttention(num_heads=num_heads, scale_value=scale_value, pre_tokens=pre_tokens,
                                            next_tokens=next_tokens, input_layout=input_layout,
                                            num_key_value_heads=num_key_value_heads, sparse_mode=sparse_mode)
        stra = strategy
        stra_q = None
        if stra:
            stra_q = (stra[0],)
        self.square = OPS.Square().shard(stra_q)
        self.fa_op.shard(stra)
        self.set_atten_mask_as_constant = set_atten_mask_as_constant
        self.atten_mask = Tensor(np.ones([1, 1, 2048, 2048]), dtype=mindspore.bool_)

    def construct(self, query, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift,
                  deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2):
        ret = self.square(query)
        if self.set_atten_mask_as_constant:
            out = self.fa_op(ret, key, value, self.atten_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift,
                             deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2)
        else:
            out = self.fa_op(ret, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift,
                             deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2)
        return self.square(out)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
@pytest.mark.parametrize('strategys', [(1, 2, 4)])
def test_prompt_flash_attention_semi_auto_parallel_sparsemode0(input_layout, strategys):
    """
    Feature: test PromptFlashAttention semi parallel
    Description: semi parallel
    Expectation: compile success
    """
    B, N, S, D = 8, 16, 1024, 128
    dp = strategys[0]
    mp = strategys[1]
    sp = strategys[2]
    optinal_inputs = [True, False, False, False, False, False, False, False, False]
    dims = [B, N, S, D]
    inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout, sparse_mode=0)
    strategies = generate_strategy(dp, mp, optinal_inputs, input_layout=input_layout, sparse_mode=0, sp=sp)
    net = Net(N, input_layout=input_layout, strategy=strategies, sparse_mode=0)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    standalone_out = net(*inputs)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, dataset_strategy="full_batch",
                                      parallel_mode="semi_auto_parallel")
    dist_out = net(*inputs)
    np.testing.assert_array_almost_equal(dist_out.asnumpy(), standalone_out.asnumpy(), decimal=4)
