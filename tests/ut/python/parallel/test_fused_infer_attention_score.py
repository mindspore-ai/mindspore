# Copyright 2023-24 Huawei Technologies Co., Ltd
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
import mindspore
import mindspore.nn as nn
from mindspore.ops import auto_generate as P
from mindspore import Tensor, context
from mindspore.common.api import _cell_graph_executor
from mindspore.context import set_auto_parallel_context
from mindspore.ops import operations as OPS

context.set_context(mode=context.GRAPH_MODE)


def generate_inputs(dims, optinal_inputs, input_layout='BSH', sparse_mode=0, is_ifa=False):
    B, N, S, D = dims
    has_pse_shift, has_atten_mask, has_actual_seq_lengths, has_actual_seq_lengths_kv, has_deq_scale1, \
    has_quant_scale1, has_deq_scale2, has_quant_scale2, has_quant_offset2, has_antiquant_scale, has_antiquant_offset, \
    has_block_table, has_query_padding_size, has_kv_padding_size = optinal_inputs
    attn_mask = None
    pse_shift = None
    actual_seq_lengths = Tensor(np.ones((B,), dtype=np.int64)) if has_actual_seq_lengths else None
    actual_seq_lengths_kv = Tensor(np.ones((B,), dtype=np.int64)) if has_actual_seq_lengths_kv else None
    deq_scale1 = Tensor(1, dtype=mindspore.uint64) if has_deq_scale1 else None
    quant_scale1 = Tensor(1, dtype=mindspore.uint64) if has_quant_scale1 else None
    deq_scale2 = Tensor(1, dtype=mindspore.uint64) if has_deq_scale2 else None
    quant_scale2 = Tensor(1, dtype=mindspore.float32) if has_quant_scale2 else None
    quant_offset2 = Tensor(1, dtype=mindspore.float32) if has_quant_offset2 else None
    antiquant_scale = Tensor(1, dtype=mindspore.float32) if has_antiquant_scale else None
    antiquant_offset = Tensor(1, dtype=mindspore.float32) if has_antiquant_offset else None
    block_table = Tensor(1, dtype=mindspore.float32) if has_block_table else None
    query_padding_size = Tensor(1, dtype=mindspore.float32) if has_query_padding_size else None
    kv_padding_size = Tensor(1, dtype=mindspore.float32) if has_kv_padding_size else None

    ret_inputs = None
    Q_S = 1 if is_ifa else S
    if input_layout == 'BSH':
        H = N * D
        query = Tensor(np.ones((B, Q_S, H), dtype=np.float16))
        key = Tensor(np.ones((B, S, H), dtype=np.float16))
        value = Tensor(np.ones((B, S, H), dtype=np.float16))
        if has_atten_mask:
            attn_mask = Tensor(np.ones((B, Q_S, S)), dtype=mindspore.float16) if sparse_mode == 0 else Tensor(
                np.ones((1, 2048, 2048)), dtype=mindspore.float16)
        if has_pse_shift:
            pse_shift = Tensor(np.zeros((B, N, Q_S, S)), dtype=mindspore.float16)
        ret_inputs = (query, key, value, pse_shift, attn_mask, actual_seq_lengths, actual_seq_lengths_kv,
                      deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale,
                      antiquant_offset, block_table, query_padding_size, kv_padding_size)
    elif input_layout == 'BNSD':
        query = Tensor(np.ones((B, N, Q_S, D), dtype=np.float16))
        key = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        value = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        if has_atten_mask:
            attn_mask = Tensor(np.ones((B, Q_S, S)), dtype=mindspore.float16) if sparse_mode == 0 else Tensor(
                np.ones((1, 2048, 2048)), dtype=mindspore.float16)
        if has_pse_shift:
            pse_shift = Tensor(np.zeros((B, N, Q_S, S)), dtype=mindspore.float16)
        ret_inputs = (query, key, value, pse_shift, attn_mask, actual_seq_lengths, actual_seq_lengths_kv,
                      deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale,
                      antiquant_offset, block_table, query_padding_size, kv_padding_size)
    elif input_layout == 'BSND':
        query = Tensor(np.ones((B, Q_S, N, D), dtype=np.float16))
        key = Tensor(np.ones((B, S, N, D), dtype=np.float16))
        value = Tensor(np.ones((B, S, N, D), dtype=np.float16))
        if has_atten_mask:
            attn_mask = Tensor(np.ones((B, Q_S, S)), dtype=mindspore.float16) if sparse_mode == 0 else Tensor(
                np.ones((1, 2048, 2048)), dtype=mindspore.float16)
        if has_pse_shift:
            pse_shift = Tensor(np.zeros((B, N, Q_S, S)), dtype=mindspore.float16)
        ret_inputs = (query, key, value, pse_shift, attn_mask, actual_seq_lengths, actual_seq_lengths_kv,
                      deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale,
                      antiquant_offset, block_table, query_padding_size, kv_padding_size)
    else:
        print("unsupported input layout ", input_layout)
    return ret_inputs


def generate_strategy(dp, mp, optinal_inputs, input_layout='BSH', sparse_mode=0, sp=1, is_ifa=False):
    has_pse_shift, has_atten_mask, has_actual_seq_lengths, has_actual_seq_lengths_kv, has_deq_scale1, \
    has_quant_scale1, has_deq_scale2, has_quant_scale2, has_quant_offset2, has_antiquant_scale, has_antiquant_offset, \
    has_block_table, has_query_padding_size, has_kv_padding_size = optinal_inputs
    if dp is None or mp is None:
        return ()
    q_sp = 1 if is_ifa else sp
    kv_sp = sp if is_ifa else 1
    if input_layout == 'BSH':
        stra = ((dp, q_sp, mp), ((dp, kv_sp, mp),), ((dp, kv_sp, mp),))
        if has_pse_shift:
            stra += ((dp, 1, 1, 1),)
        if has_atten_mask:
            if sparse_mode in [2, 3, 4]:
                sp = 1
            stra += ((dp, q_sp, sp),) if sparse_mode == 0 else ((1, q_sp, 1),)
        if has_actual_seq_lengths:
            stra += ((dp,),)
        if has_actual_seq_lengths_kv:
            stra += ((dp,),)
    if input_layout == 'BNSD':
        stra = ((dp, mp, q_sp, 1), ((dp, mp, kv_sp, 1),), ((dp, mp, kv_sp, 1),))
        if has_pse_shift:
            stra += ((dp, 1, 1, 1),)
        if has_atten_mask:
            if sparse_mode in [2, 3, 4]:
                sp = 1
            stra += ((dp, q_sp, sp),) if sparse_mode == 0 else ((1, q_sp, 1),)
        if has_actual_seq_lengths:
            stra += ((dp,),)
        if has_actual_seq_lengths_kv:
            stra += ((dp,),)
    if input_layout == 'BSND':
        stra = ((dp, q_sp, mp, 1), ((dp, kv_sp, mp, 1),), ((dp, kv_sp, mp, 1),))
        if has_pse_shift:
            stra += ((dp, 1, 1, 1),)
        if has_atten_mask:
            if sparse_mode in [2, 3, 4]:
                sp = 1
            stra += ((dp, q_sp, sp),) if sparse_mode == 0 else ((1, q_sp, 1),)
        if has_actual_seq_lengths:
            stra += ((dp,),)
        if has_actual_seq_lengths_kv:
            stra += ((dp,),)
    for i in [has_deq_scale1, has_quant_scale1, has_deq_scale2, has_quant_scale2, has_quant_offset2,\
              has_antiquant_scale, has_antiquant_offset, has_block_table, has_query_padding_size, has_kv_padding_size]:
        if i:
            stra += ((),)
    return stra


def compile_net(net, *inputs):
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()
    return phase


class Net(nn.Cell):
    def __init__(self, num_heads, scale_value=1.0, pre_tokens=2147483547, next_tokens=0, input_layout='BSH',
                 num_key_value_heads=0, strategy=None, sparse_mode=0, inner_precise=1, block_size=0,
                 antiquant_mode=0, softmax_lse_flag=False, set_atten_mask_as_constant=False):
        super(Net, self).__init__()
        self.fias_op = P.FusedInferAttentionScore(num_heads=num_heads, scale_value=scale_value, pre_tokens=pre_tokens,
                                                  next_tokens=next_tokens, input_layout=input_layout,
                                                  num_key_value_heads=num_key_value_heads, sparse_mode=sparse_mode,
                                                  inner_precise=inner_precise, block_size=block_size,
                                                  antiquant_mode=antiquant_mode, softmax_lse_flag=softmax_lse_flag)
        stra = strategy
        stra_q = None
        if stra:
            stra_q = (stra[0],)
        self.square = OPS.Square().shard(stra_q)
        self.fias_op.shard(stra)
        self.set_atten_mask_as_constant = set_atten_mask_as_constant
        self.atten_mask = Tensor(np.ones([1, 2048, 2048]), dtype=mindspore.bool_)

    def construct(self, query, key, value, pse_shift, attn_mask, actual_seq_lengths, actual_seq_lengths_kv,
                  deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale,
                  antiquant_offset, block_table, query_padding_size, kv_padding_size):
        ret = self.square(query)
        key_mut = [key,]
        value_mut = [value,]
        if self.set_atten_mask_as_constant:
            out = self.fias_op(ret, key_mut, value_mut, pse_shift, self.atten_mask, actual_seq_lengths,
                               actual_seq_lengths_kv, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                               quant_offset2, antiquant_scale, antiquant_offset, block_table,
                               query_padding_size, kv_padding_size)
        else:
            out = self.fias_op(ret, key_mut, value_mut, pse_shift, attn_mask, actual_seq_lengths,
                               actual_seq_lengths_kv, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                               quant_offset2, antiquant_scale, antiquant_offset, block_table,
                               query_padding_size, kv_padding_size)
        return self.square(out[0])


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
def test_self_attention_standalone(input_layout):
    """
    Feature: test FusedInferAttentionScore standalone
    Description: standalone
    Expectation: compile success
    """
    context.reset_auto_parallel_context()

    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    B, N, S, D = 8, 16, 1024, 128
    dims = [B, N, S, D]
    optinal_inputs = [True, False, False, False, False, False, False,
                      False, False, False, False, False, False, False]
    inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout)
    net = Net(N, input_layout=input_layout)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
def test_self_attention_standalone_ifa(input_layout):
    """
    Feature: test FusedInferAttentionScore standalone
    Description: standalone
    Expectation: compile success
    """
    context.reset_auto_parallel_context()

    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    B, N, S, D = 8, 16, 1024, 128
    dims = [B, N, S, D]
    optinal_inputs = [True, False, False, False, False, False, False,
                      False, False, False, False, False, False, False]
    inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout, is_ifa=True)
    net = Net(N, input_layout=input_layout)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
@pytest.mark.parametrize('strategys', [(4, 2), (2, 2)])
def test_fused_infer_attention_score_semi_auto_parallel(input_layout, strategys):
    """
    Feature: test FusedInferAttentionScore semi parallel
    Description: semi parallel
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    B, N, S, D = 8, 16, 1024, 128
    dp = strategys[0]
    mp = strategys[1]
    optinal_inputs = [False, True, False, False, False, False, False,
                      False, False, False, False, False, False, False]
    dims = [B, N, S, D]
    inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout)
    strategies = generate_strategy(dp, mp, optinal_inputs, input_layout=input_layout)
    net = Net(N, input_layout=input_layout, strategy=strategies)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
@pytest.mark.parametrize('strategys', [(4, 2), (2, 2)])
def test_fused_infer_attention_score_semi_auto_parallel_ifa(input_layout, strategys):
    """
    Feature: test FusedInferAttentionScore semi parallel
    Description: semi parallel
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    B, N, S, D = 8, 16, 1024, 128
    dp = strategys[0]
    mp = strategys[1]
    optinal_inputs = [False, True, False, False, False, False, False,
                      False, False, False, False, False, False, False]
    dims = [B, N, S, D]
    inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout, is_ifa=True)
    strategies = generate_strategy(dp, mp, optinal_inputs, input_layout=input_layout, is_ifa=True)
    net = Net(N, input_layout=input_layout, strategy=strategies)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BNSD'])
def test_fused_infer_attention_score_strategy_error(input_layout):
    """
    Feature: test invalid strategy for FusedInferAttentionScore
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 2, 1), ((1, 1, 1),), ((1, 1, 1),), (1, 1, 1, 1))
    B, N, S, D = 8, 16, 1024, 128
    dims = [B, N, S, D]
    net = Net(N, strategy=strategy, input_layout=input_layout)
    with pytest.raises(RuntimeError):
        optinal_inputs = [True, False, False, False, False, False, False,
                          False, False, False, False, False, False, False]
        inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout)
        compile_net(net, *inputs)
    context.reset_auto_parallel_context()


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
@pytest.mark.parametrize('strategys', [(2, 2, 1), (1, 2, 1)])
def test_fused_infer_attention_score_semi_auto_parallel_sparsemode0(input_layout, strategys):
    """
    Feature: test FusedInferAttentionScore semi parallel
    Description: semi parallel
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="full_batch")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    B, N, S, D = 8, 16, 1024, 128
    dp = strategys[0]
    mp = strategys[1]
    sp = strategys[2]
    optinal_inputs = [True, True, True, True, False, False, False,
                      False, False, False, False, False, False, False]
    dims = [B, N, S, D]
    inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout, sparse_mode=0)
    strategies = generate_strategy(dp, mp, optinal_inputs, input_layout=input_layout, sparse_mode=0, sp=sp)
    net = Net(N, input_layout=input_layout, strategy=strategies,
              sparse_mode=0)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
@pytest.mark.parametrize('strategys', [(2, 2, 2), (1, 1, 8)])
def test_fused_infer_attention_score_semi_auto_parallel_ifa_split_s(input_layout, strategys):
    """
    Feature: test FusedInferAttentionScore semi parallel
    Description: semi parallel
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="full_batch")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    B, N, S, D = 8, 16, 1024, 128
    dp = strategys[0]
    mp = strategys[1]
    sp = strategys[2]
    optinal_inputs = [True, True, False, False, False, False, False,
                      False, False, False, False, False, False, False]
    dims = [B, N, S, D]
    inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout, sparse_mode=0, is_ifa=True)
    strategies = generate_strategy(dp, mp, optinal_inputs, input_layout=input_layout, sparse_mode=0, sp=sp, is_ifa=True)
    net = Net(N, input_layout=input_layout, strategy=strategies,
              sparse_mode=0, softmax_lse_flag=True)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD', 'BSND'])
@pytest.mark.parametrize('strategys', [(1, 2, 4)])
def test_fused_infer_attention_score_semi_auto_parallel_sparsemode2_pfa(input_layout, strategys):
    """
    Feature: test FusedInferAttentionScore semi parallel
    Description: semi parallel
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="full_batch")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    B, N, S, D = 8, 16, 1024, 128
    dp = strategys[0]
    mp = strategys[1]
    sp = strategys[2]
    optinal_inputs = [True, True, False, False, False, False, False,
                      False, False, False, False, False, False, False]
    dims = [B, N, S, D]
    inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout, sparse_mode=2)
    strategies = generate_strategy(dp, mp, optinal_inputs, input_layout=input_layout, sparse_mode=2, sp=sp)
    net = Net(N, input_layout=input_layout, strategy=strategies,
              sparse_mode=2)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD', 'BSND'])
@pytest.mark.parametrize('strategys', [(1, 2, 4), (2, 2, 2)])
def test_fused_infer_attention_score_semi_auto_parallel_sparsemode0_pfa(input_layout, strategys):
    """
    Feature: test FusedInferAttentionScore semi parallel
    Description: semi parallel
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="full_batch")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    B, N, S, D = 8, 16, 1024, 128
    dp = strategys[0]
    mp = strategys[1]
    sp = strategys[2]
    optinal_inputs = [True, True, False, False, False, False, False,
                      False, False, False, False, False, False, False]
    dims = [B, N, S, D]
    inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout, sparse_mode=0)
    strategies = generate_strategy(dp, mp, optinal_inputs, input_layout=input_layout, sparse_mode=0, sp=sp)
    net = Net(N, input_layout=input_layout, strategy=strategies,
              sparse_mode=0)
    compile_net(net, *inputs)
