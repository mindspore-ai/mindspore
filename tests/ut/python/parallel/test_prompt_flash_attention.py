# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.context import set_auto_parallel_context
from mindspore.ops import operations as OPS
import mindspore.ops.operations.nn_ops as P

context.set_context(mode=context.GRAPH_MODE)


def generate_inputs(dims, optinal_inputs, input_layout='BSH'):
    B, N, S, D = dims
    has_atten_mask, has_padding_mask, has_actual_seq_lengths, has_actual_seq_lengths_kv, has_deq_scale1, \
    has_quant_scale1, has_deq_scale2, has_quant_scale2, has_quant_offset2 = optinal_inputs
    attn_mask = None
    padding_mask = None
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
        query = Tensor(np.ones((B, S, H), dtype=np.float16))
        key = Tensor(np.ones((B, S, H), dtype=np.float16))
        value = Tensor(np.ones((B, S, H), dtype=np.float16))
        if has_atten_mask:
            attn_mask = Tensor(np.ones((B, 1, S, S)), dtype=mindspore.float16)
        if has_padding_mask:
            padding_mask = Tensor(np.zeros((B, 1, S, S)), dtype=mindspore.float16)
        ret_inputs = (query, key, value, attn_mask, padding_mask, actual_seq_lengths, actual_seq_lengths_kv,
                      deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2)
    elif input_layout == 'BNSD':
        query = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        key = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        value = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        if has_atten_mask:
            attn_mask = Tensor(np.ones((B, 1, S, S)), dtype=mindspore.float16)
        if has_padding_mask:
            padding_mask = Tensor(np.zeros((B, 1, S, S)), dtype=mindspore.float16)
        ret_inputs = (query, key, value, attn_mask, padding_mask, actual_seq_lengths, actual_seq_lengths_kv,
                      deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2)
    else:
        print("unsupported input layout ", input_layout)
    return ret_inputs


def generate_strategy(dp, mp, optinal_inputs, input_layout='BSH'):
    has_atten_mask, has_padding_mask, has_actual_seq_lengths, has_actual_seq_lengths_kv, has_deq_scale1, \
    has_quant_scale1, has_deq_scale2, has_quant_scale2, has_quant_offset2 = optinal_inputs
    if dp is None or mp is None:
        return ()
    if input_layout == 'BSH':
        stra = ((dp, 1, mp), (dp, 1, mp), (dp, 1, mp))
        if has_atten_mask:
            stra += ((dp, 1, 1, 1),)
        if has_padding_mask:
            stra += ((dp, 1, 1, 1),)
        if has_actual_seq_lengths:
            stra += ((dp,),)
        if has_actual_seq_lengths_kv:
            stra += ((dp,),)
    if input_layout == 'BNSD':
        stra = ((dp, mp, 1, 1), (dp, mp, 1, 1), (dp, mp, 1, 1))
        if has_atten_mask:
            stra += ((dp, 1, 1, 1),)
        if has_padding_mask:
            stra += ((dp, 1, 1, 1),)
        if has_actual_seq_lengths:
            stra += ((dp,),)
        if has_actual_seq_lengths_kv:
            stra += ((dp,),)
    for i in [has_deq_scale1, has_quant_scale1, has_deq_scale2, has_quant_scale2, has_quant_offset2]:
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
                 num_key_value_heads=0, dp=None, mp=None, strategy=None, sparse_mode=0):
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

    def construct(self, query, key, value, attn_mask, padding_mask, actual_seq_lengths, actual_seq_lengths_kv,
                  deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2):
        ret = self.square(query)
        out = self.fa_op(ret, key, value, attn_mask, padding_mask, actual_seq_lengths, actual_seq_lengths_kv,
                         deq_scale1,
                         quant_scale1, deq_scale2, quant_scale2, quant_offset2)
        return self.square(out[0])


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
def test_self_attention_standalone(input_layout):
    """
    Feature: test PromptFlashAttention standalone
    Description: standalone
    Expectation: compile success
    """
    context.reset_auto_parallel_context()

    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    B, N, S, D = 8, 16, 1024, 128
    dims = [B, N, S, D]
    optinal_inputs = [True, True, True, True, True, True, True, True, True]
    inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout)
    net = Net(N, input_layout=input_layout)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
@pytest.mark.parametrize('strategys', [(4, 2), (2, 2)])
def test_prompt_flash_attention_semi_auto_parallel(input_layout, strategys):
    """
    Feature: test PromptFlashAttention semi parallel
    Description: semi parallel
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    B, N, S, D = 8, 16, 1024, 128
    dp = strategys[0]
    mp = strategys[1]
    optinal_inputs = [True, False, False, False, False, False, False, False, False]
    dims = [B, N, S, D]
    inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout)
    strategies = generate_strategy(dp, mp, optinal_inputs, input_layout=input_layout)
    net = Net(N, input_layout=input_layout, strategy=strategies)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
@pytest.mark.parametrize('search_mode', ['sharding_propagation', 'dynamic_programming', 'recursive_programming'])
def test_prompt_flash_attention_auto_parallel(input_layout, search_mode):
    """
    Feature: test PromptFlashAttention auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode=search_mode, device_num=8,
                                      global_rank=0)
    dp = 2
    mp = 4
    B, N, S, D = 8, 16, 1024, 128
    dims = [B, N, S, D]
    optinal_inputs = [True, False, False, False, False, False, False, False, False]
    inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout)
    strategies = generate_strategy(dp, mp, optinal_inputs, input_layout=input_layout)
    net = Net(N, input_layout=input_layout, strategy=strategies)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
def test_prompt_flash_attention_strategy_error(input_layout):
    """
    Feature: test invalid strategy for PromptFlashAttention
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 2, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1, 1))
    B, N, S, D = 8, 16, 1024, 128
    dims = [B, N, S, D]
    net = Net(N, strategy=strategy, input_layout=input_layout)
    with pytest.raises(RuntimeError):
        optinal_inputs = [True, False, False, False, False, False, False, False, False]
        inputs = generate_inputs(dims, optinal_inputs, input_layout=input_layout)
        compile_net(net, *inputs)
    context.reset_auto_parallel_context()
