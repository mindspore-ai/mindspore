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
import re

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.parallel._cost_model_context import _set_rp_matmul_mem_coef

class FATestNet(nn.Cell):
    def __init__(self, input_layout, batch_size, head_num, seq_len, head_size):
        super().__init__()

        self.batch_size = batch_size
        self.head_num = head_num
        self.seq_len = seq_len
        self.head_size = head_size

        self.input_layout = input_layout

        self.matmul = P.MatMul()
        self.flash_attention = FlashAttentionScore(head_num=head_num, input_layout=input_layout)

        self.reshape = P.Reshape()

    def _reshape_layout(self, x):
        if self.input_layout == 'BNSD':
            return self.reshape(x, (self.batch_size, self.head_num, self.seq_len, self.head_size))
        if self.input_layout == 'BSH':
            return self.reshape(x, (self.batch_size, self.seq_len, self.head_num * self.head_size))
        if self.input_layout == 'SBH':
            return self.reshape(x, (self.seq_len, self.batch_size, self.head_num * self.head_size))
        if self.input_layout == 'BSND':
            return self.reshape(x, (self.batch_size, self.seq_len, self.head_num, self.head_size))

        raise ValueError(f'Unexpected input_layout: {self.input_layout}')

    def construct(self, x, w, att_mask):
        x = self.matmul(x, w)
        x = self._reshape_layout(x)

        out = self.flash_attention(x, x, x, None, None, None, att_mask, None)
        return out

def generate_bnsd_inputs(batch_size, head_num, seq_len, head_size):
    x = Tensor(np.ones([batch_size * head_num, 64]), dtype=ms.float16)
    w = Tensor(np.ones([64, seq_len * head_size]), dtype=ms.float16)
    attn_mask = Tensor(np.ones([batch_size, 1, seq_len, seq_len]), dtype=ms.uint8)

    return x, w, attn_mask

def generate_bsh_inputs(batch_size, head_num, seq_len, head_size):
    x = Tensor(np.ones([batch_size * seq_len, 64]), dtype=ms.float16)
    w = Tensor(np.ones([64, head_num * head_size]), dtype=ms.float16)
    attn_mask = Tensor(np.ones([batch_size, 1, seq_len, seq_len]), dtype=ms.uint8)

    return x, w, attn_mask

def generate_sbh_inputs(batch_size, head_num, seq_len, head_size):
    x = Tensor(np.ones([seq_len * batch_size, 64]), dtype=ms.float16)
    w = Tensor(np.ones([64, head_num * head_size]), dtype=ms.float16)
    attn_mask = Tensor(np.ones([batch_size, 1, seq_len, seq_len]), dtype=ms.uint8)

    return x, w, attn_mask

def generate_bsnd_inputs(batch_size, head_num, seq_len, head_size):
    x = Tensor(np.ones([batch_size * seq_len, 64]), dtype=ms.float16)
    w = Tensor(np.ones([64, head_num * head_size]), dtype=ms.float16)
    attn_mask = Tensor(np.ones([batch_size, 1, seq_len, seq_len]), dtype=ms.uint8)

    return x, w, attn_mask

def get_input_generator_map():
    return {
        'BNSD': generate_bnsd_inputs,
        'BSH': generate_bsh_inputs,
        'SBH': generate_sbh_inputs,
        'BSND': generate_bsnd_inputs,
    }

def generate_inputs_for_layout(input_layout, batch_size, head_num, seq_len, head_size):
    generator_map = get_input_generator_map()
    return generator_map[input_layout](batch_size, head_num, seq_len, head_size)

def get_net_strategies(x, w, attn_mask, input_layout, batch_size, head_num, seq_len, head_size):
    net = FATestNet(input_layout=input_layout, batch_size=batch_size, head_num=head_num, seq_len=seq_len,
                    head_size=head_size)
    net.set_train()
    _cell_graph_executor.compile(net, x, w, attn_mask, phase='train')

    strategies = _cell_graph_executor._get_shard_strategy(net)

    for (k, v) in strategies.items():
        if re.search('MatMul-op0', k) is not None:
            matmul_stra = v

        if re.search('FlashAttentionScore-op0', k) is not None:
            fa_stra = v

    return matmul_stra, fa_stra

def get_layout_indexes(input_layout):
    # Returns indexes for (dp,mp,sp) depending on layout

    if input_layout == "BNSD":
        return 0, 1, 2
    if input_layout == "BSH":
        return 0, 2, 1
    if input_layout == "SBH":
        return 1, 2, 0
    if input_layout == "BSND":
        return 0, 2, 1

    raise ValueError(f'Unexpected input_layout: {input_layout}')

def check_valid_fa_strategy(input_layout, fa_stra):
    dp, mp, sp = get_layout_indexes(input_layout)

    q_stra = fa_stra[0]
    k_stra = fa_stra[1]
    v_stra = fa_stra[2]
    mask_stra = fa_stra[3]

    fa_dp = q_stra[dp]
    fa_mp = q_stra[mp]
    fa_sp = q_stra[sp]

    assert fa_dp == k_stra[dp]
    assert fa_mp == k_stra[mp]
    assert fa_dp == mask_stra[0]
    assert mask_stra[1] == 1
    assert fa_sp == mask_stra[2]
    assert k_stra == v_stra

def check_correct_strategy_propagation(fa_stra, mm_stra):
    mm_out_stra = mm_stra[0][0], mm_stra[1][1]

    # Rely on reshape propagation rule which splits parallelism on first dimension that fit

    assert mm_out_stra[0] == fa_stra[0][0]
    assert mm_out_stra[1] == fa_stra[0][2]

def run_layout_test(input_layout, mem_coef):
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")
    _set_rp_matmul_mem_coef(mem_coef)

    batch_size = 8
    seq_len = 1024
    head_num = 32
    head_size = 128

    x, w, attn_mask = generate_inputs_for_layout(input_layout, batch_size, head_num, seq_len, head_size)
    mm_stra, fa_stra = get_net_strategies(x, w, attn_mask, input_layout, batch_size, head_num, seq_len, head_size)

    check_valid_fa_strategy(input_layout, fa_stra)
    check_correct_strategy_propagation(fa_stra, mm_stra)

def test_auto_parallel_sapp_flash_attention_bnsd():
    """
    Feature: test flash attention strategy in SAPP for BNSD layout
    Description: auto parallel
    Expectation: compile success and and strategy correct
    """

    run_layout_test("BNSD", pow(2, -10))

def test_auto_parallel_sapp_flash_attention_bsh():
    """
    Feature: test flash attention strategy in SAPP for BSH layout
    Description: auto parallel
    Expectation: compile success and and strategy correct
    """

    run_layout_test("BSH", pow(2, 0))

def test_auto_parallel_sapp_flash_attention_sbh():
    """
    Feature: test flash attention strategy in SAPP for SBH layout
    Description: auto parallel
    Expectation: compile success and and strategy correct
    """

    run_layout_test("SBH", pow(2, 0))

def test_auto_parallel_sapp_flash_attention_bsnd():
    """
    Feature: test flash attention strategy in SAPP for BSND layout
    Description: auto parallel
    Expectation: compile success and and strategy correct
    """

    run_layout_test("BSND", pow(2, 0))
