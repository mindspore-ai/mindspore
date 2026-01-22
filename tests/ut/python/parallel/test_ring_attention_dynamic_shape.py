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
import os
import shutil
import subprocess

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import Symbol
from mindspore.graph.api import _cell_graph_executor
from mindspore.context import set_auto_parallel_context
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.ops.auto_generate import FusedInferAttentionScore
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


def generate_inputs_for_fias(dims, optinal_inputs, input_layout='BSH', sparse_mode=0, is_ifa=False):
    B, N, S, D = dims
    has_pse_shift, has_atten_mask, has_actual_seq_lengths, has_actual_seq_lengths_kv, has_deq_scale1, \
    has_quant_scale1, has_deq_scale2, has_quant_scale2, has_quant_offset2, has_antiquant_scale, has_antiquant_offset, \
    has_block_table, has_query_padding_size, has_kv_padding_size = optinal_inputs
    attn_mask = None
    pse_shift = None
    actual_seq_lengths = [1 for i in range(B)] if has_actual_seq_lengths else None
    actual_seq_lengths_kv = [1 for i in range(B)] if has_actual_seq_lengths_kv else None
    deq_scale1 = Tensor(1, dtype=ms.uint64) if has_deq_scale1 else None
    quant_scale1 = Tensor(1, dtype=ms.uint64) if has_quant_scale1 else None
    deq_scale2 = Tensor(1, dtype=ms.uint64) if has_deq_scale2 else None
    quant_scale2 = Tensor(1, dtype=ms.float32) if has_quant_scale2 else None
    quant_offset2 = Tensor(1, dtype=ms.float32) if has_quant_offset2 else None
    antiquant_scale = Tensor(1, dtype=ms.float32) if has_antiquant_scale else None
    antiquant_offset = Tensor(1, dtype=ms.float32) if has_antiquant_offset else None
    block_table = Tensor(1, dtype=ms.float32) if has_block_table else None
    query_padding_size = Tensor(1, dtype=ms.float32) if has_query_padding_size else None
    kv_padding_size = Tensor(1, dtype=ms.float32) if has_kv_padding_size else None

    ret_inputs = None
    Q_S = 1 if is_ifa else S
    if input_layout == 'BSH':
        H = N * D
        query = Tensor(np.ones((B, Q_S, H), dtype=np.float16))
        key = Tensor(np.ones((B, S, H), dtype=np.float16))
        value = Tensor(np.ones((B, S, H), dtype=np.float16))
        if has_atten_mask:
            attn_mask = Tensor(np.ones((B, Q_S, S)), dtype=ms.float16) if sparse_mode == 0 else Tensor(
                np.ones((1, 2048, 2048)), dtype=ms.float16)
        if has_pse_shift:
            pse_shift = Tensor(np.zeros((B, N, Q_S, S)), dtype=ms.float16)
        ret_inputs = (query, key, value, pse_shift, attn_mask, actual_seq_lengths, actual_seq_lengths_kv,
                      deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale,
                      antiquant_offset, block_table, query_padding_size, kv_padding_size)
    elif input_layout == 'BNSD':
        query = Tensor(np.ones((B, N, Q_S, D), dtype=np.float16))
        key = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        value = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        if has_atten_mask:
            attn_mask = Tensor(np.ones((B, Q_S, S)), dtype=ms.float16) if sparse_mode == 0 else Tensor(
                np.ones((1, 2048, 2048)), dtype=ms.float16)
        if has_pse_shift:
            pse_shift = Tensor(np.zeros((B, N, Q_S, S)), dtype=ms.float16)
        ret_inputs = (query, key, value, pse_shift, attn_mask, actual_seq_lengths, actual_seq_lengths_kv,
                      deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale,
                      antiquant_offset, block_table, query_padding_size, kv_padding_size)
    elif input_layout == 'BSND':
        query = Tensor(np.ones((B, Q_S, N, D), dtype=np.float16))
        key = Tensor(np.ones((B, S, N, D), dtype=np.float16))
        value = Tensor(np.ones((B, S, N, D), dtype=np.float16))
        if has_atten_mask:
            attn_mask = Tensor(np.ones((B, Q_S, S)), dtype=ms.float16) if sparse_mode == 0 else Tensor(
                np.ones((1, 2048, 2048)), dtype=ms.float16)
        if has_pse_shift:
            pse_shift = Tensor(np.zeros((B, N, Q_S, S)), dtype=ms.float16)
        ret_inputs = (query, key, value, pse_shift, attn_mask, actual_seq_lengths, actual_seq_lengths_kv,
                      deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale,
                      antiquant_offset, block_table, query_padding_size, kv_padding_size)
    else:
        print("unsupported input layout ", input_layout)
    return ret_inputs


def generate_strategy_for_fias(dp, mp, optinal_inputs, input_layout='BSH', sparse_mode=0, sp=1, is_ifa=False):
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

def generate_inputs(B, N, S, D, input_layout, use_mqa=False, with_real_shift=False, sparse_mode=0):
    N_Q = N
    N_KV = 1 if use_mqa else N
    if input_layout == "BSH":
        query = Tensor(shape=[B, S, D], dtype=ms.float16)
        key = Tensor(shape=[B, S, D], dtype=ms.float16)
        value = Tensor(shape=[B, S, D], dtype=ms.float16)
    elif input_layout == "SBH":
        H_Q = N_Q * D
        H_KV = N_KV * D
        query = Tensor(shape=[S, B, H_Q], dtype=ms.float16)
        key = Tensor(shape=[S, B, H_KV], dtype=ms.float16)
        value = Tensor(shape=[S, B, H_KV], dtype=ms.float16)
    elif input_layout == "BNSD":
        query = Tensor(shape=[B, N, S, D], dtype=ms.float16)
        key = Tensor(shape=[B, N_KV, S, D], dtype=ms.float16)
        value = Tensor(shape=[B, N_KV, S, D], dtype=ms.float16)
    elif input_layout == "BSND":
        query = Tensor(shape=[B, S, N, D], dtype=ms.float16)
        key = Tensor(shape=[B, S, N_KV, D], dtype=ms.float16)
        value = Tensor(shape=[B, S, N_KV, D], dtype=ms.float16)
    else:
        raise ValueError(f"input_layout is invalid.")
    real_shift = Tensor(shape=[B, N, S, S], dtype=np.float16) if with_real_shift else None
    attn_mask = None
    sample_num = 4
    actual_seq_qlen = Tensor(tuple(range(2048 // sample_num, 2048 + 1, 2048 // sample_num)), ms.int64)
    actual_seq_kvlen = actual_seq_qlen
    return query, key, value, real_shift, attn_mask, actual_seq_qlen, actual_seq_kvlen




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
                 with_real_shift=False, dp=None, mp=None, sp=1, enable_ring_attention=True,
                 use_send_recv=True, enable_flash_sp=False,
                 enable_ud_mask=False, reset_attn_mask=False):
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
            if reset_attn_mask:
                stra += ((dp,),)
                stra += ((dp,),)
        self.fa_op.shard(stra)
        self.fa_op.add_prim_attr("enable_ring_attention", enable_ring_attention)
        self.fa_op.add_prim_attr("enable_ra_send_recv", use_send_recv)
        self.fa_op.add_prim_attr("enable_flash_sp", enable_flash_sp)

    def construct(self, query, key, value, real_shift, attn_mask, actual_seq_qlen=None, actual_seq_kvlen=None):
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
        return self.fa_op(query, key, value, real_shift, drop_mask_bits, None, attn_mask, None,
                          actual_seq_qlen, actual_seq_kvlen)


class FiasNet(nn.Cell):
    def __init__(self, num_heads, scale_value=1.0, pre_tokens=2147483547, next_tokens=0, input_layout='BSH',
                 num_key_value_heads=0, strategy=None, sparse_mode=0, inner_precise=1, block_size=0,
                 antiquant_mode=0, softmax_lse_flag=False, set_atten_mask_as_constant=False,
                 enable_ring_attention=True):
        super(FiasNet, self).__init__()
        self.fias_op = FusedInferAttentionScore(num_heads=num_heads, scale_value=scale_value, pre_tokens=pre_tokens,
                                                next_tokens=next_tokens, input_layout=input_layout,
                                                num_key_value_heads=num_key_value_heads, sparse_mode=sparse_mode,
                                                inner_precise=inner_precise, block_size=block_size,
                                                antiquant_mode=antiquant_mode, softmax_lse_flag=softmax_lse_flag)
        stra = strategy
        stra_q = None
        if stra:
            stra_q = (stra[0],)
        self.square = P.Square().shard(stra_q)
        self.fias_op.shard(stra)
        self.set_atten_mask_as_constant = set_atten_mask_as_constant
        self.atten_mask = Tensor(np.ones([1, 2048, 2048]), dtype=ms.bool_)
        self.fias_op.add_prim_attr("enable_ring_attention", enable_ring_attention)

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


@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_ring_attention_semi_auto_parallel_send_recv(input_layout):
    """
    Features: test Ring Attention
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=4, global_rank=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=True, save_graphs_path="./test_ring_attention_semi_auto_parallel_send_recv")
    dp = 1
    mp = 1
    sp = 4
    s1 = Symbol(divisor=8)
    B, N, S, D = s1, 16, s1, 128
    query, key, value, real_shift, attn_mask, _, _ = generate_inputs(B, N, S, D,
                                                                     input_layout)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_ring_attention=True, use_send_recv=True)
    if os.path.exists("./test_ring_attention_semi_auto_parallel_send_recv"):
        shutil.rmtree("./test_ring_attention_semi_auto_parallel_send_recv")
    net.set_inputs(query, key, value, real_shift, attn_mask)
    compile_net(net, query, key, value, real_shift, attn_mask)
    file = "./test_ring_attention_semi_auto_parallel_send_recv/*validate*.ir"
    para = "PrimFunc_FlashAttentionScore"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "4"

    para = "Send("
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "3"

    para = "Receive("
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "3"
    context.reset_auto_parallel_context()

@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_flash_sp_semi_auto_parallel(input_layout):
    """
    Features: test Ring Attention
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=True, save_graphs_path="./test_flash_sp_semi_auto_parallel")
    dp = 1
    mp = 1
    sp = 8
    s1 = Symbol(divisor=8)
    B, N, S, D = s1, 16, s1, 128
    query, key, value, real_shift, attn_mask, _, _ = generate_inputs(B, N, S, D,
                                                                     input_layout)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_flash_sp=True)
    if os.path.exists("./test_flash_sp_semi_auto_parallel"):
        shutil.rmtree("./test_flash_sp_semi_auto_parallel")
    net.set_inputs(query, key, value, real_shift, attn_mask)
    compile_net(net, query, key, value, real_shift, attn_mask)
    file = "./test_flash_sp_semi_auto_parallel/*validate*.ir"
    para = "PrimFunc_FlashAttentionScore"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "9"

    para = "Send("
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "11"

    para = "Receive("
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "4"
    context.reset_auto_parallel_context()

@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_ring_attention_user_define_mask_semi_auto_parallel(input_layout):
    """
    Features: test Ring Attention with user define mask
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """

    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=True, save_graphs_path="./test_ring_attention_user_define_mask_semi_auto_parallel")
    dp = 1
    mp = 1
    sp = 8
    s1 = Symbol(divisor=8)
    B, N, S, D = s1, 16, s1, 128
    query, key, value, real_shift, attn_mask, _, _ = generate_inputs(B, N, S, D,
                                                                     input_layout)
    np.random.seed(42)
    attn_mask = Tensor(shape=[S, S], dtype=ms.uint8)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_ring_attention=True, enable_ud_mask=True)
    net.set_inputs(query, key, value, real_shift, attn_mask)
    compile_net(net, query, key, value, real_shift, attn_mask)

@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_ring_attention_semi_auto_parallel_eod_reset_attn_mask(input_layout):
    """
    Features: test Ring Attention
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    dp = 1
    mp = 1
    sp = 8
    s1 = Symbol(divisor=8)
    B, N, S, D = s1, 16, s1, 1024
    query, key, value, real_shift, _, actual_seq_qlen, actual_seq_kvlen = generate_inputs(B, N, S, D, input_layout)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_ring_attention=True, reset_attn_mask=True)
    net.set_inputs(query, key, value, real_shift, _, actual_seq_qlen, actual_seq_kvlen)
    compile_net(net, query, key, value, real_shift, None, actual_seq_qlen, actual_seq_kvlen)
