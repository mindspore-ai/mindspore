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


def generate_inputs(B, N, S, D, input_layout, use_mqa=False, with_real_shift=False, sparse_mode=0, is_eod_pad=False):
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
    sample_num = 4
    actual_seq_qlen = Tensor(tuple(range(S // sample_num, S + 1, S // sample_num)), ms.int64)
    if is_eod_pad:
        actual_seq_qlen = Tensor(tuple(range(S // sample_num, S - 2, S // sample_num)), ms.int64)
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
                 with_real_shift=False, dp=None, mp=None, sp=1, enable_ring_attention=False,
                 use_send_recv=False, use_cp=False, enable_flash_sp=False,
                 enable_ud_mask=False, reset_attn_mask=False, multi_fa=False, enable_bf16=False):
        super(Net, self).__init__()
        self.multi_fa = multi_fa
        self.enable_bf16 = enable_bf16
        self.mul = P.Mul()
        self.cast = P.Cast()
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
        self.fa_op.add_prim_attr("enable_ra_context_parallel", use_cp)
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
        if self.enable_bf16:
            query = self.cast(query, ms.bfloat16)
            key = self.cast(key, ms.bfloat16)
            value = self.cast(value, ms.bfloat16)
        if self.multi_fa:
            out1 = self.fa_op(query, key, value, real_shift, drop_mask_bits, None, attn_mask, None,
                              actual_seq_qlen, actual_seq_kvlen)
            # If the inputs of the two FA ops are completely identical, the flash_sp pass will generate two identical
            # Send nodes. The subsequent CSE pass in frontend-compilation may eliminate the second Send node (as CSE
            # considers them duplicate nodes that can be removed).
            # However, if the CSE logic changes, it could affect the number of Send nodes in this testcase and
            # potentially cause test failure, which would increase maintenance costs for parallel and CSE developers.
            # Therefore, we modify the input of the second FA op to differ from the first one to prevent this testcase
            # from triggering CSE elimination.
            query2 = query * 2
            key2 = key * 2
            value2 = value * 2
            out2 = self.fa_op(query2, key2, value2, real_shift, drop_mask_bits, None, attn_mask, None,
                              actual_seq_qlen, actual_seq_kvlen)
            return self.mul(out1[0], out2[0])
        return self.fa_op(query, key, value, real_shift, drop_mask_bits, None, attn_mask, None,
                          actual_seq_qlen, actual_seq_kvlen)


class FiasNet(nn.Cell):
    def __init__(self, num_heads, scale_value=1.0, pre_tokens=2147483647, next_tokens=0, input_layout='BSH',
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
def test_ring_attention_semi_auto_parallel_alltoallv(input_layout):
    """
    Features: test Ring Attention
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=4, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=True, save_graphs_path="./ring_attention_semi_auto_parallel_alltoallv")
    dp = 1
    mp = 1
    sp = 4
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, attn_mask, _, _ = generate_inputs(B, N, S, D, input_layout)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_ring_attention=True, use_send_recv=False)
    if os.path.exists("./ring_attention_semi_auto_parallel_alltoallv"):
        shutil.rmtree("./ring_attention_semi_auto_parallel_alltoallv")
    compile_net(net, query, key, value, real_shift, attn_mask)
    file = "./ring_attention_semi_auto_parallel_alltoallv/*validate*.ir"
    para = "PrimFunc_FlashAttentionScore"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "4"

    para = "NeighborExchange("
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "3"
    context.reset_auto_parallel_context()


@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_ring_attention_semi_auto_parallel_send_recv(input_layout):
    """
    Features: test Ring Attention
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=4, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=True, save_graphs_path="./ring_attention_semi_auto_parallel_send_recv")
    dp = 1
    mp = 1
    sp = 4
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, attn_mask, _, _ = generate_inputs(B, N, S, D,
                                                                     input_layout)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_ring_attention=True, use_send_recv=True)
    if os.path.exists("./ring_attention_semi_auto_parallel_send_recv"):
        shutil.rmtree("./ring_attention_semi_auto_parallel_send_recv")
    compile_net(net, query, key, value, real_shift, attn_mask)
    file = "./ring_attention_semi_auto_parallel_send_recv/*validate*.ir"
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
    context.set_context(save_graphs=True, save_graphs_path="./flash_sp_semi_auto_parallel")
    dp = 1
    mp = 1
    sp = 8
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, attn_mask, _, _ = generate_inputs(B, N, S, D,
                                                                     input_layout)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_flash_sp=True)
    if os.path.exists("./flash_sp_semi_auto_parallel"):
        shutil.rmtree("./flash_sp_semi_auto_parallel")
    compile_net(net, query, key, value, real_shift, attn_mask)
    file = "./flash_sp_semi_auto_parallel/*validate*.ir"
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
def test_flash_sp_semi_auto_parallel_bf16(input_layout):
    """
    Features: test Ring Attention
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=True, save_graphs_path="./flash_sp_semi_auto_parallel_bf16")
    dp = 1
    mp = 1
    sp = 8
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, attn_mask, _, _ = generate_inputs(B, N, S, D,
                                                                     input_layout)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_flash_sp=True, enable_bf16=True)
    if os.path.exists("./flash_sp_semi_auto_parallel_bf16"):
        shutil.rmtree("./flash_sp_semi_auto_parallel_bf16")
    compile_net(net, query, key, value, real_shift, attn_mask)
    file = "./flash_sp_semi_auto_parallel_bf16/*validate*.ir"
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
def test_flash_sp_semi_auto_parallel_not_full(input_layout):
    """
    Features: test Ring Attention
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=True, save_graphs_path="./test_flash_sp_semi_auto_parallel_not_full")
    dp = 1
    mp = 1
    sp = 4
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, attn_mask, _, _ = generate_inputs(B, N, S, D,
                                                                     input_layout)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_flash_sp=True)
    if os.path.exists("./test_flash_sp_semi_auto_parallel_not_full"):
        shutil.rmtree("./test_flash_sp_semi_auto_parallel_not_full")
    compile_net(net, query, key, value, real_shift, attn_mask)
    file = "./test_flash_sp_semi_auto_parallel_not_full/*validate*.ir"
    para = "PrimFunc_FlashAttentionScore"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "5"

    para = "Send("
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "5"

    para = "Receive("
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "2"
    context.reset_auto_parallel_context()


@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_flash_sp_semi_auto_parallel_multi_fa(input_layout):
    """
    Features: test Ring Attention
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=True, save_graphs_path="./flash_sp_semi_auto_parallel_multi_fa")
    dp = 1
    mp = 1
    sp = 8
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, attn_mask, _, _ = generate_inputs(B, N, S, D,
                                                                     input_layout)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_flash_sp=True, multi_fa=True)
    if os.path.exists("./flash_sp_semi_auto_parallel_multi_fa"):
        shutil.rmtree("./flash_sp_semi_auto_parallel_multi_fa")
    compile_net(net, query, key, value, real_shift, attn_mask)
    file = "./flash_sp_semi_auto_parallel_multi_fa/*validate*.ir"
    para = "PrimFunc_FlashAttentionScore"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "18"

    para = "Send("
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "22"

    para = "Receive("
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "8"
    context.reset_auto_parallel_context()


@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_ring_attention_user_define_mask_semi_auto_parallel(input_layout):
    """
    Features: test Ring Attention with user define mask
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """

    set_auto_parallel_context(device_num=4, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=False)
    dp = 1
    mp = 1
    sp = 4
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, attn_mask, _, _ = generate_inputs(B, N, S, D,
                                                                     input_layout)
    np.random.seed(42)
    attn_mask = Tensor(np.random.uniform(0, 2, size=(S, S)).astype(np.uint8), dtype=ms.uint8)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_ring_attention=True, enable_ud_mask=True)
    compile_net(net, query, key, value, real_shift, attn_mask)


@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_ring_attention_semi_auto_parallel_cp(input_layout):
    """
    Features: test Ring Attention
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=4, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=True, save_graphs_path="./ring_attention_semi_auto_parallel_cp")
    dp = 1
    mp = 1
    sp = 4
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, attn_mask, _, _ = generate_inputs(B, N, S, D,
                                                                     input_layout)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_ring_attention=True, use_cp=True)
    if os.path.exists("./ring_attention_semi_auto_parallel_cp"):
        shutil.rmtree("./ring_attention_semi_auto_parallel_cp")
    compile_net(net, query, key, value, real_shift, attn_mask)
    file = "./ring_attention_semi_auto_parallel_cp/*validate*.ir"
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

    file = "./ring_attention_semi_auto_parallel_cp/*opt_pass_2_opt_a*.ir"
    para = "call"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "3"
    context.reset_auto_parallel_context()

@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_ring_attention_semi_auto_parallel_eod_reset_attn_mask(input_layout):
    """
    Features: test Ring Attention
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=4, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=False)
    dp = 1
    mp = 1
    sp = 4
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, _, actual_seq_qlen, actual_seq_kvlen = generate_inputs(B, N, S, D, input_layout)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_ring_attention=True, reset_attn_mask=True)
    compile_net(net, query, key, value, real_shift, None, actual_seq_qlen, actual_seq_kvlen)

@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_ring_attention_semi_auto_parallel_eod_reset_pad_attn_mask(input_layout):
    """
    Features: test Ring Attention
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=4, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=False)
    dp = 1
    mp = 1
    sp = 4
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, _, actual_seq_qlen, actual_seq_kvlen = generate_inputs(B, N, S, D, input_layout,
                                                                                          is_eod_pad=True)
    net = Net(N, input_layout=input_layout, dp=dp, mp=mp, sp=sp, enable_ring_attention=True, reset_attn_mask=True)
    compile_net(net, query, key, value, real_shift, None, actual_seq_qlen, actual_seq_kvlen)

@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
@pytest.mark.parametrize('strategys', [(1, 2, 4)])
def test_ring_attention_semi_auto_parallel_for_fused_infer_attention_score(input_layout, strategys):
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
    optinal_inputs = [True, True, True, False, False, False, False,
                      False, False, False, False, False, False, False]
    dims = [B, N, S, D]
    inputs = generate_inputs_for_fias(dims, optinal_inputs, input_layout=input_layout, sparse_mode=0)
    strategies = generate_strategy_for_fias(dp, mp, optinal_inputs, input_layout=input_layout, sparse_mode=0, sp=sp)
    net = FiasNet(N, input_layout=input_layout, strategy=strategies, sparse_mode=0, enable_ring_attention=True,
                  softmax_lse_flag=True)
    compile_net(net, *inputs)
