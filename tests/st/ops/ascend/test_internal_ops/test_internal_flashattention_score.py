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

import numpy as np
import math
import mindspore as ms
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.nn import Cell

ms.set_context(mode=ms.GRAPH_MODE)


def gen_seq_len(batch, max_seq, pa=False, rand=False):
    if pa:
        max_seq_aligned = 16
        seqlen = np.ones((batch,)).astype(np.int32)
        print(seqlen)
        seqlen_aligned = np.ones((batch,)) * max_seq_aligned
        seqlen_aligned = seqlen_aligned.astype(np.int32)
    elif rand:
        max_seq_aligned = (max_seq + 15) // 16 * 16
        randnum = np.random.randint(1, max_seq, 1)
        seqlen = np.ones((batch,)) * randnum
        seqlen = seqlen.astype(np.int32)
        print("seqlen:", seqlen)
        seqlen_aligned = np.ones((batch,)) * max_seq_aligned
        seqlen_aligned = seqlen_aligned.astype(np.int32)
    else:
        max_seq_aligned = (max_seq + 15) // 16 * 16
        seqlen = np.ones((batch,)) * max_seq
        seqlen = seqlen.astype(np.int32)
        print(seqlen)
        seqlen_aligned = np.ones((batch,)) * max_seq_aligned
        seqlen_aligned = seqlen_aligned.astype(np.int32)

    ntokens = seqlen.sum()
    print("ntokens:", ntokens)
    return seqlen, seqlen_aligned, ntokens


def calc_expect_func(batch, max_seq, q_heads, kv_heads, embed, drop_prop, low_tri=True, is_dropout=False, \
                     pa=False, inc=False, input_type='float16', calc_type=np.float32, layout='BNSD', block_size=16):
    num_groups = q_heads // kv_heads
    q_seqlen, _, _ = gen_seq_len(batch, max_seq, pa | inc)
    if pa:
        kv_seqlen, _, _ = gen_seq_len(batch, max_seq, False)
    else:
        kv_seqlen, _, _ = gen_seq_len(batch, max_seq, False)
    np.random.seed(0)

    q = np.random.uniform(-1.0, 1.0, size=(batch, q_heads, q_seqlen[0], embed)).astype(np.float16)
    k = np.random.uniform(-1.0, 1.0, size=(batch, kv_heads, max_seq, embed)).astype(np.float16)
    v = np.random.uniform(-1.0, 1.0, size=(batch, kv_heads, max_seq, embed)).astype(np.float16)
    dropout_uint8 = np.zeros(batch * max_seq * max_seq >> 3).astype(np.uint8)

    amask = np.ones(shape=(batch, max_seq, max_seq)).astype(np.float16)
    amask = np.triu(amask, 1)  # 下三角
    dmask = None
    if is_dropout:
        dmask = np.random.uniform(size=(batch * max_seq * max_seq)) > drop_prop
        masks = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80)
        j = 0
        for i in range(batch * max_seq * max_seq >> 3):
            for m in masks:
                if dmask[j]:
                    dropout_uint8[i] = dropout_uint8[i] | m
                j += 1
        dmask = dmask.reshape(batch, max_seq, max_seq)

    q_offset = 0
    k_offset = 0
    v_offset = 0

    s = None
    _p = None
    out = None
    _score_max = None
    _score_sum = None

    for idx in range(batch):
        for hidx in range(q_heads):
            kv_hidx = hidx // num_groups
            q_s = q_seqlen[idx]
            q_slice = q[idx, hidx, :, :]
            k_slice = k[idx, kv_hidx, 0:kv_seqlen[0], :]
            k_slice_t = np.transpose(k_slice, (1, 0))  # get K^T
            v_slice = v[idx, kv_hidx, 0:kv_seqlen[0], :]
            if is_dropout:
                dmask_slice = dmask[idx, :, :]
            if low_tri:
                amask_slice = amask[idx, :, :]
            score = np.matmul(q_slice.astype(np.float32),
                              k_slice_t.astype(np.float32)).astype(np.float16)
            if s is None:
                s = score.reshape([-1, ])
            else:
                s = np.concatenate((s, score.reshape([-1, ])), 0)

            tor = np.float16(math.sqrt(1.0 * embed))
            score = score / tor
            if low_tri:
                for i in range(q_s):
                    score[i][:] = score[i][:] - amask_slice[i][:] * 10000
            score_max = np.max(score, axis=-1)
            if _score_max is None:
                _score_max = score_max.astype(np.float16).reshape([-1, ])
            else:
                _score_max = np.concatenate((_score_max, score_max.astype(np.float16).reshape([-1, ])), 0)
            score = score - score_max.reshape((q_s, 1))
            score_exp = np.exp(score.astype(np.float32))

            if is_dropout:
                score_exp = score_exp * dmask_slice / (1 - drop_prop)

            score_sum = np.sum(score_exp.astype(np.float16), axis=-1)
            if _score_sum is None:
                _score_sum = score_sum.astype(np.float16).reshape([-1, ])
            else:
                _score_sum = np.concatenate((_score_sum, score_sum.astype(np.float16).reshape([-1, ])), 0)

            p = score_exp.astype(np.float16) / score_sum.reshape((q_s, 1)).astype(np.float16)

            if _p is None:
                _p = p.astype(np.float16).reshape([-1, ])
            else:
                _p = np.concatenate((_p, p.astype(np.float16).reshape([-1, ])), 0)

            o = np.matmul(p.astype(np.float32),
                          v_slice.astype(np.float32)).astype(np.float16)
            o = o.reshape(q_s, embed)
            o = np.ascontiguousarray(o)

            if out is None:
                out = o
            else:
                out = np.concatenate((out, o), 0)

            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq

    table = np.reshape(np.zeros(batch * max_seq // block_size * kv_heads), (batch, max_seq // block_size, kv_heads))

    q_bsh = np.reshape(np.zeros(batch * q_seqlen[0] * embed * q_heads), (batch, q_seqlen[0], q_heads * embed))
    k_bsh = np.reshape(np.zeros(batch * max_seq * embed * kv_heads), (batch, max_seq, kv_heads * embed))
    v_bsh = np.reshape(np.zeros(batch * max_seq * embed * kv_heads), (batch, max_seq, kv_heads * embed))

    for bidx in range(batch):
        for sidx in range(q_seqlen[0]):
            for hidx in range(embed * q_heads):
                q_bsh[bidx][sidx][hidx] = q[bidx][hidx // embed][sidx][hidx % embed]
        for sidx in range(max_seq):
            for hidx in range(embed * kv_heads):
                k_bsh[bidx][sidx][hidx] = k[bidx][hidx // embed][sidx][hidx % embed]
                v_bsh[bidx][sidx][hidx] = v[bidx][hidx // embed][sidx][hidx % embed]
        if pa:
            for sidx in range(max_seq // block_size):
                for hidx in range(kv_heads):
                    table[bidx][sidx][hidx] = bidx * max_seq * kv_heads * embed \
                                              + sidx * block_size * kv_heads * embed \
                                              + hidx * embed

    return q_bsh, k_bsh, v_bsh, out


class MyFA(Cell):
    def __init__(self, q_heads):
        super().__init__()
        self.q_heads = q_heads
        self.net = FlashAttentionScore(self.q_heads)

    def construct(self, query, key, value, attn_mask, drop_mask, real_shift, padding_mask, prefix):
        output = self.net(query, key, value, attn_mask, drop_mask, real_shift, padding_mask, prefix)
        return output


def test_internal_flash_attention_score_bf16():
    """
    Feature: test internal flash attention score with bf16
    Description: test internal flash attention score with bf16
    Expectation: the result is correct
    """
    batch = 1
    q_heads = 11
    kv_heads = 1
    max_seq = 1024
    embed = 128
    drop_prop = 0.5
    q, k, v, out = \
        calc_expect_func(batch, max_seq, q_heads, kv_heads, embed, drop_prop, \
                         low_tri=True, is_dropout=False, pa=False, input_type='float32', layout="BSH")
    out = out.reshape(batch, max_seq, embed * q_heads)

    query = q.astype(np.float32)
    key = k.astype(np.float32)
    value = v.astype(np.float32)
    expect = out.astype(np.float32)

    print("query:", query)
    print("key:", key)
    print("value:", value)

    my_flashattentionscore = MyFA(q_heads)
    output = my_flashattentionscore(ms.Tensor(query).astype(ms.bfloat16),
                                    ms.Tensor(key).astype(ms.bfloat16),
                                    ms.Tensor(value).astype(ms.bfloat16),
                                    None, None, None, None, None)
    output = output[0].to(ms.float32)
    print("my output shape:", output.shape)
    print("my output:", output)

    print("expect shape:", expect.shape)
    print("expect:", expect)

    output = output.reshape(-1).numpy()
    expect = expect.reshape(-1)

    count = 0
    for index in range(0, len(output)):
        if np.abs(output[index] - expect[index]) > np.abs(expect[index]) * 0.03:
            count = count + 1

    err_ratio = count / len(output)
    print("err_ratio:", err_ratio)

    assert err_ratio < 0.05
