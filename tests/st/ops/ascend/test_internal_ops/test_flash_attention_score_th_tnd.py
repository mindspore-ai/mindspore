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

import os
import numpy as np
import math
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.nn import Cell
from mindspore import context


def group_matmul(heads, group_num, A, B):
    group_head = heads // group_num
    score = None
    for i in range(group_num):
        group_score = np.matmul(A[i * group_head: (i + 1) * group_head, :, :].astype(np.float32),
                                B[i:(i + 1), :, :].astype(np.float32)).astype(np.float16)
        if score is None:
            score = group_score
        else:
            score = np.concatenate((score, group_score), 0)
    return score

class TestFlashAttentionScore():
    def set_data_params(self, batch, max_seq, q_heads, kv_heads, embed, need_pad=True,
                        layout='BSH', input_type=ms.float16, enable_alibi=False):
        self.batch = batch
        self.max_seq = max_seq
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.embed = embed
        self.need_pad = need_pad
        self.layout = layout
        self.input_type = input_type
        self.enable_alibi = enable_alibi
        self.alibi = None

    def gen_seq_len(self):
        if self.need_pad:
            seqlen = np.ones((self.batch,)) * self.max_seq
            seqlen = seqlen.astype(np.int32)
        else:
            seqlen = np.random.randint(1, self.max_seq, (self.batch,))
            seqlen = seqlen.astype(np.int32)

        ntokens = seqlen.sum()
        return seqlen, ntokens

    def gen_input_data(self):
        self.q_seqlen, self.num_tokens = self.gen_seq_len()
        self.kv_seqlen = self.q_seqlen
        np.random.seed(0)

        if self.layout == 'BSH':
            self.q = np.random.uniform(-1.0, 1.0, size=(self.batch, self.max_seq, self.q_heads * self.embed)).astype(
                np.float32)
            self.k = np.random.uniform(-1.0, 1.0, size=(self.batch, self.max_seq, self.kv_heads * self.embed)).astype(
                np.float32)
            self.v = np.random.uniform(-1.0, 1.0, size=(self.batch, self.max_seq, self.kv_heads * self.embed)).astype(
                np.float32)
            amask = np.ones(shape=(self.batch, 1, self.max_seq, self.max_seq)).astype(np.float16)
            amask = np.triu(amask, 1)  # 下三角
        elif self.layout == 'TH':
            self.q = np.random.uniform(-1.0, 1.0, size=(self.num_tokens, self.q_heads * self.embed)).astype(np.float32)
            self.k = np.random.uniform(-1.0, 1.0, size=(self.num_tokens, self.kv_heads * self.embed)).astype(np.float32)
            self.v = np.random.uniform(-1.0, 1.0, size=(self.num_tokens, self.kv_heads * self.embed)).astype(np.float32)
            amask = np.ones(shape=(128, 128)).astype(np.float16)
            amask = np.triu(amask, 1)  # 下三角
        elif self.layout == 'TND':
            self.q = np.random.uniform(-1.0, 1.0, size=(self.num_tokens, self.q_heads, self.embed)).astype(np.float32)
            self.k = np.random.uniform(-1.0, 1.0, size=(self.num_tokens, self.kv_heads, self.embed)).astype(np.float32)
            self.v = np.random.uniform(-1.0, 1.0, size=(self.num_tokens, self.kv_heads, self.embed)).astype(np.float32)
            amask = np.ones(shape=(128, 128)).astype(np.float16)
            amask = np.triu(amask, 1)  # 下三角
        amask *= -10000.0

        if self.enable_alibi:
            alibi = np.random.uniform(-10.0, 0.0, size=(1, self.q_heads, self.max_seq, self.max_seq)).astype(np.float32)
            self.alibi = alibi

        return self.q, self.k, self.v, amask, self.q_seqlen, self.alibi

    def calc_expect_func(self):
        if self.layout == 'BSH':
            amask = np.ones(shape=(self.batch, self.max_seq, self.max_seq)).astype(np.float16)
            amask = np.triu(amask, 1)
            amask *= -10000.0
        else:
            if self.layout == 'TND':
                self.q = self.q.reshape(self.num_tokens, self.q_heads * self.embed)
                self.k = self.k.reshape(self.num_tokens, self.kv_heads * self.embed)
                self.v = self.v.reshape(self.num_tokens, self.kv_heads * self.embed)
            amask = np.ones(shape=(self.max_seq, self.max_seq)).astype(np.float16)
            amask = np.triu(amask, 1)
            if self.input_type == ms.float16:
                amask *= -10000.0
            else:
                amask *= -3e38

        q_offset = 0
        k_offset = 0
        v_offset = 0

        s = None
        _p = None
        out = None
        fp32 = True

        for idx in range(self.batch):
            q_s = self.q_seqlen[idx]
            kv_s = self.kv_seqlen[idx]
            if self.layout == 'BSH':
                q_slice = self.q[idx, :q_s, :]
            else:
                q_slice = self.q[q_offset:q_offset + q_s, :]
            q_slice = q_slice.reshape(q_s, self.q_heads, self.embed)
            q_slice = np.transpose(q_slice, (1, 0, 2))  # (self.q_heads, q_seq, self.embed)
            if self.layout == 'BSH':
                k_slice = self.k[idx, :kv_s, :]
            else:
                k_slice = self.k[k_offset:k_offset + kv_s, :]
            k_slice = k_slice.reshape(kv_s, self.kv_heads, self.embed)
            k_slice = np.transpose(k_slice, (1, 0, 2))
            k_slice_t = np.transpose(k_slice, (0, 2, 1))   # get K^T (self.kv_heads, self.embed, k_seq)
            if self.layout == 'BSH':
                v_slice = self.v[idx, :kv_s, :]
            else:
                v_slice = self.v[v_offset:v_offset + kv_s, :]
            v_slice = v_slice.reshape(kv_s, self.kv_heads, self.embed)
            v_slice = np.transpose(v_slice, (1, 0, 2))
            score = group_matmul(self.q_heads, self.kv_heads, q_slice, k_slice_t)
            if s is None:
                s = score.reshape([-1,])
            else:
                s = np.concatenate((s, score.reshape([-1,])), 0)

            tor = np.float16(1.0 / math.sqrt(1.0 * self.embed))
            score = score * tor
            if self.enable_alibi:
                score = score + self.alibi[0, :self.q_heads, :q_s, :kv_s]
            else:
                if self.layout == 'BSH':
                    score = score + amask[idx, :q_s, :kv_s]
                else:
                    score = score + amask[:q_s, :kv_s]

            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((self.q_heads, q_s, 1))
            score_exp = np.exp(score.astype(np.float32))
            if not fp32:
                score_sum = np.sum(score_exp.astype(np.float16), axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1,])
                else:
                    _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1,])), 0)
                p = score_exp.astype(np.float16) / score_sum.reshape((self.q_heads, q_s, 1)).astype(np.float16)
                out_sub = group_matmul(self.q_heads, self.kv_heads, p, v_slice)
            else:
                score_sum = np.sum(score_exp, axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1,])
                else:
                    _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1,])), 0)
                p = score_exp.astype(np.float16)
                out_sub = group_matmul(self.q_heads, self.kv_heads, p, v_slice)
                out_sub = out_sub / score_sum.reshape((self.q_heads, q_s, 1)).astype(np.float16)

            out_sub = out_sub.reshape(self.q_heads, q_s, self.embed)
            out_sub = np.transpose(out_sub, (1, 0, 2))
            out_sub = np.ascontiguousarray(out_sub)
            if out is None:
                out = out_sub
            else:
                out = np.concatenate((out, out_sub), 0)

            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s

        if self.layout == 'BSH':
            out = out.reshape(self.batch, self.max_seq, self.q_heads * self.embed)
        elif self.layout == 'TND':
            out = out.reshape(self.num_tokens, self.q_heads, self.embed)
        elif self.layout == 'TH':
            out = out.reshape(self.num_tokens, self.q_heads * self.embed)

        return out


class FlashAttentionScoreNet(Cell):
    def __init__(self, q_heads, tor=1.0, layout='BSH'):
        super().__init__()
        self.net = FlashAttentionScore(head_num=q_heads, scale_value=tor, input_layout=layout)

    def construct(self, query, key, value, real_shift, drop_mask, padding_mask, attn_mask, prefix,
                  q_seq_lens=None, batch_valid_length=None):
        output = self.net(query, key, value, real_shift, drop_mask, padding_mask, attn_mask, prefix,
                          q_seq_lens, batch_valid_length)[3]
        return output


def _test_internal_asd_flash_attention_score(parma_dict, in_layout, ms_dtype, is_dyn, enable_alibi, mode):
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"
    context.set_context(mode=mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    batch = parma_dict["batch"]
    q_heads = parma_dict["q_heads"]
    kv_heads = parma_dict["kv_heads"]
    max_seq_len = parma_dict["max_seq_len"]
    embed = parma_dict["embed"]
    tor = 1/math.sqrt(1.0 * embed)

    net = FlashAttentionScoreNet(q_heads, tor, in_layout)

    batch_seq_list = [[batch, max_seq_len]]
    if is_dyn:
        qkv_rank = len(in_layout) # 'TH'=2, 'BSH'=3, 'TND'=3
        dyn_shape = [None] * qkv_rank
        dyn_tensor = ms.Tensor(shape=dyn_shape, dtype=ms_dtype)
        dyn_seq = ms.Tensor(shape=(None), dtype=ms.int32)
        dyn_mask = ms.Tensor(shape=(None, None, None, None), dtype=ms_dtype) if in_layout == "BSH" else ms.Tensor(
            shape=(None, None), dtype=ms_dtype)  # if not enable_alibi else None
        dyn_alibi = ms.Tensor(shape=(None, None, None, None), dtype=ms_dtype) if enable_alibi else None
        net.set_inputs(dyn_tensor, dyn_tensor, dyn_tensor, dyn_alibi, None, None, dyn_mask, None, dyn_seq, dyn_seq)
        batch_seq_list = [
            [batch, max_seq_len],
            [batch + 1, max_seq_len + 256],
            [batch - 1, max_seq_len + 128]]

    for batch, max_seq in batch_seq_list:
        assert batch > 0 and max_seq > 0, (
            "batch and max_seq_len should > 0 but got " + str(batch) + " and " + str(max_seq)
        )
        test_fa = TestFlashAttentionScore()
        test_fa.set_data_params(batch, max_seq, q_heads, kv_heads, embed,
                                need_pad=True, layout=in_layout, input_type=ms_dtype, enable_alibi=enable_alibi)
        q, k, v, amask, q_seqlen, alibi = test_fa.gen_input_data()
        expect = test_fa.calc_expect_func()

        alibi_tensor = ms.Tensor(alibi).astype(ms_dtype) if enable_alibi else None
        amask_tensor = ms.Tensor(amask).astype(ms_dtype)# if not enable_alibi else None
        output = net(ms.Tensor(q).astype(ms_dtype),
                     ms.Tensor(k).astype(ms_dtype),
                     ms.Tensor(v).astype(ms_dtype),
                     alibi_tensor, None, None, amask_tensor, None,
                     ms.Tensor(q_seqlen).astype(ms.int32),
                     ms.Tensor(q_seqlen).astype(ms.int32))

        output = output.to(ms.float32)
        output = output.reshape(-1).numpy()
        expect = expect.reshape(-1)

        count = 0
        for _, (output_, expect_) in enumerate(zip(output, expect)):
            if np.abs(output_ - expect_) > np.abs(expect_) * 0.05:
                count = count + 1

        err_ratio = count / len(output)
        assert err_ratio < 0.05, "err_ratio is " + str(err_ratio)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ms_dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('enable_alibi', [False, True])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_internal_asd_flash_attention_score_bsh(ms_dtype, enable_alibi, mode):
    """
    Feature: test internal flash attention score
    Description: test internal flash attention score in BSH layout
    Expectation: the result is correct
    """
    param_dict = {
        "batch": 2,
        "q_heads": 11,
        "kv_heads": 1,
        "max_seq_len": 1024,
        "embed": 128
    }
    in_layout = "BSH"
    is_dyn = False
    _test_internal_asd_flash_attention_score(param_dict, in_layout, ms_dtype, is_dyn, enable_alibi, mode)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('enable_alibi', [False, True])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_internal_asd_flash_attention_score_bsh_bf16_dyn(enable_alibi, mode):
    """
    Feature: test internal flash attention score
    Description: test internal flash attention score in BSH layout with bfloat16 dynamic shape inputs
    Expectation: the result is correct
    """
    param_dict = {
        "batch": 2,
        "q_heads": 11,
        "kv_heads": 1,
        "max_seq_len": 1024,
        "embed": 128
    }
    in_layout = "BSH"
    ms_dtype = ms.bfloat16
    is_dyn = True
    _test_internal_asd_flash_attention_score(param_dict, in_layout, ms_dtype, is_dyn, enable_alibi, mode)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ms_dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('enable_alibi', [False, True])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_internal_asd_flash_attention_score_th(ms_dtype, enable_alibi, mode):
    """
    Feature: test internal flash attention score
    Description: test internal flash attention score in TH layout
    Expectation: the result is correct
    """
    param_dict = {
        "batch": 4,
        "q_heads": 40,
        "kv_heads": 40,
        "max_seq_len": 20,
        "embed": 128
    }
    in_layout = "TH"
    is_dyn = False
    _test_internal_asd_flash_attention_score(param_dict, in_layout, ms_dtype, is_dyn, enable_alibi, mode)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('enable_alibi', [False, True])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_internal_asd_flash_attention_score_th_fp16_dyn(enable_alibi, mode):
    """
    Feature: test internal flash attention score
    Description: test internal flash attention score in TH layout with float16 dynamic shape inputs
    Expectation: the result is correct
    """
    param_dict = {
        "batch": 4,
        "q_heads": 40,
        "kv_heads": 40,
        "max_seq_len": 20,
        "embed": 128
    }
    in_layout = "TH"
    ms_dtype = ms.float16
    is_dyn = True
    _test_internal_asd_flash_attention_score(param_dict, in_layout, ms_dtype, is_dyn, enable_alibi, mode)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ms_dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('enable_alibi', [False, True])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_internal_asd_flash_attention_score_tnd(ms_dtype, enable_alibi, mode):
    """
    Feature: test internal flash attention score
    Description: test internal flash attention score in TND layout
    Expectation: the result is correct
    """
    param_dict = {
        "batch": 4,
        "q_heads": 40,
        "kv_heads": 40,
        "max_seq_len": 20,
        "embed": 128
    }
    in_layout = "TND"
    is_dyn = False
    _test_internal_asd_flash_attention_score(param_dict, in_layout, ms_dtype, is_dyn, enable_alibi, mode)
