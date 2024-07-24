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

import mindspore as ms
from mindspore.ops.operations.nn_ops import PagedAttention, PagedAttentionMask
from mindspore.nn import Cell
from mindspore import context


class PagedAttentionNet(Cell):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = PagedAttention(*args, **kwargs)

    def construct(self, *args, **kwargs):
        return self.net(*args, **kwargs)


class PagedAttentionMaskNet(Cell):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = PagedAttentionMask(*args, **kwargs)

    def construct(self, *args, **kwargs):
        return self.net(*args, **kwargs)


class PagedAttentionTest:
    def __init__(self, i_test: dict):
        self.i_test = i_test
        self.i_construct = None
        self.i_init = None
        self.o_golden = None  # o_golden is the golden outputs to be compared
        self.o_ascend = None  # o_ascend is the outputs from Ascend
        self.calc_expect_func(**i_test)
        self.calc_actual_func()
        self.compare()

    def calc_expect_func(self, input_type, layout: str, batch: int, q_heads: int,
                         kv_heads: int, max_seq: int, embed: int, q_seqlen: int,
                         block_size=16, drop_prop=0.0, low_tri=False,
                         is_amask=False, is_alibi=False, rand_kv_seqlen=False):
        num_groups = q_heads // kv_heads
        bs = batch * max_seq
        assert bs % block_size == 0
        blk_cnt = bs // block_size
        assert 0.0 <= drop_prop <= 1.0
        is_dropout = bool(drop_prop)
        if input_type in {"float32", np.float32}:
            ms_type = ms.bfloat16
        elif input_type in {"float16", np.float16}:
            ms_type = ms.float16
        else:
            raise Exception("Wrong input_type = %s" % input_type)
        if is_alibi:
            self.i_construct = {
                "query": None,  # dtype: tensor
                "key_cache": None,  # dtype: tensor
                "value_cache": None,  # dtype: tensor
                "block_tables": None,  # dtype: tensor
                "context_lens": None,  # dtype: tensor
                "alibi_mask": None  # dtype: tensor
            }
        else:
            self.i_construct = {
                "query": None,  # dtype: tensor
                "key_cache": None,  # dtype: tensor
                "value_cache": None,  # dtype: tensor
                "block_tables": None,  # dtype: tensor
                "context_lens": None  # dtype: tensor
            }
        self.i_init = {
            "head_num": q_heads,  # dtype: int
            "scale_value": 1 / math.sqrt(float(embed)),  # dtype: float
            "kv_head_num": kv_heads  # dtype: int
        }
        self.o_golden = {
            "attention_out": None,  # dtype: tensor
        }
        np.random.seed(0)
        if rand_kv_seqlen:
            randnum = np.random.randint(1, max_seq, 1)
            kv_seqlen = np.ones((batch,)) * randnum
            kv_seqlen = kv_seqlen.astype(np.int32)
        else:
            kv_seqlen = np.ones((batch,)) * max_seq
            kv_seqlen = kv_seqlen.astype(np.int32)
        q = np.random.uniform(-1.0, 1.0, size=(batch, q_heads, q_seqlen, embed)).astype(np.float16).astype(np.float32)
        k = np.random.uniform(-1.0, 1.0, size=(batch, kv_heads, max_seq, embed)).astype(np.float16).astype(np.float32)
        v = np.random.uniform(-1.0, 1.0, size=(batch, kv_heads, max_seq, embed)).astype(np.float16).astype(np.float32)
        if is_alibi:
            alibi = np.random.uniform(-10.0, 0.0, size=(
                batch, q_heads, q_seqlen, max_seq)).astype(np.float16).astype(np.float32)

        dropout_uint8 = np.zeros(batch * q_seqlen * max_seq >> 3).astype(np.uint8)
        amask = np.ones(shape=(batch, q_seqlen, max_seq)).astype(np.float16).astype(
            np.float32)
        amask = np.triu(amask, 1)  # 下三角
        dmask = None
        if is_dropout:
            dmask = np.random.uniform(size=(batch * max_seq * max_seq)) > drop_prop
            masks = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80)
            j = 0
            for i in range(batch * q_seqlen * max_seq >> 3):
                for m in masks:
                    if dmask[j]:
                        dropout_uint8[i] = dropout_uint8[i] | m
                    j += 1
            dmask = dmask.reshape(batch, q_seqlen, max_seq)

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
                q_s = q_seqlen
                q_slice = q[idx, hidx, :, :]
                k_slice = k[idx, kv_hidx, 0:kv_seqlen[0], :]
                k_slice_t = np.transpose(k_slice, (1, 0))  # get K^T
                v_slice = v[idx, kv_hidx, 0:kv_seqlen[0], :]
                if is_dropout:
                    dmask_slice = dmask[idx, :, :]
                if low_tri | is_amask:
                    amask_slice = amask[idx, :, :]
                score = np.matmul(q_slice, k_slice_t)
                if s is None:
                    s = score.reshape([-1,])
                else:
                    s = np.concatenate((s, score.reshape([-1,])), 0)

                tor = np.float16(math.sqrt(1.0 * embed))
                score = score / tor
                if is_alibi:
                    alibi_slice = alibi[idx, hidx, :, :kv_seqlen[0]]
                    score = score + alibi_slice
                if low_tri | is_amask:
                    for i in range(q_s):
                        score[i][:] = score[i][:] - amask_slice[i][:] * 10000
                score_max = np.max(score, axis=-1)
                if _score_max is None:
                    _score_max = score_max.reshape([-1,])
                else:
                    _score_max = np.concatenate((_score_max, score_max.reshape([-1,])), 0)
                score = score - score_max.reshape((q_s, 1))
                score_exp = np.exp(score)

                if is_dropout:
                    score_exp = score_exp * dmask_slice / (1 - drop_prop)

                score_sum = np.sum(score_exp, axis=-1)
                if _score_sum is None:
                    _score_sum = score_sum.reshape([-1,])
                else:
                    _score_sum = np.concatenate((_score_sum, score_sum.reshape([-1,])), 0)

                p = score_exp / score_sum.reshape((q_s, 1))

                if _p is None:
                    _p = p.reshape([-1,])
                else:
                    _p = np.concatenate((_p, p.reshape([-1,])), 0)

                o = np.matmul(p, v_slice)
                o = o.reshape(q_s, embed)
                o = np.ascontiguousarray(o)

                if out is None:
                    out = o
                else:
                    out = np.concatenate((out, o), 0)

                q_offset += q_s
                k_offset += max_seq
                v_offset += max_seq

        table = np.zeros(shape=(batch, max_seq // block_size))
        for bidx in range(batch):
            for sidx in range(max_seq // block_size):
                table[bidx][sidx] = bidx * max_seq // block_size \
                                    + sidx
        self.i_construct["block_tables"] = ms.Tensor(table.astype(np.int32))

        if layout == "BSH":  # BNSD to BSH of o
            q_bsh = np.zeros((batch, q_seqlen, q_heads * embed))
            k_bsh = np.zeros((batch, max_seq, kv_heads * embed))
            v_bsh = np.zeros((batch, max_seq, kv_heads * embed))
            o_bsh = np.zeros((batch, q_seqlen, q_heads * embed))
            out = out.reshape(batch, q_heads, q_seqlen, embed)

            for bidx in range(batch):
                for sidx in range(q_seqlen):
                    for hidx in range(embed * q_heads):
                        q_bsh[bidx][sidx][hidx] = q[bidx][hidx // embed][sidx][hidx % embed]
                        o_bsh[bidx][sidx][hidx] = out[bidx][hidx // embed][sidx][hidx % embed]
                for sidx in range(max_seq):
                    for hidx in range(embed * kv_heads):
                        k_bsh[bidx][sidx][hidx] = k[bidx][hidx // embed][sidx][hidx % embed]
                        v_bsh[bidx][sidx][hidx] = v[bidx][hidx // embed][sidx][hidx % embed]
            self.i_construct["query"] = ms.Tensor(q_bsh).astype(ms_type)
            self.o_golden["attention_out"] = o_bsh.astype(input_type)
        else:  # BNSD of o
            out = out.reshape(batch, q_heads, q_seqlen, embed)
            self.o_golden["attention_out"] = out.astype(input_type)
            self.i_construct["query"] = ms.Tensor(q).astype(ms_type)
            k_bsh = np.zeros((batch, max_seq, kv_heads * embed))
            v_bsh = np.zeros((batch, max_seq, kv_heads * embed))
            for bidx in range(batch):
                for sidx in range(max_seq):
                    for hidx in range(embed * kv_heads):
                        k_bsh[bidx][sidx][hidx] = k[bidx][hidx // embed][sidx][
                            hidx % embed]
                        v_bsh[bidx][sidx][hidx] = v[bidx][hidx // embed][sidx][
                            hidx % embed]
        k_bsh = k_bsh.reshape(blk_cnt, block_size, kv_heads, embed)
        v_bsh = v_bsh.reshape(blk_cnt, block_size, kv_heads, embed)
        self.i_construct["key_cache"] = ms.Tensor(k_bsh).astype(ms_type)
        self.i_construct["value_cache"] = ms.Tensor(v_bsh).astype(ms_type)

        if is_alibi:
            self.i_construct["alibi_mask"] = ms.Tensor(alibi).astype(ms_type)

        self.i_construct["context_lens"] = ms.Tensor(kv_seqlen.astype(np.int32))

    def calc_actual_func(self):
        if "ASCEND_HOME_PATH" not in os.environ:
            os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
        if "is_alibi" in self.i_test.keys() and self.i_test["is_alibi"]:
            net = PagedAttentionMaskNet(**self.i_init)
        else:
            net = PagedAttentionNet(**self.i_init)

        self.o_ascend = net(*tuple(self.i_construct.values()))

    def compare(self):
        actual = self.o_ascend
        if self.i_test["input_type"] in {"float32", np.float32}:
            actual = actual.to(ms.float32)
        actual = actual.numpy()
        expect = self.o_golden["attention_out"]
        assert actual.shape == expect.shape
        actual = actual.flatten()
        expect = expect.flatten()
        data = [actual, expect]
        nan_inf = [None, None]
        for i in range(2):
            nan_inf[i] = np.isnan(data[i]) + np.isinf(data[i])
            nan_inf[i] = np.sum(nan_inf[i])
        if nan_inf[0] or nan_inf[1]:
            print("nan and inf counts of actual is %d" % nan_inf[0])
            print("nan and inf counts of expect is %d" % nan_inf[1])
            raise Exception("Nan Inf Error")
        err_ratio = 0.05
        err_gate = np.abs(expect) * err_ratio
        diff = np.abs(data[0] - data[1])
        err_cnt = int(np.sum(diff > err_gate))
        if err_cnt > expect.shape[0] * err_ratio:
            raise Exception("err_ratio = err_cnt / all = %d / %d > %f" % (
                err_cnt, expect.shape[0], err_ratio))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_bnsd():
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op in BNSD.
    Expectation: the result is correct
    """
    i_test = {
        "input_type": "float32",
        "layout": "BNSD",
        "batch": 2,
        "q_heads": 9,
        "kv_heads": 3,
        "max_seq": 512,
        "embed": 128,
        "q_seqlen": 1,
        "block_size": 16,
        "drop_prop": 0.0,
        "low_tri": False,
        "is_amask": False,
        "is_alibi": False,
        "rand_kv_seqlen": False
    }
    PagedAttentionTest(i_test)

@pytest.mark.skip
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_bsh_256():
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op in BSH.
    Expectation: the result is correct
    """
    i_test = {
        "input_type": "float16",
        "layout": "BSH",
        "batch": 2,
        "q_heads": 9,
        "kv_heads": 3,
        "max_seq": 1024,
        "embed": 256,
        "q_seqlen": 1,
        "block_size": 16,
        "drop_prop": 0.0,
        "low_tri": False,
        "is_amask": False,
        "is_alibi": True,
        "rand_kv_seqlen": True
    }
    PagedAttentionTest(i_test)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_longseq():
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op with long sequence.
    Expectation: the result is correct
    """
    i_test = {
        "input_type": "float16",
        "layout": "BSH",
        "batch": 1,
        "q_heads": 14,
        "kv_heads": 2,
        "max_seq": 8192,
        "embed": 128,
        "q_seqlen": 1,
        "block_size": 128,
        "drop_prop": 0.0,
        "low_tri": False,
        "is_amask": False,
        "is_alibi": False,
        "rand_kv_seqlen": False
    }
    PagedAttentionTest(i_test)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_rand0():
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op with random sequence.
    Expectation: the result is correct
    """
    i_test = {
        "input_type": "float16",
        "layout": "BSH",
        "batch": 1,
        "q_heads": 14,
        "kv_heads": 2,
        "max_seq": 8192,
        "embed": 128,
        "q_seqlen": 1,
        "block_size": 128,
        "drop_prop": 0.0,
        "low_tri": False,
        "is_amask": False,
        "is_alibi": False,
        "rand_kv_seqlen": True
    }
    PagedAttentionTest(i_test)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_fd_long():
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op with extra long seq.
    Expectation: the result is correct
    """
    i_test = {
        "input_type": "float16",
        "layout": "BSH",
        "batch": 1,
        "q_heads": 2,
        "kv_heads": 1,
        "max_seq": 20032,
        "embed": 128,
        "q_seqlen": 1,
        "block_size": 64,
        "drop_prop": 0.0,
        "low_tri": False,
        "is_amask": False,
        "is_alibi": False,
        "rand_kv_seqlen": False
    }
    PagedAttentionTest(i_test)

@pytest.mark.skip
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_fd_bsh_alibi():
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op with alibi_mask.
    Expectation: the result is correct
    """
    i_test = {
        "input_type": "float16",
        "layout": "BSH",
        "batch": 2,
        "q_heads": 9,
        "kv_heads": 3,
        "max_seq": 6144,
        "embed": 128,
        "q_seqlen": 1,
        "block_size": 128,
        "drop_prop": 0.0,
        "low_tri": False,
        "is_amask": False,
        "is_alibi": True,
        "rand_kv_seqlen": False
    }
    PagedAttentionTest(i_test)
