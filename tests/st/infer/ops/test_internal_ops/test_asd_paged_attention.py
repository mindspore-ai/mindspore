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
"""
Tests for asd paged attention operator.
"""
import os
import random
import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Profiler
from tests.mark_utils import arg_mark
from .test_paged_attention import PagedAttentionNet, PagedAttentionTest, QuantMethod


class PagedInputData:
    def __init__(self, query, key_cache, value_cache, block_tables, batch_valid_length):
        self.query = query
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_tables = block_tables
        self.batch_valid_length = batch_valid_length


def group_matmul(head, kv_head, A, B):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        group_score = np.matmul(A[i * group_num: (i + 1) * group_num, :, :].astype(np.float32),
                                B[i: (i + 1), :, :].astype(np.float32)).astype(np.float16)
        if score is None:
            score = group_score
        else:
            score = np.concatenate((score, group_score), 0)
    return score


def ref_masked_attention(query, key, value, scale: float):
    query = query * scale
    query = np.transpose(query, (1, 0, 2))
    key = np.transpose(key, (1, 2, 0))
    sim = group_matmul(query.shape[0], key.shape[0], query, key)

    row_max = np.max(sim, axis=-1, keepdims=True)
    sim -= row_max
    sim = sim.astype("float32")
    sim = np.exp(sim)
    row_sum = np.sum(sim, axis=-1, keepdims=True)
    p = sim / row_sum
    p = p.astype("float16")

    value = np.transpose(value, (1, 0, 2))
    out = group_matmul(query.shape[0], key.shape[0], p, value)
    out = np.transpose(out, (1, 0, 2))
    return out


def ref_single_query_cached_kv_attention(output, paged_input):
    query = paged_input.query
    key_cache = paged_input.key_cache
    value_cache = paged_input.value_cache
    block_tables = paged_input.block_tables
    batch_valid_length = paged_input.batch_valid_length
    num_heads = query.shape[1]
    kv_heads = value_cache.shape[2]
    head_size = value_cache.shape[3]
    block_size = value_cache.shape[1]

    num_input_tokens = query.shape[0]
    for i in range(num_input_tokens):
        q = np.expand_dims(query[i], 0)
        block_table = block_tables[i]
        context_len = int(batch_valid_length[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, block_offset, :, :]
            k = k.reshape(kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, block_offset, :, :]
            v = v.reshape(kv_heads, head_size)
            values.append(v)
        keys = np.stack(np.array(keys), axis=0)
        values = np.stack(np.array(values), axis=0)
        scale = 1.0 / (head_size**0.5)

        out = ref_masked_attention(q, keys, values, scale)

        out = out.reshape(num_heads, head_size)
        output[i] = out


def generate_data(
        batch_size=4,
        seq_len=1,
        n_head=11,
        kv_heads=1,
        head_dim=128,
        block_size=1024,
        max_seq_len=1024
):
    np.random.seed(0)
    random.seed(a=0)
    num_tokens = batch_size * seq_len
    num_blocks = max_seq_len * batch_size // block_size
    dtype = "float32"
    query = np.random.uniform(-1.0, 1.0,
                              size=(num_tokens, n_head, head_dim)).astype(dtype)

    key_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks,
                                  block_size, kv_heads, head_dim)).astype(dtype)
    value_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks,
                                    block_size, kv_heads, head_dim)).astype(dtype)

    batch_valid_length = [random.randint(
        1, max_seq_len) for _ in range(num_tokens)]
    max_context_len = max_seq_len

    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for i in range(num_tokens):
        n_block = (batch_valid_length[i] + block_size - 1) // block_size
        n_pad_block = max_num_blocks_per_seq - n_block
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(n_block)
        ]
        if n_pad_block != 0:
            block_table = block_table + ([0] * n_pad_block)
        block_tables.append(block_table)

    batch_valid_length = np.array(batch_valid_length).astype(np.uint64)
    block_tables = np.array(block_tables).astype(np.uint64)

    paged_input = PagedInputData(
        query, key_cache, value_cache, block_tables, batch_valid_length)
    ref_output = np.zeros_like(query)

    ref_single_query_cached_kv_attention(
        ref_output,
        paged_input
    )

    return paged_input, ref_output


class AsdPagedAttentionTest(PagedAttentionTest):
    def __init__(self, ipt: dict, mode="gen"):  # pylint: disable=super-init-not-called
        # Note: This class overrides __init__ completely and does not call
        # super().__init__() because it has different initialization logic.
        '''
        ipt: input of one testcase or dict of testcases
        '''
        os.environ["MS_ENABLE_INTERNAL_BOOST"] = "off"
        self.mode = mode

        self.names = []  # testcases name list
        self.i_test_dict = {}
        self.check_dynamic_shape(ipt)

        self.net = None
        self.npu_init(PagedAttentionNet)

        self.i_test = self.i_test_dict[self.names[0]]
        if self.i_test["is_alibi"]:
            raise Exception(
                "[Error] alibi_mask can only be in PagedAttentionTest or PagedAttentionMaskTest")
        self.set_inputs()

        # construct
        np.random.seed(0)  # lock Skv
        for k in self.names:
            self.i_test = self.i_test_dict[k]
            self.gen = {}
            self.load = {}
            self.i_construct = {}
            self.o_ascend = {}

            self.calc_expect_func()
            self.calc_actual_func()
            self.compare()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=386.00)
@pytest.mark.parametrize("ms_dtype", [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('is_dyn', [False, True])
def test_internel_asd_paged_attention(enable_asd_env_fixture, ms_dtype, is_dyn):
    """
    Feature: test internal paged attention
    Description: test internal paged attention
    Expectation: the result is correct
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    batch_size = 4
    seq_len = 1
    num_head = 32
    head_dim = 128
    kv_head = 32
    max_seq_len = 1024
    block_size = 16
    scale_value = 1.0 / (head_dim ** 0.5)

    net = PagedAttentionNet(num_head, scale_value, kv_head)

    batch_seq_list = [[batch_size, max_seq_len]]
    if is_dyn:
        dyn_q = ms.Tensor(shape=(None, None, None), dtype=ms_dtype)
        dyn_kv = ms.Tensor(shape=(None, None, None, None), dtype=ms_dtype)
        dyn_block_tables = ms.Tensor(shape=(None, None), dtype=ms.int32)
        dyn_context_lens = ms.Tensor(shape=(None,), dtype=ms.int32)
        net.set_inputs(dyn_q, dyn_kv, dyn_kv,
                       dyn_block_tables, dyn_context_lens)
        batch_seq_list = [
            [batch_size, max_seq_len],
            [batch_size + 1, max_seq_len + 256],
            [batch_size - 1, max_seq_len + 128]]

    for batch_size, max_seq_len in batch_seq_list:
        paged_attention_input, expect = generate_data(
            batch_size, seq_len, num_head, kv_head, head_dim, block_size, max_seq_len)

        query = ms.Tensor(paged_attention_input.query.reshape(
            batch_size, 1, num_head * head_dim), ms_dtype)
        key_cache = ms.Tensor(paged_attention_input.key_cache, ms_dtype)
        value_cache = ms.Tensor(paged_attention_input.value_cache, ms_dtype)
        block_tables = ms.Tensor(paged_attention_input.block_tables, ms.int32)
        batch_valid_length = ms.Tensor(
            paged_attention_input.batch_valid_length, ms.int32)

        # enable profiling and skip comparison if repeated more than once
        repeat = 1
        if len(batch_seq_list) == 1 and repeat > 1:
            profiler = Profiler(
                start_profile=False, output_path="./profiler", data_simplification=False)
            profiler.start()
            for _ in range(repeat):
                output = net(query, key_cache, value_cache,
                             block_tables, batch_valid_length)
            profiler.stop()
            profiler.analyse()
            return

        output = net(query, key_cache, value_cache,
                     block_tables, batch_valid_length)
        output = output.asnumpy()

        expect = expect.reshape(-1)
        output = output.astype(np.float32).reshape(-1)

        count = 0
        for _, (output_, expect_) in enumerate(zip(output, expect)):
            if np.abs(output_ - expect_) > np.abs(expect_) * 0.05:
                count = count + 1

        err_ratio = count / len(output)
        assert err_ratio < 0.05


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=332.00)
def test_asd_paged_attention_batch_16_seq_1024_emb_128(enable_asd_env_fixture):
    """
    Feature: test asd paged attention operator
    Description: test asd paged attention with batch_size=16,
                 max_seq_len=1024, embedding_dim=128, float16 dtype, BSH layout
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 16,
        "sq": 1,
        "skv": "max",
        "max_skv": 1024,
        "nq": 16,
        "nkv": 16,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": False,
        "is_amask": False,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    AsdPagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=304.00)
def test_asd_paged_attention_batch_100_seq_128_emb_128(enable_asd_env_fixture):
    """
    Feature: test asd paged attention operator
    Description: test asd paged attention with batch_size=100,
                 max_seq_len=128, embedding_dim=128, float16 dtype, BSH layout
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 100,
        "sq": 1,
        "skv": "max",
        "max_skv": 128,
        "nq": 16,
        "nkv": 16,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": False,
        "is_amask": False,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    AsdPagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=194.00)
@pytest.mark.parametrize('is_amask', [True])
def test_asd_paged_attention_bsh_lookahead(enable_asd_env_fixture, is_amask):
    """
    Feature: test asd paged attention operator with BSH layout lookahead
    Description: test asd paged attention with BSH layout, lookahead mode,
                 dynamic sequence length (sq=16 and sq=32), with attention mask
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test_dict = {}

    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 1,
        "sq": 16,
        "skv": "max",
        "max_skv": 8192,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": False,
        "is_amask": is_amask,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    i_test_dict["dynamic_shape0"] = i_test

    i_test = i_test.copy()
    i_test["sq"] = 32
    i_test_dict["dynamic_shape1"] = i_test

    AsdPagedAttentionTest(i_test_dict)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=170.00)
def test_asd_paged_attention_th_lookahead(enable_asd_env_fixture):
    """
    Feature: test asd paged attention operator with TH layout lookahead
    Description: test asd paged attention with TH layout, lookahead mode, random sequence length, with attention mask
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "type": "float16",
        "layout": "TH",
        "b": 2,
        "sq": 16,
        "skv": "rand-1",
        "max_skv": 1024,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": False,
        "is_amask": True,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    AsdPagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=196.00)
def test_asd_paged_attention_fake_lookahead(enable_asd_env_fixture):
    """
    Feature: test asd paged attention operator with fake lookahead
    Description: test asd paged attention with fake lookahead mode,
                 dynamic sequence length (sq=1 and sq=16), random sequence length,
                 with attention mask
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')

    i_test_dict = {}
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 8,
        "sq": 1,
        "skv": "rand-1",
        "max_skv": 1024,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": False,
        "is_amask": True,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }

    i_test_dict["dynamic_shape0"] = i_test

    i_test = i_test.copy()
    i_test["sq"] = 16
    i_test_dict["dynamic_shape1"] = i_test

    AsdPagedAttentionTest(i_test_dict)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=226.00)
def test_paged_attention_asd_quant(enable_asd_env_fixture):
    """
    Feature: test asd paged attention operator with quantization
    Description: test asd paged attention with quantization mode enabled, INT_CUBE quantization method, float16 dtype
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 8,
        "sq": 1,
        "skv": 1024,
        "max_skv": 4096,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "quant_mode": 1,
        "is_alibi": False,
        "is_amask": False,
        "blk_sparse": 0,
        "quant_method": QuantMethod.INT_CUBE,
        "msprof": int(env_var) if env_var else 0
    }
    PagedAttentionTest(i_test)  # ??? why not AsdPagedAttentionTest


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=278.00)
def test_paged_attention_asd_mla_combine_cache_norm(enable_asd_env_fixture):
    """
    Feature: test asd paged attention operator with MLA combined cache normalization
    Description: test asd paged attention with MLA (Multi-head Latent Attention),
                 combined KV cache, normalization, float32 dtype, with attention mask
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "type": "float32",
        "layout": "BSH",
        "b": 4,
        "sq": 1,
        "skv": 1024,
        "max_skv": 1024,
        "nq": 4,
        "nkv": 1,
        "d": 576,  # head_size_qk
        "d_vo": 512,  # head_size_vo
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": False,
        "is_amask": True,
        "mla_kvcombined": True,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    AsdPagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=278.00)
def test_paged_attention_asd_mla_combine_cache_norm2(enable_asd_env_fixture):
    """
    Feature: test asd paged attention operator with MLA combined cache normalization
    Description: test asd paged attention with MLA (Multi-head Latent Attention),
                 combined KV cache, normalization, float16 dtype, with attention mask
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 4,
        "sq": 1,
        "skv": 1024,
        "max_skv": 1024,
        "nq": 4,
        "nkv": 1,
        "d": 576,  # head_size_qk
        "d_vo": 512,  # head_size_vo
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": False,
        "is_amask": True,
        "mla_kvcombined": True,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    AsdPagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=272.00)
@pytest.mark.parametrize('sq', [1, 2])
def test_paged_attention_asd_mla_fp16(enable_asd_env_fixture, sq):
    """
    Feature: test asd paged attention operator with MLA in float16
    Description: test asd paged attention with MLA (Multi-head Latent Attention),
                 combined KV cache, float16 dtype, with attention mask,
                 parameterized sequence length
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 1,
        "sq": sq,
        "skv": 512,
        "max_skv": 1024,
        "nq": 64,
        "nkv": 1,
        "d": 576,  # head_size_qk
        "d_vo": 512,  # head_size_vo
        "blk_sz": 32,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": False,
        "is_amask": True,
        "mla_kvcombined": True,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    AsdPagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=272.01)
@pytest.mark.parametrize('sq', [1, 2])
def test_paged_attention_asd_mla_fp32(enable_asd_env_fixture, sq):
    """
    Feature: test asd paged attention operator with MLA in float32
    Description: test asd paged attention with MLA (Multi-head Latent Attention),
                 combined KV cache, float32 dtype, with attention mask,
                 parameterized sequence length
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "type": "float32",
        "layout": "BSH",
        "b": 1,
        "sq": sq,
        "skv": 256,
        "max_skv": 1024,
        "nq": 16,
        "nkv": 1,
        "d": 576,  # head_size_qk
        "d_vo": 512,  # head_size_vo
        "blk_sz": 16,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": False,
        "is_amask": True,
        "mla_kvcombined": True,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    AsdPagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=278.00)
def test_paged_attention_asd_mla_combine_cache_norm_dyn(enable_asd_env_fixture):
    """
    Feature: test asd paged attention operator with MLA combined cache normalization in dynamic shape
    Description: test asd paged attention with MLA (Multi-head Latent Attention),
                 combined KV cache, normalization, float16 dtype,
                 dynamic sequence length (skv=1024 and skv=512), with attention mask
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test_dict = {}
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 4,
        "sq": 1,
        "skv": 1024,
        "max_skv": 1024,
        "nq": 4,
        "nkv": 1,
        "d": 576,  # head_size_qk
        "d_vo": 512,  # head_size_vo
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": False,
        "is_amask": True,
        "mla_kvcombined": True,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }

    i_test_dict["dynamic_shape0"] = i_test

    i_test = i_test.copy()
    i_test["skv"] = 512
    i_test_dict["dynamic_shape1"] = i_test

    AsdPagedAttentionTest(i_test_dict)
