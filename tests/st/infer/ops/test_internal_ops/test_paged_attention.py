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
Test paged attention operator.
"""
import os
import csv
import numpy as np
import pytest

import mindspore as ms
from mindspore.ops.operations.nn_ops import PagedAttention
from mindspore.nn import Cell
from mindspore import Profiler, context
from .paged_attention_base import PagedAttentionBase, QuantMethod

from tests.mark_utils import arg_mark


class PagedAttentionNet(Cell):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = PagedAttention(*args, **kwargs)

    def construct(
            self,
            query,
            key_cache,
            value_cache,
            block_tables,
            batch_valid_length,
            antiquant_scale=None,
            antiquant_offset=None,
            attn_mask=None,
            q_seq_lens=None,
            alibi_mask=None):
        return self.net(
            query,
            key_cache,
            value_cache,
            block_tables,
            batch_valid_length,
            antiquant_scale,
            antiquant_offset,
            attn_mask,
            q_seq_lens,
            alibi_mask)


def to_pad_fp32(arr, dim=2):
    original_shape = arr.shape
    new_shape = original_shape[:dim] + \
        (original_shape[dim] * 2,) + original_shape[dim + 1:]
    float32_array = np.frombuffer(arr.tobytes(), dtype=np.float32)
    float32_array = float32_array.reshape(new_shape)
    return float32_array


class PagedAttentionTest(PagedAttentionBase):
    def __init__(self, ipt: dict, mode="gen"):
        '''
        ipt: input of one testcase or dict of testcases
        '''
        self.mode = mode
        self.ctx_mode = ipt.get('ctx_mode', context.GRAPH_MODE)

        self.names = []  # testcases name list
        self.i_test_dict = {}
        self.check_dynamic_shape(ipt)

        self.net = None
        self.npu_init(PagedAttentionNet)

        self.i_test = self.i_test_dict[self.names[0]]
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

    def set_inputs(self):
        if self.i_test["type"] == "float32":
            q_type = ms.bfloat16
        else:  # float16
            q_type = ms.float16
        if self.i_test["quant_mode"]:
            kv_type = ms.int8
        else:  # not quant
            kv_type = q_type

        self.i_construct = {
            "query": None,
            "key_cache": None,
            "value_cache": None,
            "block_tables": None,
            "batch_valid_length": None,
            "antiquant_scale": None,
            "antiquant_offset": None,
            "attn_mask": None,
            "q_seq_lens": None,
            "alibi_mask": None
        }

        # query
        if self.i_test["layout"] == "BSH":
            query = [None, None, None]
        elif self.i_test["layout"] == "TH":
            query = [None, None]
        else:  # BNSD
            query = [None, None, None, None]
        self.i_construct["query"] = ms.Tensor(shape=query, dtype=q_type)
        # key_cache value_cache
        if self.i_test.get("fuse_key_cache", False):
            key_value = [None, None, None]
        else:
            key_value = [None, None, None, None]
        self.i_construct["key_cache"] = ms.Tensor(
            shape=key_value, dtype=kv_type)
        self.i_construct["value_cache"] = ms.Tensor(
            shape=key_value, dtype=kv_type)
        # block_tables
        self.i_construct["block_tables"] = ms.Tensor(
            shape=[None, None], dtype=ms.int32)
        # batch_valid_length
        self.i_construct["batch_valid_length"] = ms.Tensor(
            shape=[None], dtype=ms.int32)
        # quant
        if self.i_test["quant_mode"]:
            quant_method = self.i_test.get("quant_method", 0)
            if quant_method == QuantMethod.FP16_VEC:
                scale_type = ms.float16
                offset_type = ms.float16
            elif quant_method == QuantMethod.FP32_VEC:
                scale_type = ms.float32
                offset_type = ms.float32
                raise ValueError("quant_method does not support FP32_VEC yet.")
            elif quant_method == QuantMethod.INT_CUBE:
                if self.i_test.get("antiquant_scale_int64_to_fp32", False):
                    scale_type = ms.float32  # 在动态量化场景Pertoken，需要将scale_type配合to_pad_fp32使用。
                else:
                    scale_type = ms.int64
                offset_type = ms.int32
            else:
                raise ValueError(
                    "plz set quant_method properly when quant_mode > 0.")
            self.i_construct["antiquant_scale"] = ms.Tensor(
                shape=[None, None, None, None, None, None], dtype=scale_type)
            self.i_construct["antiquant_offset"] = ms.Tensor(
                shape=[None, None, None, None, None, None], dtype=offset_type)
        # alibi_mask
        if self.i_test["is_alibi"]:
            self.i_construct["alibi_mask"] = ms.Tensor(
                shape=[None, None, None, None], dtype=q_type)
        # attn_mask
        if self.i_test["is_amask"]:
            self.i_construct["attn_mask"] = ms.Tensor(
                shape=[None, None, None], dtype=q_type)
        # q_seq_lens
        self.i_construct["q_seq_lens"] = ms.Tensor(
            shape=[None], dtype=ms.int32)

        self.net.set_inputs(*tuple(self.i_construct.values()))

    def calc_actual_func(self):
        if self.i_test["type"] == "float32":
            q_type = ms.bfloat16
        else:  # float16
            q_type = ms.float16
        if self.i_test["quant_mode"]:
            kv_type = ms.int8
        else:  # not quant
            kv_type = q_type

        self.i_construct = {
            "query": None,
            "key_cache": None,
            "value_cache": None,
            "block_tables": None,
            "batch_valid_length": None,
            "antiquant_scale": None,
            "antiquant_offset": None,
            "attn_mask": None,
            "q_seq_lens": None,
            "alibi_mask": None
        }

        if self.mode == "load":
            self.i_construct["query"] = ms.Tensor(
                self.load["query"], dtype=q_type)
            self.i_construct["key_cache"] = ms.Tensor(
                self.load["key_cache"], dtype=kv_type)
            self.i_construct["value_cache"] = ms.Tensor(
                self.load["value_cache"], dtype=kv_type)
            self.i_construct["block_tables"] = ms.Tensor(
                self.load["block_tables"])
            self.i_construct["batch_valid_length"] = ms.Tensor(
                self.load["batch_valid_length"])
            self.o_ascend = {
                "attention_out": None,  # dtype: tensor
            }
            if os.environ.get("ENABLE_PROFILING") == "on":
                self.save_running_data()
            if "msprof" in self.i_test and self.i_test["msprof"]:
                profiler = Profiler(
                    start_profile=False,
                    output_path="./profiler",
                    data_simplification=False)
                profiler.start()
                for _ in range(self.i_test["msprof"]):
                    self.o_ascend["attention_out"] = self.net(
                        *tuple(self.i_construct.values()))
                profiler.stop()
                profiler.analyse()
            else:
                self.o_ascend["attention_out"] = self.net(
                    *tuple(self.i_construct.values()))
            return
        D = self.gen["query"].shape[-1]
        D_V = self.gen["value_cache"].shape[-1]
        # query
        if self.i_test["layout"] == "BSH":
            B, Sq, N, G, D = self.gen["query"].shape
            query = np.reshape(self.gen["query"], newshape=[B, Sq, N * G * D])
        elif self.i_test["layout"] == "TH":
            B, Sq, N, G, D = self.gen["query"].shape
            query = np.reshape(self.gen["query"], newshape=[B * Sq, N * G * D])
        else:  # BNSD
            B, N, G, Sq, D = self.gen["query"].shape
            query = np.reshape(self.gen["query"], newshape=[B, N * G, Sq, D])
        self.i_construct["query"] = ms.Tensor(query, dtype=q_type)
        # key_cache value_cache
        B, MaxSkv, N, _, D = self.gen["key_cache"].shape
        block_size = self.i_test["blk_sz"]
        block_num = B * MaxSkv // block_size
        key_cache = np.reshape(
            self.gen["key_cache"],
            newshape=[block_num, block_size, N, D])
        value_cache = np.reshape(
            self.gen["value_cache"],
            newshape=[block_num, block_size, N, D_V])
        self.i_construct["key_cache"] = ms.Tensor(key_cache, dtype=kv_type)
        if self.i_test.get("fuse_key_cache", False):
            self.i_construct["key_cache"] = self.i_construct["key_cache"].reshape(block_num, block_size, N * D)
        if self.i_test.get("mla_kvcombined", False):
            self.i_construct["value_cache"] = self.i_construct["key_cache"]
        else:
            self.i_construct["value_cache"] = ms.Tensor(
                value_cache, dtype=kv_type)
        # block_tables
        self.i_construct["block_tables"] = ms.Tensor(self.gen["block_tables"])
        # batch_valid_length
        self.i_construct["batch_valid_length"] = ms.Tensor(
            self.gen["batch_valid_length"])
        # quant
        if self.i_test["quant_mode"]:
            antiquant_scale = self.gen["antiquant_scale"]
            antiquant_offset = self.gen["antiquant_offset"]
            if self.i_test.get("antiquant_scale_int64_to_fp32", False):
                antiquant_scale = to_pad_fp32(antiquant_scale)
            self.i_construct["antiquant_scale"] = ms.Tensor(antiquant_scale)
            self.i_construct["antiquant_offset"] = ms.Tensor(antiquant_offset)
        # alibi_mask
        if self.i_test["is_alibi"]:
            B, N, G, Sq, MaxSkv = self.gen["alibi_mask"].shape
            alibi_mask = np.reshape(
                self.gen["alibi_mask"],
                newshape=[B, N * G, Sq, MaxSkv])
            self.i_construct["alibi_mask"] = ms.Tensor(
                alibi_mask, dtype=q_type)
        # attn_mask
        if self.i_test["is_amask"]:
            B, _, _, Sq, MaxSkv = self.gen["attn_mask"].shape
            attn_mask = np.reshape(
                self.gen["attn_mask"],
                newshape=[B, Sq, MaxSkv])
            self.i_construct["attn_mask"] = ms.Tensor(
                attn_mask, dtype=q_type)
        self.i_construct["q_seq_lens"] = ms.Tensor(self.gen["sq"])

        self.o_ascend = {
            "attention_out": None,  # dtype: tensor
        }
        if os.environ.get("ENABLE_PROFILING") == "on":
            self.save_running_data()
        if "msprof" in self.i_test and self.i_test["msprof"]:
            profiler = Profiler(
                start_profile=False,
                output_path="./profiler",
                data_simplification=False)
            profiler.start()
            for _ in range(self.i_test["msprof"]):
                self.o_ascend["attention_out"] = self.net(
                    *tuple(self.i_construct.values()))
            profiler.stop()
            profiler.analyse()
        else:
            self.o_ascend["attention_out"] = self.net(
                *tuple(self.i_construct.values()))

    def save_running_data(self):
        shape = {
            "query": self.i_construct["query"].shape,
            "key_cache": self.i_construct["key_cache"].shape,
            "value_cache": self.i_construct["value_cache"].shape,
            "block_tables": self.i_construct["block_tables"].shape,
            "batch_valid_length_shape": self.i_construct["batch_valid_length"].shape,
            "batch_valid_length_value": self.i_construct["batch_valid_length"],
            "antiquant_scale": self.i_construct["antiquant_scale"].shape if self.i_construct[
                "antiquant_scale"] is not None else None,
            "antiquant_offset": self.i_construct["antiquant_offset"].shape if self.i_construct[
                "antiquant_offset"] is not None else None,
            "attn_mask": self.i_construct["attn_mask"].shape if self.i_construct["attn_mask"] is not None else None,
            "q_seq_lens_shape": self.i_construct["q_seq_lens"].shape,
            "q_seq_lens_value": self.i_construct["q_seq_lens"]
        }
        key = [
            ','.join(map(str, self.i_construct["query"].shape)),
            ','.join(map(str, self.i_construct["key_cache"].shape)),
            ','.join(map(str, self.i_construct["value_cache"].shape)),
            ','.join(map(str, self.i_construct["batch_valid_length"].shape)),
            ','.join(map(str, self.i_construct["block_tables"].shape)),
            ','.join(map(str, self.i_construct["antiquant_scale"].shape))
            if self.i_construct["antiquant_scale"] is not None else "1",
            ','.join(map(str, self.i_construct["antiquant_offset"].shape))
            if self.i_construct["antiquant_offset"] is not None else "1",
            ','.join(map(str, self.i_construct["attn_mask"].shape))
            if self.i_construct["attn_mask"] is not None else "1"
        ]
        test_input = self.i_test_dict
        running_data = [shape, test_input, '\"' + ';'.join(key) + '\"']
        header = ["shape", "test_input", "key"]
        # 文件名
        csv_file = 'tmp_data/pa_tmp_data.csv'

        # 写入数据到 CSV 文件
        with open(csv_file, mode='a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # 检查文件是否为空，如果是，则写入表头
            if os.stat(csv_file).st_size == 0:
                writer.writerow(header)  # 写入表头

            # 写入数据
            writer.writerow(running_data)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=496.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_bnsd(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_bnsd
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test_dict = {}

    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float32",
        "layout": "BNSD",
        "b": 2,
        "sq": 1,
        "skv": "max",
        "max_skv": 512,
        "nq": 9,
        "nkv": 3,
        "d": 128,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    i_test_dict["dynamic_shape0"] = i_test

    i_test = i_test.copy()
    i_test["b"] = 3
    i_test_dict["dynamic_shape1"] = i_test

    PagedAttentionTest(i_test_dict)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=1392.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_fd_long(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_fd_long
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 1,
        "sq": 6,
        "skv": "max",
        "max_skv": 20032,
        "nq": 2,
        "nkv": 1,
        "d": 128,
        "blk_sz": 64,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=484.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_quant0(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant0
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 2,
        "sq": 1,
        "skv": "rand-b",
        "max_skv": 1024,
        "nq": 9,
        "nkv": 3,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 1,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": QuantMethod.FP16_VEC,
        "msprof": int(env_var) if env_var else 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=968.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_quant1(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant1
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 1,
        "sq": 1,
        "skv": 63,
        "max_skv": 8192,
        "nq": 32,
        "nkv": 32,
        "d": 128,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "quant_mode": 1,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": QuantMethod.FP16_VEC,
        "msprof": int(env_var) if env_var else 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=1608.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_quant_pertoken(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_pertoken
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float32",
        "layout": "BSH",
        "b": 1,
        "sq": 1,
        "skv": 63,
        "max_skv": 8192,
        "nq": 32,
        "nkv": 32,
        "d": 128,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "quant_mode": 1,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": QuantMethod.FP16_VEC,
        "msprof": int(env_var) if env_var else 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=936.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_lookahead0(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_lookahead0
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 1,
        "sq": 32,
        "skv": "max",
        "max_skv": 8192,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": 0,
        "is_amask": 1,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=492.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_lookahead1(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_lookahead1
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test_dict = {}

    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 2,
        "sq": 11,
        "skv": "rand-1",
        "max_skv": 1024,
        "nq": 9,
        "nkv": 3,
        "d": 256,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": 0,
        "is_amask": 1,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    i_test_dict["dynamic_shape0"] = i_test

    i_test = i_test.copy()
    i_test["b"] = 3
    i_test["sq"] = 12
    i_test_dict["dynamic_shape1"] = i_test

    PagedAttentionTest(i_test_dict)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=740.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_fake_lookahead(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_fake_lookahead
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 64,
        "sq": 1,
        "skv": "rand-1",
        "max_skv": 1024,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": 0,
        "is_amask": 1,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=522.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_large_gsq(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_large_gsq
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 2,
        "sq": 177,
        "skv": "rand-b",
        "max_skv": 512,
        "nq": 384,
        "nkv": 3,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": 0,
        "is_amask": 1,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=454.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('quant_method',
                         [QuantMethod.FP16_VEC, QuantMethod.INT_CUBE])
def test_paged_attention_quant_pertoken_bsh(quant_method, ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_pertoken_bsh
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 2,
        "sq": 1,
        "skv": 512,
        "max_skv": 512,
        "nq": 9,
        "nkv": 3,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 2,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": quant_method,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=484.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('quant_method',
                         [QuantMethod.FP16_VEC, QuantMethod.INT_CUBE])
def test_paged_attention_quant_pertoken_bnsd(quant_method, ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_pertoken_bnsd
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BNSD",
        "b": 2,
        "sq": 1,
        "skv": 1024,
        "max_skv": 1024,
        "nq": 1,
        "nkv": 1,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 2,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": quant_method,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=684.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('quant_method',
                         [QuantMethod.FP16_VEC, QuantMethod.INT_CUBE])
def test_paged_attention_quant_pertoken_with_anti_shape(quant_method, ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_pertoken_with_anti_shape
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 1,
        "sq": 1,
        "skv": 4096,
        "max_skv": 4096,
        "nq": 20,
        "nkv": 20,
        "d": 128,
        "blk_sz": 16,
        "anti_max_b": 2,
        "anti_max_sq": 4096,
        "drop_prop": 0.0,
        "quant_mode": 2,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": quant_method,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=520.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('quant_method',
                         [QuantMethod.FP16_VEC, QuantMethod.INT_CUBE])
def test_paged_attention_large_gsq_pertoken(quant_method, ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_large_gsq_pertoken
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 2,
        "sq": 177,
        "skv": 512,
        "max_skv": 512,
        "nq": 384,
        "nkv": 3,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 2,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": quant_method,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=1432.00)
@pytest.mark.parametrize('ctx_mode', [context.PYNATIVE_MODE])
def test_paged_attention_quant_no_quant_pangu38b_0(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_no_quant_pangu38b
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 48,
        "sq": 1,
        "skv": 4096,
        "max_skv": 4096,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=1432.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE])
def test_paged_attention_quant_no_quant_pangu38b_1(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_no_quant_pangu38b
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 48,
        "sq": 1,
        "skv": 4096,
        "max_skv": 4096,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=1048.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE])
@pytest.mark.parametrize('quant_method',[QuantMethod.FP16_VEC])
def test_paged_attention_quant_pertoken_pangu38b_0(quant_method, ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_pertoken_pangu38b
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 48,
        "sq": 1,
        "skv": 4096,
        "max_skv": 4096,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 2,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": quant_method,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=1052.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE])
@pytest.mark.parametrize('quant_method', [QuantMethod.INT_CUBE])
def test_paged_attention_quant_pertoken_pangu38b_1(quant_method, ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_pertoken_pangu38b
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 48,
        "sq": 1,
        "skv": 4096,
        "max_skv": 4096,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 2,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": quant_method,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=1048.00)
@pytest.mark.parametrize('ctx_mode', [context.PYNATIVE_MODE])
@pytest.mark.parametrize('quant_method', [QuantMethod.FP16_VEC])
def test_paged_attention_quant_pertoken_pangu38b_2(quant_method, ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_pertoken_pangu38b
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 48,
        "sq": 1,
        "skv": 4096,
        "max_skv": 4096,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 2,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": quant_method,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=1052.00)
@pytest.mark.parametrize('ctx_mode', [context.PYNATIVE_MODE])
@pytest.mark.parametrize('quant_method', [QuantMethod.INT_CUBE])
def test_paged_attention_quant_pertoken_pangu38b_3(quant_method, ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_pertoken_pangu38b
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 48,
        "sq": 1,
        "skv": 4096,
        "max_skv": 4096,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 2,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": quant_method,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=792.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_quant_no_quant_jiutian(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_no_quant_jiutian
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 8,
        "sq": 1,
        "skv": 2048,
        "max_skv": 4096,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=728.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('quant_method',
                         [QuantMethod.FP16_VEC, QuantMethod.INT_CUBE])
def test_paged_attention_quant_pertoken_jiutian(quant_method, ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_pertoken_jiutian
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 8,
        "sq": 1,
        "skv": 2048,
        "max_skv": 4096,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 2,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": quant_method,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=728.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_quant_pertoken_antiquant_scale_int64_to_fp32(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_pertoken_antiquant_scale_int64_to_fp32
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 8,
        "sq": 1,
        "skv": 2048,
        "max_skv": 4096,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 2,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": QuantMethod.INT_CUBE,
        "antiquant_scale_int64_to_fp32": True,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=488.00)
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_quant_pertoken_antiquant_scale_int64_to_fp32_small(ctx_mode):
    """
    Feature: paged_attention operator
    Description: test_paged_attention_quant_pertoken_antiquant_scale_int64_to_fp32_small
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 1,
        "sq": 1,
        "skv": 3,
        "max_skv": 1024,
        "nq": 20,
        "nkv": 20,
        "d": 128,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "quant_mode": 2,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "quant_method": QuantMethod.INT_CUBE,
        "antiquant_scale_int64_to_fp32": True,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=2696.00)
@pytest.mark.parametrize('dtype', ["float16", "float32"])
@pytest.mark.parametrize('mask_mode', ["MASK_DEFAULT", "TRAPEZOIDAL"])
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_trapezoidal(dtype, mask_mode, ctx_mode):
    """
    Feature: test PagedAttention op in kbk and pynative enabling infer_boost
    Description: test PagedAttention op with mask_mode TRAPEZOIDAL.
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": dtype,
        "layout": "BSH",
        "b": 1,
        "sq": 1001,
        "skv": "rand-b",
        "max_skv": 32768,
        "nq": 4,
        "nkv": 2,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": 0,
        "is_amask": 0,
        "mask_mode": mask_mode,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=270.00)
@pytest.mark.parametrize('blk_sz', [16, 128])
@pytest.mark.parametrize('dtype', ["float16", "float32"])
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_paged_attention_mla_combine_cache_norm(blk_sz, dtype, ctx_mode):
    """
    Feature: test PagedAttention op in kbk and pynative enabling infer_boost
    Description: test PagedAttention op with deepseek mla.
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": dtype,
        "layout": "BSH",
        "b": 1,
        "sq": 1,
        "skv": 512,
        "max_skv": 512,
        "nq": 4,
        "nkv": 1,
        "d": 576,
        "d_vo": 512,
        "blk_sz": blk_sz,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": False,
        "is_amask": False,
        "mla_kvcombined": True,
        "blk_sparse": 0,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@arg_mark(plat_marks=['platform_ascend310p'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=22.00)
def test_paged_attention_swft():
    """
    Feature: test swft PagedAttention op in tp4 mode
    Description: test PagedAttention op with deepseek mla.
    Expectation: the result is correct
    """
    env_var = os.environ.get('KERNEL_RUN_TIMES')
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 1,
        "sq": 1,
        "skv": 5,
        "max_skv": 8192,
        "nq": 32,
        "nkv": 1,
        "d": 576,
        "d_vo": 512,
        "mla_kvcombined": True,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "quant_mode": 0,
        "is_alibi": 0,
        "is_amask": 0,
        "blk_sparse": 0,
        "msprof": int(env_var) if env_var else 0,
        "fuse_key_cache": True
    }
    PagedAttentionTest(i_test)
