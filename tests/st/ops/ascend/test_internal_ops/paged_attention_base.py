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
import sys
import numpy as np
import math
from enum import Enum
import mindspore as ms
from mindspore import context


I8MIN = -128
I8MAX = 127
I8DIFF = I8MAX - I8MIN
BLOCK_SIZE_SET = (16, 32, 64, 128)
DIM_SET = (16, 32, 64, 128, 256)
LAYOUT_SET = ("BSH", "BNSD", "TH")
INPUT_TYPE_SET = ("float16", "float32")
get_quant_mode_str = {
    0: "DEFAULT",  # no quant
    1: "DEFAULT",  # per channel
    2: "PERTOKEN"  # per token
}


class QuantMethod(Enum):
    """Use different antiquant dtype and method to finish antiquant impls"""
    FP16_VEC = 0
    FP32_VEC = 1
    INT_CUBE = 2


def data2uint8(data: np.ndarray):
    # left is low and right is high
    tmp = data.astype(bool).astype(np.uint8).flatten()
    assert tmp.shape[0] % 8 == 0
    tmp = np.reshape(tmp, newshape=(-1, 8))  # b, 8
    for i in range(8):
        tmp[:, i] = tmp[:, i] * 2 ** i
    res = np.sum(tmp, axis=-1).astype(np.uint8)
    return res


def uint82data(data: np.ndarray):
    # left is low and right is high
    assert data.dtype == np.uint8
    tmp = data.flatten()
    res = np.zeros(shape=(tmp.shape[0], 8)).astype(np.uint8)
    for i in range(8):
        res[:, i] = tmp & 1
        tmp = tmp >> 1
    res = res.flatten()
    return res


def rerandom(data: np.ndarray):
    val_max = np.max(np.abs(data))
    rand = np.random.uniform(low=val_max * -0.1, high=val_max * 0.1, size=data.shape)
    rand = rand.astype(data.dtype)
    res = data + rand
    return res


def quant(data: np.ndarray, dtype, quant_mode: int, rerand=True):
    '''
    quant_mode 不量化=0 per-channel=1 per-token=2
    '''
    assert quant_mode in (1, 2)
    assert data.ndim == 5  # B S N 1 D
    if data.dtype in (np.uint8, np.int8):
        data = data.astype(np.float32)
    if quant_mode == 1:
        reduce_axes = (0, 1)
    else:
        reduce_axes = (2, 3, 4)
    val_min = np.min(data, axis=reduce_axes, keepdims=True)
    val_max = np.max(data, axis=reduce_axes, keepdims=True)
    val_diff = val_max - val_min
    val_diff = np.where(val_diff == 0, I8DIFF, val_diff)
    quant_scale = I8DIFF / val_diff
    antiquant_scale = val_diff / I8DIFF
    quant_offset = (val_max * I8MIN - val_min * I8MAX) / val_diff
    antiquant_offset = -quant_offset
    quant_int8 = data * quant_scale + quant_offset
    legal_min = np.min(quant_int8)
    legal_max = np.max(quant_int8)
    if legal_min < I8MIN - 1:
        raise Exception("[Error] min of quant_int8 is %f" % legal_min)
    if legal_max > I8MAX + 1:
        raise Exception("[Error] max of quant_int8 is %f" % legal_max)
    if rerand:
        antiquant_scale = rerandom(antiquant_scale)
        antiquant_offset = rerandom(antiquant_offset)
    quant_int8 = quant_int8.astype(np.int8)
    antiquant_scale = antiquant_scale.astype(dtype)
    antiquant_offset = antiquant_offset.astype(dtype)
    return quant_int8, antiquant_scale, antiquant_offset


def antiquant(quant_int8, antiquant_scale, antiquant_offset, dtype):
    res = (quant_int8.astype(dtype) + antiquant_offset) * antiquant_scale
    return res


def concat_zeros(data: np.ndarray, axis: int, size: int):
    add_shape = list(data.shape)
    add_shape[axis] = size
    zeros = np.zeros(shape=add_shape, dtype=data.dtype)
    res = np.concatenate([data, zeros], axis=axis)
    return res


def sq2multi(data: np.ndarray, axis: int, sq: np.ndarray):
    b = sq.shape[0]
    assert sq.ndim == 1  # (b,)
    assert data.shape[axis] == np.sum(sq)
    res = list()
    cnt = 0
    for i in range(b):
        string = [":"] * data.ndim
        string[axis] = "%d: %d" % (cnt, cnt + sq[i])
        cnt += sq[i]
        string = "data[" + ", ".join(string) + "]"
        # pylint: disable=eval-used
        res.append(eval(string))
    return res


def skv2multi(data: np.ndarray, axis: int, skv: np.ndarray):
    b = skv.shape[0]
    assert skv.ndim == 1  # (b,)
    assert data.shape[0] == b  # same batch
    res = list()
    for i in range(b):
        string = [":"] * data.ndim
        string[0] = "i"
        string[axis] = ":%d" % skv[i]
        string = "data[" + ", ".join(string) + "]"
        # pylint: disable=eval-used
        res.append(eval(string))
    return res


def process_deq_scale(deq_scale, quant_shape) -> np.ndarray:
    new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
    return new_deq_scale.astype(np.int64).reshape(quant_shape)


def reverse_process_deq_scale(int64_array) -> np.ndarray:
    uint32_array = int64_array.astype(np.uint32)  # 取每个 int64 中的低32位
    float32_array = np.frombuffer(uint32_array.tobytes(), dtype=np.float32)  # 将其转换回 float32
    return float32_array


class PagedAttentionBase:
    def check_dynamic_shape(self, ipt: dict):
        for k in ipt:
            if isinstance(ipt[k], dict):
                self.i_test_dict = ipt
            else:
                self.i_test_dict = {"testcase": ipt}
            break
        self.names = list(self.i_test_dict.keys())
        for i in range(1, len(self.names)):
            for k in self.i_test_dict[self.names[i]]:
                if k in ("b", "sq", "skv", "max_skv", "msprof"):
                    continue
                if self.i_test_dict[self.names[i]][k] != self.i_test_dict[self.names[0]][k]:
                    raise Exception("[Error] %s of %s and %s is not equal in dynamic shape" %
                                    (k, self.names[i], self.names[0]))

    def npu_init(self, net_function):
        first_case = self.i_test_dict[self.names[0]]
        if "quant_mode" in first_case:
            first_case["kv_cache_quant_mode"] = get_quant_mode_str[first_case["quant_mode"]]
        i_init = {
            "head_num": first_case["nq"],  # dtype: int
            "scale_value": 1 / math.sqrt(first_case["d"]),  # dtype: float
            "kv_head_num": first_case["nkv"],  # dtype: int
            "kv_cache_quant_mode": first_case.get("kv_cache_quant_mode", "DEFAULT")
        }
        if "ASCEND_HOME_PATH" not in os.environ:
            os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"
        context.set_context(mode=self.ctx_mode, device_target="Ascend")
        if first_case.get("mask_mode", "MASK_DEFAULT") != "MASK_DEFAULT":
            i_init["mask_mode"] = first_case.get("mask_mode", "MASK_DEFAULT")
        if first_case.get("mla_kvcombined", False):
            i_init["mla_v_dim"] = first_case.get("d_vo", 0)
            if i_init["mla_v_dim"] == 0:
                raise ValueError("mla_v_dim == 0 when mla_kvcombined is True.")
        context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
        self.net = net_function(**i_init)

    def save_bin(self):
        if self.mode == "gen":
            self.gen["query"].tofile("i0-query.bin")
            self.gen["key_cache"].tofile("i1-key_cache.bin")
            self.gen["value_cache"].tofile("i2-value_cache.bin")
            self.gen["block_tables"].tofile("i3-block_tables.bin")
            self.gen["batch_valid_length"].tofile("i4-batch_valid_length.bin")
            if self.i_test["is_quant"]:
                self.gen["antiquant_scale"].tofile("i5-antiquant_scale.bin")
                self.gen["antiquant_offset"].tofile("i6-antiquant_offset.bin")
            if self.i_test["is_alibi"]:
                self.gen["alibi_mask"].tofile("i7-alibi_mask.bin")
            if self.i_test["is_amask"]:
                self.gen["attn_mask"].tofile("i8-attn_mask.bin")
            self.gen["sq"].tofile("q_seqlen.bin")
            if self.i_test["drop_prop"] > 0.0:
                self.gen["drop_mask"].tofile("dmask.bin")

        self.gen["s"].tofile("s.bin")
        self.gen["score_max"].tofile("_score_max.bin")
        self.gen["score_sum"].tofile("_score_sum.bin")
        self.gen["p"].tofile("p.bin")
        self.gen["attention_out"].tofile("o0-attention_out.bin")

    def load_bin(self):
        self.load["query"] = np.fromfile("i0-query.bin", dtype=self.i_test["type"])
        if self.i_test["is_quant"]:
            self.load["key_cache"] = np.fromfile("i1-key_cache.bin", dtype=np.int8)
            self.load["value_cache"] = np.fromfile("i2-value_cache.bin", dtype=np.int8)
            self.load["antiquant_scale"] = np.fromfile("i5-antiquant_scale.bin", dtype=np.float16)
            self.load["antiquant_offset"] = np.fromfile("i6-antiquant_offset.bin", dtype=np.float16)
        else:
            self.load["key_cache"] = np.fromfile("i1-key_cache.bin", dtype=self.i_test["type"])
            self.load["value_cache"] = np.fromfile("i2-value_cache.bin", dtype=self.i_test["type"])
        self.load["block_tables"] = np.fromfile("i3-block_tables.bin", dtype=np.int32)
        self.load["batch_valid_length"] = np.fromfile("i4-batch_valid_length.bin", dtype=np.int32)
        if self.i_test["is_alibi"]:
            self.load["alibi_mask"] = np.fromfile("i7-alibi_mask.bin", dtype=self.i_test["type"])
        if self.i_test["is_amask"]:
            self.load["attn_mask"] = np.fromfile("i8-attn_mask.bin", dtype=np.float16)
        self.load["sq"] = np.fromfile("q_seqlen.bin", dtype=np.int32)
        if self.i_test["drop_prop"] > 0.0:
            self.load["dmask_uint8"] = np.fromfile("dmask.bin", dtype=np.uint8)

    def calc_expect_func(self):
        # get parameters
        b = self.i_test["b"]
        multi_sq = False
        sq = self.i_test["sq"]
        if isinstance(sq, int):
            pass
        elif isinstance(sq, list):
            multi_sq = True
            sq = np.array(sq, dtype=np.int32)
            if sq.shape != (b,):
                raise Exception("[Error] sq.shape can only be "
                                "(b,) but got %s" % sq.shape)
        else:
            raise Exception("[Error] sq can only be "
                            "int/list[int] but got %s" % sq)
        max_skv = self.i_test["max_skv"]
        anti_max_sq = self.i_test.get("anti_max_sq", sys.maxsize)
        if anti_max_sq < max_skv:
            raise ValueError(f"max_skv={max_skv} should equal or less than anti_max_sq={anti_max_sq}")
        anti_max_b = self.i_test.get("anti_max_b", sys.maxsize)
        if anti_max_b < b:
            raise ValueError(f"b={b} should equal or less than anti_max_b={anti_max_b}")
        multi_skv = False
        skv = self.i_test["skv"]
        if skv == "max":
            skv = max_skv
        elif isinstance(skv, int):
            pass
        elif isinstance(skv, list):
            multi_skv = True
            skv = np.array(skv, dtype=np.int32)
            if skv.shape != (b,):
                raise Exception("[Error] skv.shape can only be "
                                "(b,) but got %s" % skv.shape)
        elif skv == "rand-1":
            skv = int(np.random.randint(1, max_skv, 1)[0])
        elif skv == "rand-b":
            multi_skv = True
            skv = np.random.randint(1, max_skv, b).astype(np.int32)
        else:
            raise Exception("[Error] skv can only be "
                            "max/int/list[int]/rand-1/rand-b but got %s" % skv)
        quant_mode = self.i_test["quant_mode"]
        if quant_mode not in (0, 1, 2):
            raise Exception("[Error] quant mode can only be quant-off=0 "
                            "per-channel=1 per-token=2 but got %s" % quant_mode)
        gen_id = multi_sq * 2 + multi_skv
        if gen_id == 2:  # go to 3
            skv = np.array([skv] * b, dtype=np.int32)
        diff_skv = max_skv - skv  # int or tensor
        nq = self.i_test["nq"]
        n = self.i_test["nkv"]
        g = nq // n
        d = self.i_test["d"]
        d_vo = self.i_test.get("d_vo", d)
        mla_kvcombined = self.i_test.get("mla_kvcombined", False)
        if (d not in DIM_SET) and (d == d_vo):
            raise Exception("[Error] embed can only be %s, but got %d" % (
                DIM_SET, d))
        bs = b * max_skv
        block_size = self.i_test["blk_sz"]
        if block_size not in BLOCK_SIZE_SET:
            raise Exception("[Error] block size can only be %s, but got %d" % (
                BLOCK_SIZE_SET, block_size))
        if bs % block_size:
            raise Exception("[Error] bs mod block_size can only be 0, "
                            "but got %d mod %d = %d" % (bs, block_size, bs % block_size))
        if d == 256 and block_size == 128:
            raise Exception("[Error] d == 256 and block_size == 128 are not legal, "
                            "try block_size = 64")
        drop_prop = self.i_test["drop_prop"]
        assert 0.0 <= drop_prop <= 1.0
        is_dropout = drop_prop > 0.0
        layout = self.i_test["layout"]
        if layout not in LAYOUT_SET:
            raise Exception("[Error] layout can only be %s, but got %s" % (
                LAYOUT_SET, layout))
        input_type = self.i_test["type"]
        if input_type not in INPUT_TYPE_SET:
            raise Exception("[Error] input_type can only be %s, but got %s" % (
                INPUT_TYPE_SET, input_type))
        if quant_mode and input_type == "float32" and block_size > 64:
            raise Exception("[Error] quant_mode and float32 and block_size > 64 "
                            "are not legal, try block_size = 64")

        np.random.seed(0)
        amask = None
        dmask = None
        alibi = None
        if self.mode == "gen":
            if layout in ["BSH", "TH"]:
                if gen_id in (0, 1):
                    sz_q = [b, sq, n, g, d]
                elif gen_id in (2, 3):
                    sz_q = [np.sum(sq), n, g, d]
                else:
                    raise Exception("gen_id can only in (0, 1, 2, 3), but got %d" % gen_id)
            else:  # BNSD
                if gen_id in (0, 1):
                    sz_q = [b, n, g, sq, d]
                elif gen_id in (2, 3):
                    sz_q = [n, g, np.sum(sq), d]
                else:
                    raise Exception("gen_id can only in (0, 1, 2, 3), but got %d" % gen_id)

            # generate qkv data
            q = np.random.uniform(-1.0, 1.0, size=sz_q)
            # BSH of k and v in pa
            k = np.random.uniform(-1.0, 1.0, size=[b, max_skv, n, 1, d])
            if mla_kvcombined:
                v = k[:, :, :, :, :d_vo].copy()
            else:
                v = np.random.uniform(-1.0, 1.0, size=[b, max_skv, n, 1, d_vo])
            q = q.astype(np.float16).astype(input_type)
            k = k.astype(np.float16).astype(input_type)
            v = v.astype(np.float16).astype(input_type)
            if quant_mode:
                if self.i_test.get("quant_method", 0) == QuantMethod.INT_CUBE:
                    kv_range = 4.0
                    quant_shape = (1, 1, n, 1, d)  # "PERCHANNEL"
                    if get_quant_mode_str[quant_mode] == "PERTOKEN":
                        quant_shape = (b, max_skv, 1, 1, 1)
                    de_scale1_fp32 = np.random.randint(-4, 5, size=quant_shape).astype(np.float32)
                    k_anti_scale = process_deq_scale(de_scale1_fp32, quant_shape)
                    de_scale2_fp32 = np.random.randint(-4, 5, size=quant_shape).astype(np.float32)
                    v_anti_scale = process_deq_scale(de_scale2_fp32, quant_shape)
                    k_anti_offset = np.random.randint(-10, 10, size=quant_shape).astype(np.int32)
                    v_anti_offset = np.random.randint(-10, 10, size=quant_shape).astype(np.int32)
                    k = np.random.uniform(-kv_range, kv_range, size=[b, max_skv, n, 1, d]).astype(np.int8)
                    v = np.random.uniform(-kv_range, kv_range, size=[b, max_skv, n, 1, d]).astype(np.int8)
                    antiquant_scale = np.stack([k_anti_scale, v_anti_scale], axis=0)
                    antiquant_offset = np.stack([k_anti_offset, v_anti_offset], axis=0)
                else:
                    k, k_anti_scale, k_anti_offset = quant(k, np.float16, quant_mode)
                    v, v_anti_scale, v_anti_offset = quant(v, np.float16, quant_mode)
                    antiquant_scale = np.stack([k_anti_scale, v_anti_scale], axis=0)
                    antiquant_offset = np.stack([k_anti_offset, v_anti_offset], axis=0)

                actual_antiquant_scale = antiquant_scale
                actual_antiquant_offset = antiquant_offset
                if get_quant_mode_str[quant_mode] == "PERTOKEN":
                    anti_max_batch = self.i_test.get("anti_max_b", b)
                    anti_max_q = self.i_test.get("anti_max_sq", max_skv)
                    actual_antiquant_scale = np.zeros(
                        shape=(2, anti_max_batch, anti_max_q, 1, 1, 1)).astype(actual_antiquant_scale.dtype)
                    actual_antiquant_offset = np.zeros(
                        shape=(2, anti_max_batch, anti_max_q, 1, 1, 1)).astype(actual_antiquant_offset.dtype)
                    min_batch = min(anti_max_batch, b)
                    min_q = min(anti_max_q, max_skv)
                    actual_antiquant_scale[:, :min_batch, :min_q, 0, 0,
                                           0] = antiquant_scale[:, :min_batch, :min_q, 0, 0, 0]
                    actual_antiquant_offset[:, :min_batch, :min_q, 0, 0,
                                            0] = antiquant_offset[:, :min_batch, :min_q, 0, 0, 0]

            if gen_id in (0, 1):
                batch_valid_length = np.ones((b,)).astype(np.int32) * skv
            elif gen_id in (2, 3):
                batch_valid_length = np.stack([skv, sq], axis=0)
            else:
                raise Exception("gen_id can only in (0, 1, 2, 3), but got %d" % gen_id)
            q_seqlen_output = np.ones(b).astype(np.int32) * sq
            # table
            max_num_blocks_per_query = max_skv // block_size
            table = np.arange(b * max_num_blocks_per_query).astype(np.int32)
            table = np.reshape(table, newshape=(b, max_num_blocks_per_query))
            # save data
            self.gen["query"] = q.copy()
            self.gen["key_cache"] = k.copy()
            self.gen["value_cache"] = v.copy()
            if quant_mode:
                self.gen["antiquant_scale"] = actual_antiquant_scale.copy()
                self.gen["antiquant_offset"] = actual_antiquant_offset.copy()
            self.gen["batch_valid_length"] = batch_valid_length.copy()
            self.gen["sq"] = q_seqlen_output.copy()
            self.gen["block_tables"] = table.copy()
            if self.i_test["is_alibi"]:
                if gen_id in (0, 1):
                    sz_alibi = [b, n, g, sq, max_skv]
                elif gen_id in (2, 3):
                    sz_alibi = [n, g, np.sum(sq), max_skv]
                else:
                    raise Exception("gen_id can only in (0, 1, 2, 3), but got %d" % gen_id)
                alibi = np.random.uniform(-10.0, 0.0, size=sz_alibi)
                alibi = alibi.astype(np.float16).astype(input_type)
                self.gen["alibi_mask"] = alibi.copy()
            if self.i_test["is_amask"]:
                if gen_id in (0, 1):
                    sz_amask = [b, 1, 1, sq, max_skv]
                elif gen_id in (2, 3):
                    sz_amask = [1, 1, np.sum(sq), max_skv]
                else:
                    raise Exception("gen_id can only in (0, 1, 2, 3), but got %d" % gen_id)
                amask = np.random.randint(0, 2, size=sz_amask)
                amask *= -10000
                amask = amask.astype(np.float16)
                self.gen["attn_mask"] = amask.copy()
            if is_dropout:  # 0 is drop, 1 is keep
                if gen_id in (0, 1):
                    sz_dmask = [b, 1, 1, sq, max_skv]
                elif gen_id in (2, 3):
                    sz_dmask = [1, 1, np.sum(sq), max_skv]
                else:
                    raise Exception("gen_id can only in (0, 1, 2, 3), but got %d" % gen_id)
                dmask = np.random.uniform(size=sz_dmask) > drop_prop
                dmask_uint8 = data2uint8(dmask)
                self.gen["drop_mask"] = dmask_uint8.copy()
        else:  # load
            q = self.load["query"].copy()
            k = self.load["key_cache"].copy()
            v = self.load["value_cache"].copy()
            if layout in ["BSH", "TH"]:
                if gen_id in (0, 1):
                    sz_q = [b, sq, n, g, d]
                elif gen_id in (2, 3):
                    sz_q = [np.sum(sq), n, g, d]
                else:
                    raise Exception("gen_id can only in (0, 1, 2, 3), but got %d" % gen_id)
            else:  # BNSD
                if gen_id in (0, 1):
                    sz_q = [b, n, g, sq, d]
                elif gen_id in (2, 3):
                    sz_q = [n, g, np.sum(sq), d]
                else:
                    raise Exception("gen_id can only in (0, 1, 2, 3), but got %d" % gen_id)
            q = np.reshape(q, newshape=sz_q)
            # BSH of k and v in pa
            k = np.reshape(k, newshape=[b, max_skv, n, 1, d])
            v = np.reshape(v, newshape=[b, max_skv, n, 1, d_vo])
            batch_valid_length = self.load["batch_valid_length"].copy()
            if self.i_test["is_alibi"]:
                alibi = self.load["alibi_mask"].copy()
                if gen_id in (0, 1):
                    sz_alibi = [b, n, g, sq, max_skv]
                elif gen_id in (2, 3):
                    sz_alibi = [n, g, np.sum(sq), max_skv]
                else:
                    raise Exception("gen_id can only in (0, 1, 2, 3), but got %d" % gen_id)
                alibi = np.reshape(alibi, newshape=sz_alibi)
            if self.i_test["is_amask"]:
                amask = self.load["attn_mask"].copy()
                if gen_id in (0, 1):
                    sz_amask = [b, 1, 1, sq, max_skv]
                elif gen_id in (2, 3):
                    sz_amask = [1, 1, np.sum(sq), max_skv]
                else:
                    raise Exception("gen_id can only in (0, 1, 2, 3), but got %d" % gen_id)
                amask = np.reshape(amask, newshape=sz_amask)
            if is_dropout:  # 0 is drop, 1 is keep
                dmask_uint8 = self.load["drop_mask"].copy()
                dmask = uint82data(dmask_uint8)
                if gen_id in (0, 1):
                    sz_dmask = [b, 1, 1, sq, max_skv]
                elif gen_id in (2, 3):
                    sz_dmask = [1, 1, np.sum(sq), max_skv]
                else:
                    raise Exception("gen_id can only in (0, 1, 2, 3), but got %d" % gen_id)
                dmask = np.reshape(dmask, newshape=sz_dmask)
            if quant_mode:
                antiquant_scale = self.load["antiquant_scale"].copy()
                antiquant_offset = self.load["antiquant_offset"].copy()

        if quant_mode:  # antiquant
            if get_quant_mode_str[quant_mode] == "DEFAULT":
                anti_shape = [2, 1, 1] + list(k.shape)[2:]
            else:
                anti_shape = [2] + list(k.shape)[:2] + [1, 1, 1]
            if self.i_test.get("quant_method", 0) == QuantMethod.INT_CUBE:
                antiquant_scale = np.reshape(
                    reverse_process_deq_scale(antiquant_scale),
                    anti_shape)  # int64 to float32
                antiquant_offset = np.reshape(antiquant_offset, anti_shape)  # int32
                k = antiquant(k, antiquant_scale[0], antiquant_offset[0], np.float16)
                v = antiquant(v, antiquant_scale[1], antiquant_offset[1], np.float16)
            else:
                antiquant_scale = np.reshape(antiquant_scale, anti_shape)
                antiquant_offset = np.reshape(antiquant_offset, anti_shape)
                k_anti_scale = antiquant_scale[0]
                v_anti_scale = antiquant_scale[1]
                k_anti_offset = antiquant_offset[0]
                v_anti_offset = antiquant_offset[1]
                k = antiquant(k, k_anti_scale, k_anti_offset, np.float16)
                v = antiquant(v, v_anti_scale, v_anti_offset, np.float16)

        if gen_id == 0:
            if diff_skv:  # max_skv to skv
                # BSH of k and v in pa
                k = k[:, :skv, :, :, :]  # [b, skv, n, 1, d]
                v = v[:, :skv, :, :, :]  # [b, skv, n, 1, d_vo]
                if self.i_test["is_alibi"]:
                    alibi = alibi[:, :, :, :, :skv]  # [b, n, g, sq, skv]
                if self.i_test["is_amask"]:
                    amask = amask[:, :, :, :, :skv]  # [b, 1, 1, sq, skv]
                if is_dropout:  # 0 is drop, 1 is keep
                    dmask = dmask[:, :, :, :, :skv]  # [b, 1, 1, sq, skv]

            if layout in ["BSH", "TH"]:
                q = np.transpose(q, axes=(0, 2, 3, 1, 4))  # [b, n, g, sq, d]
            # BSH of k and v in pa
            k = np.transpose(k, axes=(0, 2, 3, 1, 4))  # [b, n, 1, skv, d]
            v = np.transpose(v, axes=(0, 2, 3, 1, 4))  # [b, n, 1, skv, d_vo]

            # np.matmul of float32 is faster 600 times than float16
            q = q.astype(np.float32)
            k = k.astype(np.float32)
            v = v.astype(np.float32)

            k_t = np.transpose(k, axes=(0, 1, 2, 4, 3))  # [b, n, 1, d, skv]

            s0 = np.matmul(q, k_t)  # [b, n, g, sq, skv]

            tor = np.float32(math.sqrt(1.0 * d))
            s1 = s0 / tor
            if self.i_test["is_alibi"]:
                s1 = s1 + alibi.astype(np.float32)
            if self.i_test["is_amask"] and self.i_test["sq"] != 1:
                s1 = s1 + amask.astype(np.float32)
            if self.i_test.get("mask_mode", "MASK_DEFAULT") == "TRAPEZOIDAL":
                tri_mask = np.ones(shape=[b, 1, 1, sq, skv]).astype(np.float32)
                tri_mask = np.triu(tri_mask, skv - sq + 1)
                s1 = s1 - tri_mask.astype(np.float32) * np.float32(10000)

            score_max = np.max(s1, axis=-1, keepdims=True)  # [b, n, g, sq, 1]

            s2 = s1 - score_max  # [b, n, g, sq, skv]

            s3 = np.exp(s2)  # [b, n, g, sq, skv]
            if is_dropout:
                s3 = s3 * dmask.astype(np.float32) / (1 - np.float32(drop_prop))

            score_sum = np.sum(s3, axis=-1, keepdims=True)  # [b, n, g, sq, 1]

            p = s3 / score_sum  # [b, n, g, sq, skv]

            o = np.matmul(p, v)  # [b, n, g, sq, d_vo]
            o = o.astype(input_type)

            if diff_skv:
                s0 = concat_zeros(s0, 4, diff_skv)  # [b, n, g, sq, max_skv]
                p = concat_zeros(p, 4, diff_skv)  # [b, n, g, sq, max_skv]

            if layout in ["BSH", "TH"]:
                o = np.transpose(o, axes=(0, 3, 1, 2, 4))  # [b, sq, n, g, d_vo]

        elif gen_id == 1:
            if layout in ["BSH", "TH"]:
                q = np.transpose(q, axes=(0, 2, 3, 1, 4))  # [b, n, g, sq, d]
            # BSH of k and v in pa
            k = np.transpose(k, axes=(0, 2, 3, 1, 4))  # [b, n, 1, max_skv, d]
            v = np.transpose(v, axes=(0, 2, 3, 1, 4))  # [b, n, 1, max_skv, d_vo]
            # tensor to list
            q = list(q)  # [b, n, g, sq, d] to [n, g, sq, d] * b
            k = skv2multi(k, 3, skv)  # [b, n, 1, max_skv, d] to [n, 1, skv, d] * b
            v = skv2multi(v, 3, skv)  # [b, n, 1, max_skv, d_vo] to [n, 1, skv, d_vo] * b
            if self.i_test["is_alibi"]:
                # [b, n, g, sq, max_skv] to [n, g, sq, max_skv] * b
                alibi = list(alibi)
                for i in range(b):
                    if diff_skv[i]:  # max_skv to skv
                        alibi[i] = alibi[i][:, :, :, :skv[i]]  # [n, g, sq, skv]
            if self.i_test["is_amask"]:
                # [b, 1, 1, sq, max_skv] to [1, 1, sq, max_skv] * b
                amask = list(amask)
                for i in range(b):
                    if diff_skv[i]:  # max_skv to skv
                        amask[i] = amask[i][:, :, :, :skv[i]]  # [1, 1, sq, skv]
            if is_dropout:  # 0 is drop, 1 is keep
                # [b, 1, 1, sq, max_skv] to [1, 1, sq, max_skv] * b
                dmask = list(dmask)
                for i in range(b):
                    if diff_skv[i]:  # max_skv to skv
                        dmask[i] = dmask[i][:, :, :, :skv[i]]  # [1, 1, sq, skv]

            s0 = [None] * b
            score_max = [None] * b
            score_sum = [None] * b
            p = [None] * b
            o = [None] * b
            for i in range(b):
                # np.matmul of float32 is faster 600 times than float16
                q[i] = q[i].astype(np.float32)
                k[i] = k[i].astype(np.float32)
                v[i] = v[i].astype(np.float32)

                k_t = np.transpose(k[i], axes=(0, 1, 3, 2))  # [n, 1, d, skv]

                s0[i] = np.matmul(q[i], k_t)  # [n, g, sq, skv]

                tor = np.float32(math.sqrt(1.0 * d))
                s1 = s0[i] / tor
                if self.i_test["is_alibi"]:
                    s1 = s1 + alibi[i].astype(np.float32)
                if self.i_test["is_amask"] and self.i_test["sq"] != 1:
                    s1 = s1 + amask[i].astype(np.float32)
                if self.i_test.get("mask_mode", "MASK_DEFAULT") == "TRAPEZOIDAL":
                    tri_mask = np.ones(shape=[1, 1, sq, skv[i]]).astype(np.float32)
                    tri_mask = np.triu(tri_mask, skv[i] - sq + 1)
                    s1 = s1 - tri_mask.astype(np.float32) * np.float32(10000)

                score_max[i] = np.max(s1, axis=-1, keepdims=True)  # [n, g, sq, 1]

                s2 = s1 - score_max[i]  # [n, g, sq, skv]

                s3 = np.exp(s2)  # [n, g, sq, skv]
                if is_dropout:
                    s3 = s3 * dmask[i].astype(np.float32) / (1 - np.float32(drop_prop))

                score_sum[i] = np.sum(s3, axis=-1, keepdims=True)  # [n, g, sq, 1]

                p[i] = s3 / score_sum[i]  # [n, g, sq, skv]

                o[i] = np.matmul(p[i], v[i])  # [n, g, sq, d_vo]
                o[i] = o[i].astype(input_type)

                if diff_skv[i]:
                    s0[i] = concat_zeros(s0[i], 3, diff_skv[i])  # [n, g, sq, max_skv]
                    p[i] = concat_zeros(p[i], 3, diff_skv[i])  # [n, g, sq, max_skv]

            s0 = np.stack(s0, axis=0)  # [b, n, g, sq, max_skv]
            score_max = np.stack(score_max, axis=0)  # [b, n, g, sq, 1]
            score_sum = np.stack(score_sum, axis=0)  # [b, n, g, sq, 1]
            p = np.stack(p, axis=0)  # [b, n, g, sq, max_skv]
            o = np.stack(o, axis=0)  # [b, n, g, sq, d_vo]
            if layout in ["BSH", "TH"]:
                o = np.transpose(o, axes=(0, 3, 1, 2, 4))  # [b, sq, n, g, d_vo]

        elif gen_id in (2, 3):
            if layout in ["BSH", "TH"]:
                q = np.transpose(q, axes=(1, 2, 0, 3))  # [np.sum(sq), n, g, d] to [n, g, np.sum(sq), d]
            # BSH of k and v in pa
            k = np.transpose(k, axes=(0, 2, 3, 1, 4))  # [b, n, 1, max_skv, d]
            v = np.transpose(v, axes=(0, 2, 3, 1, 4))  # [b, n, 1, max_skv, d_vo]
            # tensor to list
            q = sq2multi(q, 2, sq)  # [n, g, np.sum(sq), d] to [n, g, sq, d] * b
            k = skv2multi(k, 3, skv)  # [b, n, 1, max_skv, d] to [n, 1, skv, d] * b
            v = skv2multi(v, 3, skv)  # [b, n, 1, max_skv, d_vo] to [n, 1, skv, d_vo] * b
            if self.i_test["is_alibi"]:
                # [n, g, np.sum(sq), max_skv] to [n, g, sq, max_skv] * b
                alibi = sq2multi(alibi, 2, sq)
                for i in range(b):
                    if diff_skv[i]:  # max_skv to skv
                        alibi[i] = alibi[i][:, :, :, :skv[i]]  # [n, g, sq, skv]
            if self.i_test["is_amask"]:
                # [1, 1, np.sum(sq), max_skv] to [1, 1, sq, max_skv] * b
                amask = sq2multi(amask, 2, sq)
                for i in range(b):
                    if diff_skv[i]:  # max_skv to skv
                        amask[i] = amask[i][:, :, :, :skv[i]]  # [1, 1, sq, skv]
            if is_dropout:  # 0 is drop, 1 is keep
                # [1, 1, np.sum(sq), max_skv] to [1, 1, sq, max_skv] * b
                dmask = sq2multi(dmask, 2, sq)
                for i in range(b):
                    if diff_skv[i]:  # max_skv to skv
                        dmask[i] = dmask[i][:, :, :, :skv[i]]  # [1, 1, sq, skv]

            s0 = [None] * b
            score_max = [None] * b
            score_sum = [None] * b
            p = [None] * b
            o = [None] * b
            for i in range(b):
                # np.matmul of float32 is faster 600 times than float16
                q[i] = q[i].astype(np.float32)
                k[i] = k[i].astype(np.float32)
                v[i] = v[i].astype(np.float32)

                k_t = np.transpose(k[i], axes=(0, 1, 3, 2))  # [n, 1, d, skv]

                s0[i] = np.matmul(q[i], k_t)  # [n, g, sq, skv]

                tor = np.float32(math.sqrt(1.0 * d))
                s1 = s0[i] / tor
                if self.i_test["is_alibi"]:
                    s1 = s1 + alibi[i].astype(np.float32)
                if self.i_test["is_amask"] and self.i_test["sq"] != 1:
                    s1 = s1 + amask[i].astype(np.float32)

                score_max[i] = np.max(s1, axis=-1, keepdims=True)  # [n, g, sq, 1]

                s2 = s1 - score_max[i]  # [n, g, sq, skv]

                s3 = np.exp(s2)  # [n, g, sq, skv]
                if is_dropout:
                    s3 = s3 * dmask[i].astype(np.float32) / (1 - np.float32(drop_prop))

                score_sum[i] = np.sum(s3, axis=-1, keepdims=True)  # [n, g, sq, 1]

                p[i] = s3 / score_sum[i]  # [n, g, sq, skv]

                o[i] = np.matmul(p[i], v[i])  # [n, g, sq, d_vo]
                o[i] = o[i].astype(input_type)

                if diff_skv[i]:
                    s0[i] = concat_zeros(s0[i], 3, diff_skv[i])  # [n, g, sq, max_skv]
                    p[i] = concat_zeros(p[i], 3, diff_skv[i])  # [n, g, sq, max_skv]

            s0 = np.concatenate(s0, axis=2)  # [n, g, np.sum(sq), max_skv]
            score_max = np.concatenate(score_max, axis=2)  # [n, g, np.sum(sq), 1]
            score_sum = np.concatenate(score_sum, axis=2)  # [n, g, np.sum(sq), 1]
            p = np.concatenate(p, axis=2)  # [n, g, np.sum(sq), max_skv]
            o = np.concatenate(o, axis=2)  # [n, g, np.sum(sq), d_vo]
            if layout in ["BSH", "TH"]:
                o = np.transpose(o, axes=(2, 0, 1, 3))  # [np.sum(sq), n, g, d_vo]

        else:
            raise Exception("gen_id can only in (0, 1, 2, 3), but got %d" % gen_id)

        self.gen["s"] = s0.astype(np.float32).copy()
        self.gen["score_max"] = score_max.astype(np.float32).copy()
        self.gen["score_sum"] = score_sum.astype(np.float32).copy()
        self.gen["p"] = p.astype(np.float32).copy()
        self.gen["attention_out"] = o.copy()

    def compare(self):
        expect = self.gen["attention_out"]
        actual = self.o_ascend["attention_out"]
        if self.i_test["type"] in {"float32", np.float32}:
            actual = actual.to(ms.float32)
        actual = actual.numpy()
        actual = np.reshape(actual, newshape=expect.shape)
        actual = actual.flatten()
        expect = expect.flatten()
        denom = np.sqrt(np.dot(actual, actual)) * np.sqrt(np.dot(expect, expect))
        cos = np.dot(actual, expect) / (denom + 1e-10)
        print("cosine similarity = ", cos)
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
