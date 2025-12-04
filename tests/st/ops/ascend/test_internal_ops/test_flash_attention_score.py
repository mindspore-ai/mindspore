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
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.nn import Cell
from mindspore import context, Profiler
from tests.mark_utils import arg_mark


INT8_MAX = 2 ** 7 - 1
DIM_MAX = 256
LAYOUT_SET = ("BSH", "BNSD")
INPUT_TYPE_SET = ("float16", "float32")


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


def concat_zeros(data: np.ndarray, axis: int, size: int):
    add_shape = list(data.shape)
    add_shape[axis] = size
    zeros = np.zeros(shape=add_shape, dtype=data.dtype)
    res = np.concatenate([data, zeros], axis=axis)
    return res


class FlashAttentionScoreNet(Cell):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = FlashAttentionScore(*args, **kwargs)

    def construct(self, *args, **kwargs):
        return self.net(*args, **kwargs)


class FlashAttentionScoreTest:
    def __init__(self, ipt: dict, mode="gen"):
        '''
        ipt: input of one testcase or dict of testcases
        '''
        self.mode = mode
        self.ctx_mode = ipt.get('ctx_mode', context.GRAPH_MODE)

        self.names = list()  # testcases name list
        self.i_test_dict = dict()
        self.check_dynamic_shape(ipt)

        self.net = None
        self.npu_init(FlashAttentionScoreNet)

        self.i_test = self.i_test_dict[self.names[0]]
        self.set_inputs()

        # construct
        np.random.seed(0)  # lock Skv
        for k in self.names:
            print(k, "start")
            self.i_test = self.i_test_dict[k]
            self.gen = dict()
            self.load = dict()
            self.i_construct = dict()
            self.o_ascend = dict()
            self.calc_expect_func()
            self.calc_actual_func()
            self.compare()
            print(k, "success")

    def check_dynamic_shape(self, ipt: dict):
        for k in ipt:
            if isinstance(ipt[k], dict):
                self.i_test_dict = ipt
            else:
                self.i_test_dict = {"testcase": ipt}
            break
        self.names = list(self.i_test_dict.keys())
        for i in range(1, len(self.names)):
            if self.i_test_dict[self.names[i]]["nq"] != self.i_test_dict[self.names[0]]["nq"]:
                raise Exception(
                    "[Error] nq of %s and %s is not equal in dynamic shape" % (self.names[i], self.names[0])
                )
            if self.i_test_dict[self.names[i]]["drop_prop"] != self.i_test_dict[self.names[0]]["drop_prop"]:
                raise Exception(
                    "[Error] drop_prop of %s and %s is not equal in dynamic shape" % (self.names[i], self.names[0])
                )
            if self.i_test_dict[self.names[i]]["d"] != self.i_test_dict[self.names[0]]["d"]:
                raise Exception(
                    "[Error] d of %s and %s is not equal in dynamic shape" % (self.names[i], self.names[0])
                )
            if self.i_test_dict[self.names[i]]["low_tri"] != self.i_test_dict[self.names[0]]["low_tri"]:
                raise Exception(
                    "[Error] low_tri of %s and %s is not equal in dynamic shape" % (self.names[i], self.names[0])
                )
            if self.i_test_dict[self.names[i]]["layout"] != self.i_test_dict[self.names[0]]["layout"]:
                raise Exception(
                    "[Error] layout of %s and %s is not equal in dynamic shape" % (self.names[i], self.names[0])
                )
            if self.i_test_dict[self.names[i]]["blk_sparse"] != self.i_test_dict[self.names[0]]["blk_sparse"]:
                raise Exception(
                    "[Error] blk_sparse of %s and %s is not equal in dynamic shape" % (self.names[i], self.names[0])
                )

    def npu_init(self, net_function):
        i_init = {
            "head_num": self.i_test_dict[self.names[0]]["nq"],  # dtype: int
            "keep_prob": 1 - self.i_test_dict[self.names[0]]["drop_prop"],  # dtype: float
            "scale_value": 1 / math.sqrt(float(self.i_test_dict[self.names[0]]["d"])),  # dtype: float
            "pre_tokens": 2147483647,  # dtype: int
            "next_tokens": 0 if self.i_test_dict[self.names[0]]["low_tri"] else 2147483647,  # dtype: int
            "inner_precise": 0,  # dtype: int
            "input_layout": self.i_test_dict[self.names[0]]["layout"],  # dtype: str
            "sparse_mode": self.i_test_dict[self.names[0]]["blk_sparse"]  # dtype: int
        }

        if "ASCEND_HOME_PATH" not in os.environ:
            os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"
        context.set_context(mode=self.ctx_mode, device_target="Ascend")
        context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
        self.net = net_function(**i_init)

    def save_bin(self):
        if self.mode == "gen":
            self.gen["query"].tofile("i0-query.bin")
            self.gen["key"].tofile("i1-key.bin")
            self.gen["value"].tofile("i2-value.bin")
            if self.i_test["is_alibi"]:
                self.gen["real_shift"].tofile("i3-real_shift.bin")
            if self.i_test["drop_prop"] > 0.0:
                self.gen["drop_mask"].tofile("i4-drop_mask.bin")
            # i5-padding_mask skip
            if self.i_test["is_amask"]:
                self.gen["attn_mask"].tofile("i6-attn_mask.bin")
            # i7-prefix skip
            self.gen["actual_seq_qlen"].tofile("i8-actual_seq_qlen.bin")
            self.gen["actual_seq_kvlen"].tofile("i9-actual_seq_kvlen.bin")
        self.gen["s"].tofile("s.bin")
        self.gen["softmax_max"].tofile("o0-softmax_max.bin")
        self.gen["softmax_sum"].tofile("o1-softmax_sum.bin")
        self.gen["softmax_out"].tofile("o2-softmax_out.bin")
        self.gen["attention_out"].tofile("o3-attention_out.bin")

    def load_bin(self):
        self.load["query"] = np.fromfile("i0-query.bin", dtype=self.i_test["type"])
        self.load["key"] = np.fromfile("i1-key.bin", dtype=self.i_test["type"])
        self.load["value"] = np.fromfile("i2-value.bin", dtype=self.i_test["type"])
        if self.i_test["is_alibi"]:
            self.load["real_shift"] = np.fromfile("i3-real_shift.bin", dtype=self.i_test["type"])
        if self.i_test["drop_prop"] > 0.0:
            self.load["dmask_uint8"] = np.fromfile("i4-drop_mask.bin", dtype=np.uint8)
        # i5-padding_mask skip
        if self.i_test["is_amask"]:
            self.load["attn_mask"] = np.fromfile("i6-attn_mask.bin", dtype=np.float16)
        # i7-prefix skip
        self.load["actual_seq_qlen"] = np.fromfile("i8-actual_seq_qlen.bin", dtype=np.int32)
        self.load["actual_seq_kvlen"] = np.fromfile("i9-actual_seq_kvlen.bin", dtype=np.int32)

    def calc_expect_func(self):
        # get parameters
        b = self.i_test["b"]
        sq = self.i_test["sq"]
        max_skv = self.i_test["max_skv"]
        skv = self.i_test["skv"]
        if skv == "max":
            skv = max_skv
        elif isinstance(skv, int):
            pass
        elif skv == "rand-1":
            skv = int(np.random.randint(1, max_skv, 1)[0])
        else:
            raise Exception("[Error] skv can only be max/int/rand-1 but got %s" % skv)

        diff_skv = max_skv - skv
        nq = self.i_test["nq"]
        n = self.i_test["nkv"]
        g = nq // n
        d = self.i_test["d"]
        if d > DIM_MAX:
            raise Exception("[Error] embed <= %s, but got %d" % (
                DIM_MAX, d))
        drop_prop = self.i_test["drop_prop"]
        assert 0.0 <= drop_prop <= 1.0
        is_dropout = (drop_prop > 0.0)
        layout = self.i_test["layout"]
        if layout not in LAYOUT_SET:
            raise Exception("[Error] layout can only be %s, but got %s" % (
                LAYOUT_SET, layout))
        input_type = self.i_test["type"]
        if input_type not in INPUT_TYPE_SET:
            raise Exception("[Error] input_type can only be %s, but got %s" % (
                INPUT_TYPE_SET, input_type))

        np.random.seed(0)
        amask = None
        dmask = None
        alibi = None
        if self.mode == "gen":
            if layout == "BSH":
                q = np.random.uniform(-1.0, 1.0, size=[b, sq, n, g, d])
                k = np.random.uniform(-1.0, 1.0, size=[b, max_skv, n, 1, d])
                v = np.random.uniform(-1.0, 1.0, size=[b, max_skv, n, 1, d])
            else:  # BNSD
                q = np.random.uniform(-1.0, 1.0, size=[b, n, g, sq, d])
                k = np.random.uniform(-1.0, 1.0, size=[b, n, 1, max_skv, d])
                v = np.random.uniform(-1.0, 1.0, size=[b, n, 1, max_skv, d])
            q = q.astype(np.float16).astype(input_type)
            k = k.astype(np.float16).astype(input_type)
            v = v.astype(np.float16).astype(input_type)
            kv_seqlen = np.ones((b,)).astype(np.int32) * skv
            q_seqlen_output = np.ones(b).astype(np.int32) * sq
            # save data
            self.gen["query"] = q.copy()
            self.gen["key"] = k.copy()
            self.gen["value"] = v.copy()
            self.gen["actual_seq_kvlen"] = kv_seqlen.copy()
            self.gen["actual_seq_qlen"] = q_seqlen_output.copy()
            if self.i_test["is_alibi"]:
                alibi = np.random.uniform(-10.0, 0.0, size=[1, n, g, sq, max_skv])
                alibi = alibi.astype(np.float16).astype(input_type)
                self.gen["real_shift"] = alibi.copy()
            if self.i_test["is_amask"]:
                amask = np.random.randint(0, 2, size=[b, 1, 1, sq, max_skv])
                amask *= -10000
                amask = amask.astype(np.float16)
                self.gen["attn_mask"] = amask.copy()
            if is_dropout:  # 0 is drop, 1 is keep
                dmask = np.random.uniform(size=[b, 1, 1, sq, max_skv]) > drop_prop
                dmask_uint8 = data2uint8(dmask)
                self.gen["drop_mask"] = dmask_uint8.copy()
        else:  # load
            q = self.load["query"].copy()
            k = self.load["key"].copy()
            v = self.load["value"].copy()
            if layout == "BSH":
                q = np.reshape(q, newshape=[b, sq, n, g, d])
                k = np.reshape(k, newshape=[b, max_skv, n, 1, d])
                v = np.reshape(v, newshape=[b, max_skv, n, 1, d])
            else:  # BNSD
                q = np.reshape(q, newshape=[b, n, g, sq, d])
                k = np.reshape(k, newshape=[b, n, 1, max_skv, d])
                v = np.reshape(v, newshape=[b, n, 1, max_skv, d])
            kv_seqlen = self.load["actual_seq_kvlen"].copy()
            if self.i_test["is_alibi"]:
                alibi = self.load["real_shift"].copy()
                alibi = alibi.reshape([1, n, g, sq, max_skv])
            if self.i_test["is_amask"]:
                amask = self.load["attn_mask"].copy()
                amask = np.reshape(amask, newshape=[b, 1, 1, sq, max_skv])
            if is_dropout:  # 0 is drop, 1 is keep
                dmask_uint8 = self.load["drop_mask"].copy()
                dmask = uint82data(dmask_uint8)
                dmask = np.reshape(dmask, newshape=[b, 1, 1, sq, max_skv])

        if diff_skv:  # max_skv to skv
            if layout == "BSH":
                k = k[:, :skv, :, :, :]  # [b, skv, n, 1, d]
                v = v[:, :skv, :, :, :]  # [b, skv, n, 1, d]
            else:  # BNSD
                k = k[:, :, :, :skv, :]  # [b, n, 1, skv, d]
                v = v[:, :, :, :skv, :]  # [b, n, 1, skv, d]
            if self.i_test["is_alibi"]:
                alibi = alibi[:, :, :, :, :skv]  # [b, n, g, sq, skv]
            if self.i_test["is_amask"]:
                amask = amask[:, :, :, :, :skv]  # [b, 1, 1, sq, skv]
            if is_dropout:  # 0 is drop, 1 is keep
                dmask = dmask[:, :, :, :, :skv]  # [b, 1, 1, sq, skv]

        if layout == "BSH":
            q = np.transpose(q, axes=(0, 2, 3, 1, 4))  # [b, n, g, sq, d]
            k = np.transpose(k, axes=(0, 2, 3, 1, 4))  # [b, n, 1, skv, d]
            v = np.transpose(v, axes=(0, 2, 3, 1, 4))  # [b, n, 1, skv, d]

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
        elif self.i_test["is_amask"]:
            s1 = s1 + amask.astype(np.float32)
        elif self.i_test["low_tri"]:
            tri_mask = np.ones(shape=[b, 1, 1, sq, max_skv]).astype(np.float32)
            tri_mask = np.triu(tri_mask, 1)
            s1 = s1 - tri_mask.astype(np.float32) * np.float32(10000)

        score_max = np.max(s1, axis=-1, keepdims=True)  # [b, n, g, sq, 1]

        s2 = s1 - score_max  # [b, n, g, sq, skv]

        s3 = np.exp(s2)  # [b, n, g, sq, skv]
        if is_dropout:
            s3 = s3 * dmask.astype(np.float32) / (1 - np.float32(drop_prop))

        score_sum = np.sum(s3, axis=-1, keepdims=True)  # [b, n, g, sq, 1]

        p = s3 / score_sum  # [b, n, g, sq, skv]

        o = np.matmul(p, v)  # [b, n, g, sq, d]
        o = o.astype(input_type)

        if layout == "BSH":
            o = np.transpose(o, axes=(0, 3, 1, 2, 4))  # [b, sq, n, g, d]

        if diff_skv:
            s0 = concat_zeros(s0, 4, diff_skv)  # [b, n, g, sq, max_skv]
            p = concat_zeros(p, 4, diff_skv)  # [b, n, g, sq, max_skv]

        self.gen["s"] = s0.astype(np.float32).copy()
        self.gen["softmax_max"] = score_max.astype(np.float32).copy()
        self.gen["softmax_sum"] = score_sum.astype(np.float32).copy()
        self.gen["softmax_out"] = p.astype(np.float32).copy()
        self.gen["attention_out"] = o.copy()

    def set_inputs(self):
        if self.i_test["type"] == "float32":
            ms_type = ms.bfloat16
        else:  # float16
            ms_type = ms.float16

        self.i_construct = {
            "query": None,  # dtype: tensor
            "key": None,  # dtype: tensor
            "value": None,  # dtype: tensor
            "real_shift": None,  # dtype: tensor
            "drop_mask": None,  # dtype: tensor
            "padding_mask": None,  # dtype: tensor
            "attn_mask": None,  # dtype: tensor
            "prefix": None,  # dtype: tuple[int]
            "actual_seq_qlen": None,  # dtype: tuple[int]
            "actual_seq_kvlen": None  # dtype: tuple[int]
        }
        # query key value
        if self.i_test["layout"] == "BSH":
            qkv = [None, None, None]
        else:  # BNSD
            qkv = [None, None, None, None]
        self.i_construct["query"] = ms.Tensor(shape=qkv, dtype=ms_type)
        self.i_construct["key"] = ms.Tensor(shape=qkv, dtype=ms_type)
        self.i_construct["value"] = ms.Tensor(shape=qkv, dtype=ms_type)
        # real_shift drop_mask attn_mask
        if self.i_test["is_alibi"]:
            self.i_construct["real_shift"] = ms.Tensor(
                shape=[None, None, None, None], dtype=ms_type)
        if self.i_test["is_amask"]:
            self.i_construct["attn_mask"] = ms.Tensor(
                shape=[None, None, None, None], dtype=ms.float16)
        if self.i_test["drop_prop"] > 0.0:
            self.i_construct["drop_mask"] = ms.Tensor(
                shape=[None, None, None, None], dtype=ms.float16)
        # actual_seq_qlen actual_seq_kvlen
        self.i_construct["actual_seq_qlen"] = ms.Tensor(shape=[None], dtype=ms.int32)
        self.i_construct["actual_seq_kvlen"] = ms.Tensor(shape=[None], dtype=ms.int32)

        self.net.set_inputs(*tuple(self.i_construct.values()))

    def calc_actual_func(self):
        if self.i_test["type"] == "float32":
            ms_type = ms.bfloat16
        else:  # float16
            ms_type = ms.float16

        self.i_construct = {
            "query": None,  # dtype: tensor
            "key": None,  # dtype: tensor
            "value": None,  # dtype: tensor
            "real_shift": None,  # dtype: tensor
            "drop_mask": None,  # dtype: tensor
            "padding_mask": None,  # dtype: tensor
            "attn_mask": None,  # dtype: tensor
            "prefix": None,  # dtype: tuple[int]
            "actual_seq_qlen": None,  # dtype: tuple[int]
            "actual_seq_kvlen": None  # dtype: tuple[int]
        }
        # query key value
        if self.i_test["layout"] == "BSH":
            B, Sq, N, G, D = self.gen["query"].shape
            query = np.reshape(self.gen["query"], newshape=[B, Sq, N * G * D])
            B, MaxSkv, N, _, D = self.gen["key"].shape
            key = np.reshape(self.gen["key"], newshape=[B, MaxSkv, N * D])
            value = np.reshape(self.gen["value"], newshape=[B, MaxSkv, N * D])
        else:  # BNSD
            B, N, G, Sq, D = self.gen["query"].shape
            query = np.reshape(self.gen["query"], newshape=[B, N * G, Sq, D])
            B, N, _, MaxSkv, D = self.gen["key"].shape
            key = np.reshape(self.gen["key"], newshape=[B, N, MaxSkv, D])
            value = np.reshape(self.gen["value"], newshape=[B, N, MaxSkv, D])
        self.i_construct["query"] = ms.Tensor(query, dtype=ms_type)
        self.i_construct["key"] = ms.Tensor(key, dtype=ms_type)
        self.i_construct["value"] = ms.Tensor(value, dtype=ms_type)
        # real_shift drop_mask attn_mask
        if self.i_test["is_alibi"]:
            B, N, G, Sq, MaxSkv = self.gen["real_shift"].shape
            real_shift = np.reshape(self.gen["real_shift"], newshape=[B, N * G, Sq, MaxSkv])
            self.i_construct["real_shift"] = ms.Tensor(real_shift, dtype=ms_type)
        if self.i_test["is_amask"]:
            B, _, _, Sq, MaxSkv = self.gen["attn_mask"].shape
            amask = np.reshape(self.gen["attn_mask"], newshape=[B, 1, Sq, MaxSkv])
            self.i_construct["attn_mask"] = ms.Tensor(amask, dtype=ms.float16)
        if self.i_test["drop_prop"] > 0.0:
            B, N, _, Sq, MaxSkv = self.gen["drop_mask"].shape
            drop_mask = np.reshape(self.gen["drop_mask"], newshape=[B, N, Sq, MaxSkv])
            self.i_construct["drop_mask"] = ms.Tensor(drop_mask, dtype=ms.uint8)
        # actual_seq_qlen actual_seq_kvlen
        self.i_construct["actual_seq_qlen"] = ms.Tensor(
            self.gen["actual_seq_qlen"], dtype=ms.int32)
        self.i_construct["actual_seq_kvlen"] = ms.Tensor(
            self.gen["actual_seq_kvlen"], dtype=ms.int32)

        self.o_ascend = {
            "softmax_max": None,  # dtype: tensor
            "softmax_sum": None,  # dtype: tensor
            "softmax_out": None,  # dtype: tensor
            "attention_out": None,  # dtype: tensor
        }

        if "msprof" in self.i_test and self.i_test["msprof"]:
            profiler = Profiler(start_profile=False, output_path="./profiler", data_simplification=False)
            profiler.start()
            for _ in range(self.i_test["msprof"]):
                self.o_ascend["softmax_max"], \
                self.o_ascend["softmax_sum"], \
                self.o_ascend["softmax_out"], \
                self.o_ascend["attention_out"] = self.net(*tuple(self.i_construct.values()))
            profiler.stop()
            profiler.analyse()
        else:
            self.o_ascend["softmax_max"], \
            self.o_ascend["softmax_sum"], \
            self.o_ascend["softmax_out"], \
            self.o_ascend["attention_out"] = self.net(*tuple(self.i_construct.values()))

    def compare(self):
        expect = self.gen["attention_out"]
        actual = self.o_ascend["attention_out"]
        if self.i_test["type"] in {"float32", np.float32}:
            actual = actual.to(ms.float32)
        actual = actual.numpy()
        actual = np.reshape(actual, newshape=expect.shape)
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


description = {
    "type": "Type of query/key/value/real_shift/attention_out. float16 or float32. \
        If type is float32, kernel will calculate with bfloat16.",
    "layout": "BSH or BNSD. Where H = N * D",
    "b": "Batch size. int",
    "sq": "Sequence length of query. int.",
    "skv": "Real sequence length of key_cache and value_cache. max/int/rand-1. \
        If skv is string of max, skv will be same with max_skv. If skv is int, skv will be the number you set. \
        If skv is string of rand-1, skv will generated with random.",
    "max_skv": "Max sequence length of key_cache and value_cache. int.",
    "nq": "Number head of query. int.",
    "nkv": "Number head of key_cache and value_cache. int.",
    "d": "Head dim size. int.",
    "drop_prop": "Drop probability. float. If drop_prop > 0.0, drop_mask will be use after exp, \
        the type of drop_mask is uint8. If drop_prop is 0.0, drop_mask is none.",
    "is_alibi": "Whether use real_shift. bool. If is_alibi is true, kernel will add real_shift after q@k_t/(dk)**0.5, \
        the type of real_shift is type. If is_alibi is false, real_shift is none.",
    "is_amask": "Whether use attn_mask. bool. If is_amask is true, kernel will use attn_mask after alibi_mask, \
        the type of attn_mask is float16. If is_amask is false, attn_mask is none.",
    "low_tri": "Whether open lower triangle. bool.",
    "blk_sparse": "Block sparse. int. Not use for now.",
    "msprof": "Open msprof. int. Kernel will run the val you set times, 0 is off."
}


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flash_attention_score_sd(ctx_mode):
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op in BNSD.
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BNSD",
        "b": 2,
        "sq": 1024,
        "skv": "max",
        "max_skv": 1024,
        "nq": 8,
        "nkv": 8,
        "d": 32,
        "drop_prop": 0.0,
        "is_alibi": False,
        "is_amask": False,
        "low_tri": False,
        "blk_sparse": 0,
        "msprof": 0
    }
    FlashAttentionScoreTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flash_attention_score_bnsd_64(ctx_mode):
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op in BNSD end embed 64.
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BNSD",
        "b": 2,
        "sq": 1024,
        "skv": "max",
        "max_skv": 1024,
        "nq": 9,
        "nkv": 3,
        "d": 64,
        "drop_prop": 0.0,
        "is_alibi": False,
        "is_amask": False,
        "low_tri": False,
        "blk_sparse": 0,
        "msprof": 0
    }
    FlashAttentionScoreTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flash_attention_score_bsh(ctx_mode):
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op in BSH.
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float32",
        "layout": "BSH",
        "b": 2,
        "sq": 791,
        "skv": "max",
        "max_skv": 1727,
        "nq": 9,
        "nkv": 3,
        "d": 128,
        "drop_prop": 0.0,
        "is_alibi": False,
        "is_amask": False,
        "low_tri": True,
        "blk_sparse": 0,
        "msprof": 0
    }
    FlashAttentionScoreTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flash_attention_score_bsh_mask(ctx_mode):
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op in BSH with attn_mask.
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 2,
        "sq": 1024,
        "skv": "max",
        "max_skv": 1024,
        "nq": 9,
        "nkv": 3,
        "d": 128,
        "drop_prop": 0.0,
        "is_alibi": False,
        "is_amask": True,
        "low_tri": False,
        "blk_sparse": 0,
        "msprof": 0
    }
    FlashAttentionScoreTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flash_attention_score_bsh_mask_alibi(ctx_mode):
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op with alibi_mask.
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float32",
        "layout": "BSH",
        "b": 2,
        "sq": 791,
        "skv": "max",
        "max_skv": 1234,
        "nq": 9,
        "nkv": 3,
        "d": 128,
        "drop_prop": 0.0,
        "is_alibi": True,
        "is_amask": True,
        "low_tri": False,
        "blk_sparse": 0,
        "msprof": 0
    }
    FlashAttentionScoreTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flash_attention_score_fa_bsh_small(ctx_mode):
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op in BSH with small shape.
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 2,
        "sq": 5,
        "skv": "max",
        "max_skv": 5,
        "nq": 8,
        "nkv": 8,
        "d": 64,
        "drop_prop": 0.0,
        "is_alibi": False,
        "is_amask": False,
        "low_tri": False,
        "blk_sparse": 0,
        "msprof": 0
    }
    FlashAttentionScoreTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flash_attention_score_d256_amask_fp16(ctx_mode):
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op with embed 256 with attn_mask.
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 2,
        "sq": 512,
        "skv": "max",
        "max_skv": 1024,
        "nq": 9,
        "nkv": 3,
        "d": 256,
        "drop_prop": 0.0,
        "is_alibi": False,
        "is_amask": True,
        "low_tri": False,
        "blk_sparse": 0,
        "msprof": 0
    }
    FlashAttentionScoreTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flash_attention_score_d256_low_tri_bf16(ctx_mode):
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op with low-trangle mask in bfloat16.
    Expectation: the result is correct
    """
    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float32",
        "layout": "BSH",
        "b": 2,
        "sq": 512,
        "skv": "max",
        "max_skv": 1024,
        "nq": 9,
        "nkv": 3,
        "d": 256,
        "drop_prop": 0.0,
        "is_alibi": False,
        "is_amask": False,
        "low_tri": True,
        "blk_sparse": 0,
        "msprof": 0
    }
    FlashAttentionScoreTest(i_test)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ctx_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flash_attention_score_dynamic(ctx_mode):
    """
    Feature: test FlashAttentionScore op in kbk enabling infer_boost
    Description: test FlashAttentionScore op with low-trangle mask in float16.
    Expectation: the result is correct
    """
    i_test_dict = dict()

    i_test = {
        "ctx_mode": ctx_mode,
        "type": "float16",
        "layout": "BSH",
        "b": 1,
        "sq": 32,
        "skv": "max",
        "max_skv": 64,
        "nq": 14,
        "nkv": 2,
        "d": 128,
        "drop_prop": 0.0,
        "is_alibi": False,
        "is_amask": False,
        "low_tri": True,
        "blk_sparse": 0,
        "msprof": 0
    }
    i_test_dict["dynamic_shape0"] = i_test

    i_test = i_test.copy()
    i_test["sq"] = 64
    i_test_dict["dynamic_shape1"] = i_test

    FlashAttentionScoreTest(i_test_dict)
