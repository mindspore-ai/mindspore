#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
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
"""ascend custom op: kv_cache_mgr by tik"""

import functools
from tbe import tik
import tbe.common.platform as tbe_platform
from tbe.common.utils import para_check


# 'pylint: disable=unused-argument,unused-variable,too-many-arguments,too-many-locals
def check_supported(past, cur, index, out, kernel_name="kv_cache_mgr"):
    """check data type and shape"""
    # check data type
    past_dtype = past.get("dtype").lower()
    cur_dtype = cur.get("dtype").lower()
    out_dtype = out.get("dtype").lower()

    if past_dtype != cur_dtype or past_dtype != out_dtype:
        reason = "past_dtype is %s, cur_dtype is %s, out_dtype is %s" % (past_dtype, cur_dtype, out_dtype)
        return False, reason

    support_dtype_list = ["float32", "int32", "uint32",
                          "float16", "int16", "uint16",
                          "int8", "uint8"]
    if past_dtype not in support_dtype_list:
        reason = "past_dtype(%s) is not support" % (past_dtype)
        return False, reason

    index_dtype = index.get("dtype").lower()
    if index_dtype != "int32":
        reason = "index_dtype is %s, not int32" % (index_dtype)
        return False, reason

    # check shape
    past_shape = past.get("shape")
    cur_shape = cur.get("shape")

    if len(past_shape) != 4 or len(cur_shape) != 4:
        reason = "len(past_shape) != 4 or len(cur_shape) != 4 "
        return False, reason

    # key_past shape: (bs, num_heads, size_per_head, seq_length)
    # value_past shape: (bs, num_heads, seq_length, size_per_head)
    # key shape: (bs, num_heads, 1, size_per_head)
    # value shape: (bs, num_heads, 1, size_per_head)

    if past_shape[0] != cur_shape[0] or past_shape[1] != cur_shape[1]:
        reason = "past_shape[0] != cur_shape[0] or past_shape[1] != cur_shape[1] "
        return False, reason

    if past_shape[3] != cur_shape[3]:
        reason = "past_shape[3] != cur_shape[3]"
        return False, reason

    return True, ""


def ceil_div(dividend, divisor):
    return (dividend + divisor - 1) // divisor


def get_loop_info(total_num, each_loop_num):
    loop_times = ceil_div(total_num, each_loop_num)
    last_loop_num = total_num - each_loop_num * (loop_times - 1)
    return loop_times, last_loop_num


class TilingHelper:
    """Tiling parameter"""
    def __init__(self, past, cur, index, out, kernel_name="kv_cache_mgr"):
        self.kernel_name = kernel_name

        # sys info
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

        self.past_shape = past.get("shape")
        self.cur_shape = cur.get("shape")
        self.index_shape = index.get("shape")

        self.gm_type = past.get("dtype").lower()
        self.ub_type = self.gm_type
        self.index_ub_type = "int32"
        self.int32_size = 4

        self.gm_dtype_size = 2
        if self.gm_type in ["int8", "uint8"]:
            self.gm_dtype_size = 1
        elif self.gm_type in ["float16", "int16", "uint16"]:
            self.gm_dtype_size = 2
        elif self.gm_type in ["float32", "int32", "uint32"]:
            self.gm_dtype_size = 4

        # tiling policy
        self.seq_length = self.past_shape[2]
        self.size_per_head = self.past_shape[3]
        self.update_seq_length = self.cur_shape[2]

        self.num_head = self.past_shape[1]

        self.past_elements = functools.reduce(lambda a, b: a * b, self.past_shape)
        self.cur_elements = functools.reduce(lambda a, b: a * b, self.cur_shape)
        self.index_elements = functools.reduce(lambda a, b: a * b, self.index_shape)

        # split cur
        self.cur_bs = self.cur_shape[0] * self.cur_shape[1]
        self.each_core_bs_num = ceil_div(self.cur_bs, self.core_num)
        self.core_num, self.last_core_bs_num = get_loop_info(self.cur_bs, self.each_core_bs_num)
        self.each_core_cur_elements = ceil_div(self.cur_elements, self.core_num)
        self.cur_ub_elements = self.each_core_cur_elements


class KVCacheImpl(TilingHelper):
    """KVCacheImpl"""
    def __init__(self, past, cur, index, out, kernel_name):
        super().__init__(past, cur, index, out, kernel_name)
        # key_past or value_past shape: (bs, num_heads, seq_length, size_per_head)
        # batch_valid_length
        # cur update shape: (bs, num_heads, 1, size_per_head)

        self.tik_inst = tik.Tik(disable_debug=True)
        self.past_gm = self.tik_inst.Tensor(self.gm_type, (self.past_elements,), name="past_gm", scope=tik.scope_gm)
        self.cur_gm = self.tik_inst.Tensor(self.gm_type, (self.cur_elements,), name="cur_gm", scope=tik.scope_gm)
        self.index_gm = self.tik_inst.Tensor(self.index_ub_type, (self.index_elements,), name="index_gm",
                                             scope=tik.scope_gm)
        # we use is_atomic_add=True to set the out_gm zeros. But if inplace update out_gm, no need to set this flag.
        self.out_gm = self.tik_inst.Tensor(self.gm_type, (self.past_elements,), name="out_gm", scope=tik.scope_gm)

    def valid_cur_ub_load(self, core_idx):
        """KVCacheImpl.valid_cur_ub_load"""
        cur_ub = self.tik_inst.Tensor(self.ub_type, (self.cur_ub_elements,), name="valid_cur_ub",
                                      scope=tik.scope_ubuf)
        cur_gm_offset = core_idx * self.cur_ub_elements
        self.tik_inst.data_move(cur_ub, self.cur_gm[cur_gm_offset:], 0, 1,
                                self.cur_ub_elements * self.gm_dtype_size // 32, 0, 0)
        return cur_ub

    def valid_index_ub_load(self):
        """KVCacheImpl.valid_index_ub_load"""
        index_ub = self.tik_inst.Tensor(self.index_ub_type, (self.index_elements,), name="valid_index_ub",
                                        scope=tik.scope_ubuf)
        self.tik_inst.data_move(index_ub, self.index_gm, 0, 1, self.index_elements * self.int32_size // 32, 0, 0)
        return index_ub

    def valid_pos_update(self, core_idx, cur_ub, index_ub, each_core_bs_num):
        """KVCacheImpl.valid_pos_update"""
        src_bs_stride = self.update_seq_length * self.size_per_head
        dst_bs_stride = self.seq_length * self.size_per_head
        burst_len = self.update_seq_length * self.size_per_head * self.gm_dtype_size // 32

        valid_idx = self.tik_inst.Scalar(dtype="int32")
        with self.tik_inst.for_range(0, each_core_bs_num) as each_core_bs_idx:
            bs_idx = core_idx * each_core_bs_num + each_core_bs_idx
            # because we fused bs * num_head, we need get the real bs_idx
            valid_idx.set_as(index_ub[bs_idx // self.num_head])
            dst_offset = bs_idx * dst_bs_stride + valid_idx * self.size_per_head
            src_offset = each_core_bs_idx * src_bs_stride
            if burst_len < 65536:
                self.tik_inst.data_move(self.out_gm[dst_offset], cur_ub[src_offset], 0, 1, burst_len, 0, 0)
            else:
                nburst = 1
                each_burst_len = burst_len
                while each_burst_len > 65535:
                    nburst += 1
                    each_burst_len = burst_len // nburst
                self.tik_inst.data_move(self.out_gm[dst_offset], cur_ub[src_offset], 0, nburst, each_burst_len, 0, 0)

    # 'pylint: disable=too-many-arguments
    def compute_each_core(self, core_idx, core_bs_num):
        """KVCacheImpl.compute_each_core"""
        index_ub = self.valid_index_ub_load()
        cur_ub = self.valid_cur_ub_load(core_idx)
        self.valid_pos_update(core_idx, cur_ub, index_ub, core_bs_num)

    def compute(self):
        """KVCacheImpl.compute"""
        if self.each_core_bs_num == self.last_core_bs_num:
            with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_index:
                self.compute_each_core(core_idx=core_index, core_bs_num=self.each_core_bs_num)
        else:
            with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_index:
                with self.tik_inst.if_scope(core_index < self.core_num - 1):
                    self.compute_each_core(core_idx=core_index, core_bs_num=self.each_core_bs_num)
                with self.tik_inst.else_scope():
                    self.compute_each_core(core_idx=core_index, core_bs_num=self.last_core_bs_num)

        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.past_gm, self.cur_gm, self.index_gm],
                               outputs=[self.out_gm],
                               )
        return self.tik_inst


# 'pylint: disable = unused-argument
# 'pylint: disable=too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def kv_cache_mgr(past, cur, index, out, kernel_name="kv_cache_mgr"):
    """
    :param past: key_past or value_past. shape: (bs, num_head, seq_length, size_pre_head)
    :param cur: key_current or value_current. shape: (bs, num_head, update_seq_length, size_pre_head)
    :param index: which index to update. shape * len(dtype) need be multiples of 32. Option Input.
    :param out: output shape: (bs, num_head, seq_length, size_pre_head)
    :param kernel_name: the name of the op
    :return:
    """
    obj = KVCacheImpl(past, cur, index, out, kernel_name)
    return obj.compute()
