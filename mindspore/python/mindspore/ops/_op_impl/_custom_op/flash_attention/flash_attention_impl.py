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
"""The impl of flash attention"""
from __future__ import absolute_import

from mindspore.ops import DataType
from mindspore.ops import TBERegOp
from mindspore.ops import op_info_register
from mindspore.ops._op_impl._custom_op.flash_attention.flash_attention_bwd import flash_attention_grad
from mindspore.ops._op_impl._custom_op.flash_attention.flash_attention_fwd import flash_attention

KERNEL_NAME = "flash_attention"

cus_flash_atten_op_info = TBERegOp("FlashAttentionPrimitive") \
    .fusion_type("OPAQUE") \
    .partial_flag(True) \
    .async_flag(False) \
    .binfile_name("flash_attention.so") \
    .compute_cost(10) \
    .kernel_name(KERNEL_NAME) \
    .attr("prev_block_num", "required", "int", "all", "65536") \
    .attr("next_block_num", "required", "int", "all", "65536") \
    .attr("high_precision", "required", "bool", "all", "false") \
    .attr("tiling_stgy_name", "required", "str", "all", "xunfei") \
    .input(0, "q", False, "required", "all") \
    .input(1, "k", False, "required", "all") \
    .input(2, "v", False, "required", "all") \
    .input(3, "dim_mask", False, "required", "all") \
    .input(4, "attn_mask", False, "optional", "all") \
    .input(5, "dropout_mask", False, "optional", "all") \
    .input(6, "alibi_mask", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .output(1, "l", False, "required", "all") \
    .output(2, "m", False, "required", "all") \
    .dtype_format(DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.I8_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default) \
    .dtype_format(DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.I8_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F32_Default,
                  DataType.F16_Default) \
    .get_op_info()


# Binding kernel info with the kernel implementation.
@op_info_register(cus_flash_atten_op_info)
def flash_attention_impl(query, key, value, dim_mask, attn_mask, dropout_mask, alibi_mask, y, l,
                         m, prev_block_num, next_block_num, high_precision, tiling_stgy_name):
    flash_attention(query, key, value, dim_mask, attn_mask, dropout_mask, alibi_mask,
                    y, l, m, prev_block_num, next_block_num,
                    high_precision=high_precision,
                    kernel_name=KERNEL_NAME,
                    tiling_stgy_name=tiling_stgy_name)


GRAD_KERNEL_NAME = "flash_attention_grad"

cus_flash_atten_grad_op_info = TBERegOp("FlashAttentionGradPrimitive") \
    .fusion_type("OPAQUE") \
    .partial_flag(True) \
    .async_flag(False) \
    .binfile_name("flash_attention_grad.so") \
    .compute_cost(10) \
    .kernel_name(GRAD_KERNEL_NAME) \
    .attr("prev_block_num", "required", "int", "all", "65536")\
    .attr("next_block_num", "required", "int", "all", "65536")\
    .attr("high_precision", "required", "bool", "all", "false") \
    .attr("tiling_stgy_name", "required", "str", "all", "xunfei")\
    .input(0, "q", False, "required", "all") \
    .input(1, "k", False, "required", "all") \
    .input(2, "v", False, "required", "all") \
    .input(3, "o", False, "required", "all") \
    .input(4, "do", False, "required", "all") \
    .input(5, "l", False, "required", "all") \
    .input(6, "m", False, "required", "all") \
    .input(7, "dim_mask", False, "required", "all") \
    .input(8, "attn_mask", False, "optional", "all") \
    .input(9, "dropout_mask", False, "optional", "all") \
    .input(10, "alibi_mask", False, "optional", "all") \
    .output(0, "dq", False, "required", "all") \
    .output(1, "dk", False, "required", "all") \
    .output(2, "dv", False, "required", "all") \
    .dtype_format(DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.I8_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F32_Default,
                  DataType.F32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F32_Default,
                  DataType.F16_Default,
                  DataType.I8_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F16_Default,
                  DataType.F32_Default,
                  DataType.F32_Default,
                  DataType.F32_Default) \
    .get_op_info()


# Binding kernel info with the kernel implementation.
@op_info_register(cus_flash_atten_grad_op_info)
def flash_attention_grad_impl(q, k, v, o, dout, l, m, dim_mask, attn_mask, dropout_mask, alibi_mask,
                              dq, dk, dv, prev_block_num, next_block_num,
                              high_precision, tiling_stgy_name="xunfei"):
    flash_attention_grad(q, k, v, o, dout, l, m, dim_mask, attn_mask, dropout_mask, alibi_mask,
                         dq, dk, dv, prev_block_num, next_block_num,
                         high_precision=high_precision,
                         kernel_name=GRAD_KERNEL_NAME,
                         tiling_stgy_name=tiling_stgy_name)
