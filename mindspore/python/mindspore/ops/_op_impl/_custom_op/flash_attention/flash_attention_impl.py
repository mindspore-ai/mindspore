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
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.ops import Custom
from mindspore.ops import DataType
from mindspore.ops import TBERegOp
from mindspore.ops._op_impl._custom_op.flash_attention.flash_attention_bwd import flash_attention_grad
from mindspore.ops._op_impl._custom_op.flash_attention.flash_attention_fwd import flash_attention
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like

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
    .input(0, "query", False, "required", "all") \
    .input(1, "key", False, "required", "all") \
    .input(2, "value", False, "required", "all") \
    .input(3, "dim_mask", False, "required", "all") \
    .input(4, "attn_mask", False, "optional", "all") \
    .input(5, "dropout_mask", False, "optional", "all") \
    .input(6, "alibi_mask", False, "optional", "all") \
    .output(0, "output", False, "required", "all") \
    .output(1, "rowsum", False, "required", "all") \
    .output(2, "rowmax", False, "required", "all") \
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

GRAD_KERNEL_NAME = "flash_attention_grad"

cus_flash_atten_grad_op_info = TBERegOp("FlashAttentionGradPrimitive") \
    .fusion_type("OPAQUE") \
    .partial_flag(True) \
    .async_flag(False) \
    .binfile_name("flash_attention_grad.so") \
    .compute_cost(10) \
    .kernel_name(GRAD_KERNEL_NAME) \
    .attr("prev_block_num", "required", "int", "all", "65536") \
    .attr("next_block_num", "required", "int", "all", "65536") \
    .attr("high_precision", "required", "bool", "all", "false") \
    .attr("tiling_stgy_name", "required", "str", "all", "xunfei") \
    .input(0, "query", False, "required", "all") \
    .input(1, "key", False, "required", "all") \
    .input(2, "value", False, "required", "all") \
    .input(3, "output", False, "required", "all") \
    .input(4, "do", False, "required", "all") \
    .input(5, "rowsum", False, "required", "all") \
    .input(6, "rowmax", False, "required", "all") \
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


def get_flash_attention_grad(prev_block_num=65536, next_block_num=65536,
                             tiling_stgy_name='xunfei', high_precision=False):
    """get flash attention grad"""

    def infer_shape(q_shape, k_shape, v_shape, o_shape, do_shape, l_shape, m_shape,
                    dim_mask_shape, att_mask_shape, dropout_mask_shape, alibi_mask_shape):
        return q_shape, k_shape, v_shape

    def infer_dtype(q_dtype, k_dtype, v_dtype, o_dytpe, do_dtype, l_dtype, m_dtype,
                    dim_mask_dtype, attn_mask_dtype, dropout_mask_dtype, alibi_mask_type):
        return mstype.float32, mstype.float32, mstype.float32

    fa_grad = Custom(flash_attention_grad, out_shape=infer_shape,
                     out_dtype=infer_dtype, func_type="tbe", reg_info=cus_flash_atten_grad_op_info)
    fa_grad.add_prim_attr("prev_block_num", prev_block_num)
    fa_grad.add_prim_attr("next_block_num", next_block_num)
    fa_grad.add_prim_attr("high_precision", high_precision)
    fa_grad.add_prim_attr("tiling_stgy_name", tiling_stgy_name)
    fa_grad.init_prim_io_names(
        inputs=["query", "key", "value", "output", "do", "rowsum", "rowmax", "dim_mask", "attn_mask", "dropout_mask",
                "alibi_mask"],
        outputs=["dq", "dk", "dv"]
    )

    def bprop(query, key, value, dim_mask, attn_mask, dropout_mask, alibi_mask, out, douts):
        output, rowsum, rowmax = out
        dout, _, _ = douts
        dq, dk, dv = fa_grad(query, key, value, output, dout, rowsum, rowmax, dim_mask, attn_mask, dropout_mask,
                             alibi_mask)
        dq = ops.cast(dq, mstype.float16)
        dk = ops.cast(dk, mstype.float16)
        dv = ops.cast(dv, mstype.float16)
        return dq, dk, dv, zeros_like(dim_mask), zeros_like(attn_mask), \
            zeros_like(dropout_mask), zeros_like(alibi_mask)

    return bprop


def get_flash_attention(prev_block_num=65536, next_block_num=65536, tiling_stgy_name='xunfei', high_precision=False):
    """get_flash_attention"""

    def infer_shape(q_shape, k_shape, v_shape, dim_mask_shape, attn_mask_shape=None,
                    dropout_mask_shape=None, alibi_mask_shape=None):
        """infer shape"""
        batch, hidden_size, seq_len, _ = q_shape
        l_shape = (batch, hidden_size, seq_len)
        m_shape = (batch, hidden_size, seq_len)
        return q_shape, l_shape, m_shape

    def infer_dtype(q_dtype, k_dtype, v_dtype, dim_mask_dtype, attn_mask_dtype=None,
                    dropout_mask_dtype=None, alibi_mask_type=None):
        """infer type"""
        l_dtype = mstype.float16
        if high_precision:
            l_dtype = mstype.float32
        return q_dtype, l_dtype, q_dtype

    fa_grad = get_flash_attention_grad(prev_block_num, next_block_num, tiling_stgy_name, high_precision)
    fa_forward = Custom(flash_attention, out_shape=infer_shape,
                        out_dtype=infer_dtype, func_type="tbe", bprop=fa_grad,
                        reg_info=cus_flash_atten_op_info)
    fa_forward.add_prim_attr("prev_block_num", prev_block_num)
    fa_forward.add_prim_attr("next_block_num", next_block_num)
    fa_forward.add_prim_attr("high_precision", high_precision)
    fa_forward.add_prim_attr("tiling_stgy_name", tiling_stgy_name)
    fa_forward.init_prim_io_names(
        inputs=["query", "key", "value", "dim_mask", "attn_mask", "dropout_mask", "alibi_mask"],
        outputs=["output", "rowsum", "rowmax"]
    )

    return fa_forward
