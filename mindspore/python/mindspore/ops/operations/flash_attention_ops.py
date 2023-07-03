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
"""flash attention primitive"""
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.ops import PrimitiveWithInfer
from mindspore.ops import prim_attr_register
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like


class FlashAttentionGradPrimitive(PrimitiveWithInfer):
    """The primitive of flash attention grad"""

    @prim_attr_register
    def __init__(self, name="flash_attention_grad"):
        super().__init__(name)
        # pylint: disable=unused-import
        from mindspore.ops._op_impl._custom_op.flash_attention.flash_attention_impl import flash_attention_grad_impl
        self.init_prim_io_names(
            inputs=["q", "k", "v", "o", "do", "l", "m", "dim_mask", "attn_mask", "dropout_mask", "alibi_mask"],
            outputs=["dq", "dk", "dv"]
        )

    def infer_shape(self, q_shape, k_shape, v_shape, o_shape, do_shape, l_shape, m_shape,
                    dim_mask_shape, att_mask_shape, dropout_mask_shape, alibi_mask_shape):
        return q_shape, k_shape, v_shape

    def infer_dtype(self, q_dtype, k_dtype, v_dtype, o_dytpe, do_dtype, l_dtype, m_dtype,
                    dim_mask_dtype, attn_mask_dtype, dropout_mask_dtype, alibi_mask_type):
        return mstype.float32, mstype.float32, mstype.float32


class FlashAttentionPrimitive(PrimitiveWithInfer):
    """The primitive of flash attention"""

    @prim_attr_register
    def __init__(self, prev_block_num=65536, next_block_num=65536, high_precision=False,
                 name="flash_attention", tiling_stgy_name='xunfei'):
        super().__init__(name)
        # pylint: disable=unused-import
        from mindspore.ops._op_impl._custom_op.flash_attention.flash_attention_impl import flash_attention_impl
        self.init_prim_io_names(
            inputs=["q", "k", "v", "dim_mask", "attn_mask", "dropout_mask", "alibi_mask"],
            outputs=["y", "l", "m"]
        )
        self.prev_block_num = prev_block_num
        self.next_block_num = next_block_num
        self.high_precision = high_precision
        self.tiling_stgy_name = tiling_stgy_name

    def infer_shape(self, q_shape, k_shape, v_shape, dim_mask_shape, attn_mask_shape=None,
                    dropout_mask_shape=None, alibi_mask_shape=None):
        """infer shape"""
        b, h, N, _ = q_shape
        l_shape = (b, h, N)
        m_shape = (b, h, N)
        return q_shape, l_shape, m_shape

    def infer_dtype(self, q_dtype, k_dtype, v_dtype, dim_mask_dtype, attn_mask_dtype=None,
                    dropout_mask_dtype=None, alibi_mask_type=None):
        """infer type"""
        l_dtype = mstype.float16
        if self.high_precision:
            l_dtype = mstype.float32
        return q_dtype, l_dtype, q_dtype

    def get_bprop(self):
        """get bprop"""
        flash_attention_grad = FlashAttentionGradPrimitive()
        flash_attention_grad.add_prim_attr("prev_block_num", self.prev_block_num)
        flash_attention_grad.add_prim_attr("next_block_num", self.next_block_num)
        flash_attention_grad.add_prim_attr("high_precision", self.high_precision)
        flash_attention_grad.add_prim_attr("tiling_stgy_name", self.tiling_stgy_name)

        def bprop(q, k, v, dim_mask, attn_mask, dropout_mask, alibi_mask, out, douts):
            o, l, m = out
            dout, _, _ = douts
            dq, dk, dv = flash_attention_grad(q, k, v, o, dout, l, m, dim_mask, attn_mask, dropout_mask, alibi_mask)
            dq = ops.cast(dq, mstype.float16)
            dk = ops.cast(dk, mstype.float16)
            dv = ops.cast(dv, mstype.float16)
            return dq, dk, dv, zeros_like(dim_mask), zeros_like(attn_mask), \
                zeros_like(dropout_mask), zeros_like(alibi_mask)

        return bprop
