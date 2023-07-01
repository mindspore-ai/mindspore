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
"""
A FlashAttention Layer.
"""
import math

import mindspore.numpy as mnp
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops._op_impl._custom_op.flash_attention.flash_attention_impl import get_flash_attention

__all__ = ['FlashAttention']


class FlashAttention(Cell):
    """Flash Attention Layer.

    This function contains the flash attention primitives used in FlashAttention (see paper)
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/pdf/2205.14135.pdf>`

    Specifically, it includes the following:

    1. An interface for calling flashattention operation.
    2. Two configuration parameters for enabling local block sparse of flashattention.

    Args:
        head_dim(int): The hidden size of input.
        dropout_rate(float): The dropout rate of the attention score. Default 0.0.
        prev_block_num(int): A integer to define the number of blocks to look ahead for local block sparse attention.
            Default 65536.
        next_block_num(int): A integer to define the number of blocks to look behind for local block sparse attention.
            Default 65536.
        tiling_stgy_name(str): A str to define tiling strategy of flash attention.
        dp(int): data parallel.
            Default 1.
        mp(int): model parallel.
            Default 1.
        high_precision(bool): This mode has higher precision but some performance loss.
            Default False.
        have_attention_mask_batch(bool): indicates whether attention_mask contains the batch dimension.
            Default True
        alibi(bool): This parameter indicates whether the flashattention supports the Alibi.
            Default: False


    Inputs:
      - **q** (Tensor) - Tensor query (:class:`mstype.fp16` [batch_size, head_num, seq_length, head_dim])
      - **k** (Tensor) - Tensor key (:class:`mstype.fp16` [batch_size, head_num, seq_length, head_dim])
      - **v** (Tensor) - Tensor value (:class:`mstype.fp16` [batch_size, head_num, seq_length, head_dim])
      - **attention_mask** (Tensor) - Float Tensor the mask of (:class:`mstype.fp16` [batch_size, seq_length,
          seq_length]): A matrix to pass masked information.

    Outputs:
        A Tensor. The output of the attention with shape [batch_size, head_num, seq_length, head_dim]

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import dtype as mstype
        >>> from mindspore.nn.layer.flash_attention import FlashAttention
        >>> from mindspore import Tensor
        >>> model = FlashAttention(head_dim=128,
        ...                        dropout_rate=0.1,
        ...                        prev_block_num=7,
        ...                        next_block_num=0
        ...                        )
        >>> q = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> k = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> v = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> attention_mask = Tensor(np.ones((2, 4096, 4096)), mstype.float16)
        >>> output = model(q, k, v, attention_mask)
        >>> print(output.shape)
        (2, 16, 4096, 128)
    """

    def __init__(self,
                 head_dim,
                 dropout_rate=0.0,
                 prev_block_num=65536,
                 next_block_num=65536,
                 tiling_stgy_name="xunfei",
                 dp=1,
                 mp=1,
                 high_precision=False,
                 have_attention_mask_batch=False,
                 alibi=False
                 ):
        super(FlashAttention, self).__init__()

        self.flash_attention = get_flash_attention(
            prev_block_num=prev_block_num,
            next_block_num=next_block_num,
            tiling_stgy_name=tiling_stgy_name,
            high_precision=high_precision
        )
        self.flash_attention.add_prim_attr("primitive_target", "Ascend")

        self.scale_factor = Tensor([1. / math.sqrt(head_dim)], dtype=mstype.float16)
        self.dim_mask = Tensor([1 for _ in range(head_dim)], dtype=mstype.int8)
        self.scale_mul = ops.Mul().shard(((dp, mp, 1, 1), (1,)))
        self.dropout_rate = dropout_rate
        self.have_attention_mask_batch = have_attention_mask_batch
        self.alibi = alibi
        if self.dropout_rate > 1e-5:
            self.keep_prob = Tensor(1 - self.dropout_rate, dtype=mstype.float16)
            self.fill_v2 = ops.FillV2().shard(((dp, mp, 1, 1), ()))
            self.tensor_one = Tensor(1.0, mstype.float16)
            self.drop_gen_mask = ops.DropoutGenMask()
            self.do_dropout = ops.DropoutDoMask().shard(((dp, mp, 1, 1),))
            self.depend = ops.Depend()

    def shard(self, in_strategy=None, out_strategy=None):
        """Distributed configuration of FlashAttention
        :param in_strategy: Describe the split strategy of operator input. Default: None.
        :param out_strategy: Describe the split strategy of operator output, it is only for certain operators,
                                  such as MatMul. Default: None.
        :return:
        """
        if in_strategy is not None:
            shard_stgy = list(in_strategy)
            shard_stgy.insert(3, (1,))  # dim_mask
            shard_stgy = tuple(shard_stgy)
        else:
            # default: dp=1, mp=1, construct inputs only contain q, k, v
            shard_stgy = (
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                (1,),  # dim_mask
            )
        self.flash_attention.shard(shard_stgy)
        dp = shard_stgy[0][0]
        mp = shard_stgy[0][1]
        self.flash_attention.add_prim_attr("dev_matrix_shape", [dp, mp, 1, 1])
        inputs_tensor_map = [
            [3, 2, 1, 0],
            [3, 2, 1, 0],
            [3, 2, 1, 0],
            [-1]
        ]
        if self.have_attention_mask_batch:
            inputs_tensor_map.append([3, 1, 0])
        else:
            inputs_tensor_map.append([-1, 1, 0])

        # dropout_mask
        if self.dropout_rate > 1e-5:
            inputs_tensor_map.append([3, 2, 1, 0])

        if self.alibi:
            inputs_tensor_map.append([3, 2, 1, 0])

        self.flash_attention.add_prim_attr("inputs_tensor_map", inputs_tensor_map)

        self.flash_attention.add_prim_attr("outputs_tensor_map", [
            [3, 2, 1, 0],  # O
            [3, 2, 1],  # L
            [3, 2, 1]  # M
        ])
        self.flash_attention.add_prim_attr("as_loss_divisor", 0)
        self.flash_attention.add_prim_attr("empty_mirror_ops", 1)

    def construct(self, q, k, v, attn_mask=None, alibi_mask=None):
        """FlashAttention forward
        :param q:           [bsz, head_num, seq_len, head_dim]
        :param k:           [bsz, head_num, seq_len, head_dim]
        :param v:           [bsz, head_num, seq_len, head_dim]
        :param attn_mask:   [1 or bsz, seq_len, seq_len], if not None
        :param alibi_mask: [bsz, head_num, 1, seq_len], if not None
        :return: o          [bsz, head_num, seq_len, head_dim]
        """
        q = self.scale_mul(q, self.scale_factor)
        bsz, head_num, seq_len, head_dim = q.shape
        _, k_head_num, _, _ = k.shape
        _, v_head_num, _, _ = v.shape
        if head_num != k_head_num or head_num != v_head_num:
            raise ValueError(
                "the head_num of query, key and value must be the same, "
                "If different head_num are used, users need to change themselves to be same by tile.")
        if self.dropout_rate > 1e-5:
            drop_mask_bits = self.drop_gen_mask((bsz, head_num, seq_len, seq_len), self.keep_prob)
            tensor_shape = Tensor((bsz, head_num, seq_len, seq_len), mstype.int32)
            ones = self.fill_v2(tensor_shape, self.tensor_one)
            ones = self.depend(ones, q)
            drop_mask = self.do_dropout(ones, drop_mask_bits, self.keep_prob)
        else:
            drop_mask = None
        if head_dim > 304:
            raise ValueError(
                "the head_dim must be less than 304, otherwise the ub would be OOM.")
        if head_dim % 16 != 0:
            padding_size = 16 - head_dim % 16
            q = mnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_values=0)
            k = mnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_values=0)
            v = mnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_values=0)
            o, _, _ = self.flash_attention(q, k, v, self.dim_mask, attn_mask, drop_mask, alibi_mask)
            o = ops.slice(o, [0, 0, 0, 0], [bsz, head_num, seq_len, head_dim])
        else:
            o, _, _ = self.flash_attention(q, k, v, self.dim_mask, attn_mask, drop_mask, alibi_mask)
        return o
