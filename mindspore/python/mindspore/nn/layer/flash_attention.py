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

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore._c_expression import MSContext

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
        use_mqa(bool): Using MHA if True, only take effect under 910B or 910C. Default: False.


    Inputs:
      - **query** (Tensor) - Tensor query (:class:`mstype.fp16` [batch_size, head_num, seq_length, head_dim])
      - **key** (Tensor) - Tensor key (:class:`mstype.fp16` [batch_size, head_num, seq_length, head_dim])
      - **value** (Tensor) - Tensor value (:class:`mstype.fp16` [batch_size, head_num, seq_length, head_dim])
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
        >>> query = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> key = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> value = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> attention_mask = Tensor(np.ones((2, 4096, 4096)), mstype.float16)
        >>> output = model(query, key, value, attention_mask)
        >>> print(output.shape)
        (2, 16, 4096, 128)
    """

    def __init__(self,
                 head_dim,
                 head_num,
                 dropout_rate=0.0,
                 prev_block_num=65536,
                 next_block_num=65536,
                 dp=1,
                 mp=1,
                 high_precision=False,
                 have_attention_mask_batch=True,
                 alibi=False,
                 use_mqa=False
                 ):
        super(FlashAttention, self).__init__()
        device = MSContext.get_instance().get_ascend_soc_version()
        valid_device = ["ascend910b", "ascend910c"]
        if device not in valid_device:
            raise ValueError(f"nn.FlashAttention only support {valid_device}, but current device is {device}")
        if alibi:
            raise ValueError(f"Alib is not support currently")
        scaling_constant = math.sqrt(head_dim)
        if scaling_constant == 0:
            raise ValueError("the scaling constant must not be 0.")
        self.attn_expand_dims = ops.ExpandDims().shard(((dp, 1, 1),))
        if use_mqa:
            fa_strategies = ((dp, mp, 1, 1),
                             (dp, 1, 1, 1),
                             (dp, 1, 1, 1),
                             (dp, 1, 1, 1))
        else:
            fa_strategies = ((dp, mp, 1, 1),
                             (dp, mp, 1, 1),
                             (dp, mp, 1, 1),
                             (dp, 1, 1, 1))
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 1e-5:
            self.keep_prob = Tensor(1 - self.dropout_rate, dtype=mstype.float16)
            self.drop_gen_mask = ops.DropoutGenMask()
            fa_strategies += ((dp, mp, 1, 1),)
        self.flash_attention = FlashAttentionScore(head_num=head_num, pre_tokens=prev_block_num,
                                                   next_tokens=next_block_num,
                                                   keep_prob=1 - dropout_rate,
                                                   scale_value=1. / scaling_constant,
                                                   inner_precise=0 if high_precision else 1,
                                                   input_layout="BNSD").shard(fa_strategies)

    def shard(self, in_strategy=None, out_strategy=None):
        """Distributed configuration of FlashAttention
        :param in_strategy: Describe the split strategy of operator input. Default: None.
        :param out_strategy: Describe the split strategy of operator output, it is only for certain operators,
                                  such as MatMul. Default: None.
        :return:
        """
        self.flash_attention.shard(in_strategy)

    def construct(self, query, key, value, attn_mask=None, alibi_mask=None):
        """FlashAttention forward
        :param query:           [bsz, head_num, seq_len, head_dim]
        :param key:           [bsz, head_num, seq_len, head_dim]
        :param value:           [bsz, head_num, seq_len, head_dim]
        :param attn_mask:   [1 or bsz, seq_len, seq_len], if not None
        :param alibi_mask: [bsz, head_num, 1, seq_len], if not None
        :return: output          [bsz, head_num, seq_len, head_dim]
        """
        bsz, head_num, seq_len, _ = query.shape
        # 910B/910C -- FlashAttentionScore
        if self.dropout_rate > 1e-5:
            drop_mask_bits = self.reshape(self.drop_gen_mask((bsz, head_num, seq_len, seq_len), self.keep_prob),
                                          (bsz, head_num, seq_len, seq_len // 8))
        else:
            drop_mask_bits = None
        # (B, N, S, D) -> (B, S, H)
        attn_mask = attn_mask.reshape((bsz, 1, seq_len, seq_len))
        output, _, _ = self.flash_attention(query,
                                            key,
                                            value,
                                            attn_mask,
                                            drop_mask_bits,
                                            None,
                                            None,
                                            None)
        return output
