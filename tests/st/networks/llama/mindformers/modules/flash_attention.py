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
"""Flash Attention Layer"""
import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import functional as F
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention Layer.

    This function contains the flash attention primitives used in FlashAttention (see paper)
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/pdf/2205.14135.pdf>`

    Specifically, it includes the following:

    1. An interface for calling flashattention operation.
    2. Two configuration parameters for enabling local block sparse of flashattention.

    B -- Batch size
    S1 -- Sequence length of query. The value ranges from 1 to 32768 and is a multiple of 16.
    S2 -- Sequence length of key and value. The value ranges from 1 to 32768 and is a multiple of 16.
    N1 -- Num heads of query
    N2 -- Num heads of key and value, and N2 must be a factor of N1
    D -- Head size. Support value: 64, 80, 96, 120, 128 and 256.
    H1 -- Hidden size of query, which equals to N1 * D
    H2 -- Hidden size of key and value, which equals to N2 * D
    Args:
        head_num (int): The head num of query.
        keep_prob (float): The keep probability of dropout. Default: 1.0.
        scale_value (float): The scale factor of score. Default: 1.0.
        pre_tokens (int): Parameter for sparse computation, represents how many tokens are counted forward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        next_tokens (int): Parameter for sparse computation, represents how many tokens are counted backward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        input_layout (str): Specifies the layout of input `query`, key and value. The value can be "BSH" or "BNSD".
        Default: "BSH".
        sparse_mode (int): Indicates sparse mode. Default 0.

            - 0: Indicates the defaultMask mode. If attn_mask is not passed, the mask operation is not performed,
              and preTokens and nextTokens(internally assigned as INT_MAX) are ignored. If passed in, the full attn_mask
              matrix (S1 * S2) needs to be passed in, indicating that the part between preTokens and nextTokens needs to
              be calculated.
            - 1: Represents allMask, that is, passing in the complete attn_mask matrix.
            - 2: Representing the leftUpCausal mode corresponds to the lower triangle scenario divided by the left
              vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 3: Representing the rightDownCausal model corresponds to the lower triangle scene divided by the lower
              right vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 4: Represents the band scenario, that is, the part between counting preTokens and nextTokens, and the
              optimized attn_mask matrix (2048*2048) is required..
            - 5: Represents the prefix scenario, that is, on the basis of rightDownCasual, a matrix with length S1 and
              width N is added to the left side. The value of N is obtained by the new input prefix, and the N value of
              each Batch axis is different. Not implemented yet.
            - 6: Represents the global scenario, not implemented yet.
            - 7: Represents the dilated scenario, not implemented yet.
            - 8: Represents the block_local scenario, not implemented yet.
        use_attention_mask (bool): The value is True if attention_mask is passed. Default: False.
        use_alibi_mask (bool): The value is True if alibi_mask is passed. Default: False.
        use_mqa (bool): Specifies whether using MQA. Default: False.
        dp (int): Data parallel num.
        mp (int): Model parallel num.
        sp (int): Sequence parallel num.


    Inputs:
        - **query** (Tensor[float16, bfloat16]) - The query tensor.
          Input tensor of shape :math:`(B, S1, H1)` or `(B, N1, S1, D)`.
        - **key** (Tensor[float16, bfloat16]) - The key tensor.
          Input tensor of shape :math:`(B, S2, H2)` or `(B, N2, S2, D)`.
        - **value** (Tensor[float16, bfloat16]) - The value tensor.
          Input tensor of shape :math:`(B, S2, H2)` or `(B, N2, S2, D)`.
        - **attn_mask** (Union[Tensor[uint8], None]) - The attention mask tensor. For each element, 0 indicates
          retention and 1 indicates discard. Input tensor of shape :math:`(B, N1, S1, S2)`, `(B, 1, S1, S2)`, `(S1, S2)`
          or (2048, 2048).
        - **alibi_mask** (Union[Tensor[float16, bfloat16], None]) - The position embedding code. If S is greater than
          1024 and the mask of the lower triangle is used, enter only the inverse 1024 lines of the lower triangle for
          memory optimization.
          Input tensor of shape :math:`(B, N1, S1, S2)`, `(1, N1, S1, S2)`, `(B, N1, 1024, S2)`, `(1, N1, 1024, S2)`
          or (1024, 1024).
        - **padding_mask** (None) - Reserved parameter. Not implemented yet.
        - **prefix** (Union[Tensor[int64], None]) - N value of each Batch in the prefix sparse calculation scenario.
          Not implemented yet. Input tensor of shape :math:`(B,)`.

    Outputs:
        - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
          are the same as the query.

    Supported Platforms:
        ``Ascend910B``

    Examples:
        >>> import numpy as np
        >>> import math
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> bsz, head_num, seq_len, head_size = 1, 16, 4096, 128
        >>> hidden_size = head_num * head_size
        >>> query = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
        >>> key = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
        >>> value = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
        >>> attn_mask = Tensor(np.ones((bsz, 1, seq_len, seq_len)), mstype.uint8)
        >>> model = FlashAttention(head_num,
                                   keep_prob=1.0,
                                   scale_value=1.0 / math.sqrt(head_dim),
                                   pre_tokens=2147483647,
                                   next_tokens=2147483647,
                                   input_layout="BSH",
                                   sparse_mode=0,
                                   use_attention_mask=True,
                                   use_alibi_mask=False,
                                   use_mqa=False,
                                   dp=1,
                                   mp=1,
                                   sp=1
        ...                        )
        >>> output = model(query, key, value, attn_mask)
        >>> print(output.shape)
        (1, 16, 2048)
    """

    def __init__(self,
                 head_num,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 input_layout="BSH",
                 sparse_mode=0,
                 use_attention_mask=True,
                 use_alibi_mask=False,
                 use_mqa=False,
                 dp=1,
                 mp=1,
                 sp=1,
                 ):
        super(FlashAttention, self).__init__()
        self.head_num = head_num
        self.enable_dropout = keep_prob < 1.0
        self.input_layout = input_layout
        self.sparse_mode = sparse_mode
        self.use_alibi_mask = use_alibi_mask
        self.use_attention_mask = use_attention_mask
        self.use_mqa = use_mqa
        self.dp = dp
        self.mp = mp
        self.sp = sp

        fa_strategies = self._generate_flash_attention_strategy(dp, mp, sp)
        self.flash_attention = FlashAttentionScore(head_num=head_num,
                                                   keep_prob=keep_prob,
                                                   scale_value=scale_value,
                                                   pre_tokens=pre_tokens,
                                                   next_tokens=next_tokens,
                                                   inner_precise=0,
                                                   input_layout=self.input_layout,
                                                   sparse_mode=self.sparse_mode).shard(fa_strategies)
        if self.use_alibi_mask:
            self.alibi_rescale_factor = Tensor([1.0 / scale_value], dtype=mstype.float16)
            self.alibi_rescale_mul = ops.Mul().shard(((dp, mp, sp, 1), (1,)))
        if self.enable_dropout:
            self.keep_prob_tensor = Tensor(keep_prob, dtype=mstype.float16)
            self.drop_gen_mask = ops.DropoutGenMask()

    def _generate_flash_attention_strategy(self, dp, mp, sp):
        """get FA generate strategies"""
        kv_head_split_num = 1 if self.use_mqa else mp
        if self.input_layout == "BSH":
            fa_strategies = ((dp, sp, mp),
                             (dp, 1, kv_head_split_num),
                             (dp, 1, kv_head_split_num))
        else:
            fa_strategies = ((dp, mp, sp, 1),
                             (dp, kv_head_split_num, 1, 1),
                             (dp, kv_head_split_num, 1, 1))
        if self.use_alibi_mask:
            fa_strategies += ((dp, mp, sp, 1),)
        if self.enable_dropout:
            fa_strategies += ((dp, mp, sp, 1),)
        if self.use_attention_mask:
            if self.sparse_mode in [0, 1]:
                fa_strategies += ((dp, 1, sp, 1),)
            else:
                raise RuntimeError(f"sparse_mode: {self.sparse_mode} is not support currently")

        return fa_strategies

    def construct(self, query, key, value, attn_mask=None, alibi_mask=None, prefix=None, padding_mask=None):
        """Forward process of the AttentionMaskMF"""
        if self.input_layout == "BSH":
            bsz, q_seq_len, _ = query.shape
            _, kv_seq_len, _ = key.shape
        else:
            bsz, _, q_seq_len, _ = query.shape
            _, _, kv_seq_len, _ = key.shape
        if self.enable_dropout:
            drop_mask_bits = F.reshape(
                self.drop_gen_mask((bsz, self.head_num, q_seq_len, kv_seq_len), self.keep_prob_tensor),
                (bsz, self.head_num, q_seq_len, kv_seq_len // 8))
        else:
            drop_mask_bits = None
        if self.use_alibi_mask:
            alibi_mask = self.alibi_rescale_mul(alibi_mask, F.cast(self.alibi_rescale_factor, alibi_mask.dtype))
        _, _, _, output = self.flash_attention(query,
                                               key,
                                               value,
                                               alibi_mask,
                                               drop_mask_bits,
                                               padding_mask,
                                               attn_mask,
                                               prefix)
        return output
