# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Bahdanau attention block."""
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import Uniform

INF = 65504.0


class BahdanauAttention(nn.Cell):
    """
    Constructor for the BahdanauAttention.

    Args:
        is_training (bool): Whether to train.
        query_size (int): feature dimension for query.
        key_size (int): feature dimension for keys.
        num_units (int): internal feature dimension.
        normalize (bool): Whether to normalize.
        initializer_range: range for uniform initializer parameters.

    Returns:
        Tensor, shape (t_q_length, N, D).
    """

    def __init__(self,
                 is_training,
                 query_size,
                 key_size,
                 num_units,
                 normalize=False,
                 initializer_range=0.1,
                 compute_type=mstype.float16):
        super(BahdanauAttention, self).__init__()
        self.is_training = is_training
        self.mask = None
        self.query_size = query_size
        self.key_size = key_size
        self.normalize = normalize
        self.num_units = num_units
        self.linear_att = Parameter(Tensor(np.random.uniform(-initializer_range, initializer_range, size=[num_units]),
                                           dtype=mstype.float32))
        if self.normalize:
            self.normalize_scalar = Parameter(Tensor(np.array([1.0 / num_units]), dtype=mstype.float32))
            self.normalize_bias = Parameter(Tensor(np.zeros(num_units), dtype=mstype.float32))
        self.transpose = P.Transpose()
        self.transpose_orders = (1, 0, 2)
        self.shape_op = P.Shape()

        self.linear_q = nn.Dense(query_size,
                                 num_units,
                                 has_bias=False,
                                 weight_init=Uniform(initializer_range)).to_float(compute_type)

        self.linear_k = nn.Dense(key_size,
                                 num_units,
                                 has_bias=False,
                                 weight_init=Uniform(initializer_range)).to_float(compute_type)
        self.expand = P.ExpandDims()
        self.tile = P.Tile()

        self.norm = nn.Norm(axis=-1)
        self.mul = P.Mul()
        self.matmul = P.MatMul()
        self.batchMatmul = P.BatchMatMul()
        self.tanh = nn.Tanh()

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)
        self.reshape = P.Reshape()
        self.cast = P.Cast()

    def construct(self, query, keys, attention_mask=None):
        """
        Construct attention block.

        Args:
            query (Tensor): Shape (t_q_length, N, D).
            keys (Tensor): Shape (t_k_length, N, D).
            attention_mask: Shape(N, t_k_length).
        Returns:
            Tensor, shape (t_q_length, N, D).
        """

        # (t_k_length, N, D) -> (N, t_k_length, D).
        keys = self.transpose(keys, self.transpose_orders)
        # (t_q_length, N, D) -> (N, t_q_length, D).
        query_trans = self.transpose(query, self.transpose_orders)

        query_shape = self.shape_op(query_trans)
        batch_size = query_shape[0]
        t_q_length = query_shape[1]
        t_k_length = self.shape_op(keys)[1]

        # (N, t_q_length, D)
        query_trans = self.reshape(query_trans, (batch_size * t_q_length, self.query_size))
        if self.is_training:
            query_trans = self.cast(query_trans, mstype.float16)
        processed_query = self.linear_q(query_trans)
        if self.is_training:
            processed_query = self.cast(processed_query, mstype.float32)
        processed_query = self.reshape(processed_query, (batch_size, t_q_length, self.num_units))
        # (N, t_k_length, D)
        keys = self.reshape(keys, (batch_size * t_k_length, self.key_size))
        if self.is_training:
            keys = self.cast(keys, mstype.float16)
        processed_key = self.linear_k(keys)
        if self.is_training:
            processed_key = self.cast(processed_key, mstype.float32)
        processed_key = self.reshape(processed_key, (batch_size, t_k_length, self.num_units))

        # scores: (N, t_q_length, t_k_length)
        scores = self.obtain_score(processed_query, processed_key)
        # attention_mask: (N, t_k_length)
        mask = attention_mask
        if mask is not None:
            mask = 1.0 - mask
            mask = self.tile(self.expand(mask, 1), (1, t_q_length, 1))
            scores += mask * (-INF)
        # [batch_size, t_q_length, t_k_length]
        scores_softmax = self.softmax(scores)

        keys = self.reshape(keys, (batch_size, t_k_length, self.key_size))
        if self.is_training:
            keys = self.cast(keys, mstype.float16)
            scores_softmax_fp16 = self.cast(scores_softmax, mstype.float16)
        else:
            scores_softmax_fp16 = scores_softmax

        # (b, t_q_length, D)
        context_attention = self.batchMatmul(scores_softmax_fp16, keys)
        # [t_q_length, b, D]
        context_attention = self.transpose(context_attention, self.transpose_orders)
        if self.is_training:
            context_attention = self.cast(context_attention, mstype.float32)

        return context_attention, scores_softmax

    def obtain_score(self, attention_q, attention_k):
        """
        Calculate Bahdanau score

        Args:
            attention_q: (batch_size, t_q_length, D).
            attention_k: (batch_size, t_k_length, D).

        returns:
            scores: (batch_size, t_q_length, t_k_length).
        """
        batch_size, t_k_length, D = self.shape_op(attention_k)
        t_q_length = self.shape_op(attention_q)[1]
        # (batch_size, t_q_length, t_k_length, n)
        attention_q = self.tile(self.expand(attention_q, 2), (1, 1, t_k_length, 1))
        attention_k = self.tile(self.expand(attention_k, 1), (1, t_q_length, 1, 1))
        # (batch_size, t_q_length, t_k_length, n)
        sum_qk_add = attention_q + attention_k

        if self.normalize:
            # (batch_size, t_q_length, t_k_length, n)
            sum_qk_add = sum_qk_add + self.normalize_bias
            linear_att_norm = self.linear_att / self.norm(self.linear_att)
            linear_att_norm = self.cast(linear_att_norm, mstype.float32)
            linear_att_norm = self.mul(linear_att_norm, self.normalize_scalar)
        else:
            linear_att_norm = self.linear_att

        linear_att_norm = self.expand(linear_att_norm, -1)
        sum_qk_add = self.reshape(sum_qk_add, (-1, D))

        tanh_sum_qk = self.tanh(sum_qk_add)
        if self.is_training:
            linear_att_norm = self.cast(linear_att_norm, mstype.float16)
            tanh_sum_qk = self.cast(tanh_sum_qk, mstype.float16)

        scores_out = self.matmul(tanh_sum_qk, linear_att_norm)

        # (N, t_q_length, t_k_length)
        scores_out = self.reshape(scores_out, (batch_size, t_q_length, t_k_length))
        if self.is_training:
            scores_out = self.cast(scores_out, mstype.float32)
        return scores_out
