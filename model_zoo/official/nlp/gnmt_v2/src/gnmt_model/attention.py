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
        Tensor, shape (N, T, D).
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
            query (Tensor): Shape (t_q, N, D).
            keys (Tensor): Shape (t_k, N, D).
            attention_mask: Shape(N, t_k).
        Returns:
            Tensor, shape (N, t_q, D).
        """

        # (t_k, N, D) -> (N, t_k, D).
        keys = self.transpose(keys, self.transpose_orders)
        # (t_q, N, D) -> (N, t_q, D).
        query = self.transpose(query, self.transpose_orders)

        query_shape = self.shape_op(query)
        b = query_shape[0]
        t_q = query_shape[1]
        t_k = self.shape_op(keys)[1]

        # (N, t_q, D)
        query = self.reshape(query, (b * t_q, self.query_size))
        if self.is_training:
            query = self.cast(query, mstype.float16)
        processed_query = self.linear_q(query)
        if self.is_trining:
            processed_query = self.cast(processed_query, mstype.float32)
        processed_query = self.reshape(processed_query, (b, t_q, self.num_units))
        # (N, t_k, D)
        keys = self.reshape(keys, (b * t_k, self.key_size))
        if self.is_training:
            keys = self.cast(keys, mstype.float16)
        processed_key = self.linear_k(keys)
        if self.is_trining:
            processed_key = self.cast(processed_key, mstype.float32)
        processed_key = self.reshape(processed_key, (b, t_k, self.num_units))

        # scores: (N ， T_q， T_k)
        scores = self.calc_score(processed_query, processed_key)
        # attention_mask: (N, T_k)
        mask = attention_mask
        # [N， 1]
        if mask is not None:
            mask = 1.0 - mask
            mask = self.tile(self.expand(mask, 1), (1, t_q, 1))
            scores += mask * (-INF)
        # [b, t_q, t_k]
        scores_normalized = self.softmax(scores)

        keys = self.reshape(keys, (b, t_k, self.key_size))
        if self.is_training:
            keys = self.cast(keys, mstype.float16)
            scores_normalized_fp16 = self.cast(scores_normalized, mstype.float16)
        else:
            scores_normalized_fp16 = scores_normalized

        # (b, t_q, n)
        context_attention = self.batchMatmul(scores_normalized_fp16, keys)
        # [t_q,b,D]
        context_attention = self.transpose(context_attention, self.transpose_orders)
        if self.is_training:
            context_attention = self.cast(context_attention, mstype.float32)

        return context_attention, scores_normalized

    def calc_score(self, att_query, att_keys):
        """
        Calculate Bahdanau score

        Args:
            att_query: (N, T_q, D).
            att_keys: (N, T_k, D).

        returns:
            scores: (N, T_q, T_k).
        """
        b, t_k, n = self.shape_op(att_keys)
        t_q = self.shape_op(att_query)[1]
        # (b, t_q, t_k, n)
        att_query = self.tile(self.expand(att_query, 2), (1, 1, t_k, 1))
        att_keys = self.tile(self.expand(att_keys, 1), (1, t_q, 1, 1))
        # (b, t_q, t_k, n)
        sum_qk = att_query + att_keys

        if self.normalize:
            # (b, t_q, t_k, n)
            sum_qk = sum_qk + self.normalize_bias
            linear_att = self.linear_att / self.norm(self.linear_att)
            linear_att = self.cast(linear_att, mstype.float32)
            linear_att = self.mul(linear_att, self.normalize_scalar)
        else:
            linear_att = self.linear_att

        linear_att = self.expand(linear_att, -1)
        sum_qk = self.reshape(sum_qk, (-1, n))

        tanh_sum_qk = self.tanh(sum_qk)
        if self.is_training:
            linear_att = self.cast(linear_att, mstype.float16)
            tanh_sum_qk = self.cast(tanh_sum_qk, mstype.float16)

        out = self.matmul(tanh_sum_qk, linear_att)

        # (b, t_q, t_k)
        out = self.reshape(out, (b, t_q, t_k))
        if self.is_training:
            out = self.cast(out, mstype.float32)
        return out
