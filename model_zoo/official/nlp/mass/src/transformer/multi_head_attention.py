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
"""Multi-Head Self-Attention block."""
import math

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.common.initializer import TruncatedNormal
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from .components import SaturateCast


class MultiHeadAttention(nn.Cell):
    """
    Implement of multi-head self-attention.

    In the encoder, the calculation of single-head self-attention is as below.

    Inputs: [x1, x2, x3, x4...] (xi is a word embedding, with shape T*D, Inputs's shape is N*T*D);
    Weights: Wq(D*embed_dim), Wk(D*embed_dim), Wv(D*embed_dim);

    Query, key, value are calculated in below formula:
        Q = Input * Wq (N*T*embed_dim);
        K = Input * Wk (N*T*embed_dim);
        V = Input * Wv (N*T*embed_dim);

    Then, attention score is calculated:
        A = K * Q.T (qi is doted with each ki, A's shape is N*T*T.
                     e.g. q1 is doted with k1, k2, k3, k4,
                     then vector of [a1.1, a1.2, a1.3, a1.4] will be available.
                     ai,j represent the importance of j-th word embedding to i-th.)

        A^ = Soft-max(A) (Normalize the score, N*T*T).

    Finally, the output of self-attention cell is:
        O = A^ * V (N*T*embed_dim, each word embedding was represented with self-attention.)

    Multi-head self-attention is the same with single-head self-attention except that
    Wq, Wk, Wv are repeat `head_num` times.

    In our implements, Wq = Wk = Wv = attn_embed_dim // num_attn_heads.

    Args:
        src_dim (int): Dimensions of queries.
        tgt_dim (int): Dimensions of keys and values.
        attn_embed_dim (int): Dimensions of attention weight, e.g. Q, K, V.
        num_attn_heads (int): Attention heads number. Default: 1.
        query_act (str): Activation function for Q. Default: None.
        key_act (str): Activation function for K. Default: None.
        value_act (str): Activation function for V. Default: None.
        has_attention_mask (bool): Whether has attention mask. Default: True.
        attention_dropout_prob (float): Dropout rate in attention. Default: 0.1.
        initializer_range (float): Initial range.
        do_return_2d_tensor (bool): Whether return 2d matrix. Default: True.
        compute_type (mstype): Mindspore data type. Default: mstype.float32.

    Returns:
        Tensor, with shape (N, T, D).
    """

    def __init__(self,
                 src_dim,
                 tgt_dim,
                 attn_embed_dim,
                 num_attn_heads=1,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 out_act=None,
                 has_attention_mask=True,
                 attention_dropout_prob=0.0,
                 initializer_range=0.02,
                 do_return_2d_tensor=True,
                 compute_type=mstype.float32):
        super(MultiHeadAttention, self).__init__()
        if attn_embed_dim % num_attn_heads != 0:
            raise ValueError(f"The hidden size {attn_embed_dim} is not a multiple of the "
                             f"number of attention heads {num_attn_heads}")

        self.attn_embed_dim = attn_embed_dim
        self.num_attn_heads = num_attn_heads
        self.size_per_head = attn_embed_dim // num_attn_heads
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.has_attention_mask = has_attention_mask

        if attn_embed_dim != self.num_attn_heads * self.size_per_head:
            raise ValueError("`attn_embed_dim` must be divided by num_attn_heads.")

        self.scores_mul = Tensor([1.0 / math.sqrt(float(self.size_per_head))],
                                 dtype=compute_type)
        self.reshape = P.Reshape()

        self.query_layer = nn.Dense(src_dim,
                                    attn_embed_dim,
                                    activation=query_act,
                                    has_bias=True,
                                    weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.key_layer = nn.Dense(tgt_dim,
                                  attn_embed_dim,
                                  activation=key_act,
                                  has_bias=True,
                                  weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.value_layer = nn.Dense(tgt_dim,
                                    attn_embed_dim,
                                    activation=value_act,
                                    has_bias=True,
                                    weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.out_layer = nn.Dense(attn_embed_dim,
                                  attn_embed_dim,
                                  activation=out_act,
                                  has_bias=True,
                                  weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.multiply_data = Tensor([-10000.0], dtype=compute_type)
        self.matmul = P.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(1.0 - attention_dropout_prob)

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.Add()
            self.cast = P.Cast()
            self.get_dtype = P.DType()

        self.do_return_2d_tensor = do_return_2d_tensor
        self.cast_compute_type = SaturateCast(dst_type=compute_type)
        self.softmax_cast = P.Cast()
        self.get_shape = P.Shape()
        self.transpose_orders = (0, 2, 1, 3)

    def construct(self, queries, keys, values, attention_mask):
        """
        Construct network.

        For self attention operation, T==T'.
        For encoder-decoder-attention, T!=T'

        Args:
            queries (Tensor): Input queries, with shape (N, T, D).
            keys (Tensor): Input keys, with shape (N, T', D).
            values (Tensor): Input values, with shape (N, T', D).
            attention_mask (Tensor): Mask matrix, with shape (N, T, T').

        Returns:
            Tensor, with shape (N, T, D).
        """
        q_shape = self.get_shape(queries)  # (N, T, D)
        batch_size = q_shape[0]
        src_max_len = q_shape[1]

        k_shape = self.get_shape(keys)  # (N, T', D)
        tgt_max_len = k_shape[1]

        _src_4d_shape = (batch_size, src_max_len, self.num_attn_heads, self.size_per_head)
        _tgt_4d_shape = (batch_size, tgt_max_len, self.num_attn_heads, self.size_per_head)

        queries_2d = self.reshape(queries, (-1, self.src_dim))
        keys_2d = self.reshape(keys, (-1, self.tgt_dim))
        values_2d = self.reshape(values, (-1, self.tgt_dim))

        query_out = self.query_layer(queries_2d)  # (N*T, D)*(D, D) -> (N*T, D)
        key_out = self.key_layer(keys_2d)  # (N*T, D)*(D, D) -> (N*T, D)
        value_out = self.value_layer(values_2d)  # (N*T, D)*(D, D) -> (N*T, D)

        query_out = self.multiply(query_out, self.scores_mul)

        query_layer = self.reshape(query_out, _src_4d_shape)
        query_layer = self.transpose(query_layer, self.transpose_orders)  # (N, h, T, D')
        key_layer = self.reshape(key_out, _tgt_4d_shape)
        key_layer = self.transpose(key_layer, self.transpose_orders)  # (N, h, T', D')
        value_layer = self.reshape(value_out, _tgt_4d_shape)
        value_layer = self.transpose(value_layer, self.transpose_orders)  # (N, h, T', D')

        # (N, h, T, D')*(N, h, D', T') -> (N, h, T, T')
        attention_scores = self.matmul_trans_b(query_layer, key_layer)

        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(
                self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                self.cast(attention_mask, self.get_dtype(attention_scores))
            )  # make mask position into 1, unmask position into 0.
            adder = self.multiply(multiply_out, self.multiply_data)
            adder = self.softmax_cast(adder, mstype.float32)
            attention_scores = self.softmax_cast(attention_scores, mstype.float32)
            attention_scores = self.add(adder, attention_scores)

        attention_scores = self.softmax_cast(attention_scores, mstype.float32)
        attention_prob = self.softmax(attention_scores)
        attention_prob = self.softmax_cast(attention_prob, self.get_dtype(key_layer))
        attention_prob = self.dropout(attention_prob)

        # (N, h, T, T')*(N, h, T', D') -> (N, h, T, D')
        context_layer = self.matmul(attention_prob, value_layer)
        context_layer = self.transpose(context_layer, self.transpose_orders)  # (N, T, h, D')
        context_layer = self.reshape(context_layer,
                                     (batch_size * src_max_len, self.attn_embed_dim))  # (N*T, D)

        context_layer = self.out_layer(context_layer)

        if not self.do_return_2d_tensor:
            context_layer = self.reshape(
                context_layer, (batch_size, src_max_len, self.attn_embed_dim)
            )  # (N, T, D)

        return context_layer
