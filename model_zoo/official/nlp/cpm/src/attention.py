# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Attention module"""
import math
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor

from src.util import LayerNorm
from src.util import LinearLayer, ResidualConnection, Dropout


class MaskedSelfAttention(nn.Cell):
    """
    Self-Attention module for each layer.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Length of last dim of hidden layer.
        seq_length (int): Length of input tensor sequence.
        num_attention_heads (int): Number of attention heads.
        dim_per_head (int): Size of each attention head.
        config: The config of networks.
        has_attention_mask (bool): Specifies whether to use attention mask.
        do_return_2d_tensor (bool): Whether use 2-dimension.
        attention_dropout (float): The dropout probability for attention.
        is_training (bool): Whether is training.
        compute_type (:class:`mindspore.dtype`): Compute type in attention.

    Returns:
        Tensor, with the shape [batch_size, hidden_size]
    """

    def __init__(self,
                 batch_size,
                 hidden_size,
                 seq_length,
                 num_attention_heads,
                 dim_per_head,
                 config=None,
                 has_attention_mask=True,
                 do_return_2d_tensor=True,
                 attention_dropout=0.0,
                 is_training=False,
                 compute_type=mstype.float16):
        super(MaskedSelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_heads = num_attention_heads
        self.dim_per_head = dim_per_head
        self.has_attention_mask = has_attention_mask
        self.compute_type = compute_type
        self.is_training = is_training

        self.scale = Tensor(math.sqrt(float(self.dim_per_head)), dtype=compute_type)
        self.mask_data = Tensor([-10000.0], dtype=mstype.float32)
        self.split_head_shape = (self.batch_size, self.seq_length, self.num_heads, self.dim_per_head)

        self.dense = LinearLayer(hidden_size, hidden_size)
        self.dense.matmul.shard(((config.dp, config.mp), (1, config.mp)))
        self.dense.bias_add.shard(((config.dp, 1), (1,)))
        self.dense.bias.parallel_optimizer = False

        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((config.dp, 1, config.mp, 1),))
        self.merge_transpose = P.Transpose().shard(((config.dp, config.mp, 1, 1),))
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape2 = (0, 2, 3, 1)
        self.matmul_trans_b = P.BatchMatMul().shard(((config.dp, config.mp, 1, 1), (config.dp, config.mp, 1, 1)))
        self.matmul = P.BatchMatMul().shard(((config.dp, config.mp, 1, 1), (config.dp, config.mp, 1, 1)))
        self.multiply = P.Mul().shard(((config.dp, 1, 1, 1), (1,))).add_prim_attr("_side_effect", True)
        self.realdiv = P.RealDiv().shard(((config.dp, config.mp, 1, 1), ()))

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims().shard(((config.dp, 1, 1),))
            self.sub = P.Sub().shard(((1,), (config.dp, 1, 1, 1))).add_prim_attr("_side_effect", True)
            self.add = P.TensorAdd().shard(((config.dp, 1, 1, 1), (config.dp, config.mp, 1, 1)))
        self.cast = P.Cast()
        self.get_dtype = P.DType()

        if do_return_2d_tensor:
            self.shape_return = (-1, hidden_size)
        else:
            self.shape_return = (-1, seq_length, hidden_size)

        self.softmax = nn.Softmax()
        self.softmax.softmax.shard(((config.dp, config.mp, 1, 1),))
        self.softmax_cast = P.Cast()
        self.shape = P.Shape()

        self.dropout = Dropout(1 - attention_dropout)
        self.dropout.dropout_gen_mask.shard(((config.dp, 1),))
        self.dropout.dropout_do_mask.shard(((config.dp, 1),))

        self.dropout_probs = Dropout(1 - attention_dropout)
        self.dropout_probs.dropout_gen_mask.shard(((config.dp, config.mp, 1, 1),))
        self.dropout_probs.dropout_do_mask.shard(((config.dp, config.mp, 1, 1),))

        self.use_attention_dropout = is_training
        self.dense1 = LinearLayer(self.hidden_size, self.hidden_size)
        self.dense1.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense1.bias_add.shard(((config.dp, config.mp), (config.mp,)))
        self.dense2 = LinearLayer(self.hidden_size, self.hidden_size)
        self.dense2.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense2.bias_add.shard(((config.dp, config.mp), (config.mp,)))
        self.dense3 = LinearLayer(self.hidden_size, self.hidden_size)
        self.dense3.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense3.bias_add.shard(((config.dp, config.mp), (config.mp,)))

        attention_mask = np.tril(np.ones(shape=(config.seq_length, config.seq_length),))
        attention_mask = np.expand_dims(attention_mask, 0)
        attention_mask = np.tile(attention_mask, (config.batch_size, 1, 1))
        attention_mask = np.expand_dims(attention_mask, 1)
        self.attention_mask = Tensor(attention_mask, dtype=compute_type)

    def construct(self, input_tensor, attention_mask=None):
        """do masked self-attention"""
        # input_tensor [batch_size, seq_length, hidden_size], eg:[1,571,2560].
        query = self.dense1(input_tensor)
        key = self.dense2(input_tensor)
        value = self.dense3(input_tensor)

        # split head
        query = self.reshape(query, self.split_head_shape)
        # query shape [2, 571, 32, 80] -> [2, 32, 571, 80]
        query = self.transpose(query, self.trans_shape)

        key = self.reshape(key, self.split_head_shape)
        # key shape [batch_size, num_heads, dim_per_head, seq_len]
        key = self.transpose(key, self.trans_shape2)

        value = self.reshape(value, self.split_head_shape)

        # value shape [batch_size, num_heads, seq_len, dim_per_head]
        value = self.transpose(value, self.trans_shape)

        # precision transition fp32 -> fp16
        query = self.cast(query, self.compute_type)
        key = self.cast(key, self.compute_type)
        # 8, 32, 725, 80|8, 32, 80, 725 -> 8, 32, 725, 725
        attention_scores = self.matmul_trans_b(query, key)

        attention_scores = self.cast(attention_scores, mstype.float32)
        attention_scores = self.realdiv(attention_scores, self.cast(self.scale, self.get_dtype(attention_scores)))
        attention_scores = P.Cast()(attention_scores, mstype.float32)
        if self.has_attention_mask:
            if attention_mask is None:
                attention_mask = self.attention_mask
            else:
                attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask,
                                              self.get_dtype(attention_scores)))
            # 1, 1, 725, 725
            adder = self.multiply(multiply_out, self.mask_data)
            adder = self.cast(adder, mstype.float32)
            attention_scores = self.cast(attention_scores, mstype.float32)
            # 1, 1, 725, 725|8, 32, 725, 725-》8, 32, 725, 725
            attention_scores = self.add(adder, attention_scores)

        attention_scores = self.softmax_cast(attention_scores, mstype.float32)
        # [8, 32, 725, 725] ->8, 32, 725, 725
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.softmax_cast(attention_probs, self.get_dtype(key))

        if self.use_attention_dropout:
            attention_probs = self.dropout_probs(attention_probs)

        value = self.cast(value, self.compute_type)
        attention_probs = self.cast(attention_probs, self.compute_type)
        if self.is_training:
            # 1, 8, 2, 725, 725 -> 8, 2, 725, 725
            attention_probs = self.reshape(attention_probs, (
                self.batch_size, self.num_heads, self.seq_length,
                self.seq_length))
        # 8, 32, 725, 725 | 8, 32, 725, 80 -> 8, 2, 725, 1280
        outputs = self.matmul(attention_probs, value)

        outputs = self.cast(outputs, mstype.float32)

        # merge heads, [8, 2, 725, 1280]->8, 725, 2, 1280
        outputs = self.merge_transpose(outputs, self.trans_shape)
        # 8, 725, 2, 1280->5800, 2560
        outputs = self.reshape(outputs,
                               self.shape_return)
        # project
        outputs = self.dense(outputs)
        if self.is_training:
            outputs = self.dropout(outputs)

        return outputs


class MaskedMultiHeadAttention(nn.Cell):
    """
    Constructor for the MaskedMultiHeadAttention.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input tensor sequence.
        hidden_size (int): Length of last dim of hidden layer.
        config: The config of networks.
        num_attention_heads (int): Number of attention heads.
        attention_dropout (float): The dropout probability for attention.
        hidden_dropout (float): The dropout probability for hidden layer.
        has_attention_mask (bool): Specifies whether to use attention mask.
        is_training (bool): Whether to train.
        compute_type (:class:`mindspore.dtype`): Compute type in attention.

    Returns:
        Tensor, shape (N, T).
    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 hidden_size,
                 config=None,
                 num_attention_heads=12,
                 attention_dropout=0.02,
                 hidden_dropout=0.1,
                 has_attention_mask=True,
                 is_training=False,
                 compute_type=mstype.float16
                 ):
        super(MaskedMultiHeadAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))

        self.dim_per_head = int(hidden_size / num_attention_heads)

        self.masked_self_attention = MaskedSelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            seq_length=seq_length,
            config=config,
            num_attention_heads=num_attention_heads,
            dim_per_head=self.dim_per_head,
            has_attention_mask=has_attention_mask,
            do_return_2d_tensor=True,
            attention_dropout=attention_dropout,
            is_training=is_training,
            compute_type=compute_type
        )

        self.layernorm = LayerNorm((hidden_size,), config, epsilon=1e-5).to_float(mstype.float32)
        self.layernorm.gamma.parallel_optimizer = False
        self.layernorm.beta.parallel_optimizer = False

        self.residual_connection = ResidualConnection(dropout_prob=0.1)
        self.residual_connection.add.shard(((config.dp, 1), (config.dp, 1)))

        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.new_shape = (-1, hidden_size)

    def construct(self, input_tensor, attention_mask=None):
        # input tensor shape[batch_size*sen_length, hidden_size]
        output_tensor = self.layernorm(input_tensor)
        # attention_output shape [batch_size * sen_length, hidden_size]
        attention_output = self.masked_self_attention(output_tensor, attention_mask)
        # residual connection, [5800, 2560] | [5800, 2560] -> [5800, 2560]
        output = self.residual_connection(attention_output, input_tensor)
        return output
