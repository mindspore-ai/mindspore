# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Bert model."""

import math
import copy
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter


class BertConfig:
    """
    Configuration for `BertModel`.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input sequence. Default: 128.
        vocab_size (int): The shape of each embedding vector. Default: 32000.
        hidden_size (int): Size of the bert encoder layers. Default: 768.
        num_hidden_layers (int): Number of hidden layers in the BertTransformer encoder
                           cell. Default: 12.
        num_attention_heads (int): Number of attention heads in the BertTransformer
                             encoder cell. Default: 12.
        intermediate_size (int): Size of intermediate layer in the BertTransformer
                           encoder cell. Default: 3072.
        hidden_act (str): Activation function used in the BertTransformer encoder
                    cell. Default: "gelu".
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.1.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 512.
        type_vocab_size (int): Size of token type vocab. Default: 16.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        input_mask_from_dataset (bool): Specifies whether to use the input mask that loaded from
                                 dataset. Default: True.
        token_type_ids_from_dataset (bool): Specifies whether to use the token type ids that loaded
                                     from dataset. Default: True.
        dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size,
                 seq_length=128,
                 vocab_size=32000,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 use_relative_positions=False,
                 input_mask_from_dataset=True,
                 token_type_ids_from_dataset=True,
                 dtype=mstype.float32,
                 compute_type=mstype.float32,
                 enable_fused_layernorm=False):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.input_mask_from_dataset = input_mask_from_dataset
        self.token_type_ids_from_dataset = token_type_ids_from_dataset
        self.use_relative_positions = use_relative_positions
        self.dtype = dtype
        self.compute_type = compute_type
        self.enable_fused_layernorm = enable_fused_layernorm


class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        embedding_shape (list): [batch_size, seq_length, embedding_size], the shape of
                         each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
    """
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 embedding_shape,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = Parameter(initializer
                                         (TruncatedNormal(initializer_range),
                                          [vocab_size, embedding_size]),
                                         name='embedding_table')
        self.expand = P.ExpandDims()
        self.shape_flat = (-1,)
        self.gather = P.Gather()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = tuple(embedding_shape)

    def construct(self, input_ids):
        extended_ids = self.expand(input_ids, -1)
        flat_ids = self.reshape(extended_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(
                one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)
        output = self.reshape(output_for_reshape, self.shape)
        return output, self.embedding_table.value()


class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional and token type embeddings to word embeddings.

    Args:
        embedding_size (int): The size of each embedding vector.
        embedding_shape (list): [batch_size, seq_length, embedding_size], the shape of
                         each embedding vector.
        use_token_type (bool): Specifies whether to use token type embeddings. Default: False.
        token_type_vocab_size (int): Size of token type vocab. Default: 16.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 512.
        dropout_prob (float): The dropout probability. Default: 0.1.
    """
    def __init__(self,
                 embedding_size,
                 embedding_shape,
                 use_relative_positions=False,
                 use_token_type=False,
                 token_type_vocab_size=16,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 max_position_embeddings=512,
                 dropout_prob=0.1):
        super(EmbeddingPostprocessor, self).__init__()
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.embedding_table = Parameter(initializer
                                         (TruncatedNormal(initializer_range),
                                          [token_type_vocab_size,
                                           embedding_size]),
                                         name='embedding_table')

        self.shape_flat = (-1,)
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.1, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = tuple(embedding_shape)
        self.layernorm = nn.LayerNorm((embedding_size,))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.gather = P.Gather()
        self.use_relative_positions = use_relative_positions
        self.slice = P.StridedSlice()
        self.full_position_embeddings = Parameter(initializer
                                                  (TruncatedNormal(initializer_range),
                                                   [max_position_embeddings,
                                                    embedding_size]),
                                                  name='full_position_embeddings')

    def construct(self, token_type_ids, word_embeddings):
        output = word_embeddings
        if self.use_token_type:
            flat_ids = self.reshape(token_type_ids, self.shape_flat)
            if self.use_one_hot_embeddings:
                one_hot_ids = self.one_hot(flat_ids,
                                           self.token_type_vocab_size, self.on_value, self.off_value)
                token_type_embeddings = self.array_mul(one_hot_ids,
                                                       self.embedding_table)
            else:
                token_type_embeddings = self.gather(self.embedding_table, flat_ids, 0)
            token_type_embeddings = self.reshape(token_type_embeddings, self.shape)
            output += token_type_embeddings
        if not self.use_relative_positions:
            _, seq, width = self.shape
            position_embeddings = self.slice(self.full_position_embeddings, (0, 0), (seq, width), (1, 1))
            position_embeddings = self.reshape(position_embeddings, (1, seq, width))
            output += position_embeddings
        output = self.layernorm(output)
        output = self.dropout(output)
        return output


class BertOutput(nn.Cell):
    """
    Apply a linear computation to hidden status and a residual computation to input.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        dropout_prob (float): The dropout probability. Default: 0.1.
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 initializer_range=0.02,
                 dropout_prob=0.1,
                 compute_type=mstype.float32,
                 enable_fused_layernorm=False):
        super(BertOutput, self).__init__()
        self.dense = nn.Dense(in_channels, out_channels,
                              weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout_prob = dropout_prob
        self.add = P.Add()
        self.layernorm = nn.LayerNorm((out_channels,)).to_float(compute_type)
        self.cast = P.Cast()

    def construct(self, hidden_status, input_tensor):
        output = self.dense(hidden_status)
        output = self.dropout(output)
        output = self.add(output, input_tensor)
        output = self.layernorm(output)
        return output


class RelaPosMatrixGenerator(nn.Cell):
    """
    Generates matrix of relative positions between inputs.

    Args:
        length (int): Length of one dim for the matrix to be generated.
        max_relative_position (int): Max value of relative position.
    """
    def __init__(self, length, max_relative_position):
        super(RelaPosMatrixGenerator, self).__init__()
        self._length = length
        self._max_relative_position = max_relative_position
        self._min_relative_position = -max_relative_position
        self.range_length = -length + 1

        self.tile = P.Tile()
        self.range_mat = P.Reshape()
        self.sub = P.Sub()
        self.expanddims = P.ExpandDims()
        self.cast = P.Cast()

    def construct(self):
        range_vec_row_out = self.cast(F.tuple_to_array(F.make_range(self._length)), mstype.int32)
        range_vec_col_out = self.range_mat(range_vec_row_out, (self._length, -1))
        tile_row_out = self.tile(range_vec_row_out, (self._length,))
        tile_col_out = self.tile(range_vec_col_out, (1, self._length))
        range_mat_out = self.range_mat(tile_row_out, (self._length, self._length))
        transpose_out = self.range_mat(tile_col_out, (self._length, self._length))
        distance_mat = self.sub(range_mat_out, transpose_out)

        distance_mat_clipped = ops.clip_by_value(distance_mat,
                                                 self._min_relative_position,
                                                 self._max_relative_position)

        # Shift values to be >=0. Each integer still uniquely identifies a
        # relative position difference.
        final_mat = distance_mat_clipped + self._max_relative_position
        return final_mat


class RelaPosEmbeddingsGenerator(nn.Cell):
    """
    Generates tensor of size [length, length, depth].

    Args:
        length (int): Length of one dim for the matrix to be generated.
        depth (int): Size of each attention head.
        max_relative_position (int): Maxmum value of relative position.
        initializer_range (float): Initialization value of TruncatedNormal.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,
                 length,
                 depth,
                 max_relative_position,
                 initializer_range,
                 use_one_hot_embeddings=False):
        super(RelaPosEmbeddingsGenerator, self).__init__()
        self.depth = depth
        self.vocab_size = max_relative_position * 2 + 1
        self.use_one_hot_embeddings = use_one_hot_embeddings

        self.embeddings_table = Parameter(
            initializer(TruncatedNormal(initializer_range),
                        [self.vocab_size, self.depth]),
            name='embeddings_for_position')

        self.relative_positions_matrix = RelaPosMatrixGenerator(length=length,
                                                                max_relative_position=max_relative_position)
        self.reshape = P.Reshape()
        self.one_hot = nn.OneHot(depth=self.vocab_size)
        self.shape = P.Shape()
        self.gather = P.Gather()  # index_select
        self.matmul = P.BatchMatMul()

    def construct(self):
        relative_positions_matrix_out = self.relative_positions_matrix()

        # Generate embedding for each relative position of dimension depth.
        if self.use_one_hot_embeddings:
            flat_relative_positions_matrix = self.reshape(relative_positions_matrix_out, (-1,))
            one_hot_relative_positions_matrix = self.one_hot(
                flat_relative_positions_matrix)
            embeddings = self.matmul(one_hot_relative_positions_matrix, self.embeddings_table)
            my_shape = self.shape(relative_positions_matrix_out) + (self.depth,)
            embeddings = self.reshape(embeddings, my_shape)
        else:
            embeddings = self.gather(self.embeddings_table,
                                     relative_positions_matrix_out, 0)
        return embeddings


class SaturateCast(nn.Cell):
    """
    Performs a safe saturating cast. This operation applies proper clamping before casting to prevent
    the danger that the value will overflow or underflow.

    Args:
        src_type (:class:`mindspore.dtype`): The type of the elements of the input tensor. Default: mstype.float32.
        dst_type (:class:`mindspore.dtype`): The type of the elements of the output tensor. Default: mstype.float32.
    """
    def __init__(self, src_type=mstype.float32, dst_type=mstype.float32):
        super(SaturateCast, self).__init__()
        np_type = mstype.dtype_to_nptype(dst_type)
        min_type = float(np.finfo(np_type).min)
        max_type = float(np.finfo(np_type).max)

        self.tensor_min_type = min_type
        self.tensor_max_type = max_type

        self.min_op = P.Minimum()
        self.max_op = P.Maximum()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        out = self.max_op(x, self.tensor_min_type)
        out = self.min_op(out, self.tensor_max_type)
        return self.cast(out, self.dst_type)


class BertAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".

    Args:
        batch_size (int): Batch size of input datasets.
        from_tensor_width (int): Size of last dim of from_tensor.
        to_tensor_width (int): Size of last dim of to_tensor.
        from_seq_length (int): Length of from_tensor sequence.
        to_seq_length (int): Length of to_tensor sequence.
        num_attention_heads (int): Number of attention heads. Default: 1.
        size_per_head (int): Size of each attention head. Default: 512.
        query_act (str): Activation function for the query transform. Default: None.
        key_act (str): Activation function for the key transform. Default: None.
        value_act (str): Activation function for the value transform. Default: None.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: False.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        do_return_2d_tensor (bool): True for return 2d tensor. False for return 3d
                             tensor. Default: False.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in BertAttention. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size,
                 from_tensor_width,
                 to_tensor_width,
                 from_seq_length,
                 to_seq_length,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 has_attention_mask=False,
                 attention_probs_dropout_prob=0.0,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 do_return_2d_tensor=False,
                 use_relative_positions=False,
                 compute_type=mstype.float32):

        super(BertAttention, self).__init__()
        self.batch_size = batch_size
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.has_attention_mask = has_attention_mask
        self.use_relative_positions = use_relative_positions

        self.scores_mul = 1.0 / math.sqrt(float(self.size_per_head))
        self.reshape = P.Reshape()
        self.shape_from_2d = (-1, from_tensor_width)
        self.shape_to_2d = (-1, to_tensor_width)
        weight = TruncatedNormal(initializer_range)
        units = num_attention_heads * size_per_head
        self.query_layer = nn.Dense(from_tensor_width,
                                    units,
                                    activation=query_act,
                                    weight_init=weight).to_float(compute_type)
        self.key_layer = nn.Dense(to_tensor_width,
                                  units,
                                  activation=key_act,
                                  weight_init=weight).to_float(compute_type)
        self.value_layer = nn.Dense(to_tensor_width,
                                    units,
                                    activation=value_act,
                                    weight_init=weight).to_float(compute_type)

        self.shape_from = (batch_size, from_seq_length, num_attention_heads, size_per_head)
        self.shape_to = (
            batch_size, to_seq_length, num_attention_heads, size_per_head)

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = -10000.0
        self.batch_num = batch_size * num_attention_heads
        self.matmul = P.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=attention_probs_dropout_prob)

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.Add()
            self.cast = P.Cast()
            self.get_dtype = P.DType()
        if do_return_2d_tensor:
            self.shape_return = (batch_size * from_seq_length, num_attention_heads * size_per_head)
        else:
            self.shape_return = (batch_size, from_seq_length, num_attention_heads * size_per_head)

        self.cast_compute_type = SaturateCast(dst_type=compute_type)
        if self.use_relative_positions:
            self._generate_relative_positions_embeddings = \
                RelaPosEmbeddingsGenerator(length=to_seq_length,
                                           depth=size_per_head,
                                           max_relative_position=16,
                                           initializer_range=initializer_range,
                                           use_one_hot_embeddings=use_one_hot_embeddings)

    def construct(self, from_tensor, to_tensor, attention_mask):
        # reshape 2d/3d input tensors to 2d
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)
        value_out = self.value_layer(to_tensor_2d)

        query_layer = self.reshape(query_out, self.shape_from)
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, self.shape_to)
        key_layer = self.transpose(key_layer, self.trans_shape)

        attention_scores = self.matmul_trans_b(query_layer, key_layer)

        # use_relative_position, supplementary logic
        if self.use_relative_positions:
            # 'relations_keys' = [F|T, F|T, H]
            relations_keys = self._generate_relative_positions_embeddings()
            relations_keys = self.cast_compute_type(relations_keys)
            # query_layer_t is [F, B, N, H]
            query_layer_t = self.transpose(query_layer, self.trans_shape_relative)
            # query_layer_r is [F, B * N, H]
            query_layer_r = self.reshape(query_layer_t,
                                         (self.from_seq_length,
                                          self.batch_num,
                                          self.size_per_head))
            # key_position_scores is [F, B * N, F|T]
            key_position_scores = self.matmul_trans_b(query_layer_r,
                                                      relations_keys)
            # key_position_scores_r is [F, B, N, F|T]
            key_position_scores_r = self.reshape(key_position_scores,
                                                 (self.from_seq_length,
                                                  self.batch_size,
                                                  self.num_attention_heads,
                                                  self.from_seq_length))
            # key_position_scores_r_t is [B, N, F, F|T]
            key_position_scores_r_t = self.transpose(key_position_scores_r,
                                                     self.trans_shape_position)
            attention_scores = attention_scores + key_position_scores_r_t

        attention_scores = self.multiply(self.scores_mul, attention_scores)

        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))

            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.reshape(value_out, self.shape_to)
        value_layer = self.transpose(value_layer, self.trans_shape)
        context_layer = self.matmul(attention_probs, value_layer)

        # use_relative_position, supplementary logic
        if self.use_relative_positions:
            # 'relations_values' = [F|T, F|T, H]
            relations_values = self._generate_relative_positions_embeddings()
            relations_values = self.cast_compute_type(relations_values)
            # attention_probs_t is [F, B, N, T]
            attention_probs_t = self.transpose(attention_probs, self.trans_shape_relative)
            # attention_probs_r is [F, B * N, T]
            attention_probs_r = self.reshape(
                attention_probs_t,
                (self.from_seq_length,
                 self.batch_num,
                 self.to_seq_length))
            # value_position_scores is [F, B * N, H]
            value_position_scores = self.matmul(attention_probs_r,
                                                relations_values)
            # value_position_scores_r is [F, B, N, H]
            value_position_scores_r = self.reshape(value_position_scores,
                                                   (self.from_seq_length,
                                                    self.batch_size,
                                                    self.num_attention_heads,
                                                    self.size_per_head))
            # value_position_scores_r_t is [B, N, F, H]
            value_position_scores_r_t = self.transpose(value_position_scores_r,
                                                       self.trans_shape_position)
            context_layer = context_layer + value_position_scores_r_t

        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, self.shape_return)

        return context_layer


class BertSelfAttention(nn.Cell):
    """
    Apply self-attention.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input sequence.
        hidden_size (int): Size of the bert encoder layers.
        num_attention_heads (int): Number of attention heads. Default: 12.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one_hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in BertSelfAttention. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size,
                 seq_length,
                 hidden_size,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 use_relative_positions=False,
                 compute_type=mstype.float32,
                 enable_fused_layernorm=False):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))

        self.size_per_head = int(hidden_size / num_attention_heads)

        self.attention = BertAttention(
            batch_size=batch_size,
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            from_seq_length=seq_length,
            to_seq_length=seq_length,
            num_attention_heads=num_attention_heads,
            size_per_head=self.size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            use_relative_positions=use_relative_positions,
            has_attention_mask=True,
            do_return_2d_tensor=True,
            compute_type=compute_type)

        self.output = BertOutput(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 initializer_range=initializer_range,
                                 dropout_prob=hidden_dropout_prob,
                                 compute_type=compute_type,
                                 enable_fused_layernorm=enable_fused_layernorm)
        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)

    def construct(self, input_tensor, attention_mask):
        input_tensor = self.reshape(input_tensor, self.shape)
        attention_output = self.attention(input_tensor, input_tensor, attention_mask)
        output = self.output(attention_output, input_tensor)
        return output


class BertEncoderCell(nn.Cell):
    """
    Encoder cells used in BertTransformer.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the bert encoder layers. Default: 768.
        seq_length (int): Length of input sequence. Default: 512.
        num_attention_heads (int): Number of attention heads. Default: 12.
        intermediate_size (int): Size of intermediate layer. Default: 3072.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.02.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        hidden_act (str): Activation function. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size,
                 hidden_size=768,
                 seq_length=512,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.02,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 use_relative_positions=False,
                 hidden_act="gelu",
                 compute_type=mstype.float32,
                 enable_fused_layernorm=False):
        super(BertEncoderCell, self).__init__()
        self.attention = BertSelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            seq_length=seq_length,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            use_relative_positions=use_relative_positions,
            compute_type=compute_type,
            enable_fused_layernorm=enable_fused_layernorm)
        self.intermediate = nn.Dense(in_channels=hidden_size,
                                     out_channels=intermediate_size,
                                     activation=hidden_act,
                                     weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.output = BertOutput(in_channels=intermediate_size,
                                 out_channels=hidden_size,
                                 initializer_range=initializer_range,
                                 dropout_prob=hidden_dropout_prob,
                                 compute_type=compute_type,
                                 enable_fused_layernorm=enable_fused_layernorm)

    def construct(self, hidden_states, attention_mask):
        # self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        # feed construct
        intermediate_output = self.intermediate(attention_output)
        # add and normalize
        output = self.output(intermediate_output, attention_output)
        return output


class BertTransformer(nn.Cell):
    """
    Multi-layer bert transformer.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers.
        seq_length (int): Length of input sequence.
        num_hidden_layers (int): Number of hidden layers in encoder cells.
        num_attention_heads (int): Number of attention heads in encoder cells. Default: 12.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 3072.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
        return_all_encoders (bool): Specifies whether to return all encoders. Default: False.
    """
    def __init__(self,
                 batch_size,
                 hidden_size,
                 seq_length,
                 num_hidden_layers,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 use_relative_positions=False,
                 hidden_act="gelu",
                 compute_type=mstype.float32,
                 return_all_encoders=False,
                 enable_fused_layernorm=False):
        super(BertTransformer, self).__init__()
        self.return_all_encoders = return_all_encoders

        layers = []
        for _ in range(num_hidden_layers):
            layer = BertEncoderCell(batch_size=batch_size,
                                    hidden_size=hidden_size,
                                    seq_length=seq_length,
                                    num_attention_heads=num_attention_heads,
                                    intermediate_size=intermediate_size,
                                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                                    use_one_hot_embeddings=use_one_hot_embeddings,
                                    initializer_range=initializer_range,
                                    hidden_dropout_prob=hidden_dropout_prob,
                                    use_relative_positions=use_relative_positions,
                                    hidden_act=hidden_act,
                                    compute_type=compute_type,
                                    enable_fused_layernorm=enable_fused_layernorm)
            layers.append(layer)

        self.layers = nn.CellList(layers)

        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)
        self.out_shape = (batch_size, seq_length, hidden_size)

    def construct(self, input_tensor, attention_mask):
        prev_output = self.reshape(input_tensor, self.shape)

        all_encoder_layers = ()
        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask)
            prev_output = layer_output

            if self.return_all_encoders:
                layer_output = self.reshape(layer_output, self.out_shape)
                all_encoder_layers = all_encoder_layers + (layer_output,)

        if not self.return_all_encoders:
            prev_output = self.reshape(prev_output, self.out_shape)
            all_encoder_layers = all_encoder_layers + (prev_output,)
        return all_encoder_layers


class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config (Class): Configuration for BertModel.
    """
    def __init__(self, config):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.input_mask_from_dataset = config.input_mask_from_dataset
        self.input_mask = None

        if not self.input_mask_from_dataset:
            self.input_mask = initializer(
                "ones", [config.batch_size, config.seq_length], mstype.int32).init_data()

        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = (config.batch_size, 1, config.seq_length)
        self.broadcast_ones = initializer(
            "ones", [config.batch_size, config.seq_length, 1], mstype.float32).init_data()
        self.batch_matmul = P.BatchMatMul()

    def construct(self, input_mask):
        if not self.input_mask_from_dataset:
            input_mask = self.input_mask

        input_mask = self.cast(self.reshape(input_mask, self.shape), mstype.float32)
        attention_mask = self.batch_matmul(self.broadcast_ones, input_mask)
        return attention_mask


class BertModel(nn.Cell):
    """
    Bidirectional Encoder Representations from Transformers.

    Args:
        config (Class): Configuration for BertModel.
        is_training (bool): True for training mode. False for eval mode.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=False):
        super(BertModel, self).__init__()
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.input_mask_from_dataset = config.input_mask_from_dataset
        self.token_type_ids_from_dataset = config.token_type_ids_from_dataset
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_size = config.hidden_size
        self.token_type_ids = None

        self.last_idx = self.num_hidden_layers - 1
        output_embedding_shape = [self.batch_size, self.seq_length,
                                  self.embedding_size]

        if not self.token_type_ids_from_dataset:
            self.token_type_ids = initializer(
                "zeros", [self.batch_size, self.seq_length], mstype.int32).init_data()

        self.bert_embedding_lookup = EmbeddingLookup(
            vocab_size=config.vocab_size,
            embedding_size=self.embedding_size,
            embedding_shape=output_embedding_shape,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=config.initializer_range)

        self.bert_embedding_postprocessor = EmbeddingPostprocessor(
            embedding_size=self.embedding_size,
            embedding_shape=output_embedding_shape,
            use_relative_positions=config.use_relative_positions,
            use_token_type=True,
            token_type_vocab_size=config.type_vocab_size,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=0.02,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

        self.bert_encoder = BertTransformer(
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=config.initializer_range,
            hidden_dropout_prob=config.hidden_dropout_prob,
            use_relative_positions=config.use_relative_positions,
            hidden_act=config.hidden_act,
            compute_type=config.compute_type,
            return_all_encoders=True,
            enable_fused_layernorm=config.enable_fused_layernorm)

        self.cast = P.Cast()
        self.dtype = config.dtype
        self.cast_compute_type = SaturateCast(dst_type=config.compute_type)
        self.slice = P.StridedSlice()

        self.squeeze_1 = P.Squeeze(axis=1)
        self.dense = nn.Dense(self.hidden_size, self.hidden_size,
                              activation="tanh",
                              weight_init=TruncatedNormal(config.initializer_range)).to_float(config.compute_type)
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)

    def construct(self, input_ids, token_type_ids, input_mask):

        # embedding
        if not self.token_type_ids_from_dataset:
            token_type_ids = self.token_type_ids
        word_embeddings, embedding_tables = self.bert_embedding_lookup(input_ids)
        embedding_output = self.bert_embedding_postprocessor(token_type_ids,
                                                             word_embeddings)

        # attention mask [batch_size, seq_length, seq_length]
        attention_mask = self._create_attention_mask_from_input_mask(input_mask)

        # bert encoder
        encoder_output = self.bert_encoder(self.cast_compute_type(embedding_output),
                                           attention_mask)

        sequence_output = self.cast(encoder_output[self.last_idx], self.dtype)

        # pooler
        sequence_slice = self.slice(sequence_output,
                                    (0, 0, 0),
                                    (self.batch_size, 1, self.hidden_size),
                                    (1, 1, 1))
        first_token = self.squeeze_1(sequence_slice)
        pooled_output = self.dense(first_token)
        pooled_output = self.cast(pooled_output, self.dtype)

        return sequence_output, pooled_output, embedding_tables
