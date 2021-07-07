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

import math
import copy
import numpy as np
from mindspore import nn
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter


class AlbertConfig:
    """
    Configuration for `AlbertModel`.

    Args:
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
        dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
    """

    def __init__(self,
                 seq_length=256,
                 vocab_size=21128,
                 hidden_size=312,
                 num_hidden_groups=1,
                 num_hidden_layers=4,
                 inner_group_num=1,
                 num_attention_heads=12,
                 intermediate_size=1248,
                 hidden_act="gelu",
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 hidden_dropout_prob=0.0,
                 attention_probs_dropout_prob=0.0,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 use_relative_positions=False,
                 classifier_dropout_prob=0.1,
                 embedding_size=128,
                 layer_norm_eps=1e-12,
                 has_attention_mask=True,
                 do_return_2d_tensor=True,
                 use_one_hot_embeddings=False,
                 use_token_type=True,
                 return_all_encoders=False,
                 output_attentions=False,
                 output_hidden_states=False,
                 dtype=mstype.float32,
                 compute_type=mstype.float32,
                 is_training=True,
                 num_labels=5,
                 use_word_embeddings=True):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.inner_group_num = inner_group_num
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.query_act = query_act
        self.key_act = key_act
        self.value_act = value_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.use_relative_positions = use_relative_positions
        self.classifier_dropout_prob = classifier_dropout_prob
        self.embedding_size = embedding_size
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_groups = num_hidden_groups
        self.has_attention_mask = has_attention_mask
        self.do_return_2d_tensor = do_return_2d_tensor
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.use_token_type = use_token_type
        self.return_all_encoders = return_all_encoders
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.dtype = dtype
        self.compute_type = compute_type
        self.is_training = is_training
        self.num_labels = num_labels
        self.use_word_embeddings = use_word_embeddings


class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        config (AlbertConfig): Albert Config.
    """

    def __init__(self, config):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = config.vocab_size
        self.use_one_hot_embeddings = config.use_one_hot_embeddings
        self.embedding_table = Parameter(initializer
                                         (TruncatedNormal(config.initializer_range),
                                          [config.vocab_size, config.embedding_size]),
                                         name='embedding_table')
        self.expand = P.ExpandDims()
        self.shape_flat = (-1,)
        self.gather = P.Gather()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = (-1, config.seq_length, config.embedding_size)

    def construct(self, input_ids):
        """embedding lookup"""
        flat_ids = self.reshape(input_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(
                one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)
        output = self.reshape(output_for_reshape, self.shape)
        return output, self.embedding_table


class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional and token type embeddings to word embeddings.

    Args:
        config (AlbertConfig): Albert Config.
    """

    def __init__(self, config):
        super(EmbeddingPostprocessor, self).__init__()
        self.use_token_type = config.use_token_type
        self.token_type_vocab_size = config.type_vocab_size
        self.use_one_hot_embeddings = config.use_one_hot_embeddings
        self.max_position_embeddings = config.max_position_embeddings
        self.embedding_table = Parameter(initializer
                                         (TruncatedNormal(config.initializer_range),
                                          [config.type_vocab_size,
                                           config.embedding_size]))
        self.shape_flat = (-1,)
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.1, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = (-1, config.seq_length, config.embedding_size)
        self.layernorm = nn.LayerNorm((config.embedding_size,))
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)
        self.gather = P.Gather()
        self.use_relative_positions = config.use_relative_positions
        self.slice = P.StridedSlice()
        self.full_position_embeddings = Parameter(initializer
                                                  (TruncatedNormal(config.initializer_range),
                                                   [config.max_position_embeddings,
                                                    config.embedding_size]))

    def construct(self, token_type_ids, word_embeddings):
        """embedding postprocessor"""
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


class AlbertOutput(nn.Cell):
    """
    Apply a linear computation to hidden status and a residual computation to input.

    Args:
        config (AlbertConfig): Albert Config.
    """

    def __init__(self, config):
        super(AlbertOutput, self).__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size,
                              weight_init=TruncatedNormal(config.initializer_range)).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)
        self.add = P.Add()
        self.is_gpu = context.get_context('device_target') == "GPU"
        if self.is_gpu:
            self.layernorm = nn.LayerNorm((config.hidden_size,)).to_float(mstype.float32)
            self.compute_type = config.compute_type
        else:
            self.layernorm = nn.LayerNorm((config.hidden_size,)).to_float(config.compute_type)

        self.cast = P.Cast()

    def construct(self, hidden_status, input_tensor):
        """bert output"""
        output = self.dense(hidden_status)
        output = self.dropout(output)
        output = self.add(input_tensor, output)
        output = self.layernorm(output)
        if self.is_gpu:
            output = self.cast(output, self.compute_type)
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
        self._max_relative_position = Tensor(max_relative_position, dtype=mstype.int32)
        self._min_relative_position = Tensor(-max_relative_position, dtype=mstype.int32)
        self.range_length = -length + 1
        self.tile = P.Tile()
        self.range_mat = P.Reshape()
        self.sub = P.Sub()
        self.expanddims = P.ExpandDims()
        self.cast = P.Cast()

    def construct(self):
        """position matrix generator"""
        range_vec_row_out = self.cast(F.tuple_to_array(F.make_range(self._length)), mstype.int32)
        range_vec_col_out = self.range_mat(range_vec_row_out, (self._length, -1))
        tile_row_out = self.tile(range_vec_row_out, (self._length,))
        tile_col_out = self.tile(range_vec_col_out, (1, self._length))
        range_mat_out = self.range_mat(tile_row_out, (self._length, self._length))
        transpose_out = self.range_mat(tile_col_out, (self._length, self._length))
        distance_mat = self.sub(range_mat_out, transpose_out)
        distance_mat_clipped = C.clip_by_value(distance_mat,
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
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.shape = P.Shape()
        self.gather = P.Gather()  # index_select
        self.matmul = P.BatchMatMul()

    def construct(self):
        """position embedding generation"""
        relative_positions_matrix_out = self.relative_positions_matrix()
        # Generate embedding for each relative position of dimension depth.
        if self.use_one_hot_embeddings:
            flat_relative_positions_matrix = self.reshape(relative_positions_matrix_out, (-1,))
            one_hot_relative_positions_matrix = self.one_hot(
                flat_relative_positions_matrix, self.vocab_size, self.on_value, self.off_value)
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
        min_type = np.finfo(np_type).min
        max_type = np.finfo(np_type).max
        self.tensor_min_type = Tensor([min_type], dtype=src_type)
        self.tensor_max_type = Tensor([max_type], dtype=src_type)
        self.min_op = P.Minimum()
        self.max_op = P.Maximum()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        """saturate cast"""
        out = self.max_op(x, self.tensor_min_type)
        out = self.min_op(out, self.tensor_max_type)
        return self.cast(out, self.dst_type)


class AlbertAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".

    Args:
        config (AlbertConfig): Albert Config.
    """

    def __init__(self, config):
        super(AlbertAttention, self).__init__()
        self.from_seq_length = config.seq_length
        self.to_seq_length = config.seq_length
        self.num_attention_heads = config.num_attention_heads
        self.size_per_head = int(config.hidden_size / config.num_attention_heads)
        self.has_attention_mask = config.has_attention_mask
        self.use_relative_positions = config.use_relative_positions
        self.scores_mul = Tensor([1.0 / math.sqrt(float(self.size_per_head))], dtype=config.compute_type)
        self.reshape = P.Reshape()
        self.shape_from_2d = (-1, config.hidden_size)
        self.shape_to_2d = (-1, config.hidden_size)
        weight = TruncatedNormal(config.initializer_range)

        self.query = nn.Dense(config.hidden_size,
                              config.hidden_size,
                              activation=config.query_act,
                              weight_init=weight).to_float(config.compute_type)
        self.key = nn.Dense(config.hidden_size,
                            config.hidden_size,
                            activation=config.key_act,
                            weight_init=weight).to_float(config.compute_type)
        self.value = nn.Dense(config.hidden_size,
                              config.hidden_size,
                              activation=config.value_act,
                              weight_init=weight).to_float(config.compute_type)
        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.matmul = P.BatchMatMul()
        self.shape_from = (-1, config.seq_length, config.num_attention_heads, self.size_per_head)
        self.shape_to = (-1, config.seq_length, config.num_attention_heads, self.size_per_head)
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = Tensor([-10000.0], dtype=config.compute_type)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(1 - config.attention_probs_dropout_prob)
        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.Add()
            self.cast = P.Cast()
            self.get_dtype = P.DType()
        if config.do_return_2d_tensor:
            self.shape_return = (-1, config.hidden_size)
        else:
            self.shape_return = (-1, config.seq_length, config.hidden_size)
        self.cast_compute_type = SaturateCast(dst_type=config.compute_type)
        if self.use_relative_positions:
            self._generate_relative_positions_embeddings = \
                RelaPosEmbeddingsGenerator(length=config.seq_length,
                                           depth=self.size_per_head,
                                           max_relative_position=16,
                                           initializer_range=config.initializer_range,
                                           use_one_hot_embeddings=config.use_one_hot_embeddings)

    def construct(self, from_tensor, to_tensor, attention_mask):
        """bert attention"""
        # reshape 2d/3d input tensors to 2d
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        query_out = self.query(from_tensor_2d)
        key_out = self.key(to_tensor_2d)
        value_out = self.value(to_tensor_2d)
        query_layer = self.reshape(query_out, self.shape_from)
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, self.shape_to)
        key_layer = self.transpose(key_layer, self.trans_shape)
        attention_scores = self.matmul_trans_b(query_layer, key_layer)
        # use_relative_position, supplementary logic
        if self.use_relative_positions:
            # relations_keys is [F|T, F|T, H]
            relations_keys = self._generate_relative_positions_embeddings()
            relations_keys = self.cast_compute_type(relations_keys)
            # query_layer_t is [F, B, N, H]
            query_layer_t = self.transpose(query_layer, self.trans_shape_relative)
            # query_layer_r is [F, B * N, H]
            query_layer_r = self.reshape(query_layer_t,
                                         (self.from_seq_length,
                                          -1,
                                          self.size_per_head))
            # key_position_scores is [F, B * N, F|T]
            key_position_scores = self.matmul_trans_b(query_layer_r,
                                                      relations_keys)
            # key_position_scores_r is [F, B, N, F|T]
            key_position_scores_r = self.reshape(key_position_scores,
                                                 (self.from_seq_length,
                                                  -1,
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
            # relations_values is [F|T, F|T, H]
            relations_values = self._generate_relative_positions_embeddings()
            relations_values = self.cast_compute_type(relations_values)
            # attention_probs_t is [F, B, N, T]
            attention_probs_t = self.transpose(attention_probs, self.trans_shape_relative)
            # attention_probs_r is [F, B * N, T]
            attention_probs_r = self.reshape(
                attention_probs_t,
                (self.from_seq_length,
                 -1,
                 self.to_seq_length))
            # value_position_scores is [F, B * N, H]
            value_position_scores = self.matmul(attention_probs_r,
                                                relations_values)
            # value_position_scores_r is [F, B, N, H]
            value_position_scores_r = self.reshape(value_position_scores,
                                                   (self.from_seq_length,
                                                    -1,
                                                    self.num_attention_heads,
                                                    self.size_per_head))
            # value_position_scores_r_t is [B, N, F, H]
            value_position_scores_r_t = self.transpose(value_position_scores_r,
                                                       self.trans_shape_position)
            context_layer = context_layer + value_position_scores_r_t
        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, self.shape_return)
        return context_layer, attention_scores


class AlbertSelfAttention(nn.Cell):
    """
    Apply self-attention.

    Args:
        config (AlbertConfig): Albert Config.
    """

    def __init__(self, config):
        super(AlbertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.attention = AlbertAttention(config)
        self.output = AlbertOutput(config)
        self.reshape = P.Reshape()
        self.shape = (-1, config.hidden_size)

    def construct(self, input_tensor, attention_mask):
        """bert self attention"""
        input_tensor = self.reshape(input_tensor, self.shape)
        attention_output, attention_scores = self.attention(input_tensor, input_tensor, attention_mask)
        output = self.output(attention_output, input_tensor)
        return output, attention_scores


class AlbertEncoderCell(nn.Cell):
    """
    Encoder cells used in BertTransformer.

    Args:
        config (AlbertConfig): Albert Config.
    """

    def __init__(self, config):
        super(AlbertEncoderCell, self).__init__()
        self.attention = AlbertSelfAttention(config)
        self.intermediate = nn.Dense(in_channels=config.hidden_size,
                                     out_channels=config.intermediate_size,
                                     activation=config.hidden_act,
                                     weight_init=TruncatedNormal(config.initializer_range)
                                     ).to_float(config.compute_type)
        self.output = AlbertOutput(config)

    def construct(self, hidden_states, attention_mask):
        """bert encoder cell"""
        # self-attention
        attention_output, attention_scores = self.attention(hidden_states, attention_mask)
        # feed construct
        intermediate_output = self.intermediate(attention_output)
        # add and normalize
        output = self.output(intermediate_output, attention_output)
        return output, attention_scores


class AlbertLayer(nn.Cell):
    """
    Args:
        config (AlbertConfig): Albert Config.
    """
    def __init__(self, config):
        super(AlbertLayer, self).__init__()

        self.output_attentions = config.output_attentions
        self.attention = AlbertSelfAttention(config)
        self.ffn = nn.Dense(config.hidden_size,
                            config.intermediate_size,
                            activation=config.hidden_act).to_float(config.compute_type)
        self.ffn_output = nn.Dense(config.intermediate_size, config.hidden_size)
        self.full_layer_layer_norm = nn.LayerNorm((config.hidden_size,))
        self.shape = (-1, config.seq_length, config.hidden_size)
        self.reshape = P.Reshape()

    def construct(self, hidden_states, attention_mask):
        attention_output, attention_scores = self.attention(hidden_states, attention_mask)

        ffn_output = self.ffn(attention_output)
        ffn_output = self.ffn_output(ffn_output)
        ffn_output = self.reshape(ffn_output + attention_output, self.shape)
        hidden_states = self.full_layer_layer_norm(ffn_output)

        return hidden_states, attention_scores


class AlbertLayerGroup(nn.Cell):
    """
    Args:
        config (AlbertConfig): Albert Config.
    """

    def __init__(self, config):
        super(AlbertLayerGroup, self).__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.albert_layers = nn.CellList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def construct(self, hidden_states, attention_mask):
        layer_hidden_states = ()
        layer_attentions = ()

        for _, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask)
            hidden_states = layer_output[0]
            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)
            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        return outputs


class AlbertTransformer(nn.Cell):
    """
    Multi-layer bert transformer.

    Args:
        config (AlbertConfig): Albert Config.
    """

    def __init__(self, config):
        super(AlbertTransformer, self).__init__()
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups
        self.group_idx_list = [int(_ / (config.num_hidden_layers / config.num_hidden_groups))
                               for _ in range(config.num_hidden_layers)]

        self.embedding_hidden_mapping_in = nn.Dense(config.embedding_size, config.hidden_size)
        self.return_all_encoders = config.return_all_encoders
        layers = []
        for _ in range(config.num_hidden_groups):
            layer = AlbertLayerGroup(config)
            layers.append(layer)
        self.albert_layer_groups = nn.CellList(layers)
        self.reshape = P.Reshape()
        self.shape = (-1, config.embedding_size)
        self.out_shape = (-1, config.seq_length, config.hidden_size)

    def construct(self, input_tensor, attention_mask):
        """bert transformer"""
        prev_output = self.reshape(input_tensor, self.shape)
        prev_output = self.embedding_hidden_mapping_in(prev_output)
        all_encoder_layers = ()
        all_encoder_atts = ()
        all_encoder_outputs = (prev_output,)
        # for layer_module in self.layers:
        for i in range(self.num_hidden_layers):
            # Index of the hidden group
            group_idx = self.group_idx_list[i]

            layer_output, encoder_att = self.albert_layer_groups[group_idx](prev_output, attention_mask)
            prev_output = layer_output
            if self.return_all_encoders:
                all_encoder_outputs += (layer_output,)
                layer_output = self.reshape(layer_output, self.out_shape)
                all_encoder_layers += (layer_output,)
                all_encoder_atts += (encoder_att,)
        if not self.return_all_encoders:
            prev_output = self.reshape(prev_output, self.out_shape)
            all_encoder_layers += (prev_output,)
        return prev_output


class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config (Class): Configuration for BertModel.
    """

    def __init__(self, config):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = (-1, 1, config.seq_length)

    def construct(self, input_mask):
        attention_mask = self.cast(self.reshape(input_mask, self.shape), mstype.float32)
        return attention_mask


class AlbertModel(nn.Cell):
    """
    Bidirectional Encoder Representations from Transformers.

    Args:
        config (Class): Configuration for BertModel.
    """

    def __init__(self, config):
        super(AlbertModel, self).__init__()
        config = copy.deepcopy(config)
        if not config.is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        self.seq_length = config.seq_length
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_size = config.hidden_size
        self.token_type_ids = None
        self.last_idx = self.num_hidden_layers - 1
        self.use_word_embeddings = config.use_word_embeddings
        if self.use_word_embeddings:
            self.word_embeddings = EmbeddingLookup(config)
        self.embedding_postprocessor = EmbeddingPostprocessor(config)
        self.encoder = AlbertTransformer(config)
        self.cast = P.Cast()
        self.dtype = config.dtype
        self.cast_compute_type = SaturateCast(dst_type=config.compute_type)
        self.slice = P.StridedSlice()
        self.squeeze_1 = P.Squeeze(axis=1)
        self.pooler = nn.Dense(self.hidden_size, self.hidden_size,
                               activation="tanh",
                               weight_init=TruncatedNormal(config.initializer_range)).to_float(config.compute_type)
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)

    def construct(self, input_ids, token_type_ids, input_mask):
        """bert model"""
        # embedding
        if self.use_word_embeddings:
            word_embeddings, _ = self.word_embeddings(input_ids)
        else:
            word_embeddings = input_ids
        embedding_output = self.embedding_postprocessor(token_type_ids, word_embeddings)
        # attention mask [batch_size, seq_length, seq_length]
        attention_mask = self._create_attention_mask_from_input_mask(input_mask)
        # bert encoder
        encoder_output = self.encoder(self.cast_compute_type(embedding_output), attention_mask)
        sequence_output = self.cast(encoder_output, self.dtype)
        # pooler
        batch_size = P.Shape()(input_ids)[0]
        sequence_slice = self.slice(sequence_output,
                                    (0, 0, 0),
                                    (batch_size, 1, self.hidden_size),
                                    (1, 1, 1))
        first_token = self.squeeze_1(sequence_slice)
        pooled_output = self.pooler(first_token)
        pooled_output = self.cast(pooled_output, self.dtype)
        return sequence_output, pooled_output


class AlbertMLMHead(nn.Cell):
    """
    Get masked lm output.

    Args:
        config (AlbertConfig): The config of BertModel.

    Returns:
        Tensor, masked lm output.
    """
    def __init__(self, config):
        super(AlbertMLMHead, self).__init__()

        self.layernorm = nn.LayerNorm((config.embedding_size,)).to_float(config.compute_type)
        self.dense = nn.Dense(
            config.hidden_size,
            config.embedding_size,
            weight_init=TruncatedNormal(config.initializer_range),
            activation=config.hidden_act
        ).to_float(config.compute_type)
        self.decoder = nn.Dense(
            config.embedding_size,
            config.vocab_size,
            weight_init=TruncatedNormal(config.initializer_range),
        ).to_float(config.compute_type)

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class AlbertModelCLS(nn.Cell):
    """
    This class is responsible for classification task evaluation,
    i.e. mnli(num_labels=3), qnli(num_labels=2), qqp(num_labels=2).
    The returned output represents the final logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config):
        super(AlbertModelCLS, self).__init__()
        self.albert = AlbertModel(config)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.classifier = nn.Dense(config.hidden_size, config.num_labels, weight_init=self.weight_init,
                                   has_bias=True).to_float(config.compute_type)
        self.relu = nn.ReLU()
        self.is_training = config.is_training
        if self.is_training:
            self.dropout = nn.Dropout(1 - config.classifier_dropout_prob)

    def construct(self, input_ids, input_mask, token_type_id):
        """classification albert model"""
        _, pooled_output = self.albert(input_ids, token_type_id, input_mask)
        # pooled_output = self.relu(pooled_output)
        if self.is_training:
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.cast(logits, self.dtype)
        return logits


class AlbertModelForAD(nn.Cell):
    """albert model for ad"""

    def __init__(self, config):
        super(AlbertModelForAD, self).__init__()

        # main model
        self.albert = AlbertModel(config)

        # classifier head
        self.cast = P.Cast()
        self.dtype = config.dtype
        self.classifier = nn.Dense(config.hidden_size, config.num_labels,
                                   weight_init=TruncatedNormal(config.initializer_range),
                                   has_bias=True).to_float(config.compute_type)
        self.is_training = config.is_training
        if self.is_training:
            self.dropout = nn.Dropout(1 - config.classifier_dropout_prob)

        # masked language model head
        self.predictions = AlbertMLMHead(config)

    def construct(self, input_ids, input_mask, token_type_id):
        """albert model for ad"""
        sequence_output, pooled_output = self.albert(input_ids, token_type_id, input_mask)
        if self.is_training:
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.cast(logits, self.dtype)
        prediction_scores = self.predictions(sequence_output)
        prediction_scores = self.cast(prediction_scores, self.dtype)
        return prediction_scores, logits


class AlbertModelMLM(nn.Cell):
    """albert model for mlm"""

    def __init__(self, config):
        super(AlbertModelMLM, self).__init__()
        self.cast = P.Cast()
        self.dtype = config.dtype

        # main model
        self.albert = AlbertModel(config)

        # masked language model head
        self.predictions = AlbertMLMHead(config)

    def construct(self, input_ids, input_mask, token_type_id):
        """albert model for mlm"""
        sequence_output, _ = self.albert(input_ids, token_type_id, input_mask)
        prediction_scores = self.predictions(sequence_output)
        prediction_scores = self.cast(prediction_scores, self.dtype)
        return prediction_scores
