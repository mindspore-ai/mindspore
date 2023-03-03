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
"""Transformer model."""

import math
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops.primitive import constexpr
from mindspore.ops.operations import array_ops as array_op
from mindspore.ops.operations import _inner_ops as inner_op

BATCH_SIZE = 32
HIDDEN_SIZE = 1024
NUMBER_HIDDEN_LAYERS = 6
MAX_DECODE_LENGTH = 80
VOCAB_SIZE = 36560
MAX_POSITION_EMBEDDINGS = 128
INTERMEDIATE_SIZE = 4096
INIT_RANGE = 0.02
HIDDEN_DROPOUT_PROB = 0.2
NUM_ATTENTION_HEADS = 16
ATTENTION_PROBS_DROPOUT_PROB = 0.2
HIDDEN_ACT = "relu"
COMPUTE_TYPE = ms.float16
DTYPE = ms.float32
BEAM_WIDTH = 4
SEQ_LENGTH = 128
LENGTH_PENALTY_WEIGHT = 1.0


def normal_weight(shape, num_units):
    norm = np.random.normal(0.0, num_units**-0.5, shape).astype(np.float32)
    return Tensor(norm)


def _average_units(shape):
    """
    Average shape dim.
    """
    if not shape:
        return 1.
    if len(shape) == 1:
        return float(shape[0])
    if len(shape) == 2:
        return float(shape[0] + shape[1]) / 2.
    raise RuntimeError("not support shape.")


def weight_variable(shape):
    scale_shape = shape
    avg_units = _average_units(scale_shape)
    scale = 1.0 / max(1., avg_units)
    limit = math.sqrt(3.0 * scale)
    values = np.random.uniform(-limit, limit, shape).astype(np.float32)
    return Tensor(values)


class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
    """
    def __init__(self,
                 is_graph_mode,
                 batch_size,
                 vocab_size,
                 embedding_size,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02):
        super(EmbeddingLookup, self).__init__()
        self.is_graph_mode = is_graph_mode
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = Parameter(normal_weight([vocab_size, embedding_size], embedding_size))
        self.expand = ops.ExpandDims()
        self.shape_flat = (-1,)
        self.gather = ops.Gather()
        self.one_hot = ops.OneHot()
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.array_mul = ops.MatMul()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()

    def construct(self, input_ids):
        """Get a embeddings lookup table with a fixed dictionary and size."""
        input_shape = self.shape(input_ids)

        flat_ids = self.reshape(input_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        out_shape = input_shape + (self.embedding_size,)
        output = self.reshape(output_for_reshape, out_shape)
        return output, self.embedding_table.value()


def position_encoding(length,
                      depth,
                      min_timescale=1,
                      max_timescale=1e4):
    """
    Create Tensor of sinusoids of different frequencies.

    Args:
        length (int): Length of the Tensor to create, i.e. Number of steps.
        depth (int): Hidden size.
        min_timescale (float): Default: 1.
        max_timescale (float): Default: 10000.

    Returns:
        Tensor of shape (length, depth)
    """
    depth = depth // 2
    positions = np.arange(length, dtype=np.float32)
    log_timescale_increment = (np.log(max_timescale / min_timescale) / (depth - 1))
    inv_timescales = min_timescale * np.exp(np.arange(depth, dtype=np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return x


class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional embeddings to word embeddings.

    Args:
        embedding_size (int): The size of each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 128.
        dropout_prob (float): The dropout probability. Default: 0.1.
    """
    def __init__(self,
                 is_graph_mode,
                 embedding_size,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 max_position_embeddings=128,
                 dropout_prob=0.1):
        super(EmbeddingPostprocessor, self).__init__()
        self.scores_mul = Tensor([math.sqrt(float(embedding_size))], dtype=ms.float32)
        self.multiply = ops.Mul()
        self.add = ops.Add()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.use_dropout = dropout_prob > 0
        self.expand_dims = ops.ExpandDims()
        self.position_embedding_table = Tensor(position_encoding(max_position_embeddings, embedding_size),
                                               ms.float32)
        self.shape = ops.Shape()

    def construct(self, word_embeddings):
        """Postprocessors apply positional embeddings to word embeddings."""
        input_shape = self.shape(word_embeddings)
        input_len = input_shape[1]

        output = self.multiply(word_embeddings, self.scores_mul)

        # add position embeddings
        position_embeddings = self.position_embedding_table[0:input_len:1, ::]
        position_embeddings = self.expand_dims(position_embeddings, 0)
        output = self.add(output, position_embeddings)

        if self.use_dropout:
            output = self.dropout(output)
        return output


class CastWrapper(nn.Cell):
    """
    Cast wrapper.
    """
    def __init__(self, src_type=ms.float32, dst_type=ms.float32):
        super(CastWrapper, self).__init__()
        self.cast = ops.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        return self.cast(x, self.dst_type)


class LayerPreprocess(nn.Cell):
    """
    preprocess input of each layer.
    """
    def __init__(self,
                 in_channels=None):
        super(LayerPreprocess, self).__init__()
        self.layernorm = nn.LayerNorm((in_channels,))
        self.cast = ops.Cast()
        self.get_dtype = ops.DType()

    def construct(self, input_tensor):
        output = self.cast(input_tensor, ms.float32)
        output = self.layernorm(output)
        output = self.cast(output, self.get_dtype(input_tensor))
        return output


class LayerPostprocess(nn.Cell):
    """
    postprocess output of each layer.
    """
    def __init__(self,
                 dropout_prob=0.1):
        super(LayerPostprocess, self).__init__()
        self.add = ops.Add()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.use_dropout = dropout_prob > 0

    def construct(self, hidden_tensor, input_tensor):
        output = hidden_tensor
        if self.use_dropout:
            output = self.dropout(output)
        output = self.add(output, input_tensor)
        return output


class MultiheadAttention(nn.Cell):
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
                                      MultiheadAttention. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        do_return_2d_tensor (bool): True for return 2d tensor. False for return 3d
                             tensor. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in MultiheadAttention. Default: ms.float32.
    """
    def __init__(self,
                 is_graph_mode,
                 batch_size,
                 from_tensor_width,
                 to_tensor_width,
                 out_tensor_width,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 out_act=None,
                 has_attention_mask=True,
                 attention_probs_dropout_prob=0.0,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 do_return_2d_tensor=True,
                 compute_type=ms.float32):
        super(MultiheadAttention, self).__init__()
        self.batch_size = batch_size
        self.is_graph_mode = is_graph_mode
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.has_attention_mask = has_attention_mask
        assert has_attention_mask
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor

        self.scores_mul = Tensor([1.0 / math.sqrt(float(self.size_per_head))], dtype=compute_type)
        self.reshape = ops.Reshape()
        self.shape_from_2d = (-1, from_tensor_width)
        self.shape_to_2d = (-1, to_tensor_width)
        units = num_attention_heads * size_per_head
        self.query_layer = nn.Dense(from_tensor_width,
                                    units,
                                    activation=query_act,
                                    has_bias=False,
                                    weight_init=weight_variable([units, from_tensor_width])).to_float(compute_type)
        self.key_layer = nn.Dense(to_tensor_width,
                                  units,
                                  activation=key_act,
                                  has_bias=False,
                                  weight_init=weight_variable([units, to_tensor_width])).to_float(compute_type)
        self.value_layer = nn.Dense(to_tensor_width,
                                    units,
                                    activation=value_act,
                                    has_bias=False,
                                    weight_init=weight_variable([units, to_tensor_width])).to_float(compute_type)
        self.out_layer = nn.Dense(units,
                                  out_tensor_width,
                                  activation=out_act,
                                  has_bias=False,
                                  weight_init=weight_variable([out_tensor_width, units])).to_float(compute_type)

        self.matmul_trans_b = ops.BatchMatMul(transpose_b=True)
        self.multiply = ops.Mul()
        self.transpose = ops.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = Tensor([-10000.0,], dtype=compute_type)
        self.batch_num = batch_size * num_attention_heads
        self.matmul = ops.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=attention_probs_dropout_prob)
        self.use_dropout = attention_probs_dropout_prob > 0

        if self.has_attention_mask:
            self.expand_dims = ops.ExpandDims()
            self.sub = ops.Sub()
            self.add = ops.Add()
            self.cast = ops.Cast()
            self.get_dtype = ops.DType()

        self.cast_compute_type = CastWrapper(dst_type=compute_type)
        self.softmax_cast = ops.Cast()

    def construct(self, from_tensor, to_tensor, seq_length, enc_seq_length, attention_mask=None):
        """Apply multihead attention."""
        shape_from = (self.batch_size, -1, self.num_attention_heads, self.size_per_head)
        shape_to = (self.batch_size, -1, self.num_attention_heads, self.size_per_head)
        if self.do_return_2d_tensor:
            shape_return = (-1, self.num_attention_heads * self.size_per_head)
        else:
            shape_return = (self.batch_size, -1, self.num_attention_heads * self.size_per_head)

        # reshape 2d/3d input tensors to 2d
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)
        value_out = self.value_layer(to_tensor_2d)

        query_layer = self.reshape(query_out, shape_from)
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, shape_to)
        key_layer = self.transpose(key_layer, self.trans_shape)

        attention_scores = self.matmul_trans_b(query_layer, key_layer)
        attention_scores = self.multiply(attention_scores, self.scores_mul)

        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(ops.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))
            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        attention_scores = self.softmax_cast(attention_scores, ms.float32)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.softmax_cast(attention_probs, self.get_dtype(key_layer))
        if self.use_dropout:
            attention_probs = self.dropout(attention_probs)

        value_layer = self.reshape(value_out, shape_to)
        value_layer = self.transpose(value_layer, self.trans_shape)
        context_layer = self.matmul(attention_probs, value_layer)

        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, shape_return)
        context_layer = self.out_layer(context_layer)
        return context_layer


class SelfAttention(nn.Cell):
    """
    Apply self-attention.

    Args:
        batch_size (int): Batch size of input dataset.
        from_seq_length (int): Length of query sequence.
        to_seq_length (int): Length of memory sequence.
        hidden_size (int): Size of attention layers.
        num_attention_heads (int): Number of attention heads. Default: 16.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one_hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        has_attention_mask (bool): Specifies whether has attention mask. Default: True.
        is_encdec_att (bool): Specifies whether query sequence and memory sequence are different. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in MultiheadAttention. Default: ms.float32.
    """
    def __init__(self,
                 is_graph_mode,
                 batch_size,
                 hidden_size,
                 num_attention_heads=16,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 has_attention_mask=True,
                 is_encdec_att=False,
                 compute_type=ms.float32):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))
        self.size_per_head = int(hidden_size / num_attention_heads)
        self.is_encdec_att = is_encdec_att

        self.attention = MultiheadAttention(
            is_graph_mode=is_graph_mode,
            batch_size=batch_size,
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            out_tensor_width=hidden_size,
            num_attention_heads=num_attention_heads,
            size_per_head=self.size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            has_attention_mask=has_attention_mask,
            do_return_2d_tensor=True,
            compute_type=compute_type)

        self.preprocess = LayerPreprocess(in_channels=hidden_size)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)

        self.reshape = ops.Reshape()
        self.shape = (-1, hidden_size)

    def construct(self, input_tensor, memory_tensor, attention_mask, seq_length, enc_seq_length):
        """Apply self-attention."""
        input_tensor = self.reshape(input_tensor, self.shape)
        memory_tensor = self.reshape(memory_tensor, self.shape)

        output = self.preprocess(input_tensor)

        if not self.is_encdec_att:
            memory_tensor = output

        attention_output = self.attention(output, memory_tensor, seq_length, enc_seq_length, attention_mask)
        output = self.postprocess(attention_output, input_tensor)
        return output


class FeedForward(nn.Cell):
    """
    Apply two-layer feed forward

    Args:
        in_channels (int): Size of the input layer.
        hidden_size (int): Size of the hidden layer.
        out_channels (int): Size of the output layers.
        hidden_act (str): name of the activation function. Default: relu
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        compute_type (:class:`mindspore.dtype`): Compute type in FeedForward. Default: ms.float32.
    """
    def __init__(self,
                 in_channels,
                 hidden_size,
                 out_channels,
                 hidden_act="relu",
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 compute_type=ms.float32):
        super(FeedForward, self).__init__()

        self.conv1 = nn.Dense(in_channels,
                              hidden_size,
                              activation=hidden_act,
                              weight_init=weight_variable([hidden_size, in_channels])).to_float(compute_type)
        self.conv2 = nn.Dense(hidden_size,
                              out_channels,
                              weight_init=weight_variable([out_channels, hidden_size])).to_float(compute_type)

        self.preprocess = LayerPreprocess(in_channels=in_channels)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)

        self.reshape = ops.Reshape()
        self.shape = (-1, in_channels)
        self.dropout = nn.Dropout(p=hidden_dropout_prob)
        self.use_dropout = hidden_dropout_prob > 0

    def construct(self, input_tensor):
        input_tensor = self.reshape(input_tensor, self.shape)
        output = self.preprocess(input_tensor)
        output = self.conv1(output)
        if self.use_dropout:
            output = self.dropout(output)
        output = self.conv2(output)
        output = self.postprocess(output, input_tensor)
        return output


class EncoderCell(nn.Cell):
    """
    Encoder cells used in Transformer.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers. Default: 1024.
        seq_length (int): Length of input sequence. Default: 128.
        num_attention_heads (int): Number of attention heads. Default: 16.
        intermediate_size (int): Size of intermediate layer. Default: 4096.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.02.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.1.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function. Default: "relu".
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: ms.float32.
    """
    def __init__(self,
                 is_graph_mode,
                 batch_size,
                 hidden_size=1024,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=ms.float32):
        super(EncoderCell, self).__init__()
        self.attention = SelfAttention(
            is_graph_mode=is_graph_mode,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            is_encdec_att=False,
            compute_type=compute_type)
        self.feedforward = FeedForward(
            in_channels=hidden_size,
            hidden_size=intermediate_size,
            out_channels=hidden_size,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)

    def construct(self, hidden_states, attention_mask, seq_length):
        # self-attention with ln, res
        attention_output = self.attention(hidden_states, hidden_states, attention_mask, seq_length, seq_length)
        # feed forward with ln, res
        output = self.feedforward(attention_output)
        return output


class TransformerEncoder(nn.Cell):
    """
    Multi-layer transformer encoder.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers.
        seq_length (int): Length of input sequence.
        num_hidden_layers (int): Number of hidden layers in encoder cells.
        num_attention_heads (int): Number of attention heads in encoder cells. Default: 16.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 4096.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1..
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type. Default: ms.float32.
    """
    def __init__(self,
                 is_graph_mode,
                 batch_size,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=ms.float32):
        super(TransformerEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size
        self.is_graph_mode = is_graph_mode
        self.hidden_size = hidden_size

        layers = []
        for _ in range(num_hidden_layers):
            layer = EncoderCell(is_graph_mode=is_graph_mode,
                                batch_size=batch_size,
                                hidden_size=hidden_size,
                                num_attention_heads=num_attention_heads,
                                intermediate_size=intermediate_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                use_one_hot_embeddings=use_one_hot_embeddings,
                                initializer_range=initializer_range,
                                hidden_dropout_prob=hidden_dropout_prob,
                                hidden_act=hidden_act,
                                compute_type=compute_type)
            layers.append(layer)
        self.layers = nn.CellList(layers)

        self.layer_preprocess = LayerPreprocess(in_channels=hidden_size)

        self.reshape = ops.Reshape()
        self.shape = (-1, hidden_size)

    def construct(self, input_tensor, attention_mask, seq_length):
        """Apply encoder."""
        out_shape = (self.batch_size, seq_length, self.hidden_size)
        prev_output = self.reshape(input_tensor, self.shape)

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, seq_length)
            prev_output = layer_output

        prev_output = self.layer_preprocess(prev_output)
        output = self.reshape(prev_output, out_shape)
        return output


class DecoderCell(nn.Cell):
    """
    decoder cells used in Transformer.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the Transformer decoder layers. Default: 1024.
        seq_length (int): Length of input sequence. Default: 128.
        enc_seq_length (int): Length of source sentences. Default:128
        num_attention_heads (int): Number of attention heads. Default: 12.
        intermediate_size (int): Size of intermediate layer. Default: 4096.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.02.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function. Default: "relu".
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: ms.float32.
    """
    def __init__(self,
                 is_graph_mode,
                 batch_size,
                 hidden_size=1024,
                 num_attention_heads=12,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.02,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=ms.float32):
        super(DecoderCell, self).__init__()
        self.self_attention = SelfAttention(
            is_graph_mode=is_graph_mode,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            is_encdec_att=False,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)
        self.cross_attention = SelfAttention(
            is_graph_mode=is_graph_mode,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            is_encdec_att=True,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)
        self.feedforward = FeedForward(
            in_channels=hidden_size,
            hidden_size=intermediate_size,
            out_channels=hidden_size,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)

    def construct(self, hidden_states, attention_mask, enc_states, enc_attention_mask, seq_length, enc_seq_length):
        # self-attention with ln, res
        attention_output = self.self_attention(hidden_states, hidden_states, attention_mask, seq_length, seq_length)
        # cross-attention with ln, res
        attention_output = self.cross_attention(attention_output, enc_states, enc_attention_mask,
                                                seq_length, enc_seq_length)
        # feed forward with ln, res
        output = self.feedforward(attention_output)
        return output


class TransformerDecoder(nn.Cell):
    """
    Multi-layer transformer decoder.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers.
        seq_length (int): Length of input sequence.
        enc_seq_length (int): Length of source sentences.
        num_hidden_layers (int): Number of hidden layers in encoder cells.
        num_attention_heads (int): Number of attention heads in encoder cells. Default: 16.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 4096.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type. Default: ms.float32.
    """
    def __init__(self,
                 is_graph_mode,
                 batch_size,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=ms.float32):
        super(TransformerDecoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers

        layers = []
        for _ in range(num_hidden_layers):
            layer = DecoderCell(is_graph_mode=is_graph_mode,
                                batch_size=batch_size,
                                hidden_size=hidden_size,
                                num_attention_heads=num_attention_heads,
                                intermediate_size=intermediate_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                use_one_hot_embeddings=use_one_hot_embeddings,
                                initializer_range=initializer_range,
                                hidden_dropout_prob=hidden_dropout_prob,
                                hidden_act=hidden_act,
                                compute_type=compute_type)
            layers.append(layer)
        self.layers = nn.CellList(layers)

        self.layer_preprocess = LayerPreprocess(in_channels=hidden_size)

        self.reshape = ops.Reshape()
        self.shape = (-1, hidden_size)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.is_graph_mode = is_graph_mode

    def construct(self, input_tensor, attention_mask, enc_states, enc_attention_mask, seq_length, enc_seq_length):
        """Apply decoder."""
        out_shape = (self.batch_size, seq_length, self.hidden_size)
        prev_output = self.reshape(input_tensor, self.shape)

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, enc_states, enc_attention_mask,
                                        seq_length, enc_seq_length)
            prev_output = layer_output

        prev_output = self.layer_preprocess(prev_output)
        output = self.reshape(prev_output, out_shape)
        return output


class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        is_graph_mode (bool): is graph mode.
    """
    def __init__(self, is_graph_mode):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()
        self.batch_matmul = ops.BatchMatMul()
        self.expand_dims = ops.ExpandDims()
        self.is_graph_mode = is_graph_mode

    def construct(self, input_mask):
        """Create attention mask according to input mask."""
        input_shape = self.shape(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)

        input_mask = self.cast(input_mask, ms.float32)
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.batch_matmul(mask_left, mask_right)

        return attention_mask


class PredLogProbs(nn.Cell):
    """
    Get log probs.

    Args:
        batch_size (int): Batch size.
        seq_length (int): Length of input sequence.
        width (int): Hidden size.
        compute_type (:class:`mindspore.dtype`): Compute type. Default: ms.float32.
        dtype (:class:`mindspore.dtype`): Compute type to compute log_softmax. Default: ms.float32.
    """
    def __init__(self,
                 is_graph_mode,
                 batch_size,
                 width,
                 compute_type=ms.float32,
                 dtype=ms.float32):
        super(PredLogProbs, self).__init__()
        self.is_graph_mode = is_graph_mode
        self.batch_size = batch_size
        self.width = width
        self.compute_type = compute_type
        self.dtype = dtype

        self.reshape = ops.Reshape()
        self.matmul = ops.MatMul(transpose_b=True)
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.cast = ops.Cast()

    def construct(self,
                  input_tensor,
                  output_weights,
                  seq_length):
        """Get log probs."""
        shape_flat_sequence_tensor = (self.batch_size * seq_length, self.width)

        input_tensor = self.reshape(input_tensor, shape_flat_sequence_tensor)
        input_tensor = self.cast(input_tensor, self.compute_type)
        output_weights = self.cast(output_weights, self.compute_type)

        logits = self.matmul(input_tensor, output_weights)
        logits = self.cast(logits, self.dtype)

        log_probs = self.log_softmax(logits)
        return log_probs


@constexpr
def convert_np_to_tensor_encoder(seq_length):
    ones = np.ones(shape=(seq_length, seq_length))
    return Tensor(np.tril(ones), dtype=ms.float32)


class TransformerModel(nn.Cell):
    """
    Transformer with encoder and decoder.

    Args:
        is_training (bool): True for training mode. False for eval mode.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        is_graph_mode (bool): is graph mode.
    """
    def __init__(self,
                 is_training,
                 use_one_hot_embeddings=False,
                 is_graph_mode=False):
        super(TransformerModel, self).__init__()
        self.is_training = is_training


        self.batch_size = BATCH_SIZE
        self.is_graph_mode = is_graph_mode
        self.hidden_size = HIDDEN_SIZE
        self.num_hidden_layers = NUMBER_HIDDEN_LAYERS
        self.embedding_size = HIDDEN_SIZE

        self.last_idx = self.num_hidden_layers - 1
        self.beam_width = BEAM_WIDTH
        self.max_decode_length = MAX_DECODE_LENGTH

        self.tfm_embedding_lookup = EmbeddingLookup(
            is_graph_mode=self.is_graph_mode,
            batch_size=self.batch_size,
            vocab_size=VOCAB_SIZE,
            embedding_size=self.embedding_size,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=INIT_RANGE)
        self.tfm_embedding_postprocessor_for_encoder = EmbeddingPostprocessor(
            is_graph_mode=self.is_graph_mode,
            embedding_size=self.embedding_size,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=0.02,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            dropout_prob=HIDDEN_DROPOUT_PROB)
        self.tfm_embedding_postprocessor_for_decoder = EmbeddingPostprocessor(
            is_graph_mode=self.is_graph_mode,
            embedding_size=self.embedding_size,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=0.02,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            dropout_prob=HIDDEN_DROPOUT_PROB)
        self.tfm_encoder = TransformerEncoder(
            is_graph_mode=self.is_graph_mode,
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            num_attention_heads=NUM_ATTENTION_HEADS,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=INTERMEDIATE_SIZE,
            attention_probs_dropout_prob=ATTENTION_PROBS_DROPOUT_PROB,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=INIT_RANGE,
            hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
            hidden_act=HIDDEN_ACT,
            compute_type=COMPUTE_TYPE)

        if is_training:
            self.projection = PredLogProbs(
                is_graph_mode=self.is_graph_mode,
                batch_size=self.batch_size,
                width=self.hidden_size,
                compute_type=COMPUTE_TYPE,
                dtype=DTYPE)
            self.tfm_decoder = TransformerDecoder(
                is_graph_mode=self.is_graph_mode,
                batch_size=self.batch_size,
                hidden_size=self.hidden_size,
                num_attention_heads=NUM_ATTENTION_HEADS,
                num_hidden_layers=self.num_hidden_layers,
                intermediate_size=INTERMEDIATE_SIZE,
                attention_probs_dropout_prob=ATTENTION_PROBS_DROPOUT_PROB,
                use_one_hot_embeddings=use_one_hot_embeddings,
                initializer_range=INIT_RANGE,
                hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
                hidden_act=HIDDEN_ACT,
                compute_type=COMPUTE_TYPE)

        self.cast = ops.Cast()
        self.dtype = DTYPE
        self.cast_compute_type = CastWrapper(dst_type=COMPUTE_TYPE)
        self.expand = ops.ExpandDims()
        self.multiply = ops.Mul()
        self.shape = ops.Shape()
        self.tril = array_op.Tril()
        self.dynamic_broadcast_to = inner_op.DynamicBroadcastTo()
        self.ones_like = array_op.OnesLike()
        self.stack = array_op.Stack(0)
        self.concatenate = array_op.Concat()

        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(self.is_graph_mode)

    def construct(self, source_ids, source_mask, target_ids=None, target_mask=None):
        """Transformer with encoder and decoder."""
        seq_length = self.shape(source_ids)[1]

        # process source sentence
        src_word_embeddings, embedding_tables = self.tfm_embedding_lookup(source_ids)
        src_embedding_output = self.tfm_embedding_postprocessor_for_encoder(src_word_embeddings)
        # attention mask [batch_size, seq_length, seq_length]
        enc_attention_mask = self._create_attention_mask_from_input_mask(source_mask)
        # transformer encoder
        encoder_output = self.tfm_encoder(self.cast_compute_type(src_embedding_output),
                                          self.cast_compute_type(enc_attention_mask),
                                          seq_length)


        if self.is_graph_mode:
            ones_inner = self.ones_like(source_ids[0, :])
            seq_length = self.shape(ones_inner)[0]
            ones = self.dynamic_broadcast_to(ones_inner, (seq_length, seq_length))
            future_mask = self.tril(ones)
        else:
            future_mask = convert_np_to_tensor_encoder(seq_length)


        # process target sentence
        tgt_word_embeddings, _ = self.tfm_embedding_lookup(target_ids)
        tgt_embedding_output = self.tfm_embedding_postprocessor_for_decoder(tgt_word_embeddings)
        # attention mask [batch_size, seq_length, seq_length]
        tgt_attention_mask = self._create_attention_mask_from_input_mask(target_mask)
        tgt_attention_mask = self.multiply(tgt_attention_mask, self.expand(future_mask, 0))
        # transformer decoder
        decoder_output = self.tfm_decoder(self.cast_compute_type(tgt_embedding_output),
                                          self.cast_compute_type(tgt_attention_mask),
                                          encoder_output, enc_attention_mask,
                                          seq_length, seq_length)
        # calculate logits and log_probs
        log_probs = self.projection(decoder_output, embedding_tables, seq_length)
        ret = log_probs
        return ret
