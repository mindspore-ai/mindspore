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
import mindspore.ops.function as func

BATCH_SIZE_VALUE = 32
INF = 1. * 1e9


class LengthPenalty(nn.Cell):
    """
    Normalize scores of translations according to their length.

    Args:
        weight (float): Weight of length penalty. Default: 1.0.
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: ms.float32.
    """
    def __init__(self,
                 weight=1.0,
                 compute_type=ms.float32):
        super(LengthPenalty, self).__init__()
        self.weight = weight
        self.add = ops.Add()
        self.pow = ops.Pow()
        self.div = ops.RealDiv()
        self.cast = ops.Cast()
        self.five = Tensor(5.0, ms.float32)
        self.six = Tensor(6.0, ms.float32)

    def construct(self, length_tensor):
        length_tensor = self.cast(length_tensor, ms.float32)
        output = self.add(length_tensor, self.five)
        output = self.div(output, self.six)
        output = self.pow(output, self.weight)
        return output


class Mod(nn.Cell):
    """
    Mod function.

    Args:
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: ms.float32.
    """
    def __init__(self,
                 compute_type=ms.float32):
        super(Mod, self).__init__()
        self.compute_type = compute_type
        self.floor_div = ops.FloorDiv()
        self.sub = ops.Sub()
        self.multiply = ops.Mul()

    def construct(self, input_x, input_y):
        x = self.floor_div(input_x, input_y)
        x = self.multiply(x, input_y)
        x = self.sub(input_x, x)
        return x


class BeamSearchDecoder(nn.Cell):
    """
    Beam search decoder.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input sequence.
        vocab_size (int): Size of vocabulary.
        decoder (:class:`TransformerDecoderStep`): Decoder module.
        beam_width (int): beam width setting. Default: 4.
        length_penalty_weight (float): Weight of length penalty. Default: 1.0.
        max_decode_length (int): max decode length. Default: 128.
        sos_id (int): Id of sequence start token. Default: 1.
        eos_id (int): Id of sequence end token. Default: 2.
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: ms.float32.
    """
    def __init__(self,
                 batch_size,
                 seq_length,
                 vocab_size,
                 decoder,
                 beam_width=4,
                 length_penalty_weight=1.0,
                 max_decode_length=128,
                 sos_id=1,
                 eos_id=2,
                 compute_type=ms.float32):
        super(BeamSearchDecoder, self).__init__(auto_prefix=False)
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.max_decode_length = max_decode_length
        self.decoder = decoder

        self.add = ops.Add()
        self.expand = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.shape_flat = (-1,)
        self.shape = ops.TensorShape()

        self.zero_tensor = Tensor(np.zeros([batch_size, beam_width]), ms.float32)
        self.ninf_tensor = Tensor(np.full([batch_size, beam_width], -INF), ms.float32)

        self.select = ops.Select()
        self.flat_shape = (batch_size, beam_width * vocab_size)
        self.topk = ops.TopK(sorted=True)
        self.floor_div = ops.FloorDiv()
        self.vocab_size_tensor = Tensor(self.vocab_size, ms.int32)
        self.real_div = ops.RealDiv()
        self.mod = Mod()
        self.equal = ops.Equal()
        self.eos_ids = Tensor(np.full([batch_size, beam_width], eos_id), ms.int32)

        beam_ids = np.tile(np.arange(beam_width).reshape((1, beam_width)), [batch_size, 1])
        self.beam_ids = Tensor(beam_ids, ms.int32)
        batch_ids = np.arange(batch_size*beam_width).reshape((batch_size, beam_width)) // beam_width
        self.batch_ids = Tensor(batch_ids, ms.int32)
        self.concat = ops.Concat(axis=-1)
        self.gather_nd = ops.GatherNd()

        self.greater_equal = ops.GreaterEqual()
        self.sub = ops.Sub()
        self.cast = ops.Cast()
        self.zeroslike = ops.ZerosLike()

        # init inputs and states
        self.start_ids = Tensor(np.full([batch_size * beam_width, 1], sos_id), ms.int32)
        self.init_seq = Tensor(np.full([batch_size, beam_width, 1], sos_id), ms.int32)
        init_scores = np.tile(np.array([[0.] + [-INF]*(beam_width-1)]), [batch_size, 1])
        self.init_scores = Tensor(init_scores, ms.float32)
        self.init_finished = Tensor(np.zeros([batch_size, beam_width], dtype=np.bool))
        self.init_length = Tensor(np.zeros([batch_size, beam_width], dtype=np.int32))
        self.length_penalty = LengthPenalty(weight=length_penalty_weight)
        self.one = Tensor(1, ms.int32)

    def one_step(self, cur_input_ids, enc_states, enc_attention_mask, state_log_probs,
                 state_seq, state_finished, state_length):
        """
        One step for decode
        """
        log_probs = self.decoder(cur_input_ids, enc_states, enc_attention_mask, self.seq_length)
        log_probs = self.reshape(log_probs, (self.batch_size, self.beam_width, self.vocab_size))

        # select topk indices
        total_log_probs = self.add(log_probs, self.expand(state_log_probs, -1))

        # mask finished beams
        mask_tensor = self.select(state_finished, self.ninf_tensor, self.zero_tensor)
        total_log_probs = self.add(total_log_probs, self.expand(mask_tensor, -1))

        # reshape scores to [batch, beam*vocab]
        flat_scores = self.reshape(total_log_probs, self.flat_shape)
        # select topk
        topk_scores, topk_indices = self.topk(flat_scores, self.beam_width)

        temp = topk_indices
        beam_indices = self.zeroslike(topk_indices)
        for _ in range(self.beam_width - 1):
            temp = self.sub(temp, self.vocab_size_tensor)
            res = self.cast(self.greater_equal(temp, 0), ms.int32)
            beam_indices = beam_indices + res
        word_indices = topk_indices - beam_indices * self.vocab_size_tensor
        #======================================================================

        # mask finished indices
        beam_indices = self.select(state_finished, self.beam_ids, beam_indices)
        word_indices = self.select(state_finished, self.eos_ids, word_indices)
        topk_scores = self.select(state_finished, state_log_probs, topk_scores)

        ###### put finished sequences to the end
        # sort according to scores with -inf for finished beams
        tmp_log_probs = self.select(
            self.equal(word_indices, self.eos_ids),
            self.ninf_tensor,
            topk_scores)
        _, tmp_indices = self.topk(tmp_log_probs, self.beam_width)
        # update
        tmp_gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(tmp_indices, -1)))
        beam_indices = self.gather_nd(beam_indices, tmp_gather_indices)
        word_indices = self.gather_nd(word_indices, tmp_gather_indices)
        topk_scores = self.gather_nd(topk_scores, tmp_gather_indices)

        ###### generate new beam_search states
        # gather indices for selecting alive beams
        gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(beam_indices, -1)))

        # length add 1 if not finished in the previous step
        length_add = self.add(state_length, self.one)
        state_length = self.select(state_finished, state_length, length_add)
        state_length = self.gather_nd(state_length, gather_indices)

        # concat seq
        seq = self.gather_nd(state_seq, gather_indices)
        state_seq = self.concat((seq, self.expand(word_indices, -1)))

        # new finished flag and log_probs
        state_finished = self.equal(word_indices, self.eos_ids)
        state_log_probs = topk_scores

        ###### generate new inputs and decoder states
        cur_input_ids = self.reshape(state_seq, (self.batch_size*self.beam_width, -1))
        return cur_input_ids, state_log_probs, state_seq, state_finished, state_length

    def construct(self, enc_states, enc_attention_mask):
        """Get beam search result."""
        cur_input_ids = self.start_ids
        # beam search states
        state_log_probs = self.init_scores
        state_seq = self.init_seq
        state_finished = self.init_finished
        state_length = self.init_length

        for _ in range(self.max_decode_length):
            # run one step decoder to get outputs of the current step
            cur_input_ids, state_log_probs, state_seq, state_finished, state_length = self.one_step(
                cur_input_ids, enc_states, enc_attention_mask, state_log_probs, state_seq, state_finished, state_length)

        # add length penalty scores
        penalty_len = self.length_penalty(state_length)
        # get penalty length
        log_probs = self.real_div(state_log_probs, penalty_len)

        # sort according to scores
        _, top_beam_indices = self.topk(log_probs, self.beam_width)
        gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(top_beam_indices, -1)))
        # sort sequence
        predicted_ids = self.gather_nd(state_seq, gather_indices)
        # take the first one
        predicted_ids = predicted_ids[::, 0:1:1, ::]
        return predicted_ids


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


def normal_weight(shape, num_units):
    norm = np.random.normal(0.0, num_units**-0.5, shape).astype(np.float32)
    return Tensor(norm)


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
                 batch_size,
                 vocab_size,
                 embedding_size,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02):
        super(EmbeddingLookup, self).__init__()
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
        self.shape = ops.TensorShape()

    def construct(self, input_ids):
        """Get a embeddings lookup table with a fixed dictionary and size."""
        flat_ids = self.reshape(input_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        out_shape = (self.batch_size, -1, self.embedding_size)
        output = self.reshape(output_for_reshape, out_shape)
        return output, self.embedding_table


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
        self.shape = ops.TensorShape()

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
        from_seq_length = seq_length
        shape_from = (self.batch_size, -1, self.num_attention_heads, self.size_per_head)
        shape_to = (self.batch_size, -1, self.num_attention_heads, self.size_per_head)
        if self.do_return_2d_tensor:
            shape_return = (-1, self.num_attention_heads * self.size_per_head)
        else:
            shape_return = (self.batch_size, from_seq_length, self.num_attention_heads * self.size_per_head)

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
        self.hidden_size = hidden_size

        layers = []
        for _ in range(num_hidden_layers):
            layer = EncoderCell(batch_size=batch_size,
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
        out_shape = (self.batch_size, -1, self.hidden_size)
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
            layer = DecoderCell(batch_size=batch_size,
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

    def construct(self, input_tensor, attention_mask, enc_states, enc_attention_mask, seq_length, enc_seq_length):
        """Apply decoder."""
        out_shape = (self.batch_size, -1, self.hidden_size)
        prev_output = self.reshape(input_tensor, self.shape)

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, enc_states, enc_attention_mask,
                                        seq_length, enc_seq_length)
            prev_output = layer_output

        prev_output = self.layer_preprocess(prev_output)
        output = self.reshape(prev_output, out_shape)
        return output


class CreateAttentionMaskFromInputMask(nn.Cell):
    def __init__(self):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.shape = ops.TensorShape()
        self.batch_matmul = ops.BatchMatMul()
        self.expand_dims = ops.ExpandDims()

    def construct(self, input_mask):
        """Create attention mask according to input mask."""
        input_mask = self.cast(input_mask, ms.float32)
        mask_left = self.expand_dims(input_mask, 2)
        mask_right = self.expand_dims(input_mask, 1)
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
                 batch_size,
                 width,
                 compute_type=ms.float32,
                 dtype=ms.float32):
        super(PredLogProbs, self).__init__()
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
        shape_flat_sequence_tensor = (-1, self.width)

        input_tensor = self.reshape(input_tensor, shape_flat_sequence_tensor)
        input_tensor = self.cast(input_tensor, self.compute_type)
        output_weights = self.cast(output_weights, self.compute_type)

        logits = self.matmul(input_tensor, output_weights)
        logits = self.cast(logits, self.dtype)

        log_probs = self.log_softmax(logits)
        return log_probs


class TransformerDecoderStep(nn.Cell):
    """
    Multi-layer transformer decoder step.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers.
        max_decode_length (int): Max decode length.
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
        embedding_lookup (:class:`EmbeddingLookup`): Embedding lookup module.
        embedding_processor (:class:`EmbeddingPostprocessor`) Embedding postprocessor module.
        projection (:class:`PredLogProbs`): PredLogProbs module
    """
    def __init__(self,
                 batch_size,
                 hidden_size,
                 max_decode_length,
                 num_hidden_layers,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.3,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.3,
                 hidden_act="relu",
                 compute_type=ms.float32,
                 embedding_lookup=None,
                 embedding_processor=None,
                 projection=None):
        super(TransformerDecoderStep, self).__init__(auto_prefix=False)
        self.num_hidden_layers = num_hidden_layers

        self.tfm_embedding_lookup = embedding_lookup
        self.tfm_embedding_processor = embedding_processor
        self.projection = projection

        self.tfm_decoder = TransformerDecoder(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            compute_type=compute_type)

        self.ones_like = ops.OnesLike()
        self.shape = ops.TensorShape()

        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask()
        self.expand = ops.ExpandDims()
        self.multiply = ops.Mul()

        ones = np.ones(shape=(max_decode_length, max_decode_length))
        self.future_mask = Tensor(np.tril(ones), dtype=ms.float32)

        self.cast_compute_type = CastWrapper(dst_type=compute_type)

    def construct(self, input_ids, enc_states, enc_attention_mask, seq_length):
        """
        Multi-layer transformer decoder step.
        input_ids: [batch_size * beam_width]
        """
        # process embedding
        input_embedding, embedding_tables = self.tfm_embedding_lookup(input_ids)
        input_embedding = self.tfm_embedding_processor(input_embedding)
        input_embedding = self.cast_compute_type(input_embedding)

        input_shape = self.shape(input_ids)
        input_len = input_shape[1]
        future_mask = self.future_mask[0:input_len:1, 0:input_len:1]

        input_mask = self.ones_like(input_ids)
        input_mask = self._create_attention_mask_from_input_mask(input_mask)
        input_mask = self.multiply(input_mask, self.expand(future_mask, 0))
        input_mask = self.cast_compute_type(input_mask)

        enc_attention_mask = enc_attention_mask[::, 0:input_len:1, ::]

        # call TransformerDecoder
        decoder_output = self.tfm_decoder(input_embedding, input_mask, enc_states, enc_attention_mask, -1, seq_length)

        # take the last step
        decoder_output = decoder_output[::, input_len-1:input_len:1, ::]

        # projection and log_prob
        log_probs = self.projection(decoder_output, embedding_tables, 1)

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
    """
    def __init__(self,
                 is_training,
                 use_one_hot_embeddings=False):
        super(TransformerModel, self).__init__()
        self.is_training = is_training

        self.batch_size = BATCH_SIZE_VALUE
        self.hidden_size = 1024
        self.num_hidden_layers = 6
        self.embedding_size = 1024

        self.last_idx = self.num_hidden_layers - 1
        self.beam_width = 4
        self.max_decode_length = 80

        self.tfm_embedding_lookup = EmbeddingLookup(
            batch_size=self.batch_size,
            vocab_size=36560,
            embedding_size=self.embedding_size,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=0.02)
        self.tfm_embedding_postprocessor_for_encoder = EmbeddingPostprocessor(
            embedding_size=self.embedding_size,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=0.02,
            max_position_embeddings=128,
            dropout_prob=0.2)
        self.tfm_embedding_postprocessor_for_decoder = EmbeddingPostprocessor(
            embedding_size=self.embedding_size,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=0.02,
            max_position_embeddings=128,
            dropout_prob=0.2)
        self.tfm_encoder = TransformerEncoder(
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            num_attention_heads=16,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=4096,
            attention_probs_dropout_prob=0.2,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=0.02,
            hidden_dropout_prob=0.2,
            hidden_act="relu",
            compute_type=ms.float16)

        if is_training:
            self.projection = PredLogProbs(
                batch_size=self.batch_size,
                width=self.hidden_size,
                compute_type=ms.float16,
                dtype=ms.float32)
            self.tfm_decoder = TransformerDecoder(
                batch_size=self.batch_size,
                hidden_size=self.hidden_size,
                num_attention_heads=16,
                num_hidden_layers=self.num_hidden_layers,
                intermediate_size=4096,
                attention_probs_dropout_prob=0.2,
                use_one_hot_embeddings=use_one_hot_embeddings,
                initializer_range=0.02,
                hidden_dropout_prob=0.2,
                hidden_act="relu",
                compute_type=ms.float16)
        else:
            self.projection = PredLogProbs(
                batch_size=self.batch_size * 4,
                width=self.hidden_size,
                compute_type=ms.float16,
                dtype=ms.float32)
            self.tfm_decoder = TransformerDecoderStep(
                batch_size=self.batch_size * 4,
                hidden_size=self.hidden_size,
                max_decode_length=80,
                num_hidden_layers=6,
                num_attention_heads=16,
                intermediate_size=4096,
                attention_probs_dropout_prob=0.2,
                use_one_hot_embeddings=False,
                initializer_range=0.02,
                hidden_dropout_prob=0.2,
                hidden_act="relu",
                compute_type=ms.float16,
                embedding_lookup=self.tfm_embedding_lookup,
                embedding_processor=self.tfm_embedding_postprocessor_for_decoder,
                projection=self.projection)
            self.tfm_decoder = BeamSearchDecoder(
                batch_size=BATCH_SIZE_VALUE,
                seq_length=128,
                vocab_size=36560,
                decoder=self.tfm_decoder,
                beam_width=4,
                length_penalty_weight=1.0,
                max_decode_length=80)

            self.tfm_decoder.add_flags(loop_can_unroll=True)
            ones = np.ones(shape=(self.batch_size, self.max_decode_length))
            self.encdec_mask = Tensor(ones, ms.float32)

        self.cast = ops.Cast()
        self.dtype = ms.float32
        self.cast_compute_type = CastWrapper(dst_type=ms.float16)
        self.expand = ops.ExpandDims()
        self.multiply = ops.Mul()
        self.shape = ops.TensorShape()
        self.tril = array_op.Tril()
        self.ones_like = array_op.OnesLike()
        self.stack = array_op.Stack(0)
        self.concatenate = array_op.Concat()

        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask()

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

        ones_inner = self.ones_like(source_ids[0, :])
        seq_length = self.shape(ones_inner)
        broadcast_shape = self.concatenate([seq_length, seq_length])
        ones = func.broadcast_to(ones_inner, broadcast_shape)
        future_mask = self.tril(ones)

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
