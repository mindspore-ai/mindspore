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
"""Beam search decoder."""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.ops import composite as C

INF = 65536.0


class LengthPenalty(nn.Cell):
    """
    Length penalty.

    Args:
        weight (float): The length penalty weight.
        compute_type (mstype): Mindspore data type. Default: mstype.float32.
    """

    def __init__(self, weight=1.0, compute_type=mstype.float32):
        super(LengthPenalty, self).__init__()
        self.weight = weight
        self.add = P.Add()
        self.pow = P.Pow()
        self.div = P.RealDiv()
        self.five = Tensor(5.0, mstype.float32)
        self.six = Tensor(6.0, mstype.float32)
        self.cast = P.Cast()

    def construct(self, length_tensor):
        """
        Process source sentence

        Inputs:
            length_tensor (Tensor):  the input tensor.

        Returns:
            Tensor, after punishment of length.
        """
        length_tensor = self.cast(length_tensor, mstype.float32)
        output = self.add(length_tensor, self.five)
        output = self.div(output, self.six)
        output = self.pow(output, self.weight)
        return output


class TileBeam(nn.Cell):
    """
    Beam Tile operation.

    Args:
        beam_width (int): The Number of beam.
        compute_type (mstype): Mindspore data type. Default: mstype.float32.
    """

    def __init__(self, beam_width, compute_type=mstype.float32):
        super(TileBeam, self).__init__()
        self.beam_width = beam_width

        self.expand = P.ExpandDims()
        self.tile = P.Tile()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, input_tensor):
        """
        Process source sentence

        Inputs:
            input_tensor (Tensor):  with shape (N, T, D).

        Returns:
            Tensor, tiled tensor.
        """
        shape = self.shape(input_tensor)
        # add an dim
        input_tensor = self.expand(input_tensor, 1)
        # get tile shape: [1, beam, ...]
        tile_shape = (1,) + (self.beam_width,)
        for _ in range(len(shape) - 1):
            tile_shape = tile_shape + (1,)
        # tile
        output = self.tile(input_tensor, tile_shape)
        # reshape to [batch*beam, ...]
        out_shape = (shape[0] * self.beam_width,) + shape[1:]
        output = self.reshape(output, out_shape)

        return output


class Mod(nn.Cell):
    """
    Mod operation.

    Args:
        compute_type (mstype): Mindspore data type. Default: mstype.float32.
    """

    def __init__(self,
                 compute_type=mstype.float32):
        super(Mod, self).__init__()
        self.compute_type = compute_type

        self.floor_div = P.FloorDiv()
        self.sub = P.Sub()
        self.multiply = P.Mul()

    def construct(self, input_x, input_y):
        """
        Get the remainder of input_x and input_y.

        Inputs:
            input_x (Tensor): Divisor.
            input_y (Tensor): Dividend.

        Returns:
            Tensor, remainder.
        """
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
        vocab_size (int): The shape of each embedding vector.
        decoder (Cell): The GNMT decoder.
        beam_width (int): Beam width for beam search in inferring. Default: 4.
        decoder_layers_nums (int): The nums of decoder layers.
        length_penalty_weight (float): Penalty for sentence length. Default: 0.6.
        max_decode_length (int): Max decode length for inferring. Default: 64.
        sos_id (int): The index of start label <SOS>. Default: 1.
        eos_id (int): The index of end label <EOS>. Default: 2.
        compute_type (:class:`mindspore.dtype`): Compute type. Default: mstype.float32.

    Returns:
        Tensor, predictions output.
    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 vocab_size,
                 decoder,
                 beam_width=4,
                 decoder_layers_nums=4,
                 length_penalty_weight=0.6,
                 cov_penalty_factor=0.1,
                 hidden_size=1024,
                 max_decode_length=64,
                 sos_id=2,
                 eos_id=3,
                 is_using_while=True,
                 compute_type=mstype.float32):
        super(BeamSearchDecoder, self).__init__()

        self.encoder_length = seq_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.beam_width = beam_width
        self.decoder_layers_nums = decoder_layers_nums
        self.length_penalty_weight = length_penalty_weight
        self.cov_penalty_factor = cov_penalty_factor
        self.max_decode_length = max_decode_length
        self.decoder = decoder
        self.is_using_while = is_using_while

        self.add = P.Add()
        self.expand = P.ExpandDims()
        self.reshape = P.Reshape()
        self.shape_flat = (-1,)
        self.shape = P.Shape()

        self.zero_tensor = Tensor(np.zeros([batch_size, beam_width]), mstype.float32)
        self.ninf_tensor = Tensor(np.full([batch_size, beam_width], -INF), mstype.float32)

        self.select = P.Select()
        self.flat_shape = (batch_size, beam_width * vocab_size)
        self.topk = P.TopK(sorted=True)
        self.floor_div = P.FloorDiv()
        self.vocab_size_tensor = Tensor(self.vocab_size, mstype.int32)
        self.real_div = P.RealDiv()
        self.mod = Mod()
        self.equal = P.Equal()
        self.eos_ids = Tensor(np.full([batch_size, beam_width], eos_id), mstype.int32)

        beam_ids = np.tile(np.arange(beam_width).reshape((1, beam_width)), [batch_size, 1])
        self.beam_ids = Tensor(beam_ids, mstype.int32)

        batch_ids = np.arange(batch_size * beam_width).reshape((batch_size, beam_width)) // beam_width
        self.batch_ids = Tensor(batch_ids, mstype.int32)

        self.concat = P.Concat(axis=-1)
        self.gather_nd = P.GatherNd()

        self.start_ids = Tensor(np.full([batch_size * beam_width, 1], sos_id), mstype.int32)
        if self.is_using_while:
            self.start = Tensor(0, dtype=mstype.int32)
            self.init_seq = Tensor(np.full([batch_size, beam_width, self.max_decode_length + 1], sos_id),
                                   mstype.int32)
        else:
            self.init_seq = Tensor(np.full([batch_size, beam_width, 1], sos_id), mstype.int32)

        init_scores = np.tile(np.array([[0.] + [-INF] * (beam_width - 1)]), [batch_size, 1])
        self.init_scores = Tensor(init_scores, mstype.float32)
        self.init_finished = Tensor(np.zeros([batch_size, beam_width], dtype=np.bool))
        self.init_length = Tensor(np.zeros([batch_size, beam_width], dtype=np.int32))

        self.length_penalty = LengthPenalty(weight=length_penalty_weight)

        self.one = Tensor(1, mstype.int32)
        self.prob_concat = P.Concat(axis=1)
        self.cast = P.Cast()
        self.decoder_hidden_state = Tensor(np.zeros([self.decoder_layers_nums, 2,
                                                     self.batch_size * self.beam_width,
                                                     hidden_size]), mstype.float32)

        self.zeros_scores = Tensor(np.zeros([batch_size, beam_width], dtype=np.float))
        self.active_index = Tensor(np.ones([batch_size, beam_width], dtype=np.int32))
        self.init_zeros = Tensor(np.zeros([batch_size, beam_width], dtype=np.int32))
        self.init_ones = Tensor(np.ones([batch_size, beam_width], dtype=np.float32))

        self.accu_attn_scores = Tensor(np.zeros([batch_size, beam_width, self.encoder_length], dtype=np.float32))

        self.zeros = Tensor([0], mstype.int32)
        self.eos_tensor = Tensor(np.full([batch_size, beam_width, beam_width], eos_id), mstype.int32)

        self.ones_3d = Tensor(np.full([batch_size, beam_width, self.encoder_length], 1), mstype.float32)
        self.neg_inf_3d = Tensor(np.full([batch_size, beam_width, self.encoder_length], -INF), mstype.float32)
        self.zeros_3d = Tensor(np.full([batch_size, beam_width, self.encoder_length], 0), mstype.float32)
        self.zeros_2d = Tensor(np.full([batch_size * beam_width, self.encoder_length], 0), mstype.int32)
        self.argmin = P.ArgMinWithValue(axis=1)
        self.reducesum = P.ReduceSum()
        self.div = P.Div()
        self.shape_op = P.Shape()
        self.mul = P.Mul()
        self.log = P.Log()
        self.less = P.Less()
        self.tile = P.Tile()
        self.noteq = P.Neg()
        self.zeroslike = P.ZerosLike()
        self.greater_equal = P.GreaterEqual()
        self.sub = P.Sub()

    def one_step(self, cur_input_ids, enc_states, enc_attention_mask, state_log_probs,
                 state_seq, state_length, idx=None, decoder_hidden_state=None, accu_attn_scores=None,
                 state_finished=None):
        """
        Beam search one_step output.

        Inputs:
            cur_input_ids (Tensor):  with shape (batch_size * beam_width, 1).
            enc_states (Tensor):  with shape (batch_size * beam_width, T, D).
            enc_attention_mask (Tensor):  with shape (batch_size * beam_width, T).
            state_log_probs (Tensor):  with shape (batch_size, beam_width).
            state_seq (Tensor):  with shape (batch_size, beam_width, m).
            state_length (Tensor):  with shape (batch_size, beam_width).
            idx (Tensor):  with shape ().
            decoder_hidden_state (Tensor): with shape (decoder_layer_num, 2, batch_size * beam_width, D).
            accu_attn_scores (Tensor): with shape (batchsize, beam_width, seq_length).
            state_finished (Tensor):  with shape (batch_size, beam_width).
        """

        # log_probs, [batch_size * beam_width, 1, V]
        log_probs, all_decoder_state, attn = self.decoder(cur_input_ids, enc_states, enc_attention_mask,
                                                          decoder_hidden_state)
        # consider attention_scores
        attn = self.reshape(attn, (-1, self.beam_width, self.encoder_length))
        state_finished_attn = self.cast(state_finished, mstype.int32)
        attn_mask_0 = self.tile(self.expand(state_finished_attn, 2), (1, 1, self.encoder_length))
        attn_mask_0 = self.cast(attn_mask_0, mstype.bool_)
        attn_new = self.select(attn_mask_0, self.zeros_3d, attn)
        accu_attn_scores = self.add(accu_attn_scores, attn_new)

        # log_probs: [batch_size, beam_width, V]
        log_probs = self.reshape(log_probs, (-1, self.beam_width, self.vocab_size))
        # select topk indices, [batch_size, beam_width, V]
        total_log_probs = self.add(log_probs, self.expand(state_log_probs, -1))
        # mask finished beams, [batch_size, beam_width]
        # t-1 has finished
        mask_tensor = self.select(state_finished, self.ninf_tensor, self.zero_tensor)
        # save the t-1 probability
        total_log_probs = self.add(total_log_probs, self.expand(mask_tensor, -1))
        # [batch, beam*vocab]
        flat_scores = self.reshape(total_log_probs, (-1, self.beam_width * self.vocab_size))
        # select topk, [batch, beam]
        topk_scores, topk_indices = self.topk(flat_scores, self.beam_width)

        # convert to beam and word indices, [batch, beam]
        # beam_indices = self.floor_div(topk_indices, self.vocab_size_tensor)
        # word_indices = self.mod(topk_indices, self.vocab_size_tensor)
        # ======================================================================
        # replace floor_div and mod op, since these two ops only support fp16 on
        # Ascend310, which will cause overflow.
        temp = topk_indices
        beam_indices = self.zeroslike(topk_indices)
        for _ in range(self.beam_width - 1):
            temp = self.sub(temp, self.vocab_size_tensor)
            res = self.cast(self.greater_equal(temp, 0), mstype.int32)
            beam_indices = beam_indices + res
        word_indices = topk_indices - beam_indices * self.vocab_size_tensor
        # ======================================================================

        # mask finished indices, [batch, beam]
        beam_indices = self.select(state_finished, self.beam_ids, beam_indices)
        word_indices = self.select(state_finished, self.eos_ids, word_indices)
        topk_scores = self.select(state_finished, state_log_probs, topk_scores)

        # sort according to scores with -inf for finished beams, [batch, beam]
        # t ends
        tmp_log_probs = self.select(
            self.equal(word_indices, self.eos_ids),
            self.ninf_tensor,
            topk_scores)

        _, tmp_indices = self.topk(tmp_log_probs, self.beam_width)
        # update, [batch_size, beam_width, 2]
        tmp_gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(tmp_indices, -1)))
        # [batch_size, beam_width]
        beam_indices = self.gather_nd(beam_indices, tmp_gather_indices)
        word_indices = self.gather_nd(word_indices, tmp_gather_indices)
        topk_scores = self.gather_nd(topk_scores, tmp_gather_indices)

        # gather indices for selecting alive beams
        gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(beam_indices, -1)))

        # length add 1 if not finished in the previous step, [batch_size, beam_width]
        length_add = self.add(state_length, self.one)
        state_length = self.select(state_finished, state_length, length_add)
        state_length = self.gather_nd(state_length, gather_indices)
        # concat seq
        seq = self.gather_nd(state_seq, gather_indices)
        # update accu_attn_scores
        accu_attn_scores = self.gather_nd(accu_attn_scores, gather_indices)
        # update all_decoder_state
        all_decoder_state = self.reshape(all_decoder_state,
                                         (self.decoder_layers_nums * 2, self.batch_size, self.beam_width,
                                          self.hidden_size))
        for i in range(self.decoder_layers_nums * 2):
            all_decoder_state[i, :, :, :] = self.gather_nd(all_decoder_state[i, :, :, :], gather_indices)
        all_decoder_state = self.reshape(all_decoder_state,
                                         (self.decoder_layers_nums, 2, self.batch_size * self.beam_width,
                                          self.hidden_size))

        # update state_seq
        if self.is_using_while:
            state_seq_new = self.cast(seq, mstype.float32)
            word_indices_fp32 = self.cast(word_indices, mstype.float32)
            state_seq_new[:, :, idx] = word_indices_fp32
            state_seq = self.cast(state_seq_new, mstype.int32)
        else:
            state_seq = self.concat((seq, self.expand(word_indices, -1)))

        cur_input_ids = self.reshape(word_indices, (-1, 1))
        state_log_probs = topk_scores
        state_finished = self.equal(word_indices, self.eos_ids)

        return cur_input_ids, state_log_probs, state_seq, state_length, \
               all_decoder_state, accu_attn_scores, state_finished

    def construct(self, enc_states, enc_attention_mask):
        """
        Process source sentence

        Inputs:
            enc_states (Tensor): Output of transformer encoder with shape (batch_size * beam_width, T, D).
            enc_attention_mask (Tensor): encoder attention mask with shape (batch_size * beam_width, T).

        Returns:
            Tensor, predictions output.
        """
        # beam search start
        cur_input_ids = self.start_ids
        state_log_probs = self.init_scores
        state_seq = self.init_seq
        state_finished = self.init_finished
        state_length = self.init_length
        decoder_hidden_state = self.decoder_hidden_state
        accu_attn_scores = self.accu_attn_scores

        if not self.is_using_while:
            for _ in range(self.max_decode_length):
                cur_input_ids, state_log_probs, state_seq, state_length, decoder_hidden_state, accu_attn_scores, \
                state_finished = self.one_step(cur_input_ids, enc_states, enc_attention_mask, state_log_probs,
                                               state_seq, state_length, None, decoder_hidden_state, accu_attn_scores,
                                               state_finished)
        else:
            # At present, only ascend910 supports while operation.
            idx = self.start + 1
            ends = self.start + self.max_decode_length + 1
            while idx < ends:
                cur_input_ids, state_log_probs, state_seq, state_length, decoder_hidden_state, accu_attn_scores, \
                state_finished = self.one_step(cur_input_ids, enc_states, enc_attention_mask, state_log_probs,
                                               state_seq, state_length, idx, decoder_hidden_state, accu_attn_scores,
                                               state_finished)
                idx = idx + 1

        # add length penalty scores
        penalty_len = self.length_penalty(state_length)
        log_probs = self.real_div(state_log_probs, penalty_len)
        penalty_cov = C.clip_by_value(accu_attn_scores, 0.0, 1.0)
        penalty_cov = self.log(penalty_cov)
        penalty_less = self.less(penalty_cov, self.neg_inf_3d)
        penalty = self.select(penalty_less, self.zeros_3d, penalty_cov)
        penalty = self.reducesum(penalty, 2)
        log_probs = log_probs + penalty * self.cov_penalty_factor
        # sort according to scores
        _, top_beam_indices = self.topk(log_probs, self.beam_width)
        gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(top_beam_indices, -1)))
        # sort sequence and attention scores
        predicted_ids = self.gather_nd(state_seq, gather_indices)
        if not self.is_using_while:
            predicted_ids = predicted_ids[:, 0:1, 1:(self.max_decode_length + 1)]
        else:
            predicted_ids = predicted_ids[:, 0:1, :self.max_decode_length]

        return predicted_ids
