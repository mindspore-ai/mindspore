# Copyright 2022 Huawei Technologies Co., Ltd
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
import argparse
import copy
import logging
import math
import time
import pytest
import numpy as np

import mindspore
import mindspore.dataset.engine as de
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.initializer import One
from mindspore.context import ParallelMode
from mindspore.nn import Adam
from mindspore.nn.loss.loss import _Loss
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops import composite as C, functional as F
from mindspore.ops.functional import stop_gradient
from mindspore.parallel._utils import _get_parallel_mode, _get_device_num, _get_gradients_mean
from mindspore.train import Model, Callback

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


IGNORE_ID = -1


class MultiheadAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".
    """

    def __init__(self,
                 batch_size,
                 from_tensor_width,
                 to_tensor_width,
                 num_attention_heads=1,
                 hidden_size=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 has_attention_mask=False,
                 attention_probs_dropout_prob=0.0,
                 do_return_2d_tensor=False,
                 compute_type=mstype.float32):

        super(MultiheadAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))

        self.has_attention_mask = has_attention_mask
        self.size_per_head = int(hidden_size / num_attention_heads)
        self.scores_mul = Tensor(
            [1.0 / math.sqrt(float(self.size_per_head))], dtype=compute_type)
        self.reshape = P.Reshape()
        self.shape_from_2d = (-1, from_tensor_width)
        self.shape_to_2d = (-1, to_tensor_width)

        weight = "zeros"
        self.query_layer = CustomDense(from_tensor_width, hidden_size, activation=query_act, weight_init=weight)
        self.key_layer = CustomDense(to_tensor_width, hidden_size, activation=key_act, weight_init=weight)
        self.value_layer = CustomDense(to_tensor_width, hidden_size, activation=value_act, weight_init=weight)

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.shape_from = (batch_size, -1, num_attention_heads, self.size_per_head)
        self.shape_to = (batch_size, -1, num_attention_heads, self.size_per_head)

        self.multiply_data = Tensor([-10000.0], dtype=compute_type)
        self.matmul = P.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=attention_probs_dropout_prob)
        self.sub = P.Sub()
        self.add = P.TensorAdd()
        self.cast = P.Cast()
        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.get_dtype = P.DType()
        if do_return_2d_tensor:
            self.shape_return = (-1, hidden_size)
        else:
            self.shape_return = (batch_size, -1, hidden_size)
        self.shape = P.Shape()
        self.print = P.Print()

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
        attention_scores = self.multiply(attention_scores, self.scores_mul)

        # calculate mask
        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(P.TupleToArray()((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))
            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.reshape(value_out, self.shape_to)
        value_layer = self.transpose(value_layer, self.trans_shape)

        context_layer = self.matmul(attention_probs, value_layer)

        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, self.shape_return)

        return context_layer


class SelfAttention(nn.Cell):
    """
    Apply self-attention.
    including self attention and residual connections
    """

    def __init__(self,
                 batch_size,
                 hidden_size,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 has_attention_mask=True,
                 compute_type=mstype.float32):
        super(SelfAttention, self).__init__()

        self.attention = MultiheadAttention(
            batch_size=batch_size,
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            has_attention_mask=has_attention_mask,
            do_return_2d_tensor=True,
            compute_type=compute_type)
        self.output = ResidualNorm(hidden_size, dropout_prob=hidden_dropout_prob)
        self.reshape = P.Reshape()
        self.shape_2d = (-1, hidden_size)
        self.shape_to = (batch_size, -1, hidden_size)

    def construct(self, input_tensor, attention_mask):
        attention_output = self.attention(
            input_tensor, input_tensor, attention_mask)
        attention_output = self.reshape(attention_output, self.shape_2d)
        input_tensor = self.reshape(input_tensor, self.shape_2d)
        output = self.output(attention_output, input_tensor)
        output = self.reshape(output, self.shape_to)

        return output


class ResidualNorm(nn.Cell):
    """
    Apply a linear computation to hidden status and a residual computation to input.
    """

    def __init__(self, size, dropout_prob=0.1):
        super(ResidualNorm, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.add = P.TensorAdd()
        self.layernorm = nn.LayerNorm([size])
        self.out_shape = (-1, size)
        self.cast = P.Cast()

    def construct(self, hidden_status, input_tensor):
        output = self.dropout(hidden_status)
        output = self.add(output, input_tensor)
        # TODO Temp change for dynamic length sequence.
        if F.is_sequence_value_unknown(P.Shape()(output)):
            output = P.ExpandDims()(output, 1)
        output = self.layernorm(output)
        output = P.Reshape()(output, self.out_shape)
        return output


class FeedForward(nn.Cell):
    def __init__(self, attention_size, intermediate_size,
                 hidden_act, hidden_dropout_prob):
        super(FeedForward, self).__init__()
        self.dropout = nn.Dropout(p=hidden_dropout_prob)
        self.linear1 = CustomDense(in_channels=attention_size,
                                   out_channels=intermediate_size,
                                   activation=hidden_act,
                                   weight_init="zeros")
        self.linear2 = CustomDense(in_channels=intermediate_size,
                                   out_channels=attention_size,
                                   weight_init="zeros")

    def construct(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Conv2dSubsampling(nn.Cell):
    """
    Convolutional 2D subsampling (to 1/4 length).

    """

    def __init__(self, idim, odim, pad=2):
        """
        Construct an Conv2dSubsampling object.
        :param int idim: input dim
        :param int odim: output dim
        """
        super(Conv2dSubsampling, self).__init__()
        self.conv1 = nn.Conv2d(1, odim, 3, 2, pad_mode="pad", padding=pad)
        self.conv2 = nn.Conv2d(odim, odim, 3, 2, pad_mode="pad", padding=pad)
        shape1 = 1 + (idim + 2 * pad - 3) // 2
        shape2 = 1 + (shape1 + 2 * pad - 3) // 2
        self.linear = CustomDense(odim * shape2, odim)

        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.relu = nn.ReLU()
        self.transpose = P.Transpose()
        self.expanddim = P.ExpandDims()
        self.odim = odim

    def construct(self, x):
        """
        :param mindspore.Tensor x: input audio feats (B, time, idim)
        :return: subsampled x (B, new_time, odim)
        :rtype: mindspore.Tensor
        """
        x = self.expanddim(x, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        # TODO: Temp change for dynamic length sequence.
        (b, c, _, f) = self.shape(x)
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (-1, c*f))
        x = self.linear(x)
        # TODO: Temp change for dynamic length sequence.
        x = self.reshape(x, (b, -1, self.odim))
        return x


class EncoderCell(nn.Cell):
    """
    Transformer encoder cell.
    """

    def __init__(self, batch_size, size,
                 num_attention_heads=4,
                 intermediate_size=2048,
                 attention_probs_dropout_prob=0.0,
                 hidden_dropout_prob=0.1,
                 has_attention_mask=False,
                 hidden_act="relu",
                 compute_type=mstype.float32):
        """
        Construct an EncoderCell object.
        """
        super(EncoderCell, self).__init__()
        self.attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            has_attention_mask=has_attention_mask,
            compute_type=compute_type)

        self.dropout = nn.Dropout(p=hidden_dropout_prob)
        self.intermediate = CustomDense(in_channels=size, out_channels=intermediate_size,
                                        activation=hidden_act, weight_init="zeros")
        self.res_norm = ResidualNorm(size, dropout_prob=hidden_dropout_prob)
        self.feedforward = FeedForward(size, intermediate_size, hidden_act, hidden_dropout_prob)
        self.shape_2d = (-1, size)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.print = P.Print()

    def construct(self, x, attention_mask):
        """
        :param mindspore.Tensor x: embedeed inputs
        :param mindspore.Tensor attention_mask: mask of x. required input of BertAttention, but not used here.
        could be any Tensor with shape (batch_size, seq_length, seq_length)
        :return: encoder cell output (-1, size)
        """
        shape_out = self.shape(x)
        attention_output = self.attention(x, attention_mask)
        attention_output = self.reshape(attention_output, self.shape_2d)
        fc_output = self.feedforward(attention_output)
        output = self.res_norm(fc_output, attention_output)
        # TODO Temp change for dynamic length sequence.
        if F.is_sequence_value_unknown(shape_out):
            shape_out = P.TensorShape()(x)
        return self.reshape(output, shape_out)


class PositionalEncoding(nn.Cell):
    """Positional encoding.

    :param int dim: embedding dim
    :param int time: input sequence length
    :param float dropout_rate: dropout rate

    """

    def __init__(self, dim, maxlen=10000, dropout_rate=0.1):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()

        xscale = math.sqrt(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.mul = P.Mul()
        self.add = P.TensorAdd()
        self.shape = P.Shape()

        self.pe = self.postion_encoding_table(maxlen, dim)
        self.te = Tensor([xscale], mstype.float32)
        self.print = P.Print()

    def construct(self, x):
        """
        Add positional encoding
        :param mindspore.Tensor x: batches of embedded inputs (B, time, dim)
        :return: Encoded x (B, time, dim)
        """
        _, l, _ = self.shape(x)
        if not F.isconstant(l):
            l = P.DynamicShape()(x)[1]
        pos = self.pe[:, :l, :]
        x = self.mul(x, self.te)
        x = self.add(x, pos)
        x = self.dropout(x)
        return x

    def postion_encoding_table(self, max_length, dims):
        pe = np.zeros((max_length, dims))
        position = np.arange(0, max_length).reshape((max_length, 1))
        div_term = np.exp(np.arange(0, dims, 2) * (-(math.log(10000.0) / dims)))
        div_term = div_term.reshape((1, div_term.shape[0]))
        pe[:, 0::2] = np.sin(np.matmul(position, div_term))
        pe[:, 1::2] = np.cos(np.matmul(position, div_term))
        pe = pe.reshape((1, max_length, dims))
        pe = Tensor(pe, mstype.float32)
        return pe


class DecoderCell(nn.Cell):
    """
    Single decoder layer module.
    """

    def __init__(self, batch_size, size,
                 num_attention_heads=4,
                 attention_drop_out_prob=0.0,
                 hidden_dropout_prob=0.1,
                 intermediate_size=2048,
                 hidden_act="relu",
                 compute_type=mstype.float32):
        """Construct an DecoderLayer object."""
        super(DecoderCell, self).__init__()

        self.size = size
        self.batch_size = batch_size
        self.self_attn = SelfAttention(batch_size=batch_size,
                                       hidden_size=size,
                                       num_attention_heads=num_attention_heads,
                                       attention_probs_dropout_prob=attention_drop_out_prob,
                                       hidden_dropout_prob=hidden_dropout_prob,
                                       has_attention_mask=True,
                                       compute_type=compute_type)
        self.src_attn2 = MultiheadAttention(batch_size=batch_size,
                                            from_tensor_width=size,
                                            to_tensor_width=size,
                                            attention_probs_dropout_prob=attention_drop_out_prob,
                                            num_attention_heads=num_attention_heads,
                                            hidden_size=size,
                                            has_attention_mask=False)

        self.output1 = ResidualNorm(size, dropout_prob=hidden_dropout_prob)
        self.output2 = ResidualNorm(size, dropout_prob=hidden_dropout_prob)
        self.feedforward = FeedForward(size, intermediate_size, hidden_act, hidden_dropout_prob)
        self.cat = P.Concat(axis=1)
        self.shape = P.Shape()
        self.gather = P.GatherV2()
        self.any_tensor = Tensor([1], mstype.float32)
        self.reshape = P.Reshape()
        self.shape_to = (batch_size, -1, size)

    def construct(self, tgt, tgt_mask, memory, memory_mask):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out,max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, max_time_in, max_time_in)
        """
        x = self.self_attn(tgt, tgt_mask)

        residual = self.reshape(x, (-1, self.size))
        x = self.src_attn2(x, memory, memory_mask)
        x = self.reshape(x, (-1, self.size))
        x = self.output1(x, residual)

        residual = x
        x = self.feedforward(x)
        x = self.output2(x, residual)
        x = self.reshape(x, self.shape_to)

        return x


class CustomDense(nn.Dense):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='zeros',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None):
        """Initialize Dense."""
        super(CustomDense, self).__init__(in_channels,
                                          out_channels,
                                          weight_init,
                                          bias_init,
                                          has_bias,
                                          activation)
        self.cast = P.Cast()

    def construct(self, x):
        x_shape = self.shape_op(x)
        weight = self.weight
        if len(x_shape) != 2:
            x = self.reshape(x, (-1, x_shape[-1]))
        x = self.cast(x, mstype.float16)
        weight = self.cast(weight, mstype.float16)
        x = self.matmul(x, weight)
        x = self.cast(x, mstype.float32)
        if self.has_bias:
            x = self.bias_add(x, self.bias)
        if self.activation_flag:
            x = self.activation(x)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (-1,)
            x = self.reshape(x, out_shape)

        return x


class Encoder(nn.Cell):
    def __init__(self,
                 batch_size,
                 attention_dim,
                 feat_dim,
                 num_hidden_layers=2,
                 num_attention_heads=4,
                 intermediate_size=2048,
                 attention_probs_dropout_prob=0.0,
                 positional_dropout_rate=0.1,
                 hidden_dropout_prob=0.1,
                 pad=2):
        super(Encoder, self).__init__()
        self.reshape = P.Reshape()
        self.batch_size = batch_size
        self.subsampling = Conv2dSubsampling(feat_dim, attention_dim, pad=pad)
        self.pos_enc = PositionalEncoding(attention_dim, maxlen=10000,
                                          dropout_rate=positional_dropout_rate)
        layers = []
        for _ in range(num_hidden_layers):
            layer = EncoderCell(batch_size, attention_dim,
                                num_attention_heads=num_attention_heads,
                                intermediate_size=intermediate_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                hidden_dropout_prob=hidden_dropout_prob)
            layers.append(layer)
        self.layers_e = nn.CellList(layers)

    def construct(self, audio, mask):
        # audio embedding
        conv_emb = self.subsampling(audio)
        position_enc = self.pos_enc(conv_emb)

        prev_output = position_enc
        for layer_module in self.layers_e:
            layer_output = layer_module(prev_output, mask)
            prev_output = layer_output

        return prev_output


class Decoder(nn.Cell):
    """
    Transformer decoder module.
    """

    def __init__(self, batch_size, attention_dim, odim,
                 num_attention_heads=4,
                 intermediate_size=2048,
                 num_hidden_layers=6,
                 hidden_dropout_prob=0.1,
                 attention_drop_out_prob=0.0,
                 positional_dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.reshape = P.Reshape()
        self.att_dim = attention_dim
        self.cast = P.Cast()
        self.batch_size = batch_size
        self.odim = odim

        self.embed = nn.Embedding(odim, attention_dim)
        self.pos_dec = PositionalEncoding(attention_dim, maxlen=1000, dropout_rate=positional_dropout_rate)
        layers = []
        for _ in range(num_hidden_layers):
            layer = DecoderCell(batch_size, attention_dim,
                                num_attention_heads=num_attention_heads,
                                attention_drop_out_prob=attention_drop_out_prob,
                                hidden_dropout_prob=hidden_dropout_prob,
                                intermediate_size=intermediate_size)
            layers.append(layer)
        self.layers_d = nn.CellList(layers)
        self.linear_d = CustomDense(attention_dim, odim)
        self.softmax = nn.Softmax()
        self.print = P.Print()

    def construct(self, tgt_in, tgt_mask, memory, memory_mask):
        # tgt embedding
        tgt_in = self.cast(tgt_in, mstype.int32)
        embeddings = self.embed(tgt_in)
        position_dec = self.pos_dec(embeddings)

        # decoder cells
        tgt_mask_use = tgt_mask
        memo_mask_use = memory_mask
        prev_output = position_dec
        for layer_module in self.layers_d:
            layer_output = layer_module(
                prev_output, tgt_mask_use, memory, memo_mask_use)
            prev_output = layer_output

        # post linear
        dec_reshape = self.reshape(prev_output, (-1, self.att_dim))
        output = self.linear_d(dec_reshape)

        output = self.reshape(output, (self.batch_size, -1, self.odim))
        output = self.softmax(output)
        return output


class CTC(nn.Cell):
    def __init__(self, batch_size, adim, odim, dropout_prob, ignore_id=-1):
        super(CTC, self).__init__()
        self.ctc = P.CTCLoss(ignore_longer_outputs_than_inputs=False)
        self.transpose = P.Transpose()
        self.linear_c = nn.Dense(adim, odim)
        self.reshape = P.Reshape()
        self.adim = adim
        self.odim = odim
        self.dropout = nn.Dropout(p=dropout_prob)
        self.cast = P.Cast()
        self.not_equal = P.NotEqual()
        self.ignore_id = ignore_id
        self.mul = P.Mul()
        self.shape = P.Shape()
        self.mean = P.ReduceMean()
        self.batch_size = batch_size
        self.softmax = nn.Softmax()
        self.div = P.RealDiv()
        self.equal = P.Equal()
        self.blk = odim - 1
        self.add = P.TensorAdd()
        self.layernorm = nn.LayerNorm([odim])

    def construct(self, hs_pad, hlens, ys_pad, label_indices, label_values):
        hs = self.reshape(hs_pad, (-1, self.adim))
        hs = self.dropout(hs)
        hs = self.linear_c(hs)
        # TODO Temp change for dynamic length sequence.
        if F.is_sequence_value_unknown(self.shape(hs)):
            hs = P.ExpandDims()(hs, 1)
        hs = self.layernorm(hs)
        # TODO Temp change for dynamic length sequence.
        hs = self.reshape(hs, (self.batch_size, -1, self.odim))
        ys_hat = self.transpose(hs, (1, 0, 2))

        if self.ignore_id != self.blk:
            sign_bool = self.not_equal(ys_pad, self.ignore_id)
            sign_op = self.cast(self.equal(
                ys_pad, self.ignore_id), mstype.int32)
            sign = self.cast(sign_bool, mstype.int32)
            ys_pad = self.mul(ys_pad, sign)
            blks = self.mul(sign_op, self.blk)
            ys_pad = self.add(ys_pad, blks)

        label_indices = self.cast(label_indices, mstype.int64)
        label_values = self.cast(label_values, mstype.int32)
        seq_length = self.cast(hlens, mstype.int32)

        losses = self.ctc(ys_hat, label_indices, label_values, seq_length)
        loss = self.mean(losses[0])
        return loss


class KLDivLoss(_Loss):
    def __init__(self, eps=1e-4, kl_temperature=1.0):
        super(KLDivLoss, self).__init__()
        self.kl_temperature = Tensor(kl_temperature, dtype=mstype.float32)
        self.reshape = P.Reshape()
        self.log = P.Log()
        self.exp = P.Exp()
        self.cast = P.Cast()
        self.eps_const = Tensor(eps, dtype=mstype.float32)
        self.add = P.TensorAdd()
        self.div = P.RealDiv()
        self.mul = P.Mul()
        self.shape = P.Shape()
        self.tensor_shape = P.TensorShape()

    def construct(self, s_logit, t_logit):
        # student
        shape_ori = self.shape(s_logit)
        if F.is_sequence_value_unknown(shape_ori):
            shape_ori = self.tensor_shape(s_logit)
        s_1d = self.reshape(s_logit, (-1,))
        s = self.cast(s_1d/self.kl_temperature, mstype.float32)

        # teacher
        t_1d = self.reshape(t_logit, (-1,))
        t_1d_detach = stop_gradient(t_1d)
        t = self.cast(t_1d_detach/self.kl_temperature, mstype.float32)

        div_denom = self.add(s, self.eps_const)
        p_div_q = self.div(t, div_denom)
        log_p_div_q = self.log(self.add(p_div_q, self.eps_const))
        p_log_p_div_q = self.mul(t, log_p_div_q)
        p_log_p_div_q = self.reshape(p_log_p_div_q, shape_ori)

        return p_log_p_div_q


class LabelSmoothingLoss(_Loss):
    def __init__(self, odim, smooth_factor=0.0, ignore_id=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.ignore_id = ignore_id
        self.mul = P.Mul()
        self.onehot = P.OneHot()
        self.num_classes = odim
        self.cast = P.Cast()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (self.num_classes - 1), mstype.float32)
        self.kldiv = KLDivLoss()
        self.sum = P.ReduceSum()
        self.not_equal = P.NotEqual()
        self.blk_id = odim-1
        self.add = P.TensorAdd()
        self.equal = P.Equal()

    def construct(self, pred, ys_out_pad):
        batch = self.shape(pred)[0]
        pred = self.reshape(pred, (-1, self.num_classes))
        true_dist = self.reshape(ys_out_pad, (-1,))
        sign_bool = self.not_equal(true_dist, self.ignore_id)
        sign_op = self.cast(self.equal(
            true_dist, self.ignore_id), mstype.int32)

        sign = self.cast(sign_bool, mstype.int32)
        true_dist = self.mul(true_dist, sign)
        blks = self.mul(sign_op, self.blk_id)
        true_dist = self.add(true_dist, blks)

        target_one_hot = self.onehot(self.cast(true_dist, mstype.int32),
                                     self.num_classes, self.on_value, self.off_value)
        target_one_hot = self.cast(target_one_hot, mstype.float32)
        loss = self.kldiv(pred, target_one_hot)
        loss = self.sum(loss, 1)
        sign = self.cast(sign_bool, mstype.float32)
        mat = self.mul(loss, sign)
        loss = self.sum(mat, ()) / batch

        return loss


class MultiTaskWithLoss(nn.Cell):
    """
    E2E module
    """

    def __init__(self, batch_size, atten_dim, odim, feat_dim, alpha,
                 num_attention_heads=4,
                 eunits=2048,
                 dunits=2048,
                 encoder_blocks=6,
                 decoder_blocks=6,
                 hidden_dropout_prob=0.1,
                 positional_dropout_rate=0.1,
                 attention_drop_out_prob=0.0,
                 ignore_id=-1,
                 pad=2,
                 lsm_factor=0.0):
        """
        Construct an E2E object
        """
        super(MultiTaskWithLoss, self).__init__()
        self.encoder = Encoder(batch_size, atten_dim, feat_dim,
                               num_hidden_layers=encoder_blocks,
                               num_attention_heads=num_attention_heads,
                               intermediate_size=eunits,
                               attention_probs_dropout_prob=attention_drop_out_prob,
                               hidden_dropout_prob=hidden_dropout_prob,
                               positional_dropout_rate=positional_dropout_rate,
                               pad=pad)
        self.decoder = Decoder(batch_size, atten_dim, odim,
                               num_attention_heads=num_attention_heads,
                               intermediate_size=dunits,
                               num_hidden_layers=decoder_blocks,
                               hidden_dropout_prob=hidden_dropout_prob,
                               attention_drop_out_prob=attention_drop_out_prob,
                               positional_dropout_rate=positional_dropout_rate)
        self.att_loss = LabelSmoothingLoss(odim, smooth_factor=lsm_factor, ignore_id=ignore_id)
        self.ctc_loss = CTC(batch_size, atten_dim, odim, hidden_dropout_prob, ignore_id=ignore_id)

        self.add = P.TensorAdd()
        self.cast = P.Cast()
        self.reshape = P.Reshape()

        self.alpha = alpha
        self.adim = atten_dim
        self.batch_size = batch_size

    def construct(self, audio, tgt_in, tgt_mask, enc_mask, memory_mask,
                  tgt_out, hlens, ys_pad, label_indices, label_values):
        enc_out = self.encoder(audio, enc_mask)
        dec_out = self.decoder(tgt_in, tgt_mask, enc_out, memory_mask)
        aloss = self.att_loss(dec_out, tgt_out)
        aloss = self.cast(aloss, mstype.float32)
        hs_pad = self.reshape(enc_out, (self.batch_size, -1, self.adim))
        closs = self.ctc_loss(hs_pad, hlens, ys_pad, label_indices, label_values)
        closs = self.cast(closs, mstype.float32)
        loss = self.add(self.alpha * closs, (1 - self.alpha) * aloss)
        return (loss, aloss, closs)


class MultiTaskTrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(MultiTaskTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)

    def construct(self, *inputs):
        weights = self.weights
        (loss, aloss, closs) = self.network(*inputs)
        sens = (P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens),
                P.Fill()(P.DType()(aloss), P.Shape()(aloss), 0.0),
                P.Fill()(P.DType()(closs), P.Shape()(closs), 0.0))
        grads = self.grad(self.network, weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        return (F.depend(loss, self.optimizer(grads)), aloss, closs)


def warmup_lr(init_lr, warmup_steps, total_steps):
    warmup_steps = float(warmup_steps)
    init_lr = float(init_lr)
    lr = []
    for step in range(1, total_steps + 1):
        step = float(step)
        v = init_lr * warmup_steps ** (0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
        lr.append(v)
    return np.array(lr).astype(np.float32)


def create_dataset(batch_size=16, feats_dim=83, text_dim=6368):
    feats_widths = (1024, 1015)
    feats2_widths = ((1024, 1024), (1015, 1014))
    token_widths = (24, 22)
    text_widths = ((24, 22), (21, 22))
    seq_shape_list = [((batch_size, 895, feats_dim), (batch_size, 23)), ((
        batch_size, 896, feats_dim), (batch_size, 26))]
    seq_shape_list = []

    for feats_wid, feats2_wid, token_wid, text_wid in zip(feats_widths, feats2_widths, token_widths, text_widths):
        seq_shape_list.append(((batch_size, feats_wid, feats_dim),
                               ((feats2_wid[0], feats_dim), (feats2_wid[1], feats_dim)),
                               (batch_size, token_wid),
                               ((text_wid[0], text_dim), (text_wid[1], text_dim))))

    np.random.seed(0)
    data_list = []
    for feats_shape1, feats_shape_temp, token_shape, text_shape_temp in seq_shape_list:
        label_indices = []
        label_values = []
        text_shape = np.array(text_shape_temp).astype(np.int32)
        blank_token = text_shape[0][1] - 1

        token = np.random.randn(*token_shape).astype(np.int32)

        for batch_id, item in enumerate(token):
            tmp_token = copy.deepcopy(item)
            tmp_token[tmp_token == -1] = blank_token
            label_values.extend(tmp_token)
            for i, _ in enumerate(tmp_token):
                label_indices.append([batch_id, i])
        label_values, label_indices = np.array(
            label_values), np.array(label_indices)

        sos_pad = np.ones((len(token), 1))
        tgt_in = np.concatenate((sos_pad, token), axis=1)

        tgt_in[tgt_in == -1] = 0
        tgt_out = []
        for i, _ in enumerate(token):
            line = copy.deepcopy(token[i])
            idx = text_shape[i][0]
            if idx < len(line):
                line = np.insert(line, idx, 2)
            else:
                line = np.append(line, 2)
            tgt_out.append(line)
        tgt_out = np.array(tgt_out)
        length = tgt_in.shape[1]
        tmp_token = tgt_in != -1
        tmp_token = np.repeat(tmp_token, length, axis=0).reshape((len(tgt_in), length, length))
        mask = np.tile(np.tril(np.ones((length, length))), (len(tgt_in), 1, 1))
        mask = np.multiply(tmp_token, mask)
        tgt_mask = mask
        in_len = feats_shape1[1]
        out_len = token_shape[1]
        batch_size = feats_shape1[0]
        memory_mask = np.ones((batch_size, in_len, out_len))
        enc_mask = np.ones((batch_size, in_len, in_len))

        feats_shape = np.array(feats_shape_temp).astype(np.int32)
        hlens_ori = feats_shape[:, 0]
        hlens_ori = 1 + (hlens_ori + 2 * 2 - 3) // 2
        hlens = 1 + (hlens_ori + 2 * 2 - 3) // 2
        feats = np.random.randn(*feats_shape1).astype(np.float32)

        data_list.append((feats, tgt_in.astype(np.int32), tgt_mask.astype(np.float32),
                          enc_mask, memory_mask, tgt_out.astype(np.int32), hlens,
                          token, label_indices.astype(np.int32), label_values.astype(np.int32)))

    ds = de.GeneratorDataset(data_list,
                             ["feats", "tgt_in", "tgt_mask", "enc_mask",
                              "memory_mask", "tgt_out", "hlens",
                              "token", "label_indices", "label_values"])
    return ds


class TimeMonitor(Callback):
    """
    Monitor the time in training.

    Args:
        data_size (int): How many steps are the intervals between print information each time.
            if the program get `batch_num` during training, `data_size` will be set to `batch_num`,
            otherwise `data_size` will be used. Default: None.

    Raises:
        ValueError: If data_size is not positive int.
    """

    def __init__(self, steps_size):
        super(TimeMonitor, self).__init__()
        self.step = 0
        self.steps_size = steps_size
        self.step_time = 0.0
        self.loss = []

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        step_seconds = (time.time() - self.step_time) * 1000
        cb_params = run_context.original_args()
        # TrainOneStepWithLossScaleCell returns tuple while TrainOneStepCell returns loss directly
        step_loss = cb_params.net_outputs[0].asnumpy()
        scale = cb_params.net_outputs[2]
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        if not isinstance(step_size, int) or step_size < 1:
            raise ValueError("data_size must be positive int.")

        step_seconds = step_seconds / 1000

        logging.debug(
            "Epoch: %d, Step: %d, Step Time: %s sec, Total Loss: %i, Scale: %i.",
            int(self.step / self.steps_size),
            self.step % self.steps_size,
            str(step_seconds)[:5],
            step_loss,
            scale
        )

        self.loss.append(step_loss)
        self.step += 1


def asr_run(input_dict=None):
    args_dict = {
        "adim": 256,
        "aheads": 1,
        "alpha": 0.3,
        "attention_dropout_prob": 0.0,
        "batch_size": 2,
        "dblocks": 1,
        "dunits": 2048,
        "conv_pad": 2,
        "eblocks": 1,
        "epochs": 2,
        "eunits": 2048,
        "feats_dim": 83,
        "hidden_dropout_prob": 0.1,
        "init_lr": 0.0005,
        "lsm_factor": 0.1,
        "positional_dropout_prob": 0.1,
        "text_dim": 6368,
        "warmup_steps": 2
    }

    if input_dict:
        args_dict.update(input_dict)

    args = argparse.Namespace(**args_dict)
    time.sleep(3)
    context.set_context(mode=context.GRAPH_MODE)

    mindspore.set_seed(0)
    dataset = create_dataset(args.batch_size, args.feats_dim, args.text_dim)
    loss_net = MultiTaskWithLoss(args.batch_size, args.adim, args.text_dim, args.feats_dim, args.alpha,
                                 num_attention_heads=args.aheads,
                                 eunits=args.eunits,
                                 dunits=args.dunits,
                                 encoder_blocks=args.eblocks,
                                 decoder_blocks=args.dblocks,
                                 hidden_dropout_prob=args.hidden_dropout_prob,
                                 attention_drop_out_prob=args.attention_dropout_prob,
                                 positional_dropout_rate=args.positional_dropout_prob,
                                 ignore_id=IGNORE_ID,
                                 pad=args.conv_pad,
                                 lsm_factor=args.lsm_factor)
    lr = warmup_lr(args.init_lr, args.warmup_steps, args.epochs * dataset.get_dataset_size())
    opt = Adam(params=loss_net.trainable_params(), learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.999)

    train_net = MultiTaskTrainOneStepCell(loss_net, opt)

    dyn_feats = Tensor(shape=[args.batch_size, None, args.feats_dim], dtype=mindspore.float32)
    dyn_tgt_in = Tensor(shape=[args.batch_size, None], dtype=mindspore.int32)
    dyn_tgt_mask = Tensor(shape=[args.batch_size, None, None], dtype=mindspore.float32)
    dyn_enc_mask = Tensor(shape=[args.batch_size, None, None], dtype=mindspore.float64)
    dyn_memory_mask = Tensor(shape=[args.batch_size, None, None], dtype=mindspore.float64)
    dyn_tgt_out = Tensor(shape=[args.batch_size, None], dtype=mindspore.int32)
    dyn_hlens = Tensor(shape=[args.batch_size], dtype=mindspore.int32, init=One())
    dyn_token = Tensor(shape=[args.batch_size, None], dtype=mindspore.int32)
    dyn_label_indices = Tensor(shape=[None, 2], dtype=mindspore.int32)
    dyn_label_values = Tensor(shape=[None], dtype=mindspore.int32)
    train_net.set_inputs(dyn_feats, dyn_tgt_in, dyn_tgt_mask, dyn_enc_mask, dyn_memory_mask, dyn_tgt_out, dyn_hlens,
                         dyn_token, dyn_label_indices, dyn_label_values)

    train_net.set_train(True)

    callback_list = [TimeMonitor(dataset.get_dataset_size())]

    model = Model(train_net)
    epochs_step = dataset.get_dataset_size() * args.epochs
    model.train(epochs_step, dataset, callbacks=callback_list, sink_size=1, dataset_sink_mode=True)
    return callback_list[0].loss


def _compare_result(outputs, expects):
    if len(outputs) != len(expects):
        raise RuntimeError("Result size error, should be {}, but got {}!".format(len(expects), len(outputs)))
    for output, expect in zip(outputs, expects):
        if not np.allclose(output, expect, 0.0001, 0.0001):
            raise RuntimeError(
                "[ERROR] compare as followings:\n ==> outputs: {},\n ==> expects: {}".format(output, expect))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ascend_train():
    """
    Feature: Test the simplified dynamic shape ASR network with small data in Ascend.
    Description:  The sequence length of inputs is dynamic.
    Expectation: Assert that the training loss of fixed data is consistent with the expected loss.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    start_time = time.time()
    losses = asr_run()
    current_time = time.time()
    logging.info("Run asr with %f s.", current_time - start_time)
    expect_losses = [np.array(106.237755, dtype=np.float32), np.array(90.78951, dtype=np.float32),
                     np.array(89.331894, dtype=np.float32), np.array(102.211105, dtype=np.float32)]
    _compare_result(losses[:1], expect_losses[:1])
    logging.info("Test asr done.")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_train():
    """
    Feature: Test the simplified dynamic shape ASR network with small data in GPU.
    Description:  The sequence length of inputs is dynamic.
    Expectation: Assert that the training loss of fixed data is consistent with the expected loss.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    start_time = time.time()
    losses = asr_run()
    current_time = time.time()
    logging.info("Run asr with %f s.", current_time - start_time)
    expect_losses = [np.array(727.7395, dtype=np.float32), np.array(518.216, dtype=np.float32),
                     np.array(107.88617, dtype=np.float32), np.array(139.66273, dtype=np.float32)]
    _compare_result(losses, expect_losses)
    logging.info("Test asr done.")
