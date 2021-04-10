"""ipt"""
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
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter


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
        compute_type (:class:`mindspore.dtype`): Compute type in MultiheadAttention. Default: mstype.float32.
    """

    def __init__(self,
                 q_tensor_width,
                 k_tensor_width,
                 v_tensor_width,
                 hidden_width,
                 out_tensor_width,
                 num_attention_heads=1,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 out_act=None,
                 has_attention_mask=True,
                 attention_probs_dropout_prob=0.0,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 do_return_2d_tensor=False,
                 compute_type=mstype.float32,
                 same_dim=True):
        super(MultiheadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = int(hidden_width / num_attention_heads)
        self.has_attention_mask = has_attention_mask
        assert has_attention_mask
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor
        self.same_dim = same_dim

        self.scores_mul = Tensor(
            [1.0 / math.sqrt(float(self.size_per_head))], dtype=compute_type)
        self.reshape = P.Reshape()
        self.shape_q_2d = (-1, q_tensor_width)
        self.shape_k_2d = (-1, k_tensor_width)
        self.shape_v_2d = (-1, v_tensor_width)
        self.hidden_width = hidden_width
        if self.same_dim:
            self.in_proj_layer = \
                Parameter(Tensor(np.random.rand(hidden_width * 3,
                                                q_tensor_width), dtype=compute_type), name="weight")
        else:
            self.query_layer = nn.Dense(q_tensor_width,
                                        hidden_width,
                                        activation=query_act,
                                        has_bias=False).to_float(compute_type)
            self.key_layer = nn.Dense(k_tensor_width,
                                      hidden_width,
                                      activation=key_act,
                                      has_bias=False).to_float(compute_type)
            self.value_layer = nn.Dense(q_tensor_width,
                                        hidden_width,
                                        activation=value_act,
                                        has_bias=False).to_float(compute_type)
        self.out_proj = nn.Dense(hidden_width,
                                 out_tensor_width,
                                 activation=out_act,
                                 has_bias=False).to_float(compute_type)

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = Tensor([-10000.0,], dtype=compute_type)
        self.matmul = P.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(1. - attention_probs_dropout_prob)
        self.use_dropout = attention_probs_dropout_prob > 0

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.TensorAdd()
            self.cast = P.Cast()
            self.get_dtype = P.DType()

        self.softmax_cast = P.Cast()
        self.matmul_dense = P.MatMul(transpose_b=True)
        self.split = P.Split(0, 3)

    def construct(self, tensor_q, tensor_k, tensor_v, batch_size, seq_length, attention_mask=None):
        """Apply multihead attention."""
        self.batch_size = batch_size
        shape_qkv = (self.batch_size, -1,
                     self.num_attention_heads, self.size_per_head)
        shape_linear = (self.batch_size * seq_length,
                        self.num_attention_heads * self.size_per_head)
        if self.do_return_2d_tensor:
            shape_return = (self.batch_size * seq_length,
                            self.num_attention_heads * self.size_per_head)
            if seq_length == -1:
                shape_return = (-1, self.num_attention_heads *
                                self.size_per_head)
        else:
            shape_return = (self.batch_size, seq_length,
                            self.num_attention_heads * self.size_per_head)

        tensor_q_2d = self.reshape(tensor_q, self.shape_q_2d)
        tensor_k_2d = self.reshape(tensor_k, self.shape_k_2d)
        tensor_v_2d = self.reshape(tensor_v, self.shape_v_2d)

        if P.Equal()(tensor_q_2d, tensor_v_2d)[0][0]:
            x = self.matmul_dense(self.in_proj_layer, tensor_q_2d)
            query_out, key_out, value_out = self.split(x)

        elif self.same_dim:
            _start = int(0)
            _end = int(self.hidden_width)
            _w = self.in_proj_layer[_start:_end, :]
            query_out = self.matmul_dense(_w, tensor_q_2d)

            _start = int(self.hidden_width)
            _end = int(self.hidden_width * 2)
            _w = self.in_proj_layer[_start:_end, :]
            key_out = self.matmul_dense(_w, tensor_k_2d)

            _start = int(self.hidden_width * 2)
            _end = None
            _w = self.in_proj_layer[_start:]
            value_out = self.matmul_dense(_w, tensor_v_2d)
        else:
            query_out = self.query_layer(tensor_q_2d)
            key_out = self.key_layer(tensor_k_2d)
            value_out = self.value_layer(tensor_v_2d)
        query_out = self.transpose(query_out, (1, 0))
        key_out = self.transpose(key_out, (1, 0))
        value_out = self.transpose(value_out, (1, 0))
        query_layer = self.reshape(query_out, shape_qkv)
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, shape_qkv)
        key_layer = self.transpose(key_layer, self.trans_shape)

        attention_scores = self.matmul_trans_b(query_layer, key_layer)
        attention_scores = self.multiply(attention_scores, self.scores_mul)

        attention_scores = self.softmax_cast(attention_scores, mstype.float32)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.softmax_cast(
            attention_probs, self.get_dtype(key_layer))
        if self.use_dropout:
            attention_probs = self.dropout(attention_probs)

        value_layer = self.reshape(value_out, shape_qkv)
        value_layer = self.transpose(value_layer, self.trans_shape)
        context_layer = self.matmul(attention_probs, value_layer)

        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, shape_linear)

        context_layer = self.out_proj(context_layer)
        context_layer = self.reshape(context_layer, shape_return)
        return context_layer


class TransformerEncoderLayer(nn.Cell):
    """ipt"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()

        self.self_attn = MultiheadAttention(q_tensor_width=d_model,
                                            k_tensor_width=d_model,
                                            v_tensor_width=d_model,
                                            hidden_width=d_model,
                                            out_tensor_width=d_model,
                                            num_attention_heads=nhead,
                                            attention_probs_dropout_prob=dropout)
        self.linear1 = nn.Dense(d_model, dim_feedforward)
        self.dropout = nn.Dropout(1. - dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])
        self.dropout1 = nn.Dropout(1. - dropout)
        self.dropout2 = nn.Dropout(1. - dropout)
        self.reshape = P.Reshape()

        self.activation = P.ReLU()

    def with_pos_embed(self, tensor, pos):
        """ipt"""
        return tensor if pos is None else tensor + pos

    def construct(self, src, pos=None):
        """ipt"""
        b, n, d = src.shape
        permute_linear = (b * n, d)
        permute_recover = (b, n, d)
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2, batch_size=b, seq_length=n)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.reshape(src2, permute_linear)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = self.reshape(src2, permute_recover)
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Cell):
    """ipt"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(q_tensor_width=d_model,
                                            k_tensor_width=d_model,
                                            v_tensor_width=d_model,
                                            hidden_width=d_model,
                                            out_tensor_width=d_model,
                                            num_attention_heads=nhead,
                                            attention_probs_dropout_prob=dropout)
        self.multihead_attn = MultiheadAttention(q_tensor_width=d_model,
                                                 k_tensor_width=d_model,
                                                 v_tensor_width=d_model,
                                                 hidden_width=d_model,
                                                 out_tensor_width=d_model,
                                                 num_attention_heads=nhead,
                                                 attention_probs_dropout_prob=dropout)
        self.linear1 = nn.Dense(d_model, dim_feedforward)
        self.dropout = nn.Dropout(1. - dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])
        self.norm3 = nn.LayerNorm([d_model])
        self.dropout1 = nn.Dropout(1. - dropout)
        self.dropout2 = nn.Dropout(1. - dropout)
        self.dropout3 = nn.Dropout(1. - dropout)
        self.reshape = P.Reshape()
        self.activation = P.ReLU()

    def with_pos_embed(self, tensor, pos):
        """ipt"""
        return tensor if pos is None else tensor + pos

    def construct(self, tgt, memory, pos=None, query_pos=None):
        """ipt"""
        b, n, d = tgt.shape
        permute_linear = (b * n, d)
        permute_recover = (b, n, d)
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tensor_v=tgt2, batch_size=b, seq_length=n)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(tensor_q=self.with_pos_embed(tgt2, query_pos),
                                   tensor_k=self.with_pos_embed(memory, pos),
                                   tensor_v=memory,
                                   batch_size=b, seq_length=n)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.reshape(tgt2, permute_linear)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt2 = self.reshape(tgt2, permute_recover)
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerEncoder(nn.Cell):
    """ipt"""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def construct(self, src, pos=None):
        """ipt"""
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output


class TransformerDecoder(nn.Cell):
    """ipt"""

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def construct(self, tgt, memory, pos=None, query_pos=None):
        """ipt"""
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
        return output


def _get_clones(module, N):
    """ipt"""
    return nn.CellList([copy.deepcopy(module) for i in range(N)])


class LearnedPositionalEncoding(nn.Cell):
    """ipt"""

    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(
            max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.position_ids = Tensor(np.arange(self.seq_length).astype(np.int32))
        self.reshape = P.Reshape()
        self.position_ids = self.reshape(
            self.position_ids, (1, self.seq_length))

    def construct(self, x, position_ids=None):
        """ipt"""
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings


class VisionTransformer(nn.Cell):
    """ipt"""

    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            num_queries,
            positional_encoding_type="learned",
            dropout_rate=0,
            norm=False,
            mlp=False,
            pos_every=False,
            no_pos=False
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.norm = norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.out_dim = patch_dim * patch_dim * num_channels

        self.no_pos = no_pos

        self.unf = _unfold_(patch_dim)
        self.fold = _fold_(patch_dim)

        if self.mlp is not True:
            self.linear_encoding = nn.Dense(
                self.flatten_dim, embedding_dim)
            self.mlp_head = nn.SequentialCell(
                nn.Dense(embedding_dim, hidden_dim),
                nn.Dropout(1. - dropout_rate),
                nn.ReLU(),
                nn.Dense(hidden_dim, self.out_dim),
                nn.Dropout(1. - dropout_rate)
            )

        self.query_embed = nn.Embedding(
            num_queries, embedding_dim * self.seq_length)

        encoder_layer = TransformerEncoderLayer(
            embedding_dim, num_heads, hidden_dim, dropout_rate)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(
            embedding_dim, num_heads, hidden_dim, dropout_rate)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        self.reshape = P.Reshape()
        self.tile = P.Tile()
        self.transpose = P.Transpose()
        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(1. - dropout_rate)

    def construct(self, x, query_idx):
        """ipt"""
        B, _, _, _ = x.shape
        x = self.unf(x)
        B, N, _ = x.shape

        if self.mlp is not True:
            x = self.reshape(x, (int(B * N), -1))
            x = self.dropout_layer1(self.linear_encoding(x)) + x
            x = self.reshape(x, (B, N, -1))
        query_embed = self.tile(
            self.reshape(self.query_embed.embedding_table[int(
                query_idx)], (1, self.seq_length, self.embedding_dim)),
            (B, 1, 1))

        if not self.no_pos:
            pos = self.position_encoding(x)

        x = self.encoder(x + pos)
        x = self.decoder(x, x, query_pos=query_embed)

        if self.mlp is not True:
            x = self.reshape(x, (int(B * N), -1))
            x = self.mlp_head(x) + x
            x = self.reshape(x, (B, N, -1))
        x = self.fold(x)

        return x


def default_conv(in_channels, out_channels, kernel_size, has_bias=True):
    """ipt"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, has_bias=has_bias)


class MeanShift(nn.Conv2d):
    """ipt"""

    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.reshape = P.Reshape()
        self.eye = P.Eye()
        std = Tensor(rgb_std, mstype.float32)
        self.weight.set_data(
            self.reshape(self.eye(3, 3, mstype.float32), (3, 3, 1, 1)) / self.reshape(std, (3, 1, 1, 1)))
        self.weight.requires_grad = False
        self.bias = Parameter(
            sign * rgb_range * Tensor(rgb_mean, mstype.float32) / std, name='bias', requires_grad=False)
        self.has_bias = True


class ResBlock(nn.Cell):
    """ipt"""

    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, has_bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.SequentialCell(*m)
        self.res_scale = res_scale

        self.mul = P.Mul()

    def construct(self, x):
        """ipt"""
        res = self.mul(self.body(x), self.res_scale)
        res += x

        return res


def _pixelsf_(x, scale):
    """ipt"""
    N, C, iH, iW = x.shape
    oH = int(iH * scale)
    oW = int(iW * scale)
    oC = int(C // (scale ** 2))

    output = P.Reshape()(x, (N, oC, scale, scale, iH, iW))

    output = P.Transpose()(output, (0, 1, 5, 3, 4, 2))

    output = P.Reshape()(output, (N, oC, oH, oW))

    output = P.Transpose()(output, (0, 1, 3, 2))

    return output


class SmallUpSampler(nn.Cell):
    """ipt"""

    def __init__(self, conv, upsize, n_feats, bn=False, act=False, bias=True):
        super(SmallUpSampler, self).__init__()
        self.conv = conv(n_feats, upsize * upsize * n_feats, 3, bias)
        self.reshape = P.Reshape()
        self.upsize = upsize

    def construct(self, x):
        """ipt"""
        x = self.conv(x)
        output = _pixelsf_(x, self.upsize)
        return output


class Upsampler(nn.Cell):
    """ipt"""

    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        super(Upsampler, self).__init__()
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(SmallUpSampler(conv, 2, n_feats, bias=bias))

        elif scale == 3:
            m.append(SmallUpSampler(conv, 3, n_feats, bias=bias))
        self.net = nn.SequentialCell(m)

    def construct(self, x):
        """ipt"""
        return self.net(x)


class IPT(nn.Cell):
    """ipt"""

    def __init__(self, args, conv=default_conv):
        super(IPT, self).__init__()

        self.scale_idx = 0

        self.args = args

        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU()

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        self.head = nn.CellList([
            nn.SequentialCell(
                conv(args.n_colors, n_feats, kernel_size),
                ResBlock(conv, n_feats, 5, act=act),
                ResBlock(conv, n_feats, 5, act=act)
            ) for _ in args.scale
        ])

        self.body = VisionTransformer(img_dim=args.patch_size,
                                      patch_dim=args.patch_dim,
                                      num_channels=n_feats,
                                      embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                      num_heads=args.num_heads,
                                      num_layers=args.num_layers,
                                      hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                      num_queries=args.num_queries,
                                      dropout_rate=args.dropout_rate,
                                      mlp=args.no_mlp,
                                      pos_every=args.pos_every,
                                      no_pos=args.no_pos)

        self.tail = nn.CellList([
            nn.SequentialCell(
                Upsampler(conv, s, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ) for s in args.scale
        ])

        self.reshape = P.Reshape()
        self.tile = P.Tile()
        self.transpose = P.Transpose()

    def construct(self, x):
        """ipt"""
        x = self.sub_mean(x)
        x = self.head[self.scale_idx](x)
        res = self.body(x, self.scale_idx)
        res += x
        x = self.tail[self.scale_idx](res)
        x = self.add_mean(x)

        return x

    def set_scale(self, scale_idx):
        """ipt"""
        self.scale_idx = scale_idx

    def infrc(self, x):
        """ipt"""
        forward_function = self.forward_chop_new

        return forward_function(x)

    def forward_chop_new(self, x, shave=12, batchsize=64):
        """ipt"""
        h, w = x.shape[-2:]
        padsize = int(self.args.patch_size)
        shave = int(self.args.patch_size / 4)
        scale = self.args.scale[self.scale_idx]

        h_cut = (h - padsize) % (padsize - shave)
        w_cut = (w - padsize) % (padsize - shave)

        unf_1 = _stride_unfold_(padsize, stride=padsize - shave)
        x_unfold = unf_1(x)
        x_unfold = self.transpose(x_unfold, (1, 0, 2))  # transpose(0,2)

        x_hw_cut = x[:, :, (h - padsize):, (w - padsize):]
        y_hw_cut = self.construct(x_hw_cut)

        x_h_cut = x[:, :, (h - padsize):, :]
        x_w_cut = x[:, :, :, (w - padsize):]
        y_h_cut = self.cut_h_new(x_h_cut, h, w, h_cut,
                                 w_cut, padsize, shave, scale, batchsize)
        y_w_cut = self.cut_w_new(x_w_cut, h, w, h_cut,
                                 w_cut, padsize, shave, scale, batchsize)

        x_h_top = x[:, :, :padsize, :]
        x_w_top = x[:, :, :, :padsize]
        y_h_top = self.cut_h_new(x_h_top, h, w, h_cut,
                                 w_cut, padsize, shave, scale, batchsize)
        y_w_top = self.cut_w_new(x_w_top, h, w, h_cut,
                                 w_cut, padsize, shave, scale, batchsize)
        x_unfold = self.reshape(
            x_unfold, (x_unfold.shape[0], -1, padsize, padsize))
        x_range = x_unfold.shape[0] // batchsize + \
            (x_unfold.shape[0] % batchsize != 0)

        cc_0 = P.Concat(axis=0)
        for i in range(x_range):
            if i == 0:
                y_unfold = self.construct(
                    x_unfold[i * batchsize:(i + 1) * batchsize, :, :, :])
            else:
                y_unfold = cc_0((y_unfold, self.construct(
                    x_unfold[i * batchsize:(i + 1) * batchsize, :, :, :])))
        y_unf_shape_0 = y_unfold.shape[0]
        fold_1 = \
            _stride_fold_(padsize * scale, output_shape=((h - h_cut) * scale, (w - w_cut) * scale),
                          stride=padsize * scale - shave * scale)
        y = fold_1(self.transpose(self.reshape(
            y_unfold, (y_unf_shape_0, -1, 1)), (2, 0, 1)))
        cc_2 = P.Concat(axis=2)
        cc_3 = P.Concat(axis=3)
        y = cc_2((y_h_top, y[:, :, padsize * scale:, :]))
        y = cc_3((y_w_top, y[:, :, :, padsize * scale:]))
        y_unfold = y_unfold[:, :, int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                            int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)]
        fold_2 = _stride_fold_(padsize * scale - shave * scale,
                               output_shape=((h - h_cut - shave) *
                                             scale, (w - w_cut - shave) * scale),
                               stride=padsize * scale - shave * scale)
        y_inter = fold_2(self.transpose(self.reshape(
            y_unfold, (y_unf_shape_0, -1, 1)), (2, 0, 1)))
        y = cc_3((cc_3((y[:, :, :, :int(shave / 2 * scale)], cc_2((cc_2((y[:, :, :int(shave / 2 * scale), int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)], y_inter)), y[:, :, (h - h_cut) * scale - int(shave / 2 * scale):, int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)])))), y[:, :, :, (w - w_cut) * scale - int(shave / 2 * scale):])) #pylint: disable=line-too-long
        y = cc_2((y[:, :, :y.shape[2] - int((padsize - h_cut) / 2 * scale), :],
                  y_h_cut[:, :, int((padsize - h_cut) / 2 * scale + 0.5):, :]))
        y_w_cat = cc_2((y_w_cut[:, :, :y_w_cut.shape[2] - int((padsize - h_cut) / 2 * scale), :],
                        y_hw_cut[:, :, int((padsize - h_cut) / 2 * scale + 0.5):, :]))
        y = cc_3((y[:, :, :, :y.shape[3] - int((padsize - w_cut) / 2 * scale)],
                  y_w_cat[:, :, :, int((padsize - w_cut) / 2 * scale + 0.5):]))
        return y

    def cut_h_new(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        """ipt"""
        unf_1 = _stride_unfold_(padsize, stride=padsize - shave)
        x_h_cut_unfold = unf_1(x_h_cut)
        x_h_cut_unfold = self.transpose(x_h_cut_unfold, (1, 0, 2))

        x_h_cut_unfold = self.reshape(
            x_h_cut_unfold, (x_h_cut_unfold.shape[0], -1, padsize, padsize))
        x_range = x_h_cut_unfold.shape[0] // batchsize + \
            (x_h_cut_unfold.shape[0] % batchsize != 0)
        cc_0 = P.Concat(axis=0)
        for i in range(x_range):
            if i == 0:
                y_h_cut_unfold = self.construct(
                    x_h_cut_unfold[i * batchsize:(i + 1) * batchsize, :, :, :])
            else:
                y_h_cut_unfold = \
                    cc_0((y_h_cut_unfold, self.construct(
                        x_h_cut_unfold[i * batchsize:(i + 1) * batchsize, :, :, :])))
        y_h_cut_unfold_shape_0 = y_h_cut_unfold.shape[0]
        fold_1 = \
            _stride_fold_(padsize * scale, output_shape=(padsize * scale, (w - w_cut) * scale),
                          stride=padsize * scale - shave * scale)
        y_h_cut = fold_1(self.transpose(self.reshape(
            y_h_cut_unfold, (y_h_cut_unfold_shape_0, -1, 1)), (2, 0, 1)))
        y_h_cut_unfold = y_h_cut_unfold[:, :, :, int(
            shave / 2 * scale):padsize * scale - int(shave / 2 * scale)]
        fold_2 = _stride_fold_((padsize * scale, padsize * scale - shave * scale),
                               output_shape=(padsize * scale,
                                             (w - w_cut - shave) * scale),
                               stride=padsize * scale - shave * scale)
        y_h_cut_inter = fold_2(self.transpose(self.reshape(
            y_h_cut_unfold, (y_h_cut_unfold_shape_0, -1, 1)), (2, 0, 1)))
        cc_3 = P.Concat(axis=3)
        y_h_cut = cc_3((cc_3((y_h_cut[:, :, :, :int(shave / 2 * scale)],
                              y_h_cut_inter)), y_h_cut[:, :, :, (w - w_cut) * scale - int(shave / 2 * scale):]))
        return y_h_cut

    def cut_w_new(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        """ipt"""
        unf_1 = _stride_unfold_(padsize, stride=padsize - shave)
        x_w_cut_unfold = unf_1(x_w_cut)
        x_w_cut_unfold = self.transpose(x_w_cut_unfold, (1, 0, 2))

        x_w_cut_unfold = self.reshape(
            x_w_cut_unfold, (x_w_cut_unfold.shape[0], -1, padsize, padsize))
        x_range = x_w_cut_unfold.shape[0] // batchsize + \
            (x_w_cut_unfold.shape[0] % batchsize != 0)
        cc_0 = P.Concat(axis=0)
        for i in range(x_range):
            if i == 0:
                y_w_cut_unfold = self.construct(
                    x_w_cut_unfold[i * batchsize:(i + 1) * batchsize, :, :, :])
            else:
                y_w_cut_unfold = cc_0((y_w_cut_unfold,
                                       self.construct(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize, :, :, :])))
        y_w_cut_unfold_shape_0 = y_w_cut_unfold.shape[0]
        fold_1 = _stride_fold_(padsize * scale,
                               output_shape=((h - h_cut) * scale,
                                             padsize * scale),
                               stride=padsize * scale - shave * scale)
        y_w_cut = fold_1(self.transpose(self.reshape(
            y_w_cut_unfold, (y_w_cut_unfold_shape_0, -1, 1)), (2, 0, 1)))
        y_w_cut_unfold = y_w_cut_unfold[:, :, int(
            shave / 2 * scale):padsize * scale - int(shave / 2 * scale), :]
        fold_2 = _stride_fold_((padsize * scale - shave * scale, padsize * scale),
                               output_shape=((h - h_cut - shave)
                                             * scale, padsize * scale),
                               stride=padsize * scale - shave * scale)
        y_w_cut_inter = fold_2(self.transpose(self.reshape(
            y_w_cut_unfold, (y_w_cut_unfold_shape_0, -1, 1)), (2, 0, 1)))
        cc_2 = P.Concat(axis=2)
        y_w_cut = cc_2((cc_2((y_w_cut[:, :, :int(shave / 2 * scale), :],
                              y_w_cut_inter)), y_w_cut[:, :, (h - h_cut) * scale - int(shave / 2 * scale):, :]))
        return y_w_cut


class _stride_unfold_(nn.Cell):
    """ipt"""

    def __init__(
            self, kernel_size, stride=-1):

        super(_stride_unfold_, self).__init__()
        if stride == -1:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.kernel_size = kernel_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.unfold = _unfold_(kernel_size)

    def construct(self, x):
        """ipt"""
        N, C, H, W = x.shape
        leftup_idx_x = []
        leftup_idx_y = []
        nh = int((H - self.kernel_size) / self.stride + 1)
        nw = int((W - self.kernel_size) / self.stride + 1)
        for i in range(nh):
            leftup_idx_x.append(i * self.stride)
        for i in range(nw):
            leftup_idx_y.append(i * self.stride)
        NumBlock_x = len(leftup_idx_x)
        NumBlock_y = len(leftup_idx_y)
        zeroslike = P.ZerosLike()
        cc_2 = P.Concat(axis=2)
        cc_3 = P.Concat(axis=3)
        unf_x = P.Zeros()((N, C, NumBlock_x * self.kernel_size,
                           NumBlock_y * self.kernel_size), mstype.float32)
        N, C, H, W = unf_x.shape
        for i in range(NumBlock_x):
            for j in range(NumBlock_y):
                unf_i = i * self.kernel_size
                unf_j = j * self.kernel_size
                org_i = leftup_idx_x[i]
                org_j = leftup_idx_y[j]
                fills = x[:, :, org_i:org_i + self.kernel_size,
                          org_j:org_j + self.kernel_size]
                unf_x += cc_3((cc_3((zeroslike(unf_x[:, :, :, :unf_j]),
                                     cc_2(
                                         (cc_2((zeroslike(unf_x[:, :, :unf_i, unf_j:unf_j + self.kernel_size]), fills)),
                                          zeroslike(unf_x[:, :, unf_i + self.kernel_size:,
                                                          unf_j:unf_j + self.kernel_size]))))),
                               zeroslike(unf_x[:, :, :, unf_j + self.kernel_size:])))
        y = self.unfold(unf_x)
        return y

class _stride_fold_(nn.Cell):
    """ipt"""

    def __init__(
            self, kernel_size, output_shape=(-1, -1), stride=-1):

        super(_stride_fold_, self).__init__()

        if isinstance(kernel_size, (list, tuple)):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size, kernel_size]

        if stride == -1:
            self.stride = kernel_size[0]
        else:
            self.stride = stride

        self.output_shape = output_shape
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.fold = _fold_(kernel_size)

    def construct(self, x):
        """ipt"""
        cc_2 = P.Concat(axis=2)
        cc_3 = P.Concat(axis=3)
        zeroslike = P.ZerosLike()
        if self.output_shape[0] == -1:
            large_x = self.fold(x)
            N, C, H, _ = large_x.shape
            leftup_idx = []
            for i in range(0, H, self.kernel_size[0]):
                leftup_idx.append(i)
            NumBlock = len(leftup_idx)
            fold_x = P.Zeros()((N, C, (NumBlock - 1) * self.stride + self.kernel_size[0],
                                (NumBlock - 1) * self.stride + self.kernel_size[0]), mstype.float32)

            for i in range(NumBlock):
                for j in range(NumBlock):
                    fold_i = i * self.stride
                    fold_j = j * self.stride
                    org_i = leftup_idx[i]
                    org_j = leftup_idx[j]
                    fills = large_x[:, :, org_i:org_i + self.kernel_size[0],
                                    org_j:org_j + self.kernel_size[1]]
                    fold_x += cc_3((cc_3((zeroslike(fold_x[:, :, :, :fold_j]), cc_2((cc_2((zeroslike(fold_x[:, :, :fold_i, fold_j:fold_j + self.kernel_size[1]]), fills)), zeroslike(fold_x[:, :, fold_i  + self.kernel_size[0]:, fold_j:fold_j + self.kernel_size[1]]))))), zeroslike(fold_x[:, :, :, fold_j + self.kernel_size[1]:]))) #pylint: disable=line-too-long
            y = fold_x
        else:
            NumBlock_x = int(
                (self.output_shape[0] - self.kernel_size[0]) / self.stride + 1)
            NumBlock_y = int(
                (self.output_shape[1] - self.kernel_size[1]) / self.stride + 1)
            large_shape = [NumBlock_x * self.kernel_size[0],
                           NumBlock_y * self.kernel_size[1]]
            self.fold = _fold_(self.kernel_size, large_shape)
            large_x = self.fold(x)
            N, C, H, _ = large_x.shape
            leftup_idx_x = []
            leftup_idx_y = []
            for i in range(NumBlock_x):
                leftup_idx_x.append(i * self.kernel_size[0])
            for i in range(NumBlock_y):
                leftup_idx_y.append(i * self.kernel_size[1])
            fold_x = P.Zeros()((N, C, (NumBlock_x - 1) * self.stride + self.kernel_size[0],
                                (NumBlock_y - 1) * self.stride + self.kernel_size[1]), mstype.float32)
            for i in range(NumBlock_x):
                for j in range(NumBlock_y):
                    fold_i = i * self.stride
                    fold_j = j * self.stride
                    org_i = leftup_idx_x[i]
                    org_j = leftup_idx_y[j]
                    fills = large_x[:, :, org_i:org_i + self.kernel_size[0],
                                    org_j:org_j + self.kernel_size[1]]
                    fold_x += cc_3((cc_3((zeroslike(fold_x[:, :, :, :fold_j]), cc_2((cc_2((zeroslike(fold_x[:, :, :fold_i, fold_j:fold_j + self.kernel_size[1]]), fills)), zeroslike(fold_x[:, :, fold_i + self.kernel_size[0]:, fold_j:fold_j + self.kernel_size[1]]))))), zeroslike(fold_x[:, :, :, fold_j + self.kernel_size[1]:]))) #pylint: disable=line-too-long
            y = fold_x
        return y


class _unfold_(nn.Cell):
    """ipt"""

    def __init__(
            self, kernel_size, stride=-1):

        super(_unfold_, self).__init__()
        if stride == -1:
            self.stride = kernel_size
        self.kernel_size = kernel_size

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, x):
        """ipt"""
        N, C, H, W = x.shape
        numH = int(H / self.kernel_size)
        numW = int(W / self.kernel_size)
        if numH * self.kernel_size != H or numW * self.kernel_size != W:
            x = x[:, :, :numH * self.kernel_size, :, numW * self.kernel_size]
        output_img = self.reshape(x, (N, C, numH, self.kernel_size, W))

        output_img = self.transpose(output_img, (0, 1, 2, 4, 3))

        output_img = self.reshape(output_img, (N, C, int(
            numH * numW), self.kernel_size, self.kernel_size))

        output_img = self.transpose(output_img, (0, 2, 1, 4, 3))

        output_img = self.reshape(output_img, (N, int(numH * numW), -1))
        return output_img


class _fold_(nn.Cell):
    """ipt"""

    def __init__(
            self, kernel_size, output_shape=(-1, -1), stride=-1):

        super(_fold_, self).__init__()

        if isinstance(kernel_size, (list, tuple)):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size, kernel_size]

        if stride == -1:
            self.stride = self.kernel_size[0]
        self.output_shape = output_shape

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, x):
        """ipt"""
        N, C, L = x.shape
        org_C = int(L / self.kernel_size[0] / self.kernel_size[1])
        if self.output_shape[0] == -1:
            numH = int(np.sqrt(C))
            numW = int(np.sqrt(C))
            org_H = int(numH * self.kernel_size[0])
            org_W = org_H
        else:
            org_H = int(self.output_shape[0])
            org_W = int(self.output_shape[1])
            numH = int(org_H / self.kernel_size[0])
            numW = int(org_W / self.kernel_size[1])

        output_img = self.reshape(
            x, (N, C, org_C, self.kernel_size[0], self.kernel_size[1]))

        output_img = self.transpose(output_img, (0, 2, 1, 3, 4))
        output_img = self.reshape(
            output_img, (N, org_C, numH, numW, self.kernel_size[0], self.kernel_size[1]))

        output_img = self.transpose(output_img, (0, 1, 2, 4, 3, 5))

        output_img = self.reshape(output_img, (N, org_C, org_H, org_W))
        return output_img
