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
"""
modules for wavenet
"""
from __future__ import with_statement, print_function, absolute_import
import math
import numpy as np
from wavenet_vocoder import conv
from mindspore import nn
from mindspore.ops import operations as P


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    m = conv.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    return m


def Conv1d1x1(in_channels, out_channels, has_bias=True):
    return Conv1d(in_channels, out_channels, kernel_size=1, pad_mode='pad', padding=0, dilation=1, has_bias=has_bias)


def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    return m


def _conv1x1_forward(conv_, x, is_incremental, is_numpy=True):
    """
    Conv1x1 forward
    """
    if is_incremental:
        x = conv_.incremental_forward(x, is_numpy=is_numpy)
    else:
        x = conv_(x)
    return x


class ResidualConv1dGLU(nn.Cell):
    """Residual dilated conv1d with gated activation units

    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size
        skip_out_channels (int): Skip connection channels. If None, it will set to the same as residual_channels.
        cin_channels (int): Local conditioning channels. If given negative value, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If given negative value, global conditioning is disabled.
        dropout (float): Dropout rate.
        padding (int): Padding for convolution layers. If None, padding value will be computed according to dilation
        and kernel_size.
        dilation (int): Dilation factor.

    """

    def __init__(self, residual_channels=None, gate_channels=None, kernel_size=None, skip_out_channels=None, bias=True,
                 dropout=1 - 0.95, dilation=1, cin_channels=-1, gin_channels=-1, padding=None, causal=True):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        self.dropout_op = nn.Dropout(keep_prob=1. - self.dropout)
        self.eval_split_op = P.Split(axis=-1, output_num=2)
        self.train_split_op = P.Split(axis=1, output_num=2)
        self.tanh = P.Tanh()
        self.sigmoid = P.Sigmoid()
        self.mul = P.Mul()
        self.add = P.Add()

        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        self.conv = Conv1d(residual_channels, gate_channels, kernel_size, pad_mode='pad',
                           padding=padding, dilation=dilation, has_bias=bias)

        # local conditioning
        if cin_channels > 0:
            self.conv1x1c = Conv1d1x1(cin_channels, gate_channels, has_bias=False)
        else:
            self.conv1x1c = None

        # global conditioning
        if gin_channels > 0:
            self.conv1x1g = Conv1d(gin_channels, gate_channels, has_bias=False, kernel_size=1, dilation=1)
        else:
            self.conv1x1g = None

        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, has_bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels, has_bias=bias)
        self.factor = math.sqrt(0.5)

    def construct(self, x, c=None, g=None):
        """

        Args:
            x(Tensor): One-hot audio signal, the shape is B x C x T
            c(Tensor): local conditional feature, the shape is B x cin_channels x T
            g(Tensor): global conditional feature, not used currently

        Returns:
            Tensor: Output tensor

        """

        residual = x
        x = self.dropout_op(x)
        x = self.conv(x)
        # remove future time steps
        x = x[:, :, :residual.shape[-1]] if self.causal else x
        split_op = self.train_split_op

        a, b = split_op(x)

        # local conditioning
        if c is not None:
            c = _conv1x1_forward(self.conv1x1c, c, is_incremental=False)
            ca, cb = split_op(c)
            a, b = a + ca, b + cb

        # global conditioning
        if g is not None:
            g = _conv1x1_forward(self.conv1x1g, g, is_incremental=False)
            ga, gb = self.split(g)
            a, b = a + ga, b + gb

        x = self.mul(self.tanh(a), self.sigmoid(b))

        # For skip connection
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental=False)

        # For residual connection
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental=False)

        x = self.add(x, residual) * self.factor
        return x, s

    def sigmoid_numpy(self, x):
        return 1. / (1 + np.exp(-x))

    def incremental_forward(self, x, c=None, g=None, is_numpy=True):
        """
        Incremental forward. Used for inference stage

        Args:
            x (Tensor): One-hot audio signal, the shape is B x C x T
            c (Tensor): local conditional feature, the shape is B x cin_channels x T
            g (Tensor): global conditional feature, not used currently

        Returns:
            ndarray
        """
        residual = x
        x = self.conv.incremental_forward(x, is_numpy=is_numpy)
        if is_numpy:
            a, b = np.split(x, indices_or_sections=2, axis=-1)
        else:
            a, b = self.eval_split_op(x)

        # local conditioning
        if c is not None:
            c = _conv1x1_forward(self.conv1x1c, c, is_incremental=True, is_numpy=is_numpy)
            if is_numpy:
                ca, cb = np.split(c, indices_or_sections=2, axis=-1)
            else:
                ca, cb = self.eval_split_op(c)
            a, b = a + ca, b + cb

        # global conditioning
        if g is not None:
            g = _conv1x1_forward(self.conv1x1g, g, is_incremental=True, is_numpy=is_numpy)
            if is_numpy:
                ga, gb = np.split(g, indices_or_sections=2, axis=-1)
            else:
                ga, gb = self.eval_split_op(c)
            a, b = a + ga, b + gb

        if is_numpy:
            x = np.tanh(a) * self.sigmoid_numpy(b)
        else:
            x = self.mul(self.tanh(a), self.sigmoid(b))

        # For skip connection
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental=True, is_numpy=is_numpy)

        # For residual connection
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental=True, is_numpy=is_numpy)

        x = (x + residual) * self.factor
        return x, s

    def clear_buffer(self):
        """clear buffer"""
        for c in [self.conv, self.conv1x1_out, self.conv1x1_skip,
                  self.conv1x1c, self.conv1x1g]:
            if c is not None:
                c.clear_buffer()
