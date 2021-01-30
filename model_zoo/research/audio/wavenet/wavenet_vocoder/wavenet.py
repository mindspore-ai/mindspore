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
"""WaveNet construction"""
from __future__ import with_statement, print_function, absolute_import

import math
import numpy as np

from mindspore import nn, Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from wavenet_vocoder import upsample
from .modules import Embedding
from .modules import Conv1d1x1
from .modules import ResidualConv1dGLU
from .mixture import sample_from_discretized_mix_logistic
from .mixture import sample_from_mix_gaussian
from .mixture import sample_from_mix_onehotcategorical


class WaveNet(nn.Cell):
    """
    WaveNet model definition. Only local condition is supported

    Args:
        out_channels (int): Output channels. If input_type is mu-law quantized one-hot vecror, it should equal to the
        quantize channels. Otherwise, it equals to num_mixtures x 3. Default: 256.
        layers (int): Number of ResidualConv1dGLU layers
        stacks (int): Number of dilation cycles
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        skip_out_channels (int): Skip connection channels.
        kernel_size (int): Kernel size .
        dropout (float): Dropout rate.
        cin_channels (int): Local conditioning channels. If given negative value, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If given negative value, global conditioning is disabled.
        n_speakers (int): Number of speakers. This is used when global conditioning is enabled.
        upsample_conditional_features (bool): Whether upsampling local conditioning features by resize_nearestneighbor
            and conv or not.
        scalar_input (Bool): If True, scalar input ([-1, 1]) is expected, otherwise, quantized one-hot vector
            is expected.
        use_speaker_embedding (Bool): Use speaker embedding or Not.

    """

    def __init__(self, out_channels=256, layers=20, stacks=2,
                 residual_channels=512,
                 gate_channels=512,
                 skip_out_channels=512,
                 kernel_size=3, dropout=1 - 0.95,
                 cin_channels=-1, gin_channels=-1, n_speakers=None,
                 upsample_conditional_features=False,
                 upsample_net="ConvInUpsampleNetwork",
                 upsample_params=None,
                 scalar_input=False,
                 use_speaker_embedding=False,
                 output_distribution="Logistic",
                 cin_pad=0,
                 ):
        super(WaveNet, self).__init__()
        self.transpose_op = P.Transpose()
        self.softmax = P.Softmax(axis=1)
        self.reshape_op = P.Reshape()
        self.zeros_op = P.Zeros()
        self.ones_op = P.Ones()
        self.relu_op = P.ReLU()
        self.squeeze_op = P.Squeeze()
        self.expandim_op = P.ExpandDims()
        self.transpose_op = P.Transpose()
        self.tile_op = P.Tile()
        self.scalar_input = scalar_input
        self.out_channels = out_channels
        self.cin_channels = cin_channels
        self.output_distribution = output_distribution
        self.fack_data = P.Zeros()
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        if scalar_input:
            self.first_conv = Conv1d1x1(1, residual_channels)
        else:
            self.first_conv = Conv1d1x1(out_channels, residual_channels)

        conv_layers = []
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualConv1dGLU(
                residual_channels, gate_channels,
                kernel_size=kernel_size,
                skip_out_channels=skip_out_channels,
                bias=True,
                dropout=dropout,
                dilation=dilation,
                cin_channels=cin_channels,
                gin_channels=gin_channels)
            conv_layers.append(conv)
        self.conv_layers = nn.CellList(conv_layers)
        self.last_conv_layers = nn.CellList([
            nn.ReLU(),
            Conv1d1x1(skip_out_channels, skip_out_channels),
            nn.ReLU(),
            Conv1d1x1(skip_out_channels, out_channels)])

        if gin_channels > 0 and use_speaker_embedding:
            assert n_speakers is not None
            self.embed_speakers = Embedding(
                n_speakers, gin_channels, padding_idx=None, std=0.1)
        else:
            self.embed_speakers = None

        if upsample_conditional_features:
            self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
        else:
            self.upsample_net = None

        self.factor = math.sqrt(1.0 / len(self.conv_layers))

    def _expand_global_features(self, batch_size, time_step, g_fp, is_expand=True):
        """Expand global conditioning features to all time steps

        Args:
            batch_size (int): Batch size.
            time_step (int): Time length.
            g_fp (Tensor): Global features, (B x C) or (B x C x 1).
            is_expand (bool) : Expanded global conditioning features

        Returns:
            Tensor: B x C x T or B x T x C or None
        """
        if g_fp is None:
            return None
        if len(g_fp.shape) == 2:
            g_fp = self.expandim_op(g_fp, -1)
        else:
            g_fp = g_fp

        if is_expand:
            expand_fp = self.tile_op(g_fp, (batch_size, 1, time_step))
            return expand_fp
        expand_fp = self.tile_op(g_fp, (batch_size, 1, time_step))
        expand_fp = self.transpose_op(expand_fp, (0, 2, 1))
        return expand_fp

    def construct(self, x, c=None, g=None, softmax=False):
        """

        Args:
            x (Tensor): One-hot encoded audio signal
            c (Tensor): Local conditioning feature
            g (Tensor): Global conditioning feature
            softmax (bool): Whether use softmax or not

        Returns:
            Tensor: Net output

        """
        g = None
        B, _, T = x.shape
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(self.reshape_op(g, (B, -1)))
                g = self.transpose_op(g, (0, 2, 1))
        g_bct = self._expand_global_features(B, T, g, is_expand=True)

        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)

        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c, g_bct)
            skips += h
        skips *= self.factor

        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        x = self.softmax(x) if softmax else x

        return x

    def relu_numpy(self, inX):
        """numpy relu function"""
        return np.maximum(0, inX)

    def softmax_numpy(self, x):
        """ numpy softmax function """
        x -= np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def incremental_forward(self, initial_input=None, c=None, g=None,
                            T=100, test_inputs=None,
                            tqdm=lambda x: x, softmax=True, quantize=True,
                            log_scale_min=-50.0, is_numpy=True):
        """
        Incremental forward. Current output depends on last output.

        Args:
            initial_input (Tensor): Initial input, the shape is B x C x 1
            c (Tensor): Local conditioning feature, the shape is B x C x T
            g (Tensor): Global conditioning feature, the shape is B x C or B x C x 1
            T (int): decoding time step.
            test_inputs: Teacher forcing inputs (for debugging)
            tqdm (lamda): tqmd
            softmax (bool): Whether use softmax or not
            quantize (bool): Whether quantize softmax output in last step when decoding current step
            log_scale_min (float): Log scale minimum value

        Returns:
             Tensor: Predicted on-hot encoded samples or scalar vector depending on loss type

        """

        self.clear_buffer()
        B = 1

        if test_inputs is not None:
            if self.scalar_input:
                if test_inputs.shape[1] == 1:
                    test_inputs = self.transpose_op(test_inputs, (0, 2, 1))
            else:
                if test_inputs.shape[1] == self.out_channels:
                    test_inputs = self.transpose_op(test_inputs, (0, 2, 1))

            B = test_inputs.shape[0]
            if T is None:
                T = test_inputs.shape[1]
            else:
                T = max(T, test_inputs.shape[1])
        T = int(T)

        # Global conditioning
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(self.reshape_op(g, (B, -1)))
                g = self.transpose_op(g, (0, 2, 1))
                assert g.dim() == 3
        g_btc = self._expand_global_features(B, T, g, is_expand=False)

        # Local conditioning
        if c is not None:
            B = c.shape[0]
            if self.upsample_net is not None:
                c = self.upsample_net(c)
                assert c.shape[-1] == T
            if c.shape[-1] == T:
                c = self.transpose_op(c, (0, 2, 1))

        outputs = []
        if initial_input is None:
            if self.scalar_input:
                initial_input = self.zeros_op((B, 1, 1), mstype.float32)
            else:
                initial_input = np.zeros((B, 1, self.out_channels), np.float32)
                initial_input[:, :, 127] = 1
                initial_input = Tensor(initial_input)
        else:
            if initial_input.shape[1] == self.out_channels:
                initial_input = self.transpose_op(initial_input, (0, 2, 1))

        if is_numpy:
            current_input = initial_input.asnumpy()
        else:
            current_input = initial_input

        for t in tqdm(range(T)):
            if test_inputs is not None and t < test_inputs.shape[1]:
                current_input = self.expandim_op(test_inputs[:, t, :], 1)
            else:
                if t > 0:
                    if not is_numpy:
                        current_input = Tensor(outputs[-1])
                    else:
                        current_input = outputs[-1]

            # Conditioning features for single time step
            ct = None if c is None else self.expandim_op(c[:, t, :], 1)
            gt = None if g is None else self.expandim_op(g_btc[:, t, :], 1)

            x = current_input

            if is_numpy:
                ct = ct.asnumpy()
            x = self.first_conv.incremental_forward(x, is_numpy=is_numpy)

            skips = 0
            for f in self.conv_layers:
                x, h = f.incremental_forward(x, ct, gt, is_numpy=is_numpy)
                skips += h
            skips *= self.factor
            x = skips

            for f in self.last_conv_layers:
                try:
                    x = f.incremental_forward(x, is_numpy=is_numpy)
                except AttributeError:
                    if is_numpy:
                        x = self.relu_numpy(x)
                    else:
                        x = self.relu_op(x)

            # Generate next input by sampling
            if not is_numpy:
                x = x.asnumpy()
            if self.scalar_input:
                if self.output_distribution == "Logistic":
                    x = sample_from_discretized_mix_logistic(x.reshape((B, -1, 1)), log_scale_min=log_scale_min)

                elif self.output_distribution == "Normal":
                    x = sample_from_mix_gaussian(x.reshape((B, -1, 1)), log_scale_min=log_scale_min)
                else:
                    assert False
            else:
                x = self.softmax_numpy(np.reshape(x, (B, -1))) if softmax else np.reshape(x, (B, -1))
                if quantize:
                    x = sample_from_mix_onehotcategorical(x)

            outputs += [x]
        # T x B x C
        outputs = np.stack(outputs, 0)
        # B x C x T
        outputs = np.transpose(outputs, (1, 2, 0))
        self.clear_buffer()
        return outputs

    def clear_buffer(self):
        """clear buffer"""
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass
