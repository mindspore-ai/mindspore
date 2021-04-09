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
"""Extended Conv1D."""

import math
import numpy as np
from mindspore import nn, Tensor
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore import context

class Conv1d(nn.Conv1d):
    """
    Extended nn.Conv1d to adapt to incremental dilated convolutions.
    During training, initial Conv1D is used and during evaluation, incremental_forward is called.
    To improve the inference speed, tensor will be converted as numpy and the following calculation is based on numpy.
    These operation will be replaced with MindSpore ops in the future. Currently, some operation is not supported by
    MindSpore and a mixed use of numpy and MindSpore will take a long time.

    """

    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        self.transpose_op = P.Transpose()
        self.reshape_op = P.Reshape()
        self.squeeze_op = P.Squeeze(-2)
        self.zeros = P.Zeros()
        self.concat_op = P.Concat(axis=1)
        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()
        self.get_weight = None
        self.get_bias = None

    def incremental_forward(self, inputs, is_numpy=True):
        if is_numpy:
            return self.incremental_forward_numpy(inputs)
        return self.incremental_forward_pynative(inputs)

    def incremental_forward_pynative(self, inputs):
        """
        Incremental forward.

        Args:
            inputs: B x T x C

        Returns:
            ndarray

        """
        # input: (B, T, C)
        if self.training:
            raise RuntimeError('incremental_forward only supports eval mode')

        if self.get_weight is None:
            self.get_weight = self._get_linearized_weight()

        if self.get_bias is None and self.bias is not None:
            self.get_bias = self.bias

        # Note mindspore uses Conv2D to construct Conv1D
        kw = self.kernel_size[1]
        dilation = self.dilation[1]

        bsz = inputs.shape[0]  # input: bsz x len x dim
        if kw > 1:
            if self.input_buffer is None:
                init_buffer = self.zeros((bsz, kw + (kw - 1) * (dilation - 1), inputs.shape[2]), mstype.float32)
                self.input_buffer = self.concat_op((init_buffer[:, 1:, :], inputs[:, 0:1, :]))
            else:
                # shift buffer
                self.input_buffer = self.concat_op((self.input_buffer[:, 1:, :], inputs[:, 0:1, :]))
            inputs = self.input_buffer
            if dilation > 1:
                if context.get_context("device_target") == "CPU":
                    inputs = self.transpose_op(inputs, (1, 0, 2))
                    inputs = inputs[0::dilation, :, :]
                    inputs = self.transpose_op(inputs, (1, 0, 2))
                else:
                    inputs = inputs[:, 0::dilation, :]

        output = self.matmul(self.reshape_op(inputs, (bsz, -1)), self.get_weight)
        if self.bias is not None:
            output = self.bias_add(output, self.bias)
        return self.reshape_op(output, (bsz, 1, -1))

    def incremental_forward_numpy(self, inputs):
        """
        Incremental forward.

        Args:
            inputs: B x T x C

        Returns:
            ndarray

        """
        # input: (B, T, C)
        if self.training:
            raise RuntimeError('incremental_forward only supports eval mode')

        if self.get_weight is None:
            weight = self._get_linearized_weight()
            self.get_weight = weight.asnumpy()

        if self.get_bias is None and self.bias is not None:
            bias = self.bias
            self.get_bias = bias.asnumpy()

        # Note mindspore uses Conv2D to construct Conv1D
        kw = self.kernel_size[1]
        dilation = self.dilation[1]

        bsz = inputs.shape[0]  # input: bsz x len x dim
        if kw > 1:
            if self.input_buffer is None:
                self.input_buffer = np.zeros((bsz, kw + (kw - 1) * (dilation - 1), inputs.shape[2]), dtype=np.float32)
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :]
            # append next
            self.input_buffer[:, -1, :] = inputs[:, -1, :]
            inputs = self.input_buffer
            if dilation > 1:
                inputs = inputs[:, 0::dilation, :]
        output = inputs.reshape(bsz, -1).dot(self.get_weight.T)
        if self.bias is not None:
            output = output + np.expand_dims(self.get_bias, 0)
        return np.reshape(output, (bsz, 1, -1))

    def clear_buffer(self):
        self.input_buffer = None

    def _get_linearized_weight(self):
        """
        get linearized weight
        """
        weight = self.squeeze_op(self.weight)
        if self._linearized_weight is None:
            # Note mindspore uses Conv2D to construct Conv1D
            kw = self.kernel_size[1]
            if weight.shape == (self.out_channels, self.in_channels, kw):
                weight = self.transpose_op(weight, (0, 2, 1))
            else:
                weight = self.transpose_op(weight, (2, 0, 1))
            self._linearized_weight = self.reshape_op(weight, (self.out_channels, -1))
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None

    def _initialize_weights(self):
        """
        weight initialization
        """
        self.init_parameters_data()
        std_mul = 4.0
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv1d):
                std = math.sqrt((std_mul * 0.1) / (m.kernel_size[1] * self.in_channels))
                m.weight.set_data(Tensor(np.random.normal(0, std, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(
                    Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
