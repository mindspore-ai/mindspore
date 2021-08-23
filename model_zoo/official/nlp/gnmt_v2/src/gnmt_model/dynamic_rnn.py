# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""DynamicRNN."""
import numpy as np

import mindspore.ops.operations as P
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor


class DynamicRNNCell(nn.Cell):
    """
    DynamicRNN Cell.

    Args:
        num_setp (int): Lengths of setences.
        batch_size (int): Batch size.
        word_embed_dim (int): Input size.
        hidden_size (int): Hidden size .
        initializer_range (float): Initial range. Default: 0.02
    """

    def __init__(self,
                 num_setp=50,
                 batch_size=128,
                 word_embed_dim=1024,
                 hidden_size=1024,
                 initializer_range=0.1):
        super(DynamicRNNCell, self).__init__()
        self.num_step = num_setp
        self.batch_size = batch_size
        self.input_size = word_embed_dim
        self.hidden_size = hidden_size
        # w
        dynamicRNN_w = np.random.uniform(-initializer_range, initializer_range,
                                         size=[self.input_size + self.hidden_size, 4 * self.hidden_size])
        self.dynamicRNN_w = Parameter(Tensor(dynamicRNN_w, mstype.float32))
        # b
        dynamicRNN_b = np.random.uniform(-initializer_range, initializer_range, size=[4 * self.hidden_size])
        self.dynamicRNN_b = Parameter(Tensor(dynamicRNN_b, mstype.float32))

        self.dynamicRNN_h = Tensor(np.zeros((1, self.batch_size, self.hidden_size)), mstype.float32)
        self.dynamicRNN_c = Tensor(np.zeros((1, self.batch_size, self.hidden_size)), mstype.float32)
        self.cast = P.Cast()
        self.is_ascend = context.get_context("device_target") == "Ascend"
        if self.is_ascend:
            self.compute_type = mstype.float16
            self.rnn = P.DynamicRNN()
        else:
            self.compute_type = mstype.float32
            self.lstm = nn.LSTM(self.input_size,
                                self.hidden_size,
                                num_layers=1,
                                has_bias=True,
                                batch_first=False,
                                dropout=0,
                                bidirectional=False)

    def construct(self, x, init_h=None, init_c=None):
        """DynamicRNNCell Network."""
        if init_h is None or init_c is None:
            init_h = self.cast(self.dynamicRNN_h, self.compute_type)
            init_c = self.cast(self.dynamicRNN_c, self.compute_type)
        if self.is_ascend:
            w = self.cast(self.dynamicRNN_w, self.compute_type)
            b = self.cast(self.dynamicRNN_b, self.compute_type)
            output, hn, cn = self.rnn(x, w, b, None, init_h, init_c)
        else:
            output, (hn, cn) = self.lstm(x, (init_h, init_c))
        return output, hn, cn


class DynamicRNNNet(nn.Cell):
    """
    DynamicRNN Network.

    Args:
        seq_length (int): Lengths of setences.
        batchsize (int): Batch size.
        word_embed_dim (int): Input size.
        hidden_size (int): Hidden size.
    """

    def __init__(self,
                 seq_length=80,
                 batchsize=128,
                 word_embed_dim=1024,
                 hidden_size=1024):
        super(DynamicRNNNet, self).__init__()
        self.max_length = seq_length
        self.hidden_size = hidden_size
        self.cast = P.Cast()
        self.concat = P.Concat(axis=0)
        self.get_shape = P.Shape()
        self.net = DynamicRNNCell(num_setp=seq_length,
                                  batch_size=batchsize,
                                  word_embed_dim=word_embed_dim,
                                  hidden_size=hidden_size)
        self.is_ascend = context.get_context("device_target") == "Ascend"
        if self.is_ascend:
            self.compute_type = mstype.float16
        else:
            self.compute_type = mstype.float32

    def construct(self, inputs, init_state=None):
        """DynamicRNN Network."""
        inputs = self.cast(inputs, self.compute_type)
        if init_state is not None:
            init_h = self.cast(init_state[0:1, :, :], self.compute_type)
            init_c = self.cast(init_state[-1:, :, :], self.compute_type)
            out, state_h, state_c = self.net(inputs, init_h, init_c)
        else:
            out, state_h, state_c = self.net(inputs)
        out = self.cast(out, mstype.float32)
        state = self.concat((state_h[-1:, :, :], state_c[-1:, :, :]))
        state = self.cast(state, mstype.float32)
        # out:[T,b,D], state:[2,b,D]
        return out, state
