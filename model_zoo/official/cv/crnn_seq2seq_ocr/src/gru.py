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
#" ============================================================================
"""
GRU cell
"""
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype
from src.weight_init import gru_default_state


class GRU(nn.Cell):
    '''
    GRU model

    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
    '''
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_i, self.weight_h, self.bias_i, self.bias_h = gru_default_state(self.input_size, self.hidden_size)
        self.rnn = P.DynamicGRUV2()
        self.cast = P.Cast()

    def construct(self, x, h):
        '''
        GRU construction

        Args:
            x(Tensor): GRU input
            h(Tensor): GRU hidden state

        Returns:
            output(Tensor): rnn output
            hidden(Tensor): hidden state
        '''
        x = self.cast(x, mstype.float16)
        h = self.cast(h, mstype.float16)
        y1, h1, _, _, _, _ = self.rnn(x, self.weight_i, self.weight_h, self.bias_i, self.bias_h, None, h)
        return y1, h1
