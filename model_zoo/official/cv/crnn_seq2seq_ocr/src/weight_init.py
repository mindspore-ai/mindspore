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
weights initialization
"""
import math
import numpy as np
from mindspore import Tensor, Parameter


def lstm_default_state(batch_size, hidden_size, bidirectional, num_layers=1):
    """init default input."""
    num_directions = 2 if bidirectional else 1
    h = Tensor(np.zeros((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32))
    c = Tensor(np.zeros((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32))
    return h, c


def gru_default_state(input_size, hidden_size):
    stdv = 1 / math.sqrt(hidden_size)
    weight_i = Parameter(Tensor(np.random.uniform(-stdv, stdv, (input_size, 3*hidden_size)).astype(np.float32)),
                         name='weight_i')
    weight_h = Parameter(Tensor(np.random.uniform(-stdv, stdv, (input_size, 3*hidden_size)).astype(np.float32)),
                         name='weight_h')
    bias_i = Parameter(Tensor(np.random.uniform(-stdv, stdv, (3*hidden_size)).astype(np.float32)),
                       name='bias_i')
    bias_h = Parameter(Tensor(np.random.uniform(-stdv, stdv, (3*hidden_size)).astype(np.float32)),
                       name='bias_h')
    return weight_i, weight_h, bias_i, bias_h
