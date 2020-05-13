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
""" test Dense """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from ..ut_filter import non_graph_engine


class Net(nn.Cell):
    """Net definition"""

    def __init__(self,
                 input_channels,
                 output_channels,
                 weight='normal',
                 bias='zeros',
                 has_bias=True):
        super(Net, self).__init__()
        self.fc = nn.Dense(input_channels,
                           output_channels,
                           weight,
                           bias,
                           has_bias)

    def construct(self, input_x):
        return self.fc(input_x)


@non_graph_engine
def test_compile():
    weight = Tensor(np.ones([12, 8], np.float32))
    bias = Tensor(np.ones([12], np.float32))
    net = Net(8, 12, weight=weight, bias=bias)
    input_data = Tensor(np.ones([1, 8], np.float32))
    # since simulator currently not support matMul
    output = net(input_data)
    print(output.asnumpy())


@non_graph_engine
def test_compile_nobias():
    weight = Tensor(np.ones([12, 8], np.float32))
    net = Net(8, 12, weight=weight, has_bias=False)
    input_data = Tensor(np.ones([1, 8], np.float32))
    # since simulator currently not support matMu
    # enable it when staging function is ready
    output = net(input_data)
    print(output.asnumpy())
