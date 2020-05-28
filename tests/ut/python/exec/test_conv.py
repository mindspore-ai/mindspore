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
"""test conv"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from ..ut_filter import non_graph_engine

weight = Tensor(np.ones([2, 2]))
in_channels = 3
out_channels = 64


class Net(nn.Cell):
    """Net definition"""

    def __init__(self,
                 cin,
                 cout,
                 kernel_size,
                 stride=1,
                 pad_mode='pad',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros'):
        super(Net, self).__init__()
        Tensor(np.ones([6, 3, 3, 3]).astype(np.float32) * 0.01)
        self.conv = nn.Conv2d(cin,
                              cout,
                              kernel_size,
                              stride,
                              pad_mode,
                              padding,
                              dilation,
                              group,
                              has_bias,
                              weight_init,
                              bias_init)

    def construct(self, input_x):
        return self.conv(input_x)


@non_graph_engine
def test_compile():
    net = Net(3, 6, (3, 3), bias_init='zeros')
    input_data = Tensor(np.ones([3, 3, 32, 32]).astype(np.float32) * 0.01)
    output = net(input_data)
    print(output.asnumpy())


@non_graph_engine
def test_compile2():
    net = Net(3, 1, (3, 3), bias_init='zeros')
    input_data = Tensor(np.ones([1, 3, 32, 32]).astype(np.float32) * 0.01)
    output = net(input_data)
    print(output.asnumpy())


@non_graph_engine
def test_compile3():
    net = Net(3, 1, (3, 3), weight_init='ONES')
    input_data = Tensor(np.ones([1, 3, 32, 32]).astype(np.float32) * 0.01)
    output = net(input_data)
    print(output.asnumpy())
