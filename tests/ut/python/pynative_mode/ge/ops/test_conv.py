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
""" test_conv """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from ....ut_filter import non_graph_engine

we = Tensor(np.ones([2, 2]))
in_channels = 3
out_channels = 64
ks = 3


def get_me_conv_output(input_data, weight, in_channel, out_channel, kernel_size,
                       stride=1, padding=0, has_bias=False, bias=None):
    """ get_me_conv_output """

    class Net(nn.Cell):
        """ Net definition """

        def __init__(self, weight, in_channel, out_channel, kernel_size,
                     stride=1, padding=0, has_bias=False, bias=None):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(in_channels=in_channel,
                                  out_channels=out_channel,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  has_bias=has_bias,
                                  weight_init=weight,
                                  bias_init=bias)

        def construct(self, input_x):
            return self.conv(input_x)

    net = Net(weight, in_channel, out_channel, kernel_size, stride, padding, has_bias, bias)
    out = net.construct(input_data)
    return out.asnumpy()


@non_graph_engine
def test_ge_conv():
    """ test_ge_conv """
    input_data = np.random.randn(2, 3, 244, 244).astype(np.float32)
    kernel = np.random.randn(6, 3, 7, 7).astype(np.float32)
    out = get_me_conv_output(Tensor(input_data), Tensor(kernel), in_channel=3,
                             out_channel=6, kernel_size=7, stride=7, padding=0)
    print(out)


@non_graph_engine
def test_ge_conv_with_bias():
    """ test_ge_conv_with_bias """
    input_data = np.random.randn(2, 3, 244, 244).astype(np.float32)
    kernel = np.random.randn(6, 3, 7, 7).astype(np.float32)
    np.random.randn(2, 6, 35, 35).astype(np.float32)
    out = get_me_conv_output(Tensor(input_data), Tensor(kernel), in_channel=3,
                             out_channel=6, kernel_size=7, stride=7, padding=0)
    print(out)
