# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


class NetConv2d(nn.Cell):
    "NetConv2d"
    def __init__(self):
        super(NetConv2d, self).__init__()
        out_channel = 2
        kernel_size = 1
        self.conv = P.Conv2D(out_channel,
                             kernel_size,
                             mode=1,
                             pad_mode="valid",
                             pad=0,
                             stride=1,
                             dilation=1,
                             group=1)

    def construct(self, x, w):
        return self.conv(x, w)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conv2d_acl():
    """
    Feature: Test conv2d op with acl
    Description: Test conv2d op with acl
    Expectation: The value is processed as expected
    """
    x = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))
    w = Tensor(np.arange(2 * 3 * 1 * 1).reshape(2, 3, 1, 1).astype(np.float32))
    expect = np.array([[[[45, 48, 51],
                         [54, 57, 60],
                         [63, 66, 69]],
                        [[126, 138, 150],
                         [162, 174, 186],
                         [198, 210, 222]]]]).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    conv2d = NetConv2d()
    dynamic_x = Tensor(shape=[1, 3, None, None], dtype=mindspore.float32)
    dynamic_w = Tensor(shape=[2, 3, None, None], dtype=mindspore.float32)
    conv2d.set_inputs(dynamic_x, dynamic_w)
    output = conv2d(x, w)
    assert (output.asnumpy() == expect).all()


class NetConv2dWithGroup(nn.Cell):
    "NetConv2dWithGroup"
    def __init__(self):
        super(NetConv2dWithGroup, self).__init__()
        out_channel = 64
        kernel_size = 3
        self.conv = P.Conv2D(out_channel,
                             kernel_size,
                             mode=1,
                             pad_mode="valid",
                             pad=0,
                             stride=1,
                             dilation=1,
                             group=1)

    def construct(self, x, w):
        return self.conv(x, w)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_conv2d_with_groupacl():
    """
    Feature: Test conv2d op with acl
    Description: Test conv2d op with acl
    Expectation: The value is processed as expected
    """
    x = Tensor(np.arange(256 * 3 * 3 * 3).reshape(256, 3, 3, 3).astype(np.float32))
    w = Tensor(np.arange(64 * 3 * 3 * 3).reshape(64, 3, 3, 3).astype(np.float32))
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    conv2d = NetConv2dWithGroup()
    dynamic_x = Tensor(shape=[256, 3, None, None], dtype=mindspore.float32)
    dynamic_w = Tensor(shape=[64, 3, None, None], dtype=mindspore.float32)
    conv2d.set_inputs(dynamic_x, dynamic_w)
    output = conv2d(x, w)
    print(output)
