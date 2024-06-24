# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore.common import dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.array_ops import LowerBound


class NetLowerBound(nn.Cell):

    def __init__(self, out_type):
        super(NetLowerBound, self).__init__()
        self.lowerbound = LowerBound(out_type=out_type)

    def construct(self, x, y):
        return self.lowerbound(x, y)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lowerbound_2d_input_int32_output_int32():
    """
    Feature: LowerBound gpu TEST.
    Description: 2d test case for LowerBound
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_ms = Tensor(np.array([[0, 3, 9, 9, 10], [1, 2, 3, 4, 5]]).astype(np.int32))
    y_ms = Tensor(np.array([[2, 4, 9], [0, 2, 6]]).astype(np.int32))
    net = NetLowerBound(out_type=mstype.int32)
    z_ms = net(x_ms, y_ms)
    expect = np.array([[1, 2, 2], [0, 1, 5]]).astype(np.int32)

    assert (z_ms.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lowerbound_2d_input_float32_output_int64():
    """
    Feature: LowerBound gpu TEST.
    Description: 2d test case for LowerBound
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_ms = Tensor(np.array([[0, 3, 9, 9, 10], [1, 2, 3, 4, 5]]).astype(np.float32))
    y_ms = Tensor(np.array([[2, 4, 9], [0, 2, 6]]).astype(np.float32))
    net = NetLowerBound(out_type=mstype.int64)
    z_ms = net(x_ms, y_ms)
    expect = np.array([[1, 2, 2], [0, 1, 5]]).astype(np.int64)

    assert (z_ms.asnumpy() == expect).all()
