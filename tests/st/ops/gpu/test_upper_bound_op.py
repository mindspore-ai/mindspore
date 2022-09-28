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

import numpy as np
import pytest
from mindspore.common import dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.array_ops import UpperBound


class NetUpperBound(nn.Cell):

    def __init__(self, out_type):
        super(NetUpperBound, self).__init__()
        self.upperbound = UpperBound(out_type=out_type)

    def construct(self, x, y):
        return self.upperbound(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_upperbound_2d_input_int32_output_int32():
    """
    Feature: UpperBound gpu TEST.
    Description: 2d test case for UpperBound
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_ms = Tensor(np.array([[0, 3, 9, 9, 10], [1, 2, 3, 4, 5]]).astype(np.int32))
    y_ms = Tensor(np.array([[2, 4, 9], [0, 2, 6]]).astype(np.int32))
    net = NetUpperBound(out_type=mstype.int32)
    z_ms = net(x_ms, y_ms)
    expect = np.array([[1, 2, 4], [0, 2, 5]]).astype(np.int32)

    assert (z_ms.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_upperbound_2d_input_float16_output_int64():
    """
    Feature: UpperBound gpu TEST.
    Description: 2d test case for UpperBound
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_ms = Tensor(np.array([[0, 3, 9, 9, 10], [1, 2, 3, 4, 5]]).astype(np.float16))
    y_ms = Tensor(np.array([[2, 4, 9], [0, 2, 6]]).astype(np.float16))
    net = NetUpperBound(out_type=mstype.int64)
    z_ms = net(x_ms, y_ms)
    expect = np.array([[1, 2, 4], [0, 2, 5]]).astype(np.int64)

    assert (z_ms.asnumpy() == expect).all()
