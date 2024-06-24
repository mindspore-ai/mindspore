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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetFloorDiv(nn.Cell):
    def __init__(self):
        super(NetFloorDiv, self).__init__()
        self.floordiv = P.FloorDiv()

    def construct(self, x, y):
        return self.floordiv(x, y)


@pytest.mark.skip(reason="never run on ci or smoke test")
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32,
                                   np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
def testtype_floor_div_int_float(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for FloorDiv
    Expectation: the result match to numpy
    """
    x_np = np.random.rand(1, 5).astype(dtype)
    y_np = np.random.rand(1, 5).astype(dtype)
    expect = np.floor_divide(x_np, y_np)
    x_input = Tensor(x_np)
    y_input = Tensor(y_np)
    floor_div = NetFloorDiv()
    output = floor_div(x_input, y_input)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.skip(reason="never run on ci or smoke test")
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def testtype_floor_div_complex(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for FloorDiv
    Expectation: the result match to numpy
    """
    x_np = np.random.rand(1, 5).astype(dtype)
    x_np = x_np + 0.5j * x_np
    y_np = np.random.rand(1, 5).astype(dtype)
    y_np = y_np + 0.4j * y_np
    expect = np.floor_divide(x_np, y_np)
    x_input = Tensor(x_np)
    y_input = Tensor(y_np)
    floor_div = NetFloorDiv()
    output = floor_div(x_input, y_input)
    assert np.allclose(output.asnumpy(), expect)
