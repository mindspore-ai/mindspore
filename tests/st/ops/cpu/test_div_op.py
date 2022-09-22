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
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class NetDiv(nn.Cell):
    def __init__(self):
        super(NetDiv, self).__init__()
        self.div = P.Div()

    def construct(self, x, y):
        return self.div(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_two_tensors_add():
    """
    Feature: ALL To ALL
    Description: test cases for Div of two tensors
    Expectation: the result match to numpy
    """
    x = np.arange(2 * 3 * 2).reshape((2, 3, 2))
    y = np.arange(88, 2 * 3 * 2 + 88).reshape((2, 3, 2))
    dtypes = (np.int8, np.int16, np.int32, np.int64, np.float16,
              np.float32, np.float64, np.uint16, np.uint32, np.uint64)
    for dtype in dtypes:
        output = Tensor(x.astype(dtype)) / Tensor(y.astype(dtype))
        expect_result = (x / y).astype(dtype)
        assert output.asnumpy().dtype == expect_result.dtype
        assert np.array_equal(output.asnumpy(), expect_result)

    # Test for dynamic shape of div.
    input_x_dyn = Tensor(shape=[2, None, 2], dtype=mstype.float32)
    input_y_dyn = Tensor(shape=[2, 3, None], dtype=mstype.float32)
    div_dyn_net = NetDiv()
    div_dyn_net.set_inputs(input_x_dyn, input_y_dyn)
    dyn_output = div_dyn_net(Tensor(x.astype(np.float32)), Tensor(y.astype(np.float32)))
    expect_dync_result = (x / y).astype(np.float32)
    assert np.array_equal(dyn_output.asnumpy(), expect_dync_result)
    