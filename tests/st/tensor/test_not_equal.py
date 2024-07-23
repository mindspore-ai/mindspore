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
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class NotEqualNet(nn.Cell):
    def construct(self, x, other):
        return x.not_equal(other)


class NotEqualNet1(nn.Cell):
    def construct(self, x, other):
        return x != other


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_not_equal(mode):
    """
    Feature: test Tensor.not_equal
    Description: Verify the result of Tensor.not_equal
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode)
    x_np = np.array([1, 2, 3]).astype(np.float32)
    y_np = np.array([1, 2, 4]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    y = Tensor(y_np, ms.float32)
    net = NotEqualNet()
    output_ms_case_1 = net(x, 2.0)
    expect_output_case_1 = np.not_equal(x_np, 2.0)
    output_ms_case_2 = net(x, y)
    expect_output_case_2 = np.not_equal(x_np, y_np)
    np.testing.assert_array_equal(output_ms_case_1.asnumpy(), expect_output_case_1)
    np.testing.assert_array_equal(output_ms_case_2.asnumpy(), expect_output_case_2)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_ne(mode):
    """
    Feature: __ne__ in Tensor.
    Description: Verify the result of __ne__ in Tensor.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = NotEqualNet1()

    inputs = Tensor(np.array([[True, False], [False, True]]))
    other = Tensor(np.array([[True, True], [True, True]]))
    value = net(inputs, other)
    expect_value = np.array([[False, True], [True, False]])
    assert np.allclose(expect_value, value.asnumpy())
