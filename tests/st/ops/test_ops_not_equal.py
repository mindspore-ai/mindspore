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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.common import dtype as mstype


class Net(nn.Cell):
    def construct(self, x, other):
        return ops.not_equal(x, other)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_not_equal(mode):
    """
    Feature: ops.not_equal
    Description: Verify the result of ops.not_equal
    Expectation: success
    """
    ms.set_context(mode=mode)
    x_np = np.array([1, 2, 3]).astype(np.float32)
    y_np = np.array([1, 2, 4]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    y = Tensor(y_np, ms.float32)
    net = Net()
    output_ms_case_1 = net(x, 2.0)
    expect_output_case_1 = np.not_equal(x_np, 2.0)
    output_ms_case_2 = net(x, y)
    expect_output_case_2 = np.not_equal(x_np, y_np)
    np.testing.assert_array_equal(output_ms_case_1.asnumpy(), expect_output_case_1)
    np.testing.assert_array_equal(output_ms_case_2.asnumpy(), expect_output_case_2)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_not_equal_api_bfloat16(mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    ms.set_context(mode=mode, device_target="Ascend")
    net = Net()
    x = Tensor(np.array([1.0, 2.5, 3.8]), mstype.bfloat16)
    y = Tensor(np.array([0.8, 2.5, 4.0]), mstype.bfloat16)
    tensor_output = net(x, y)
    tensor_expected = np.array([True, False, True])
    np.testing.assert_array_equal(tensor_output.asnumpy(), tensor_expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_f_not_equal_bool():
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    net = Net()
    x = Tensor(np.full((2, 2), False))
    y = True
    tensor_output = net(x, y)
    tensor_expected = np.array([[True, True], [True, True]])
    np.testing.assert_array_equal(tensor_output.asnumpy(), tensor_expected)
