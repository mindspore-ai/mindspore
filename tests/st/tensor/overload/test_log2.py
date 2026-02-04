# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test the overload functional method"""
import numpy as np
import pytest
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils

import mindspore as ms
import mindspore.nn as nn


class Log2Net(nn.Cell):
    def construct(self, x):
        return x.log2()


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def log2_forward_func(x):
    return x.log2()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_log2(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.log2.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    net = Log2Net()
    x = ms.Tensor(np.array([3.0, 5.0, 7.0]), ms.float32)
    output = net(x)
    expect_output = np.array([1.5849625, 2.321928, 2.807355], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
    x = ms.Tensor(np.array([2, 4, 8]).astype(np.float16))
    output = net(x)
    expect_output = np.array([1., 2., 3.], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=3072.0)
def test_tensor_log2_dynamic():
    """
    Feature: Test log2 op.
    Description: Test log2 dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 3, 6), np.float32))
    ms_data2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    TEST_OP(log2_forward_func, [[ms_data1], [ms_data2]])
