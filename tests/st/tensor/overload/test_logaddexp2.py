# Copyright 2025 Huawei Technologies Co., Ltd
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


class Net(nn.Cell):
    def construct(self, x, other):
        return x.logaddexp2(other)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def logaddexp2_forward_func(x, other):
    return x.logaddexp2(other)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_logaddexp2(mode):
    """
    Feature: Tensor.logaddexp2.
    Description: Test functional feature with Tensor.logaddexp2.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    net = Net()
    x = ms.Tensor([-100, 1, 10], ms.float32)
    other = ms.Tensor([-1, -1, 3], ms.float32)
    output = net(x, other)
    expect_output = np.array([-1.0000042, 1.3219349, 10.011246], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_tensor_logaddexp2_dynamic():
    """
    Feature: Test min op.
    Description: Test min dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 3, 6), np.float32))
    ms_data2 = ms.Tensor(generate_random_input((4, 3, 6), np.float32))
    ms_data3 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    ms_data4 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    TEST_OP(logaddexp2_forward_func, [[ms_data1, ms_data2], [ms_data3, ms_data4]],
            disable_mode=["GRAPH_MODE_GE"], disable_case=['Deterministic'], case_config={'all_dim_zero': True})
