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
import mindspore as ms
import mindspore.nn as nn
import numpy as np
import pytest

from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils


class IsFiniteNet(nn.Cell):
    def construct(self, x):
        return x.isfinite()


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def isfinite_forward_func(x):
    return x.isfinite()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_isfinite(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.isfinite.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    net = IsFiniteNet()
    x = ms.Tensor(np.array([np.log(-1), 1, np.log(0)]), ms.float16)
    output = net(x)
    expected = np.array([False, True, False], dtype=np.bool_)
    assert np.allclose(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=3072.01)
def test_tensor_isfinite_dynamic():
    """
    Feature: Test isfinite op.
    Description: Test isfinite dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    ms_data2 = ms.Tensor(generate_random_input((6, 2, 5), np.float32))
    TEST_OP(isfinite_forward_func, [[ms_data1], [ms_data2]])
