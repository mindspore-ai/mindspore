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
import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark
import mindspore as ms


def generate_random_input(shape, dtype):
    return np.random.uniform(0.9, 1.0, size=shape).astype(dtype)


@test_utils.run_with_cell
def float_power_func(x, exp):
    return x.float_power(exp)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu',
                      'platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_float_power_tensor(context_mode):
    """
    Feature: tensor.float_power
    Description: Verify the result of float_power(tensor)
    Expectation: success
    """
    ms.context.set_context(mode=context_mode)
    x_np = generate_random_input((2, 3, 4, 5), np.float32)
    y_np = generate_random_input((2, 3, 4, 5), np.float32)
    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    out = float_power_func(x, y)
    expect_out = np.float_power(x_np, y_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu',
                      'platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_float_power_scalar(context_mode):
    """
    Feature: tensor.float_power
    Description: Verify the result of float_power(scalar)
    Expectation: success
    """
    ms.context.set_context(mode=context_mode)
    x_np = generate_random_input((2, 3, 4, 5), np.float32)
    y_np = 12.3
    x = ms.Tensor(x_np)
    y = y_np
    out = float_power_func(x, y)
    expect_out = np.float_power(x_np, y_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)
