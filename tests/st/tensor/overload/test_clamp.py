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

"""test Tensor.clamp"""
import numpy as np
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.common.api import _pynative_executor


class ClampNet(nn.Cell):
    def construct(self, x, min_value, max_value):
        return x.clamp(min_value, max_value)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_clamp(mode):
    """
    Feature: test Tensor.clamp
    Description: Verify the result of Tensor.clamp
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x_np = np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    net = ClampNet()
    output_ms_case_1 = net(x, 5, 20)
    expect_output_case_1 = np.clip(x_np, 5, 20)
    output_ms_case_2 = net(x, 20, 5)
    expect_output_case_2 = np.clip(x_np, 20, 5)
    assert np.allclose(output_ms_case_1.asnumpy(), expect_output_case_1)
    assert np.allclose(output_ms_case_2.asnumpy(), expect_output_case_2)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_clamp_with_input_type_error(mode):
    """
    Feature: test Tensor.clamp
    Description: Verify the result of Tensor.clamp
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x_np = np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    net = ClampNet()
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        net(x, Tensor([5, 5], ms.float32), 20)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_clamp_with_input_value_error(mode):
    """
    Feature: test Tensor.clamp
    Description: Verify the result of Tensor.clamp
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x_np = np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    net = ClampNet()
    with pytest.raises(ValueError) as error_info:
        net(x, 5, True)
        _pynative_executor.sync()
    assert "For Clamp, the dtype of 'input', 'min' and 'max' must not be bool." in str(error_info.value)
