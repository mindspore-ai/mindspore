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
import numpy as np
import pytest
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
import mindspore as ms
from mindspore import Tensor


@test_utils.run_with_cell
def tensor_double_forward_func(input_x):
    return input_x.double()


@test_utils.run_with_cell
def tensor_double_grad_func(input_x):
    return ms.grad(tensor_double_forward_func)(input_x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_double(mode):
    """
    Feature: tensor.double
    Description: Verify the result of double.
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.ones([2, 2]), ms.int32)
    output = tensor_double_forward_func(x)
    assert x.shape == output.shape
    assert output.dtype == ms.float64


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_double_grad(mode):
    """
    Feature: tensor.double grad
    Description: Verify the result of double.
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.ones([2, 2]), ms.int32)
    output = tensor_double_grad_func(x)
    assert np.allclose(output.asnumpy(), np.ones([2, 2]))
