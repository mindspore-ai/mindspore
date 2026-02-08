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
    return input_x.bfloat16()


@test_utils.run_with_cell
def tensor_double_grad_func(input_x):
    return ms.grad(tensor_double_forward_func)(input_x)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_bfloat16(mode):
    """
    Feature: tensor.bfloat16
    Description: Verify the result of bfloat16.
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.ones([2, 2]), ms.float32)
    output = tensor_double_forward_func(x)
    assert x.shape == output.shape
    assert output.dtype == ms.bfloat16


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_bfloat16_grad(mode):
    """
    Feature: tensor.bfloat16 grad
    Description: Verify the result of bfloat16.
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.ones([2, 2]), ms.float32)
    output = tensor_double_grad_func(x)
    assert np.allclose(output.asnumpy(), np.ones([2, 2]))
