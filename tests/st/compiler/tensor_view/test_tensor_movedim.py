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
# ==============================================================================
"""Test tensor view."""
import mindspore as ms
from mindspore import Tensor, jit
from tests.mark_utils import arg_mark


@jit
def movedim_func_1(x):
    output = x.movedim(0, -1)
    return output


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_tensor_movedim_1():
    """
    Feature: Support tensor view api.
    Description: Support tensor movedim api with view operator.
    Expectation: Run success and expected to use the TransposeExtView operator.
    """
    x = ms.tensor(ms.ops.zeros((3, 4, 5)))
    output = movedim_func_1(x)
    assert output.shape == (4, 5, 3)


@jit
def movedim_func_2(x):
    output = Tensor.movedim(x, 0, -1)
    return output


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_tensor_movedim_2():
    """
    Feature: Support tensor view api.
    Description: Support tensor movedim api with view operator.
    Expectation: Run success and expected to use the TransposeExtView operator.
    """
    x = ms.tensor(ms.ops.zeros((3, 4, 5)))
    output = movedim_func_2(x)
    assert output.shape == (4, 5, 3)
