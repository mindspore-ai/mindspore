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
import pytest
import mindspore as ms
from mindspore import Tensor, jit
from tests.mark_utils import arg_mark


@jit
def expand_func_1(x, shape):
    output = x.expand(shape)
    return output


@pytest.mark.skip(reason="has not supported")
@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024)
def test_tensor_expand_1():
    """
    Feature: Support tensor view api.
    Description: Support tensor expand api with view operator.
    Expectation: Run success and expected to use the TransposeExtView operator.
    """
    x = ms.tensor([1, 2, 3], ms.float32)
    shape = (2, 3)
    output = expand_func_1(x, shape)
    assert (output.asnumpy() == [[1, 2, 3], [1, 2, 3]]).all()


@jit
def expand_func_2(x, shape):
    output = Tensor.expand(x, shape)
    return output


@pytest.mark.skip(reason="has not supported")
@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024)
def test_tensor_expand_2():
    """
    Feature: Support tensor view api.
    Description: Support tensor expand api with view operator.
    Expectation: Run success and expected to use the TransposeExtView operator.
    """
    x = ms.tensor([1, 2, 3], ms.float32)
    shape = (2, 3)
    output = expand_func_2(x, shape)
    assert (output.asnumpy() == [[1, 2, 3], [1, 2, 3]]).all()
