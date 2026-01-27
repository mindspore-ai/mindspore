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
def tensor_split_func_1(x):
    output = x.tensor_split(3)
    return output


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024)
def test_tensor_tensor_split_1():
    """
    Feature: Support tensor view api.
    Description: Support tensor tensor_split api with view operator.
    Expectation: Run success and expected to use the SplitWithSizeView and SplitTensorView operator.
    """
    a = ms.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    output = tensor_split_func_1(a)
    assert (output[0].asnumpy() == [0, 1, 2]).all()
    assert (output[1].asnumpy() == [3, 4, 5]).all()
    assert (output[2].asnumpy() == [6, 7]).all()


@jit
def tensor_split_func_2(x):
    output = Tensor.tensor_split(x, 3)
    return output


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024)
def test_tensor_tensor_split_2():
    """
    Feature: Support tensor view api.
    Description: Support tensor tensor_split api with view operator.
    Expectation: Run success and expected to use the SplitWithSizeView and SplitTensorView operator.
    """
    a = ms.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    output = tensor_split_func_2(a)
    assert (output[0].asnumpy() == [0, 1, 2]).all()
    assert (output[1].asnumpy() == [3, 4, 5]).all()
    assert (output[2].asnumpy() == [6, 7]).all()
