# Copyright 2025-2026 Huawei Technologies Co., Ltd
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


@jit(backend="ms_backend")
def unbind_func_1(x):
    output = x.unbind(0)
    return output


# unsupported op [UnstackExtView] on GPU
@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048)
def test_tensor_unbind_1():
    """
    Feature: Support tensor view api.
    Description: Support tensor unbind api with view operator.
    Expectation: Run success and expected to use the UnstackExtView operator.
    """
    a = ms.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = unbind_func_1(a)
    assert (output[0].asnumpy() == [1, 2, 3]).all()
    assert (output[1].asnumpy() == [4, 5, 6]).all()
    assert (output[2].asnumpy() == [7, 8, 9]).all()


@jit(backend="ms_backend")
def unbind_func_2(x):
    output = Tensor.unbind(x, 0)
    return output


# unsupported op [UnstackExtView] on GPU
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048)
def test_tensor_unbind_2():
    """
    Feature: Support tensor view api.
    Description: Support tensor unbind api with view operator.
    Expectation: Run success and expected to use the UnstackExtView operator.
    """
    a = ms.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = unbind_func_2(a)
    assert (output[0].asnumpy() == [1, 2, 3]).all()
    assert (output[1].asnumpy() == [4, 5, 6]).all()
    assert (output[2].asnumpy() == [7, 8, 9]).all()
