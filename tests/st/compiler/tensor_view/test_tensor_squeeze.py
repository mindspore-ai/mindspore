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
def squeeze_func_1(x):
    output = x.squeeze()
    return output


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_tensor_squeeze_1():
    """
    Feature: Support tensor view api.
    Description: Support tensor squeeze api with no_view operator.
    Expectation: Run success and expected to use the Squeeze operator with a warnning information.
    """
    x = ms.ops.ones(shape=[3, 2, 1])
    output = squeeze_func_1(x)
    assert (output.asnumpy() == [[1, 1], [1, 1], [1, 1]]).all()


@jit
def squeeze_func_2(x):
    output = Tensor.squeeze(x)
    return output


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_tensor_squeeze_2():
    """
    Feature: Support tensor view api.
    Description: Support tensor squeeze api with no_view operator.
    Expectation: Run success and expected to use the Squeeze operator with a warnning information.
    """
    x = ms.ops.ones(shape=[3, 2, 1])
    output = squeeze_func_2(x)
    assert (output.asnumpy() == [[1, 1], [1, 1], [1, 1]]).all()
