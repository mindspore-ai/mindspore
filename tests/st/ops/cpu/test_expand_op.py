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
import pytest
import numpy as np
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


def test_expand_functional():
    """
    Feature: test expand functional API.
    Description: test case for expand functional API.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.array([[1], [2], [3]]), mstype.float32)
    size = Tensor(np.array([3, 4]), mstype.int32)
    output = F.expand(input_x, size)
    expected = np.array([[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_expand_functional_modes():
    """
    Feature: test expand functional API in PyNative and Graph modes.
    Description: test case for expand functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_expand_functional()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_expand_functional()


if __name__ == '__main__':
    test_expand_functional_modes()
