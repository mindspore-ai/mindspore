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


def test_blackman_window_functional():
    """
    Feature: test blackman_window functional API.
    Description: test case for blackman_window functional API.
    Expectation: the result match with expected result.
    """
    window_length = Tensor(10, mstype.int32)
    output = F.blackman_window(window_length, periodic=True, dtype=mstype.float32)
    expected = np.array([-2.9802322e-08, 4.0212840e-02, 2.0077014e-01, 5.0978714e-01,
                         8.4922993e-01, 1.0000000e+00, 8.4922981e-01, 5.0978690e-01,
                         2.0077008e-01, 4.0212870e-02]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_blackman_window_functional_modes():
    """
    Feature: test blackman_window functional API in PyNative and Graph modes.
    Description: test case for blackman_window functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_blackman_window_functional()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_blackman_window_functional()


if __name__ == '__main__':
    test_blackman_window_functional_modes()
