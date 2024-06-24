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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor


def test_add_tensor_api(nptype):
    """
    Feature: test add tensor api.
    Description: test inputs given their dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.array([1, 2, 3]).astype(nptype))
    input_y = Tensor(np.array([4, 5, 6]).astype(nptype))
    output = input_x.add(input_y)
    expected = np.array([5, 7, 9]).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_add_float32_tensor_api():
    """
    Feature: test add tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_add_tensor_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_add_tensor_api(np.float32)


if __name__ == '__main__':
    test_add_float32_tensor_api()
