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


def test_addr_tensor_api(nptype):
    """
    Feature: test addr tensor api.
    Description: test inputs given their dtype.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([[2., 2.], [3., 2.], [3., 4.]]).astype(nptype))
    vec1 = Tensor(np.array([2., 3., 2.]).astype(nptype))
    vec2 = Tensor(np.array([3., 4.]).astype(nptype))
    output = x.addr(vec1, vec2)
    expected = np.array([[8., 10.], [12., 14.], [9., 12.]]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_addr_float32_tensor_api():
    """
    Feature: test addr tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_addr_tensor_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_addr_tensor_api(np.float32)
