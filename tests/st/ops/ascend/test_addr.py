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

import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor

# all cases tested against dchip


def addr_tensor_api(nptype):
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


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_addr_float32_tensor_api():
    """
    Feature: test addr tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    addr_tensor_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    addr_tensor_api(np.float32)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_addr_invalid_dtypes():
    """
    Feature: test addr invalid dtypes.
    Description: test invalid dtypes inputs.
    Expectation: the result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with pytest.raises(TypeError):
        addr_tensor_api(np.uint16)
    with pytest.raises(TypeError):
        addr_tensor_api(np.int8)
