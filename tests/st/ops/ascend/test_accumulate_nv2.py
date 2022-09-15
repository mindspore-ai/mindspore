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
from mindspore.ops import functional as F

# all cases tested against dchip


def accumulate_n_forward_functional(nptype):
    input_x = Tensor(np.array([1, 2, 3]).astype(nptype))
    input_y = Tensor(np.array([4, 5, 6]).astype(nptype))

    output = F.accumulate_n([input_x, input_y, input_x, input_y])
    expected = np.array([10., 14., 18.])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_accumulate_n_forward_float32_functional():
    """
    Feature: test accumulate_n forward.
    Description: test float32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    accumulate_n_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    accumulate_n_forward_functional(np.float32)
