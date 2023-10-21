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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mvlgamma_functional_api_modes(mode):
    """
    Feature: Test mvlgamma functional api.
    Description: Test mvlgamma functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor([[3, 4, 5], [4, 2, 6]], mstype.float32)
    output = F.mvlgamma(x, p=3)
    expected = np.array([[2.694925, 5.402975, 9.140645], [5.402975, 1.596312, 13.64045]], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mvlgamma_tensor_api_modes(mode):
    """
    Feature: Test mvlgamma tensor api.
    Description: Test mvlgamma tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor([[3, 4, 5], [4, 2, 6]], mstype.float32)
    output = x.mvlgamma(p=3)
    expected = np.array([[2.694925, 5.402975, 9.140645], [5.402975, 1.596312, 13.64045]], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)
