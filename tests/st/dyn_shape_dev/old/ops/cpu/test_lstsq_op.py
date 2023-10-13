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
def test_lstsq_functional_api_modes(mode):
    """
    Feature: Test lstsq functional api.
    Description: Test lstsq functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor([[2, 1, 5], [3, 5, 1], [1, 1, 1]], mstype.float32)
    a = Tensor([[10, 5], [15, 8], [7, 4]], mstype.float32)
    output = F.lstsq(x, a)
    expected = np.array([[17.000002, 11.000002], [-6.5000005, -4.500001], [-3.500002, -2.5000017]], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_lstsq_tensor_api_modes(mode):
    """
    Feature: Test lstsq tensor api.
    Description: Test lstsq tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor([[2, 1, 5], [3, 5, 1], [1, 1, 1]], mstype.float32)
    a = Tensor([[10, 5], [15, 8], [7, 4]], mstype.float32)
    output = x.lstsq(a)
    expected = np.array([[17.000002, 11.000002], [-6.5000005, -4.500001], [-3.500002, -2.5000017]], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)
