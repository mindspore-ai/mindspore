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

from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


def test_gelu_functional_api():
    """
    Feature: test gelu functional API.
    Description: test gelu functional API and compare with expected output.
    Expectation: output should be equal to expected value.
    """
    input_x = Tensor([1.0, 2.0, 3.0], mstype.float32)
    output = F.gelu(input_x, approximate='tanh')
    expected = np.array([0.841192, 1.9545976, 2.9963627], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gelu_functional_api_modes():
    """
    Feature: test gelu functional API for different modes.
    Description: test gelu functional API and compare with expected output.
    Expectation: output should be equal to expected value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_gelu_functional_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_gelu_functional_api()
