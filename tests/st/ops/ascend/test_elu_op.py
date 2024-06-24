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


def test_elu_functional_api():
    """
    Feature: test elu functional API.
    Description: test elu functional API and compare with expected output.
    Expectation: output should be equal to expected value.
    """
    input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mstype.float32)
    output = F.elu(input_x)
    expected = np.array([[-0.63212055, 4.0, -0.99966455], [2.0, -0.99326205, 9.0]], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_elu_functional_api_modes():
    """
    Feature: test elu functional API for different modes.
    Description: test elu functional API and compare with expected output.
    Expectation: output should be equal to expected value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_elu_functional_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_elu_functional_api()
