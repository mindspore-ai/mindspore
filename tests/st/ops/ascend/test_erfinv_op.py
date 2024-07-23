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
import pytest
import numpy as np
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


def test_erfinv_functional_api():
    """
    Feature: test erfinv functional API.
    Description: test case for erfinv functional API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([0, 0.5, -0.9]), mstype.float32)
    output = F.erfinv(x)
    expected = np.array([0, 0.47693628, -1.1630871], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


def test_erfinv_tensor_api():
    """
    Feature: test erfinv tensor API.
    Description: test case for erfinv tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([0, 0.5, -0.9]), mstype.float32)
    output = x.erfinv()
    expected = np.array([0, 0.47693628, -1.1630871], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_erfinv_functional_tensor_modes():
    """
    Feature: test erfinv functional and tensor APIs in PyNative and Graph modes.
    Description: test case for erfinv functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_erfinv_functional_api()
    test_erfinv_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_erfinv_functional_api()
    test_erfinv_tensor_api()
