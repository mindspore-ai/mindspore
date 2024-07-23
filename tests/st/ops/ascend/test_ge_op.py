# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, ops
import mindspore.context as context
from mindspore.common import dtype as mstype

def test_f_ge_api():
    """
    Feature: test ge functional API.
    Description: testcase for ge functional API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([1.2, 2.5, 3.8]), mstype.bfloat16)
    y = Tensor(np.array([1.2, 1.8, 4.0]), mstype.bfloat16)
    output = ops.ge(x, y)
    expected = np.array([True, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_f_ge_api_modes():
    """
    Feature: test ge functional API in PyNative and Graph modes.
    Description: test case for ge functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_f_ge_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_f_ge_api()
