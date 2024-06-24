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
from mindspore import context
from mindspore.common import dtype as mstype


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_sort_tensor_api_modes(mode):
    """
    Feature: Test sort tensor api.
    Description: Test sort tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")
    x = Tensor([[8, 2, 1], [5, 9, 3], [4, 6, 7]], mstype.float16)
    (output_1, output_2) = x.sort()
    expected_1 = np.array([[1, 2, 8], [3, 5, 9], [4, 6, 7]], np.float16)
    expected_2 = np.array([[2, 1, 0], [2, 0, 1], [0, 1, 2]])
    np.testing.assert_array_equal(output_1.asnumpy(), expected_1)
    np.testing.assert_array_equal(output_2.asnumpy(), expected_2)
