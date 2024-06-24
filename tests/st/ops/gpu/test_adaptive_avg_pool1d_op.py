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
from mindspore.ops import functional as F

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


def adaptive_avg_pool1d_forward_functional(nptype):
    input_x = Tensor(np.ones((1, 3, 6)).astype(nptype))
    output = F.adaptive_avg_pool1d(input_x, output_size=2)
    expected = np.ones((1, 3, 2)).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_adaptive_avg_pool1d_forward_float32_functional():
    """
    Feature: test adaptive_avg_pool1d forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    adaptive_avg_pool1d_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    adaptive_avg_pool1d_forward_functional(np.float32)
