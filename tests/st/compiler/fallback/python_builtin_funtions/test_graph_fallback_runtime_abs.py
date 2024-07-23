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

import pytest
import numpy as np
from mindspore import Tensor, jit, context
import mindspore.nn as nn
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_runtime_abs():
    """
    Feature: JIT Fallback
    Description: Test abs() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo(z):
        y = np.array([3]) + z.asnumpy()
        x = abs(Tensor(y)) * 2 + z
        return abs(x).asnumpy()

    z = Tensor(np.array([5]))
    res = foo(z)
    assert res == 21


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_runtime_abs_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test abs() in fallback runtime
    Expectation: No exception
    """

    class TestCell(nn.Cell):
        def construct(self):
            x = Tensor([-1, 2]).asnumpy()
            return abs(x)

    test_cell = TestCell()
    assert np.all(test_cell() == np.array([1, 2]))
