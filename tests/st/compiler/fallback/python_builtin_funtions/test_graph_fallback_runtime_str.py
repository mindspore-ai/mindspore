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
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_runtime_str():
    """
    Feature: JIT Fallback
    Description: Test str() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo(x):
        return str(x.asnumpy())

    x = Tensor(np.array([1, 2, 3]))
    assert foo(x) == "[1 2 3]"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_str_with_input_tensor():
    """
    Feature: JIT Fallback
    Description: Test str() in graph mode with tensor input.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return str(x)
    out = foo(Tensor([1, 2, 4]))
    assert out == "[1 2 4]"
