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
def test_fallback_runtime_type_numpy():
    """
    Feature: JIT Fallback
    Description: Test type() in fallback runtime
    Expectation: No exception
    """
    @jit
    def foo(x):
        return type(x.asnumpy())

    x = Tensor(np.array([1, 2, 3]))
    out = foo(x)
    assert str(out) == "<class 'numpy.ndarray'>"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_runtime_type_dict():
    """
    Feature: JIT Fallback
    Description: Test type() in fallback runtime
    Expectation: No exception
    """
    @jit
    def foo(x):
        dict_x = {"1": 1, "2": x}
        return type(dict_x)

    x = Tensor(np.array([1, 2, 3]))
    out = foo(x)
    assert str(out) == "<class 'dict'>"
