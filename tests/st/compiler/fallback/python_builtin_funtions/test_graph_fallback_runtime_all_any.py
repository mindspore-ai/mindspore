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
import mindspore as ms
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_all_dict():
    """
    Feature: JIT Fallback
    Description: Test all(dict) in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo(x, y):
        dict_x = {"1": 1, "2": x}
        dict_y = {"1": y, "2": None}
        return all(dict_x), all(dict_y)

    x = ms.Tensor(np.array(10, np.float64))
    y = ms.Tensor(np.array(20, np.float64))
    out = foo(x, y)
    assert out[0] and out[1]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_any_dict():
    """
    Feature: JIT Fallback
    Description: Test any(dict) in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo(x, y):
        dict_x = {"1": 1, "2": x}
        dict_y = {"1": y, "2": None}
        return any(dict_x), any(dict_y)

    x = ms.Tensor(np.array(10, np.float64))
    y = ms.Tensor(np.array(20, np.float64))
    out = foo(x, y)
    assert out[0] and out[1]


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_all_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test all(numpy.array) in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo():
        x = Tensor(np.array([0, 1, 2, 3]))
        y = x.asnumpy()
        y[0] = 1
        return all(x), all(y)

    out = foo()
    assert out[0] and out[1]


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_any_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test any(numpy.array) in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo():
        x = Tensor(np.array([0, 1, 2, 3]))
        y = x.asnumpy()
        y[0] = 1
        return any(x), any(y)

    out = foo()
    assert out[0] and out[1]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_all_dict_get():
    """
    Feature: JIT Fallback
    Description: Test all(dict) in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo(x, y):
        dict_y = {"1": (x, None), "2": y}
        return all(dict_y["1"]), any((dict_y["1"], None)) # pylint: disable=get-dict-value-exception

    x = ms.Tensor(np.array(10, np.float64))
    y = ms.Tensor(np.array(20, np.float64))
    out = foo(x, y)
    assert not out[0] and out[1]
