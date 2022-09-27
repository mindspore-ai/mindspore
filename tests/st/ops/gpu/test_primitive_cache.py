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
""" test primitive cache """
import pytest
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import ms_function
from mindspore.ops import operations as P
from mindspore.ops._primitive_cache import _get_cache_prim


# pylint: disable=W0235


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_function_run_in_pynative():
    """
    Feature: test ms_function run in PyNative.
    Description: test ms_function run in PyNative.
    Expectation: Success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

    @ms_function
    def pow_function(x, y):
        _pow = _get_cache_prim(P.Pow)()
        return _pow(x, y)

    class Pow(nn.Cell):
        def __init__(self):
            super(Pow, self).__init__()

        def construct(self, x1, x2):
            return pow_function(x1, x2)

    x = Tensor(np.array([1.0, 2.0, 4.0]), ms.float32)
    y = 3
    output = Pow()(x, y)
    expect_output = np.array([1.0, 8.0, 64.0], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_run_pynative_and_then_run_graph():
    """
    Feature: test the cache key must be a str.
    Description: test run_pynative and then run_graph.
    Expectation: Success.
    """

    class Pow(nn.Cell):
        def __init__(self):
            super(Pow, self).__init__()

        def construct(self, x1, x2):
            _pow = _get_cache_prim(P.Pow)()
            return _pow(x1, x2)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x = Tensor(np.array([1.0, 2.0, 4.0]), ms.float32)
    y = 3
    output1 = Pow()(x, y)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x = Tensor(np.array([1.0, 2.0, 4.0]), ms.float32)
    y = 3
    output2 = Pow()(x, y)
    np.testing.assert_almost_equal(output1.asnumpy(), output2.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_continuous_cache():
    """
    Feature: test continuous cache.
    Description: test continuous cache.
    Expectation: Success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

    class AddSub(nn.Cell):
        def __init__(self):
            super(AddSub, self).__init__()

        def construct(self, x, y):
            y = y + 1
            add = _get_cache_prim(P.Add)()
            sub = _get_cache_prim(P.Sub)()
            z = add(x, y)
            out = sub(z, x)
            return out

    x = Tensor(np.array([2, 2, 1]), dtype=ms.int32)
    y = Tensor(np.array([1, 1, 1]), dtype=ms.int32)
    output = AddSub()(x, y)
    expect_output = np.array([2, 2, 2], dtype=np.int32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)
