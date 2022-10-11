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
"""test python built-in functions in graph mode"""
import pytest
import numpy as np
from mindspore import Tensor, context, nn, ms_function

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.skip(reason='Not support yet')
def test_fallback_all_tensor():
    """
    Feature: JIT Fallback
    Description: Test all(Tensor) with a variable tensor in graph mode
    Expectation: No exception
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return all(x), all(y)

    net = Net()
    x = Tensor(np.array([0, 1, 2, 3]))
    y = Tensor(np.array([1, 1]))
    out1, out2 = net(x, y)
    assert (not out1) and out2


@pytest.mark.skip(reason='Not support yet')
def test_fallback_all_list_hybrid():
    """
    Feature: JIT Fallback
    Description: Test all(List) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo(a, b):
        x = [a, np.array([1]), Tensor(1)]
        y = [a, np.array([0]), Tensor(1)]
        z = [b, np.array([1]), Tensor(1)]
        return all(x), all(y), all(z)

    x, y, z = foo(Tensor([1]), Tensor([0]))
    assert x and (not y) and (not z)


@pytest.mark.skip(reason='Not support yet')
def test_fallback_any_tensor():
    """
    Feature: JIT Fallback
    Description: Test any(Tensor) with a variable tensor in graph mode
    Expectation: No exception
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return any(x), any(y)

    net = Net()
    x = Tensor(np.array([0, 0]))
    y = Tensor(np.array([1, 0]))
    out1, out2 = net(x, y)
    assert (not out1) and out2


@pytest.mark.skip(reason='Not support yet')
def test_fallback_any_list_hybrid():
    """
    Feature: JIT Fallback
    Description: Test any(List) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo(a, b):
        x = [a, np.array([1]), Tensor(1)]
        y = [a, np.array([0]), Tensor(1)]
        z = [b, np.array([1]), Tensor(1)]
        return any(x), any(y), any(z)

    x, y, z = foo(Tensor([1]), Tensor([0]))
    assert x and y and z
