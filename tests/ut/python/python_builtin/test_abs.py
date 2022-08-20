# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test graph fallback """
import math
import numpy as np

from mindspore import ms_function, context, Tensor, nn

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_abs_integer():
    """
    Feature: JIT Fallback
    Description: Test abs(int) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = -1
        return abs(x)

    assert foo() == 1


def test_fallback_abs_float():
    """
    Feature: JIT Fallback
    Description: Test abs(float) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = -1.0
        return abs(x)

    assert math.isclose(foo(), 1.0, abs_tol=1e-5)


def test_fallback_abs_complex():
    """
    Feature: JIT Fallback
    Description: Test abs(complex) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = complex(-1, 2)
        return abs(x)

    assert math.isclose(foo(), abs(-1 + 2j), abs_tol=1e-5)


def test_fallback_abs_numpy():
    """
    Feature: JIT Fallback
    Description: Test abs(np.array) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = abs(np.array([1, -2, 3])) + 1
        return Tensor(x)

    assert np.all(foo().asnumpy() == abs(np.array([-1, 2, -3]))) + 1


def test_fallback_abs_cell_construct_tensor():
    """
    Feature: JIT Fallback
    Description: Test abs(Tensor) the tensor is construct in construct function in graph mode
    Expectation: No exception
    """

    class TestCell(nn.Cell):
        def construct(self):
            x = Tensor([-1, 2])
            return abs(x)

    test_cell = TestCell()
    assert np.all(test_cell().asnumpy() == np.array([1, 2]))


def test_fallback_abs_cell_init_tensor():
    """
    Feature: JIT Fallback
    Description: Test abs(Tensor) the tensor is construct in construct function in graph mode
    Expectation: No exception
    """

    class TestCell(nn.Cell):
        def __init__(self):
            super(TestCell, self).__init__()
            self.x = Tensor([-1, 2])

        def construct(self):
            return abs(self.x)

    test_cell = TestCell()
    assert np.allclose(test_cell().asnumpy(), np.array([1, 2]))


def test_fallback_abs_ms_function_tensor():
    """
    Feature: JIT Fallback
    Description: Test abs(Tensor) the tensor is construct in ms_function
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = abs(Tensor(np.array([1, -2, 3])))
        return x

    assert np.allclose(foo().asnumpy(), abs(np.array([-1, 2, -3])))


def test_fallback_isolated_node():
    """
    Feature: JIT Fallback
    Description: Test abs(Tensor) the tensor is construct in ms_function
    Expectation: No exception
    """

    @ms_function
    def foo():
        a = abs(-1)
        abs(2)
        return a
    assert foo() == 1


def test_fallback_tensor_one_element():
    """
    Feature: JIT Fallback
    Description: Test abs(Tensor) the tensor is construct in ms_function
    Expectation: No exception
    """

    @ms_function
    def foo():
        a = abs(Tensor(-1))
        return a
    assert foo() == 1
