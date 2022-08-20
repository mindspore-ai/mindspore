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
from mindspore import ms_function, Tensor


def test_fallback_bool_int():
    """
    Feature : JIT Fallback
    Description: Test bool(int) in graph mode.
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = bool(int)
        return x

    assert foo()


def test_fallback_bool_empty():
    """
    Feature : JIT Fallback
    Description: Test bool() in graph mode.
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = bool()
        return x

    assert not foo()


def test_fallback_bool_seq():
    """
    Feature : JIT Fallback
    Description: Test bool(sequence) in graph mode.
    Expectation: No exception
    """
    @ms_function
    def foo():
        x1 = bool([1, 2, 3, 4])
        y1 = bool((1, 2))
        x2 = bool([])
        y2 = bool(tuple())
        return x1, y1, x2, y2
    x1, y1, x2, y2 = foo()
    assert x1 and y1 and not x2 and not y2


def test_fallback_bool_str():
    """
    Feature : JIT Fallback
    Description: Test bool(str) in graph mode.
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = bool("")
        y = bool("123")
        return x, y

    x, y = foo()
    assert not x and y


def test_fallback_bool_none_and_complex():
    """
    Feature : JIT Fallback
    Description: Test bool(None) and bool(complex) in graph mode.
    Expectation: No exception
    """

    @ms_function
    def foo():
        x1 = bool(None)
        x2 = bool(complex(0, 0))
        x3 = bool(complex(1, 0))
        x4 = bool(complex(0, 1))
        return x1, x2, x3, x4

    x1, x2, x3, x4 = foo()
    assert (not x1) and (not x2) and x3 and x4


def test_fallback_bool_tensor():
    """
    Feature : JIT Fallback
    Description: Test bool(Tensor) in graph mode.
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = bool(Tensor([1]))
        y = bool(Tensor([0]))
        return x, y

    x, y = foo()
    assert x and not y


def test_fallback_bool_tensor_construct():
    """
    Feature : JIT Fallback
    Description: Test bool(Tensor) in graph mode.
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = Tensor([1])
        y = Tensor([0])
        x = bool(x)
        y = bool(y)
        return x, y
    x, y = foo()
    assert x and not y
