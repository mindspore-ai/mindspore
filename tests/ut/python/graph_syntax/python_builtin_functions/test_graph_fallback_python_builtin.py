# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
from mindspore import jit, context, Tensor

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_bin():
    """
    Feature: JIT Fallback
    Description: Test bin() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = bin(3)
        return x
    assert foo() == '0b11'


def test_fallback_chr():
    """
    Feature: JIT Fallback
    Description: Test chr() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = chr(0x61)
        return x
    assert foo() == 'a'


def test_fallback_complex():
    """
    Feature: JIT Fallback
    Description: Test complex() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = complex(1, 2)
        return Tensor(x)
    res = foo()
    expect_res = np.array(1 + 2j)
    assert isinstance(res, Tensor)
    assert np.all(res.asnumpy() == expect_res)


def test_fallback_divmod():
    """
    Feature: JIT Fallback
    Description: Test divmod() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = divmod(7, 2)
        return x
    assert foo() == (3, 1)


def test_fallback_hash():
    """
    Feature: JIT Fallback
    Description: Test hash() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = hash(1)
        return x
    assert foo() == 1


def test_fallback_hex():
    """
    Feature: JIT Fallback
    Description: Test hex() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = hex(255)
        return x
    assert foo() == '0xff'


def test_fallback_oct():
    """
    Feature: JIT Fallback
    Description: Test oct() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = oct(8)
        return x
    assert foo() == '0o10'


def test_fallback_ord():
    """
    Feature: JIT Fallback
    Description: Test ord() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = ord('a')
        return x
    assert foo() == 97


def test_fallback_reversed():
    """
    Feature: JIT Fallback
    Description: Test reversed() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = reversed([1, 2, 3])
        return list(x)
    assert foo() == [3, 2, 1]


def test_fallback_set():
    """
    Feature: JIT Fallback
    Description: Test set() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = set([1, 2, 1])
        return x
    assert list(foo()) == [1, 2]


def test_fallback_slice():
    """
    Feature: JIT Fallback
    Description: Test slice() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        slice_x = slice(5)
        arr = range(10)
        return arr[slice_x]
    assert list(foo()) == [0, 1, 2, 3, 4]


def test_fallback_sorted():
    """
    Feature: JIT Fallback
    Description: Test sorted() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sorted([5, 3, 1, 4, 2])
        return x
    assert list(foo()) == [1, 2, 3, 4, 5]


def test_fallback_str():
    """
    Feature: JIT Fallback
    Description: Test str() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = str(10)
        return x
    assert foo() == '10'


def test_fallback_unsupported_builtin_type():
    """
    Feature: JIT Fallback
    Description: Test input() in graph mode and JIT Fallback.
    Expectation: No exception.
    """
    @jit
    def func(x):
        input("input x:")
        return x * 2

    with pytest.raises(TypeError,
                       match="'<built-in function input>' is not supported both in JIT Fallback and graph mode."):
        input_x = Tensor([1])
        res = func(input_x)
        assert res == 2
