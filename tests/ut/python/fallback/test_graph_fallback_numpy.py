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
import pytest
import numpy as np
from mindspore import ms_function, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_1():
    """
    Feature: JIT Fallback
    Description: Test numpy with ndarray in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_1():
        a = np.array([1, 2, 3])
        return a
    res = np_array_1()
    assert res == (1, 2, 3)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_2():
    """
    Feature: JIT Fallback
    Description: Test numpy with ndarray in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_2():
        a = np.array([[1, 2], [3, 4]])
        return a
    res = np_array_2()
    assert res == ([1, 2], [3, 4])


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_3():
    """
    Feature: JIT Fallback
    Description: Test numpy with ndarray in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_3():
        a = np.array([1, 2, 3, 4, 5], ndmin=2)
        return a
    res = np_array_3()
    assert res == ([1, 2, 3, 4, 5],)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_4():
    """
    Feature: JIT Fallback
    Description: Test numpy with ndarray in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_4():
        a = np.array([1, 2, 3], dtype=complex)
        return a
    res = np_array_4()
    assert res == ((1+0j), (2+0j), (3+0j))


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_dtype_1():
    """
    Feature: JIT Fallback
    Description: Test numpy with dtype in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_dtype_1():
        t = np.dtype(np.int32)
        return t
    res = np_dtype_1()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_dtype_2():
    """
    Feature: JIT Fallback
    Description: Test numpy with dtype in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_dtype_2():
        t = np.dtype('i4')
        return t
    res = np_dtype_2()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_dtype_3():
    """
    Feature: JIT Fallback
    Description: Test numpy with dtype in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_dtype_3():
        t = np.dtype([('age', np.int8)])
        return t
    res = np_dtype_3()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_dtype_4():
    """
    Feature: JIT Fallback
    Description: Test numpy with dtype in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_dtype_4():
        student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
        a = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
        return a
    res = np_dtype_4()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_ndim():
    """
    Feature: JIT Fallback
    Description: Test numpy with array ndim in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_ndim():
        a = np.arange(24)
        return a.ndim
    res = np_array_ndim()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_reshape_1():
    """
    Feature: JIT Fallback
    Description: Test numpy with array reshape in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_reshape_1():
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = a.reshape(3, 2)
        return b.ndim
    res = np_array_reshape_1()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_reshape_2():
    """
    Feature: JIT Fallback
    Description: Test numpy with array reshape in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_reshape_2():
        a = np.array([[1, 2, 3], [4, 5, 6]])
        a.shape = (3, 2)
        return a
    res = np_array_reshape_2()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_itemsize():
    """
    Feature: JIT Fallback
    Description: Test numpy with array reshape in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_itemsize():
        a = np.array([1, 2, 3, 4, 5], dtype=np.int8)
        return a.itemsize
    res = np_array_itemsize()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_flags():
    """
    Feature: JIT Fallback
    Description: Test numpy with array flags in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_flags():
        a = np.array([1, 2, 3, 4, 5])
        return a.flags
    res = np_array_flags()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_empty_zeros_ones():
    """
    Feature: JIT Fallback
    Description: Test numpy with array empty, zeros, ones in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_empty_zeros_ones():
        x = np.empty([3, 2], dtype=np.int)
        y = np.zeros(x.shape, dtype=np.int)
        z = np.ones(x.shape, dtype=np.int)
        return y + z
    res = np_empty_zeros_ones()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_asarray_list():
    """
    Feature: JIT Fallback
    Description: Test numpy with list to array in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_asarray_list():
        x = [1, 2, 3]
        y = np.asarray(x)
        return y
    res = np_asarray_list()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_asarray_tuple():
    """
    Feature: JIT Fallback
    Description: Test numpy with tuple to array in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_asarray_tuple():
        x = (1, 2, 3)
        y = np.asarray(x)
        return y
    res = np_asarray_tuple()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_asarray_tuple_list():
    """
    Feature: JIT Fallback
    Description: Test numpy with tuple list to array in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_asarray_tuple_list():
        x = [(1, 2, 3), (4, 5)]
        y = np.asarray(x)
        return y
    res = np_asarray_tuple_list()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_frombuffer():
    """
    Feature: JIT Fallback
    Description: Test numpy with frombuffer in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_frombuffer():
        s = b'Hello World'
        a = np.frombuffer(s, dtype='S1')
        return a
    res = np_frombuffer()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_fromiter():
    """
    Feature: JIT Fallback
    Description: Test numpy with fromiter in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_fromiter():
        l = range(5)
        it = iter(l)
        x = np.fromiter(it, dtype=float)
        return x
    res = np_fromiter()
    print("res:", res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_arange():
    """
    Feature: JIT Fallback
    Description: Test numpy with arange in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_arange():
        x = np.arange(5, dtype=float)
        y = np.arange(10, 20, 2)
        return x, y
    res1, res2 = np_arange()
    print("res1:", res1)
    print("res2:", res2)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_linspace():
    """
    Feature: JIT Fallback
    Description: Test numpy with linspace in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_linspace():
        a = np.linspace(1, 10, 10)
        b = np.linspace(1, 1, 10)
        c = np.linspace(10, 20, 5, endpoint=False)
        d = np.linspace(10, 20, 5, endpoint=True)
        e = np.linspace(1, 10, 10, retstep=True)
        f = np.linspace(1, 10, 10).reshape([10, 1])
        return a, b, c, d, e, f
    a, b, c, d, e, f = np_linspace()
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)
    print("e:", e)
    print("f:", f)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_logspace():
    """
    Feature: JIT Fallback
    Description: Test numpy with logspace in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_logspace():
        a = np.logspace(1.0, 2.0, num=10)
        b = np.logspace(0, 9, 10, base=2)
        return a, b
    a, b = np_logspace()
    print("a:", a)
    print("b:", b)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_arange_slice_1():
    """
    Feature: JIT Fallback
    Description: Test numpy with arange slice in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_arange_slice_1():
        x = np.arange(10)
        index = slice(2, 7, 2)
        a = x[index]
        b = x[2:7:2]
        c = x[5]
        d = x[2:]
        e = x[2:5]
        return a, b, c, d, e
    a, b, c, d, e = np_arange_slice_1()
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)
    print("e:", e)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_arange_slice_2():
    """
    Feature: JIT Fallback
    Description: Test numpy with arange slice in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_arange_slice_2():
        x = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        a = x[1:]
        b = x[..., 1]
        c = x[1, ...]
        d = x[..., 1:]
        return a, b, c, d
    a, b, c, d = np_arange_slice_2()
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_advanced_index_1():
    """
    Feature: JIT Fallback
    Description: Test numpy with array advanced index in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_advanced_index_1():
        x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        a = x[[0, 1, 2], [0, 1, 0]]
        rows = np.array([[0, 0], [3, 3]])
        cols = np.array([[0, 2], [0, 2]])
        b = x[rows, cols]
        c = x[1:3, 1:3]
        d = x[1:3, [1, 2]]
        e = x[..., 1:]
        return a, b, c, d, e
    a, b, c, d, e = np_array_advanced_index_1()
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)
    print("e:", e)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_advanced_index_2():
    """
    Feature: JIT Fallback
    Description: Test numpy with array advanced index in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_advanced_index_2():
        x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        y = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])
        z = np.array([1, 2 + 6j, 5, 3.5 + 5j])
        a = x[x > 5]
        b = y[~np.isnan(y)]
        c = z[np.iscomplex(z)]
        return a, b, c
    a, b, c = np_array_advanced_index_2()
    print("a:", a)
    print("b:", b)
    print("c:", c)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_advanced_index_3():
    """
    Feature: JIT Fallback
    Description: Test numpy with array advanced index in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_advanced_index_3():
        x = np.arange(32).reshape((8, 4))
        a = x[[4, 2, 1, 7]]
        y = np.arange(32).reshape((8, 4))
        b = y[[-4, -2, -1, -7]]
        z = np.arange(32).reshape((8, 4))
        c = z[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]
        return a, b, c
    a, b, c = np_array_advanced_index_3()
    print("a:", a)
    print("b:", b)
    print("c:", c)
