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
from mindspore import ms_function, context, Tensor

context.set_context(mode=context.GRAPH_MODE)


def test_np_array_1():
    """
    Feature: JIT Fallback
    Description: Test numpy with ndarray in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_1():
        a = np.array([1, 2, 3])
        return Tensor(a)
    res = np_array_1()
    expect_res = Tensor(np.array([1, 2, 3]))
    assert np.all(res.asnumpy() == expect_res.asnumpy())


def test_np_array_2():
    """
    Feature: JIT Fallback
    Description: Test numpy with ndarray in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_2():
        a = np.array([[1, 2], [3, 4]])
        return Tensor(a)
    res = np_array_2()
    expect_res = Tensor(np.array([[1, 2], [3, 4]]))
    assert np.all(res.asnumpy() == expect_res.asnumpy())


def test_np_array_3():
    """
    Feature: JIT Fallback
    Description: Test numpy with ndarray in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_3():
        a = np.array([1, 2, 3, 4, 5], ndmin=2)
        return Tensor(a)
    res = np_array_3()
    expect_res = Tensor(np.array([[1, 2, 3, 4, 5]]))
    assert np.all(res.asnumpy() == expect_res.asnumpy())


def test_np_array_4():
    """
    Feature: JIT Fallback
    Description: Test numpy with ndarray in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_4():
        a = np.array([1, 2, 3], dtype=complex)
        return Tensor(a)
    res = np_array_4()
    assert np.all(res.asnumpy() == Tensor(np.array([1+0j, 2+0j, 3+0j])).asnumpy())


def test_np_dtype_1():
    """
    Feature: JIT Fallback
    Description: Test numpy with dtype in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_dtype_1():
        t = np.dtype(np.int32)
        return Tensor(np.array([1, 2, 3], dtype=t))
    res = np_dtype_1()
    assert np.all(res.asnumpy() == Tensor(np.array([1, 2, 3], dtype=np.int32)).asnumpy())


def test_np_dtype_2():
    """
    Feature: JIT Fallback
    Description: Test numpy with dtype in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_dtype_2():
        t = np.dtype('i4')
        return Tensor(np.array([1, 2, 3], dtype=t))
    res = np_dtype_2()
    assert np.all(res.asnumpy() == Tensor(np.array([1, 2, 3], dtype=np.int32)).asnumpy())


def test_np_array_ndim():
    """
    Feature: JIT Fallback
    Description: Test numpy with array ndim in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_ndim():
        a = np.arange(24)
        return Tensor(a.ndim)
    res = np_array_ndim()
    assert res == 1


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
        return Tensor(b.ndim)
    res = np_array_reshape_1()
    assert res == 2


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


def test_np_array_itemsize():
    """
    Feature: JIT Fallback
    Description: Test numpy with array reshape in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_itemsize():
        a = np.array([1, 2, 3, 4, 5], dtype=np.int8)
        return Tensor(a.itemsize)
    res = np_array_itemsize()
    print("res:", res)
    assert res == 1


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
        return Tensor(y + z)
    res = np_empty_zeros_ones()
    except_res = Tensor(np.ones([3, 2], dtype=np.int))
    assert np.all(res.asnumpy() == except_res.asnumpy())


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
        return Tensor(y)
    res = np_asarray_list()
    except_res = Tensor(np.asarray([1, 2, 3]))
    assert np.all(res.asnumpy() == except_res.asnumpy())


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
        return Tensor(y)
    res = np_asarray_tuple()
    except_res = Tensor(np.asarray((1, 2, 3)))
    assert np.all(res.asnumpy() == except_res.asnumpy())


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
        return Tensor(x)
    res = np_fromiter()
    except_res = Tensor(np.asarray([0., 1., 2., 3., 4.]))
    assert np.all(res.asnumpy() == except_res.asnumpy())


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
        return Tensor(x + y)
    res = np_arange()
    except_res = Tensor(np.asarray([10., 13., 16., 19., 22.]))
    assert np.all(res.asnumpy() == except_res.asnumpy())


def test_np_logspace():
    """
    Feature: JIT Fallback
    Description: Test numpy with logspace in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_logspace():
        a = np.logspace(0, 9, 10, base=2)
        return Tensor(a)
    res = np_logspace()
    except_res = Tensor(np.array([1., 2., 4., 8., 16., 32., 64., 128., 256., 512.]))
    assert np.all(res.asnumpy() == except_res.asnumpy())


def test_np_array_shape():
    """
    Feature: JIT Fallback
    Description: Test numpy with array shape in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_shape():
        a = np.array([[1, 2, 3], [4, 5, 6]])
        return Tensor(a.shape)
    res = np_array_shape()
    print("res:", res)


def test_np_array_size():
    """
    Feature: JIT Fallback
    Description: Test numpy with array size in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_size():
        a = np.array([[1, 2, 3], [4, 5, 6]])
        return Tensor(a.size)
    res = np_array_size()
    print("res:", res)


def test_np_array_real():
    """
    Feature: JIT Fallback
    Description: Test numpy with complex in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_real():
        a = np.array([1, 2, 3], dtype=complex)
        return Tensor(a.real)
    res = np_array_real()
    print("res:", res)


def test_np_array_imag():
    """
    Feature: JIT Fallback
    Description: Test numpy with complex in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_imag():
        a = np.array([1, 2, 3], dtype=complex)
        return Tensor(a.imag)
    res = np_array_imag()
    print("res:", res)


def test_np_binop():
    """
    Feature: JIT Fallback
    Description: Test numpy's binary operation in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_binop():
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = a + b
        return Tensor(c)
    res = np_binop()
    assert np.all(res.asnumpy() == np.array([5, 7, 9]))


def test_np_compare():
    """
    Feature: JIT Fallback
    Description: Test numpy's compare operation in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_compare():
        a = np.array([1, 2, 3])
        b = np.array([0, 2, 4])
        c = a > b
        return Tensor(c)
    res = np_compare()
    assert np.all(res.asnumpy() == np.array([True, False, False]))


def test_np_bool_and():
    """
    Feature: JIT Fallback
    Description: Test AND operation in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_bool_and():
        a = np.bool_(True)
        b = np.bool_(False)
        c = a and b
        return Tensor(c)
    res = np_bool_and()
    assert not res.asnumpy()


def test_np_bool_or():
    """
    Feature: JIT Fallback
    Description: Test OR operation in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_bool_or():
        a = np.bool_(True)
        b = np.bool_(False)
        c = a or b
        return Tensor(c)
    res = np_bool_or()
    assert res.asnumpy()


def test_np_bool_not():
    """
    Feature: JIT Fallback
    Description: Test NOT operation in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_bool_not():
        a = np.bool_(True)
        b = not a
        return Tensor(b)
    res = np_bool_not()
    assert not res.asnumpy()
