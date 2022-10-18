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
import numpy.random as rand
from mindspore import jit, context, Tensor
import mindspore.nn as nn

context.set_context(mode=context.GRAPH_MODE)


def test_np_array_1():
    """
    Feature: JIT Fallback
    Description: Test numpy with ndarray in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_array_1():
        a = np.array([1, 2, 3])
        return Tensor(a)
    res = np_array_1()
    expect_res = np.array([1, 2, 3])
    assert np.all(res.asnumpy() == expect_res)


def test_np_array_2():
    """
    Feature: JIT Fallback
    Description: Test numpy with ndarray in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_array_2():
        a = np.array([[1, 2], [3, 4]])
        return Tensor(a)
    res = np_array_2()
    expect_res = np.array([[1, 2], [3, 4]])
    assert np.all(res.asnumpy() == expect_res)


def test_np_array_3():
    """
    Feature: JIT Fallback
    Description: Test numpy with ndarray in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_array_3():
        a = np.array([1, 2, 3, 4, 5], ndmin=2)
        return Tensor(a)
    res = np_array_3()
    expect_res = np.array([[1, 2, 3, 4, 5]])
    assert np.all(res.asnumpy() == expect_res)


def test_np_array_4():
    """
    Feature: JIT Fallback
    Description: Test numpy with ndarray in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_array_4():
        a = np.array([1, 2, 3], dtype=complex)
        return Tensor(a)
    res = np_array_4()
    expect_res = np.array([1+0j, 2+0j, 3+0j])
    assert np.all(res.asnumpy() == expect_res)


def test_np_dtype_1():
    """
    Feature: JIT Fallback
    Description: Test numpy with dtype in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_dtype_1():
        t = np.dtype(np.int32)
        return Tensor(np.array([1, 2, 3], dtype=t))
    res = np_dtype_1()
    expect_res = np.array([1, 2, 3], dtype=np.int32)
    assert np.all(res.asnumpy() == expect_res)


def test_np_dtype_2():
    """
    Feature: JIT Fallback
    Description: Test numpy with dtype in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_dtype_2():
        t = np.dtype('i4')
        return Tensor(np.array([1, 2, 3], dtype=t))
    res = np_dtype_2()
    expect_res = np.array([1, 2, 3], dtype=np.int32)
    assert np.all(res.asnumpy() == expect_res)


def test_np_array_ndim():
    """
    Feature: JIT Fallback
    Description: Test numpy with array ndim in graph mode.
    Expectation: No exception.
    """
    @jit
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
    @jit
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
    @jit
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
    @jit
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
    @jit
    def np_empty_zeros_ones():
        x = np.empty([3, 2], dtype=np.int)
        y = np.zeros(x.shape, dtype=np.int)
        z = np.ones(x.shape, dtype=np.int)
        return Tensor(y + z)
    res = np_empty_zeros_ones()
    except_res = np.ones([3, 2], dtype=np.int)
    assert np.all(res.asnumpy() == except_res)


def test_np_asarray_list():
    """
    Feature: JIT Fallback
    Description: Test numpy with list to array in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_asarray_list():
        x = [1, 2, 3]
        y = np.asarray(x)
        return Tensor(y)
    res = np_asarray_list()
    except_res = np.asarray([1, 2, 3])
    assert np.all(res.asnumpy() == except_res)


def test_np_asarray_tuple():
    """
    Feature: JIT Fallback
    Description: Test numpy with tuple to array in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_asarray_tuple():
        x = (1, 2, 3)
        y = np.asarray(x)
        return Tensor(y)
    res = np_asarray_tuple()
    except_res = np.asarray((1, 2, 3))
    assert np.all(res.asnumpy() == except_res)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_frombuffer():
    """
    Feature: JIT Fallback
    Description: Test numpy with frombuffer in graph mode.
    Expectation: No exception.
    """
    @jit
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
    @jit
    def np_fromiter():
        l = range(5)
        it = iter(l)
        x = np.fromiter(it, dtype=float)
        return Tensor(x)
    res = np_fromiter()
    except_res = np.asarray([0., 1., 2., 3., 4.])
    assert np.all(res.asnumpy() == except_res)


def test_np_arange():
    """
    Feature: JIT Fallback
    Description: Test numpy with arange in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_arange():
        x = np.arange(5, dtype=float)
        y = np.arange(10, 20, 2)
        return Tensor(x + y)
    res = np_arange()
    except_res = np.asarray([10., 13., 16., 19., 22.])
    assert np.all(res.asnumpy() == except_res)


def test_np_logspace():
    """
    Feature: JIT Fallback
    Description: Test numpy with logspace in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_logspace():
        a = np.logspace(0, 9, 10, base=2)
        return Tensor(a)
    res = np_logspace()
    except_res = np.array([1., 2., 4., 8., 16., 32., 64., 128., 256., 512.])
    assert np.all(res.asnumpy() == except_res)


def test_np_array_shape():
    """
    Feature: JIT Fallback
    Description: Test numpy with array shape in graph mode.
    Expectation: No exception.
    """
    @jit
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
    @jit
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
    @jit
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
    @jit
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
    @jit
    def np_binop():
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = a + b
        return Tensor(c)
    res = np_binop()
    assert np.all(res.asnumpy() == np.array([5, 7, 9]))


def test_np_binop_2():
    """
    Feature: JIT Fallback
    Description: Test numpy's binary operation in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_binop():
        a = np.int_(1)
        b = 4 + a
        return Tensor(b)
    res = np_binop()
    assert res == 5


def test_np_compare():
    """
    Feature: JIT Fallback
    Description: Test numpy's compare operation in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_compare():
        a = np.array([1, 2, 3])
        b = np.array([0, 2, 4])
        c = a > b
        return Tensor(c)
    res = np_compare()
    assert np.all(res.asnumpy() == np.array([True, False, False]))


def test_np_compare_2():
    """
    Feature: JIT Fallback
    Description: Test numpy's compare operation in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_compare():
        a = 1
        b = np.int_(3)
        c = a < b
        return Tensor(c)
    res = np_compare()
    assert res


def test_np_bool_and():
    """
    Feature: JIT Fallback
    Description: Test AND operation in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_bool_and():
        a = np.bool_(True)
        b = np.bool_(False)
        c = a and b
        return Tensor(c)
    res = np_bool_and()
    assert not res


def test_np_bool_or():
    """
    Feature: JIT Fallback
    Description: Test OR operation in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_bool_or():
        a = np.bool_(True)
        b = np.bool_(False)
        c = a or b
        return Tensor(c)
    res = np_bool_or()
    assert res


def test_np_bool_or_2():
    """
    Feature: JIT Fallback
    Description: Test OR operation in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_bool_or():
        out = 0 or np.bool_(True)
        return Tensor(out)
    res = np_bool_or()
    assert res


def test_np_bool_not():
    """
    Feature: JIT Fallback
    Description: Test NOT operation in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_bool_not():
        a = np.bool_(True)
        b = not a
        return Tensor(b)
    res = np_bool_not()
    assert not res


def test_np_augassign():
    """
    Feature: JIT Fallback
    Description: Test augassign method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_augassign():
        value_add = np.array([1, 2, 3])
        value_add += np.array([4, 5, 6])
        value_sub = np.array([5, 5, 5])
        value_sub -= np.array([1, 2, 3])
        value_mul = np.int_(2)
        value_mul *= np.int_(3)
        value_div = np.int_(10)
        value_div /= np.int_(5)
        value_floordiv = np.int_(5)
        value_floordiv //= np.int_(2)
        return Tensor(value_add), Tensor(value_sub), Tensor(value_mul), Tensor(value_div), Tensor(value_floordiv)

    out_add, out_sub, out_mul, out_div, out_floordiv = np_augassign()
    assert np.all(out_add.asnumpy() == np.array([5, 7, 9]))
    assert np.all(out_sub.asnumpy() == np.array([4, 3, 2]))
    assert out_mul == 6
    assert out_div == 2
    assert out_floordiv == 2


def test_np_augassign_2():
    """
    Feature: JIT Fallback
    Description: Test augassign method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_augassign():
        value_mod = np.int_(5)
        value_mod %= np.int_(2)
        value_pow = np.int_(3)
        value_pow **= np.int_(2)
        value_lshift = np.int_(4)
        value_lshift <<= 1
        value_rshift = np.int_(4)
        value_rshift >>= 1
        value_bitxor = np.int_(0)
        value_bitxor ^= 1
        return Tensor(value_mod), Tensor(value_pow), Tensor(value_lshift), Tensor(value_rshift), Tensor(value_bitxor)

    out_mod, out_pow, out_lshift, out_rshift, out_bitxor = np_augassign()
    assert out_mod == 1
    assert out_pow == 9
    assert out_lshift == 8
    assert out_rshift == 2
    assert out_bitxor == 1


def test_np_augassign_3():
    """
    Feature: JIT Fallback
    Description: Test augassign method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_augassign():
        value_bitand = np.int_(6)
        value_bitand &= 3
        value_bitor = np.int_(0)
        value_bitor |= 1
        return Tensor(value_bitand), Tensor(value_bitor)

    out_bitand, out_bitor = np_augassign()
    assert out_bitand == 2
    assert out_bitor == 1


def test_np_subscript():
    """
    Feature: JIT Fallback
    Description: Test subscript method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_subscript():
        a = np.array([1, 2, 3])
        b = a[np.int32(1)]
        return Tensor(b)
    res = np_subscript()
    assert res == 2


def test_np_slice():
    """
    Feature: JIT Fallback
    Description: Test slice method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_slice():
        a = np.arange(10)
        b = a[1:5]
        return Tensor(b)
    res = np_slice()
    assert np.all(res.asnumpy() == np.array([1, 2, 3, 4]))


def test_np_random():
    """
    Feature: JIT Fallback
    Description: Test numpy.random module in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_random():
        a = rand.randint(100, size=(5))
        b = a[1:5]
        return Tensor(b)
    res = np_random()
    print(res)


def test_np_init():
    """
    Feature: JIT Fallback
    Description: Test numpy defined in init in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = np.array([1, 2])

        def construct(self):
            y = np.array([3, 4])
            print(self.x)
            return Tensor(y + self.x)

    net = Net()
    res = net()
    assert (res.asnumpy() == [4, 6]).all()
