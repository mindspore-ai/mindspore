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


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_linspace():
    """
    Feature: JIT Fallback
    Description: Test numpy with linspace in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_linspace():
        a = Tensor(np.linspace(1, 10, 10))
        b = Tensor(np.linspace(1, 1, 10))
        c = Tensor(np.linspace(10, 20, 5, endpoint=False))
        d = Tensor(np.linspace(10, 20, 5, endpoint=True))
        e = Tensor(np.linspace(1, 10, 10, retstep=True))
        f = Tensor(np.linspace(1, 10, 10).reshape([10, 1]))
        return a, b, c, d, e, f
    a, b, c, d, e, f = np_linspace()
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)
    print("e:", e)
    print("f:", f)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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
        a = Tensor(x[index])
        b = Tensor(x[2:7:2])
        c = Tensor(x[5])
        d = Tensor(x[2:])
        e = Tensor(x[2:5])
        return a, b, c, d, e
    a, b, c, d, e = np_arange_slice_1()
    assert np.all(a.asnumpy() == Tensor(np.array([2, 4, 6])).asnumpy())
    assert np.all(b.asnumpy() == Tensor(np.array([2, 4, 6])).asnumpy())
    assert np.all(c.asnumpy() == Tensor(np.array([5])).asnumpy())
    assert np.all(d.asnumpy() == Tensor(np.array([2, 3, 4, 5, 6, 7, 8, 9])).asnumpy())
    assert np.all(e.asnumpy() == Tensor(np.array([2, 3, 4])).asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_arange_slice_2():
    """
    Feature: JIT Fallback
    Description: Test numpy with arange slice in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_arange_slice_2():
        x = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        a = Tensor(x[1:])
        b = Tensor(x[..., 1])
        c = Tensor(x[1, ...])
        d = Tensor(x[..., 1:])
        return a, b, c, d
    a, b, c, d = np_arange_slice_2()
    assert np.all(a.asnumpy() == Tensor(np.array([[3, 4, 5], [4, 5, 6]])).asnumpy())
    assert np.all(b.asnumpy() == Tensor(np.array([2, 4, 5])).asnumpy())
    assert np.all(c.asnumpy() == Tensor(np.array([3, 4, 5])).asnumpy())
    assert np.all(d.asnumpy() == Tensor(np.array([[2, 3], [4, 5], [5, 6]])).asnumpy())



@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_array_advanced_index_1():
    """
    Feature: JIT Fallback
    Description: Test numpy with array advanced index in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_advanced_index_1():
        x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        a = Tensor(x[[0, 1, 2], [0, 1, 0]])
        rows = np.array([[0, 0], [3, 3]])
        cols = np.array([[0, 2], [0, 2]])
        b = Tensor(x[rows, cols])
        c = Tensor(x[1:3, 1:3])
        d = Tensor(x[1:3, [1, 2]])
        e = Tensor(x[..., 1:])
        return a, b, c, d, e
    a, b, c, d, e = np_array_advanced_index_1()
    assert np.all(a.asnumpy() == Tensor(np.array([0, 4, 6])).asnumpy())
    assert np.all(b.asnumpy() == Tensor(np.array([[0, 2], [9, 11]])).asnumpy())
    assert np.all(c.asnumpy() == Tensor(np.array([[4, 5], [7, 8]])).asnumpy())
    assert np.all(d.asnumpy() == Tensor(np.array([[4, 5], [7, 8]])).asnumpy())
    assert np.all(e.asnumpy() == Tensor(np.array([[1, 2], [4, 5], [7, 8], [10, 11]])).asnumpy())


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
        a = Tensor(x[x > 5])
        b = Tensor(y[~np.isnan(y)])
        c = Tensor(z[np.iscomplex(z)])
        return a, b, c
    a, b, c = np_array_advanced_index_2()
    assert np.all(a.asnumpy() == Tensor(np.array([6, 7, 8, 9, 10, 11])).asnumpy())
    assert np.all(b.asnumpy() == Tensor(np.array([1., 2., 3., 4., 5.])).asnumpy())
    assert np.all(c.asnumpy() == Tensor(np.array([2. + 6.j, 3.5 + 5.j])).asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_array_advanced_index_3():
    """
    Feature: JIT Fallback
    Description: Test numpy with array advanced index in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def np_array_advanced_index_3():
        x = np.arange(32).reshape((8, 4))
        a = Tensor(x[[4, 2, 1, 7]])
        y = np.arange(32).reshape((8, 4))
        b = Tensor(y[[-4, -2, -1, -7]])
        z = np.arange(32).reshape((8, 4))
        c = Tensor(z[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])
        return a, b, c
    a, b, c = np_array_advanced_index_3()
    print("a:", a)
    print("b:", b)
    print("c:", c)
