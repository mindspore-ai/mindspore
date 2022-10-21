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
from mindspore import jit, context, Tensor

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_linspace():
    """
    Feature: JIT Fallback
    Description: Test numpy with linspace in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_linspace():
        a = Tensor(np.linspace(1, 10, 10))
        b = Tensor(np.linspace(1, 1, 10))
        c = Tensor(np.linspace(10, 20, 5, endpoint=False))
        d = Tensor(np.linspace(10, 20, 5, endpoint=True))
        e = Tensor(np.linspace(1, 10, 10).reshape([10, 1]))
        return a, b, c, d, e
    a, b, c, d, e = np_linspace()
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)
    print("e:", e)


@pytest.mark.level1
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
    @jit
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
    assert np.all(a.asnumpy() == np.array([2, 4, 6]))
    assert np.all(b.asnumpy() == np.array([2, 4, 6]))
    assert np.all(c.asnumpy() == np.array([5]))
    assert np.all(d.asnumpy() == np.array([2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.all(e.asnumpy() == np.array([2, 3, 4]))


@pytest.mark.level1
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
    @jit
    def np_arange_slice_2():
        x = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        a = Tensor(x[1:])
        b = Tensor(x[..., 1])
        c = Tensor(x[1, ...])
        d = Tensor(x[..., 1:])
        return a, b, c, d
    a, b, c, d = np_arange_slice_2()
    assert np.all(a.asnumpy() == np.array([[3, 4, 5], [4, 5, 6]]))
    assert np.all(b.asnumpy() == np.array([2, 4, 5]))
    assert np.all(c.asnumpy() == np.array([3, 4, 5]))
    assert np.all(d.asnumpy() == np.array([[2, 3], [4, 5], [5, 6]]))


@pytest.mark.level1
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
    @jit
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
    assert np.all(a.asnumpy() == np.array([0, 4, 6]))
    assert np.all(b.asnumpy() == np.array([[0, 2], [9, 11]]))
    assert np.all(c.asnumpy() == np.array([[4, 5], [7, 8]]))
    assert np.all(d.asnumpy() == np.array([[4, 5], [7, 8]]))
    assert np.all(e.asnumpy() == np.array([[1, 2], [4, 5], [7, 8], [10, 11]]))


# Not support <class 'complex'> yet.
@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_advanced_index_2():
    """
    Feature: JIT Fallback
    Description: Test numpy with array advanced index in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_array_advanced_index_2():
        x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        y = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])
        z = np.array([1, 2 + 6j, 5, 3.5 + 5j])
        a = Tensor(x[x > 5])
        b = Tensor(y[~np.isnan(y)])
        c = Tensor(z[np.iscomplex(z)])
        return a, b, c
    a, b, c = np_array_advanced_index_2()
    assert np.all(a.asnumpy() == np.array([6, 7, 8, 9, 10, 11]))
    assert np.all(b.asnumpy() == np.array([1., 2., 3., 4., 5.]))
    assert np.all(c.asnumpy() == np.array([2. + 6.j, 3.5 + 5.j]))


@pytest.mark.level1
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
    @jit
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_reshape():
    """
    Feature: JIT Fallback
    Description: Test numpy.reshape() method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_reshape():
        x = np.arange(8)
        y = x.reshape(2, 4)
        return Tensor(y)
    assert np.all(np_reshape().asnumpy() == np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_ndarray_flatten():
    """
    Feature: JIT Fallback
    Description: Test numpy.flatten() method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_ndarray_flatten():
        x = np.arange(8).reshape(2, 4)
        y = x.flatten()
        return Tensor(y)
    assert np.all(np_ndarray_flatten().asnumpy() == np.array([0, 1, 2, 3, 4, 5, 6, 7]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_ravel():
    """
    Feature: JIT Fallback
    Description: Test numpy.ravel() method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_ravel():
        x = np.arange(8).reshape(2, 4)
        y = x.ravel(order='F')
        return Tensor(y)
    assert np.all(np_ravel().asnumpy() == np.array([0, 4, 1, 5, 2, 6, 3, 7]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_transpose():
    """
    Feature: JIT Fallback
    Description: Test numpy.transpose() method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_transpose():
        x = np.arange(4).reshape(4, 1)
        y = np.transpose(x)
        return Tensor(y)
    assert np.all(np_transpose().asnumpy() == np.array([0, 1, 2, 3]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_rollaxis():
    """
    Feature: JIT Fallback
    Description: Test numpy.rollaxis() method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_rollaxis():
        x = np.arange(8).reshape(2, 2, 2)
        tensor_x = Tensor(x)
        y = np.rollaxis(x, 2, 0)
        tensor_y = Tensor(y)
        return tensor_x[1, 1, 0], tensor_y[1, 1, 0]
    x, y = np_rollaxis()
    assert x == 6 and y == 5


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_swapaxes():
    """
    Feature: JIT Fallback
    Description: Test numpy.swapaxes() method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_swapaxes():
        x = np.arange(8).reshape(2, 2, 2)
        tensor_x = Tensor(x)
        y = np.swapaxes(x, 2, 0)
        tensor_y = Tensor(y)
        return tensor_x[1, 1, 0], tensor_y[1, 1, 0]
    x, y = np_swapaxes()
    assert x == 6 and y == 3


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_broadcast():
    """
    Feature: JIT Fallback
    Description: Test numpy.broadcast() method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_broadcast():
        x = np.array([[1], [2], [3]])
        y = np.array([4, 5, 6])
        z = np.broadcast(x, y)
        return Tensor(z.shape)
    assert np.all(np_broadcast().asnumpy() == np.array([3, 3]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_broadcast_to():
    """
    Feature: JIT Fallback
    Description: Test numpy.broadcast_to() method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_broadcast_to():
        x = np.arange(4).reshape(1, 4)
        y = np.broadcast_to(x, (2, 4))
        return Tensor(y)
    assert np.all(np_broadcast_to().asnumpy() == np.array([[0, 1, 2, 3], [0, 1, 2, 3]]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_expand_dims():
    """
    Feature: JIT Fallback
    Description: Test numpy.expand_dims() method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_expand_dims():
        x = np.array(([1, 2], [3, 4]))
        y = np.expand_dims(x, axis=0)
        return Tensor(y)
    assert np.all(np_expand_dims().asnumpy() == np.array([[[1, 2], [3, 4]]]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_squeeze():
    """
    Feature: JIT Fallback
    Description: Test numpy.squeeze() method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_squeeze():
        x = np.arange(4).reshape(1, 2, 2)
        y = np.squeeze(x)
        return Tensor(y)
    assert np.all(np_squeeze().asnumpy() == np.array([[0, 1], [2, 3]]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_concat():
    """
    Feature: JIT Fallback
    Description: Test numpy method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_concat():
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        concatenate = np.concatenate((x, y))
        stack = np.stack((x, y), 0)
        hstack = np.hstack((x, y))
        vstack = np.vstack((x, y))
        return Tensor(concatenate), Tensor(stack), Tensor(hstack), Tensor(vstack)

    out_concatenate, out_stack, out_hstack, out_vstack = np_concat()
    assert np.all(out_concatenate.asnumpy() == np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    assert np.all(out_stack.asnumpy() == np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    assert np.all(out_hstack.asnumpy() == np.array([[1, 2, 5, 6], [3, 4, 7, 8]]))
    assert np.all(out_vstack.asnumpy() == np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_split():
    """
    Feature: JIT Fallback
    Description: Test numpy split method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_split():
        x = np.arange(4).reshape(2, 2)
        split = np.split(x, 2)
        hsplit = np.hsplit(x, 2)
        vsplit = np.vsplit(x, 2)
        return Tensor(split), Tensor(hsplit), Tensor(vsplit)

    out_split, out_hsplit, out_vsplit = np_split()
    assert np.all(out_split.asnumpy() == np.array([[[0, 1]], [[2, 3]]]))
    assert np.all(out_hsplit.asnumpy() == np.array([[[0], [2]], [[1], [3]]]))
    assert np.all(out_vsplit.asnumpy() == np.array([[[0, 1]], [[2, 3]]]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_element():
    """
    Feature: JIT Fallback
    Description: Test numpy method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_element():
        resize = np.resize(np.array([[1, 2, 3], [4, 5, 6]]), (3, 2))
        append = np.append(np.array([[1, 2, 3], [4, 5, 6]]), [[7, 8, 9]], axis=0)
        insert = np.insert(np.array([[1, 2], [3, 4], [5, 6]]), 3, [7, 8], axis=0)
        delete = np.delete(np.arange(6).reshape(2, 3), 0, axis=0)
        unique = np.unique(np.array([5, 2, 6, 2, 7, 5, 6, 8, 2, 9]))
        return Tensor(resize), Tensor(append), Tensor(insert), Tensor(delete), Tensor(unique)

    out_resize, out_append, out_insert, out_delete, out_unique = np_element()
    assert np.all(out_resize.asnumpy() == np.array([[1, 2], [3, 4], [5, 6]]))
    assert np.all(out_append.asnumpy() == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert np.all(out_insert.asnumpy() == np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    assert np.all(out_delete.asnumpy() == np.array([3, 4, 5]))
    assert np.all(out_unique.asnumpy() == np.array([2, 5, 6, 7, 8, 9]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_bitwise():
    """
    Feature: JIT Fallback
    Description: Test numpy bitwise method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_bitwise():
        bitwise_and = np.bitwise_and(13, 17)
        bitwise_or = np.bitwise_or(13, 17)
        invert = np.invert(np.array([13], dtype=np.uint8))
        left_shift = np.left_shift(10, 2)
        right_shift = np.right_shift(40, 2)
        return Tensor(bitwise_and), Tensor(bitwise_or), Tensor(invert), Tensor(left_shift), Tensor(right_shift)

    bitwise_and, bitwise_or, invert, left_shift, right_shift = np_bitwise()
    assert bitwise_and.asnumpy() == 1
    assert bitwise_or.asnumpy() == 29
    assert np.all(invert.asnumpy() == np.array([242]))
    assert left_shift.asnumpy() == 40
    assert right_shift.asnumpy() == 10


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_char_1():
    """
    Feature: JIT Fallback
    Description: Test numpy char method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_char():
        char_add = np.char.add(['MindSpore'], [' fallback'])
        char_multiply = np.char.multiply('fallback ', 3)
        char_center = np.char.center('fallback', 10, fillchar='*')
        char_capitalize = np.char.capitalize('fallback')
        char_title = np.char.title('fallback')
        char_lower = np.char.lower('FALLBACK')
        char_upper = np.char.upper('fallback')
        return Tensor(char_add), Tensor(char_multiply), Tensor(char_center), Tensor(char_capitalize), \
               Tensor(char_title), Tensor(char_lower), Tensor(char_upper)

    char_add, char_multiply, char_center, char_capitalize, char_title, char_lower, char_upper = np_char()
    assert char_add.asnumpy() == 'MindSpore fallback'
    assert char_multiply.asnumpy() == 'fallback fallback fallback '
    assert char_center.asnumpy() == '*fallback*'
    assert char_capitalize.asnumpy() == 'Fallback'
    assert char_title.asnumpy() == 'Fallback'
    assert char_lower.asnumpy() == 'fallback'
    assert char_upper.asnumpy() == 'FALLBACK'


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_char_2():
    """
    Feature: JIT Fallback
    Description: Test numpy char method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_char():
        char_split = np.char.split('MindSpore fallback')
        out_split = np.char.join(' ', char_split)

        char_splitlines = np.char.splitlines('MindSpore\nfallback')
        out_splitlines = np.char.join(',', char_splitlines)

        out_strip = np.char.strip('abc acd', 'a')
        out_replace = np.char.replace('faooback', 'oo', 'll')
        char_encode = np.char.encode('runoob', 'cp500')
        out_decode = np.char.decode(char_encode, 'cp500')
        return Tensor(out_split), Tensor(out_splitlines), Tensor(out_strip), Tensor(out_replace), Tensor(out_decode)

    char_split, char_splitlines, char_strip, char_replace, char_decode = np_char()
    assert char_split.asnumpy() == 'MindSpore fallback'
    assert char_splitlines.asnumpy() == 'MindSpore,fallback'
    assert char_strip.asnumpy() == 'bc acd'
    assert char_replace.asnumpy() == 'fallback'
    assert char_decode.asnumpy() == 'runoob'


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_degree():
    """
    Feature: JIT Fallback
    Description: Test numpy method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_degree():
        out_sin = np.sin(30 * np.pi / 180)
        out_arcsin = np.degrees(np.arcsin(out_sin))
        out_cos = np.cos(60 * np.pi / 180)
        out_arccos = np.degrees(np.arccos(out_cos))
        out_tan = np.tan(45 * np.pi / 180)
        out_arctan = np.degrees(np.arctan(out_tan))
        return Tensor(out_sin), Tensor(out_arcsin), Tensor(out_cos), \
               Tensor(out_arccos), Tensor(out_tan), Tensor(out_arctan)

    out_sin, out_arcsin, out_cos, out_arccos, out_tan, out_arctan = np_degree()
    assert np.isclose(out_sin.asnumpy(), 0.5)
    assert np.isclose(out_arcsin.asnumpy(), 30)
    assert np.isclose(out_cos.asnumpy(), 0.5)
    assert np.isclose(out_arccos.asnumpy(), 60)
    assert np.isclose(out_tan.asnumpy(), 1)
    assert np.isclose(out_arctan.asnumpy(), 45)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_math_1():
    """
    Feature: JIT Fallback
    Description: Test numpy math method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_math():
        x = np.array([6, 12])
        y = np.array([3, 5])
        out_add = np.add(x, y)
        out_subtract = np.subtract(x, y)
        out_multiply = np.multiply(x, y)
        out_divide = np.divide(x, y)
        out_mod = np.mod(x, y)
        out_remainder = np.remainder(x, y)
        return Tensor(out_add), Tensor(out_subtract), Tensor(out_multiply), \
               Tensor(out_divide), Tensor(out_mod), Tensor(out_remainder)

    out_add, out_subtract, out_multiply, out_divide, out_mod, out_remainder = np_math()
    assert np.all(out_add.asnumpy() == np.array([9, 17]))
    assert np.all(out_subtract.asnumpy() == np.array([3, 7]))
    assert np.all(out_multiply.asnumpy() == np.array([18, 60]))
    assert np.allclose(out_divide.asnumpy(), np.array([2, 2.4]))
    assert np.all(out_mod.asnumpy() == np.array([0, 2]))
    assert np.all(out_remainder.asnumpy() == np.array([0, 2]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_math_2():
    """
    Feature: JIT Fallback
    Description: Test numpy math method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_math():
        x = np.array([0.1, 1.4, 2.51, 3.3])
        out_around = np.around(x)
        out_floot = np.floor(x)
        out_ceil = np.ceil(x)
        out_reciprocal = np.reciprocal(np.array([0.25, 1, 2]))
        out_power = np.power(np.array([1.0, 2.0, 3.0]), 2)
        return Tensor(out_around), Tensor(out_floot), Tensor(out_ceil), Tensor(out_reciprocal), Tensor(out_power)

    out_around, out_floot, out_ceil, out_reciprocal, out_power = np_math()
    assert np.allclose(out_around.asnumpy(), np.array([0, 1, 3, 3]))
    assert np.allclose(out_floot.asnumpy(), np.array([0, 1, 2, 3]))
    assert np.allclose(out_ceil.asnumpy(), np.array([1, 2, 3, 4]))
    assert np.allclose(out_reciprocal.asnumpy(), np.array([4, 1, 0.5]))
    assert np.allclose(out_power.asnumpy(), np.array([1, 4, 9]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_statistic():
    """
    Feature: JIT Fallback
    Description: Test numpy statistic method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_statistic():
        x = np.array([1, 2, 3, 4, 5])
        out_amin = np.amin(x)
        out_amax = np.amax(x)
        out_ptp = np.ptp(x)
        out_percentile = np.percentile(x, 50)
        out_median = np.median(x)
        out_mean = np.mean(x)
        out_average = np.average(x)
        out_sqrt = np.std(x)
        out_var = np.var(x)
        return Tensor(out_amin), Tensor(out_amax), Tensor(out_ptp), Tensor(out_percentile), \
               Tensor(out_median), Tensor(out_mean), Tensor(out_average), Tensor(out_sqrt), Tensor(out_var)

    out_amin, out_amax, out_ptp, out_percentile, out_median, out_mean, out_average, out_sqrt, out_var = np_statistic()
    assert out_amin.asnumpy() == 1
    assert out_amax.asnumpy() == 5
    assert out_ptp.asnumpy() == 4
    assert np.isclose(out_percentile.asnumpy(), 3.0)
    assert out_median.asnumpy() == 3
    assert out_mean.asnumpy() == 3
    assert out_average.asnumpy() == 3
    assert np.allclose(out_sqrt.asnumpy(), np.array([1.41421356]))
    assert np.isclose(out_var.asnumpy(), 2.0)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_sort():
    """
    Feature: JIT Fallback
    Description: Test numpy method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_sort():
        x = np.array([3, 1, 2, 4, 5])
        out_sort = np.sort(x)
        out_argsort = np.argsort(x)
        out_argmax = np.argmax(x)
        out_argmin = np.argmin(x)
        out_nonzero = np.nonzero(x)
        out_where = np.where(x > 4)
        return Tensor(out_sort), Tensor(out_argsort), Tensor(out_argmax), \
               Tensor(out_argmin), Tensor(out_nonzero), Tensor(out_where)

    out_sort, out_argsort, out_argmax, out_argmin, out_nonzero, out_where = np_sort()
    assert np.all(out_sort.asnumpy() == np.array([1, 2, 3, 4, 5]))
    assert np.all(out_argsort.asnumpy() == np.array([1, 2, 0, 3, 4]))
    assert out_argmax.asnumpy() == 4
    assert out_argmin.asnumpy() == 1
    assert np.all(out_nonzero.asnumpy() == np.array([0, 1, 2, 3, 4]))
    assert np.all(out_where.asnumpy() == np.array([4]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_extract():
    """
    Feature: JIT Fallback
    Description: Test numpy extract method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_extract():
        x = np.array([3, 1, 2, 4, 5])
        condition = x % 2 == 0
        out_extract = np.extract(condition, x)
        return Tensor(out_extract)

    out_extract = np_extract()
    assert np.all(out_extract.asnumpy() == np.array([2, 4]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_matrix():
    """
    Feature: JIT Fallback
    Description: Test numpy matrix method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_matrix():
        x = np.arange(4).reshape(2, 2)
        y = np.array([[2, 2], [3, 3]])
        out_t = x.T
        out_dot = np.dot(x, y)
        out_vdot = np.vdot(x, y)
        out_inner = np.inner(x, y)
        out_matmul = np.matmul(x, y)
        out_det = np.linalg.det(x)
        out_inv = np.linalg.inv(x)
        return Tensor(out_t), Tensor(out_dot), Tensor(out_vdot), Tensor(out_inner), \
               Tensor(out_matmul), Tensor(out_det), Tensor(out_inv)

    out_t, out_dot, out_vdot, out_inner, out_matmul, out_det, out_inv = np_matrix()
    assert np.all(out_t.asnumpy() == np.array([[0, 2], [1, 3]]))
    assert np.all(out_dot.asnumpy() == np.array([[3, 3], [13, 13]]))
    assert out_vdot.asnumpy() == 17
    assert np.all(out_inner.asnumpy() == np.array([[2, 3], [10, 15]]))
    assert np.all(out_matmul.asnumpy() == np.array([[3, 3], [13, 13]]))
    assert np.isclose(out_det.asnumpy(), -2.0)
    assert np.allclose(out_inv.asnumpy(), np.array([[-1.5, 0.5], [1, 0]]))
