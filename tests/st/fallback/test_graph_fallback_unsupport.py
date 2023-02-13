# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
""" syntax that JIT Fallback not support yet """

import math
from collections import deque
import pytest
import numpy as np

from mindspore import context
from mindspore import Tensor, jit
from mindspore.common import mutable

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_interpret_object():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Do not support to return interpret object yet.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return [1, x, np.array([1, 2, 3, 4])]

    output = foo(Tensor([2]))
    assert len(output) == 3
    assert output[0] == 1
    assert output[2] == Tensor([2])
    assert np.all(output[3], np.array([1, 2, 3, 4]))


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_error_in_variable_scene():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Do not support to raise error in variable scene.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        if x == y:
            raise ValueError("x and y is equal")
        return x - y

    output = foo(Tensor([2]), Tensor([1]))
    assert output == Tensor([1])


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_str_format_in_variable_scene():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Do not support to add variable in str.format.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        return "{}, {}".format(x, y)

    output = foo(Tensor([2]), Tensor([1]))
    assert output == "[2], [1]"


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_numpy_asarray_with_variable_scene():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: np.asarray() function can not use variable as input.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        a = [x, y]
        return np.asarray(a)

    output = foo(mutable(1), mutable(2))
    assert output == np.array([1, 2])


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_in_with_none():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: syntax 'in' do not support None input.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = [1, 2, None]
        return None in a

    assert foo()


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sequence_in_sequence():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: syntax 'in' do not support sequence in sequence check.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = [1, 2, [3, 4]]
        return [3, 4] in a

    assert foo()


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_all_with_variable():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: the label in all statement can not be parsed
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        a = x.asnumpy()
        b = all(ele < y for ele in a)
        return b

    assert foo(Tensor([1, 2, 3, 4]), Tensor([10]))


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_slice_with_variable():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Abstract slice can not in valid graph.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        a = slice(0, 3, 1)
        return x[a]

    assert foo(Tensor([1, 2, 3, 4, 5])) == Tensor([1, 2, 3])


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_slice_with_mutable_input():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: name 'mutable' is not defined.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        a = slice(mutable(0), 3, 1)
        return a.step

    assert foo(Tensor([1, 2, 3, 4, 5])) == 1


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_compress_with_mutable_input():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: the value including list is not correct.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        cond = [1, 0, 1, 1]
        a = [x, x+1, x+2, x+3]
        z = np.compress(cond, a)
        return z

    assert foo(Tensor([1])) == [Tensor([1]), Tensor([3]), Tensor([4])]


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_star_to_compress_input():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: use star to compress assigned input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [1, 2, 3, 4]
        a, *b = x
        return a, b

    ret = foo()
    assert len(ret) == 2
    assert ret[0] == 1
    assert ret[1] == [2, 3, 4]


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_math_ceil_with_variable():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: math.ceil(x) with variable x.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return math.ceil(x)

    ret = foo(mutable(10.75))
    assert ret == 11


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_unpack_interpret_node():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: * operator can not unpack a interpret object.
    Expectation: No exception.
    """

    def sum_func(a, b, c, d):
        return a + b + c + d

    def test(shape):
        reverse = reversed(shape)
        return sum_func(*reverse)

    @jit
    def foo(x):
        return test([1, 2, 3, 4])

    ret = foo([1, 2, 3, 4])
    assert ret == 10


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_starred_to_unpack_input():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: * operator can not unpack a list.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return f"output is {*a, }"

    ret = foo([1, 2, 3, 4])
    assert ret == "output is (1, 2, 3, 4)"


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_call_third_party_class():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: call third party class is not support in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        ret = deque()
        for i in x:
            ret.appendleft(i)
        return list(ret)

    ret = foo([1, 2, 3, 4])
    assert ret == [1, 2, 3, 4]


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_ix_with_variable():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: call third party class is not support in graph mode.
    Expectation: No exception.
    """

    def convert(start, stop, step):
        shape = [5,]
        grids = ([np.array(list(range(start, stop, step)), dtype=np.int64)] +
                 [np.array(list(range(dim_size)), dtype=np.int64) for dim_size in shape[1:]])
        mesh = np.ix_(*grids)
        return Tensor(np.stack(np.broadcast_arrays(*mesh), axis=-1))

    @jit
    def foo():
        return convert(mutable(0), mutable(5), mutable(1))

    ret = foo()
    assert ret == [[0], [1], [2], [3], [4]]


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_generate_tensor_using_variable_numpy_array():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: When using variable numpy array to generate Tensor, numpy array type is wrong.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable(2)
        ret = np.arange(a)
        return Tensor(ret)

    ret = foo()
    assert ret == Tensor([0, 1])


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_numpy_prod_with_variable_axis():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: numpy function do not support variable int input.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        a = x.asnumpy()
        return np.prod(a, axis=y)

    ret = foo(Tensor([1, 2], [3, 4]), mutable(1))
    assert np.all(ret == np.array([2, 12]))


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_with_interpret_object():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: interpret object can not get __len__ attribute.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        a = 0
        for i, j in zip(reversed(x), reversed(y)):
            a = a + i + j
        return a

    ret = foo([1, 2, 3], [4, 5, 6])
    assert ret == 21


@pytest.mark.skip(reason="not support now")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_with_interpret_object_2():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Value is wrong.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        a = []
        for i, j in list(zip(reversed(x), reversed(y))):
            a.append(i+j)
        return a

    ret = foo(mutable([1, 2, 3]), mutable([4, 5, 6]))
    assert ret == [9, 7, 5]
