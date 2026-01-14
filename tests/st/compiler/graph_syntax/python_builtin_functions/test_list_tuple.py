# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
import numpy as np
import operator
import os
import mindspore as ms
from mindspore import Tensor, context, jit
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_list_with_input_constant_tensor():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with constant tensor.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = list(Tensor([1, 2, 3]))
        x.append(Tensor([4]))
        return x

    out = foo()
    assert isinstance(out, list)
    assert len(out) == 4
    assert isinstance(out[0], Tensor)
    assert out[0].asnumpy() == 1
    assert isinstance(out[1], Tensor)
    assert out[1].asnumpy() == 2
    assert isinstance(out[2], Tensor)
    assert out[2].asnumpy() == 3
    assert isinstance(out[3], Tensor)
    assert out[3].asnumpy() == 4


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_list_with_input_constant_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with constant tensor.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = list(Tensor([[1, 2], [3, 4]]))
        x.append(Tensor([5, 6]))
        return x

    out = foo()
    assert isinstance(out, list)
    assert len(out) == 3
    assert isinstance(out[0], Tensor)
    assert np.allclose(out[0].asnumpy(), np.array([1, 2]))
    assert isinstance(out[1], Tensor)
    assert np.allclose(out[1].asnumpy(), np.array([3, 4]))
    assert isinstance(out[2], Tensor)
    assert np.allclose(out[2].asnumpy(), np.array([5, 6]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.25)
def test_builtin_function_list_with_non_constant_tensor():
    """
    Feature: Graph list function.
    Description: When the input to list() is non constant tensor, list function will return correct result.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return list(x)

    ret = foo(Tensor([[1, 2, 3], [4, 5, 6]]))
    assert len(ret) == 2
    assert np.all(ret[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(ret[1].asnumpy() == np.array([4, 5, 6]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_tuple_with_input_constant_tensor():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with constant tensor.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = tuple(Tensor([1, 2, 3]))
        return x

    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert isinstance(out[0], Tensor)
    assert out[0].asnumpy() == 1
    assert isinstance(out[1], Tensor)
    assert out[1].asnumpy() == 2
    assert isinstance(out[2], Tensor)
    assert out[2].asnumpy() == 3


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_tuple_with_input_constant_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode with constant tensor.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = list(Tensor([[1, 2], [3, 4]]))
        return x

    out = foo()
    assert isinstance(out, list)
    assert len(out) == 2
    assert isinstance(out[0], Tensor)
    assert np.allclose(out[0].asnumpy(), np.array([1, 2]))
    assert isinstance(out[1], Tensor)
    assert np.allclose(out[1].asnumpy(), np.array([3, 4]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_list_with_input_numpy_array():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode with numpy array.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = list(np.array([1, 2, 3]))
        x.append(4)
        return Tensor(x)

    out = foo()
    assert np.allclose(np.array([1, 2, 3, 4]), out.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_list_append_in_tensor_index():
    """
    Feature: list().
    Description: Test list() in graph mode with new tensor index.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def func():
        x = list(Tensor([1, 2]))
        x.append(3)
        y = list(())
        y.append(3)
        z = list([Tensor([1, 2]), Tensor([2, 3])])
        z.append(3)
        return x, y, z

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        ms.set_context(jit_config={"jit_level": "O0"})
        out_type = list
        out, out_y, out_z = func()
        assert isinstance(out, out_type)
        assert operator.eq(out, out_type([Tensor(1), Tensor(2), 3]))
        assert isinstance(out_y, out_type)
        assert operator.eq(out_y, out_type([3]))
        assert isinstance(out_z, out_type)
        assert (out_z[0] == Tensor([1, 2])).all()
        assert (out_z[1] == Tensor([2, 3])).all()
        assert out_z[2] == 3
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tuple_in_tensor_index():
    """
    Feature: tuple().
    Description: Test tuple() in graph mode with new tensor index.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def func():
        x = tuple(Tensor([1, 2]))
        y = ()
        return x, y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        ms.set_context(jit_config={"jit_level": "O0"})
        out, out_y = func()
        assert isinstance(out, tuple)
        assert operator.eq(out, (Tensor(1), Tensor(2)))
        assert isinstance(out_y, tuple)
        assert not out_y
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]
