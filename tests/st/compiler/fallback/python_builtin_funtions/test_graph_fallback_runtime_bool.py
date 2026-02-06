# Copyright 2025 Huawei Technologies Co., Ltd
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
""" bool test """
import pytest
import mindspore as ms
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_bool_tensor_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test bool() in fallback runtime
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor([1, 2, 3]).asnumpy()
        return bool(all(x - [1, 2, 3]))

    out = foo()
    assert not out


@pytest.mark.skip(reason="RebuildKernelSelectBackoffOp Unsupported op[Shape].")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_fallback_bool_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test bool() in fallback runtime
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return bool(x.asnumpy())

    x = Tensor([-1.0], ms.float32)
    res = foo(x)
    assert res


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_fallback_bool_int():
    """
    Feature: JIT Fallback
    Description: Test bool() in fallback runtime
    Expectation: No exception.
    """
    @jit
    def func():
        x1 = bool(int)
        x2 = bool(1)
        x3 = bool(0)
        return x1, x2, x3

    x1, x2, x3 = func()
    assert x1 and x2 and not x3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_fallback_bool_empty():
    """
    Feature: JIT Fallback
    Description: Test bool() in fallback runtime
    Expectation: No exception.
    """
    @jit
    def func():
        x = bool()
        return x

    assert not func()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_fallback_bool_seq():
    """
    Feature: JIT Fallback
    Description: Test bool() in fallback runtime
    Expectation: No exception.
    """
    @jit
    def func():
        x1 = bool([1, 2, 3, 4])
        x2 = bool((1, 2))
        x3 = bool([])
        x4 = bool(tuple())
        return x1, x2, x3, x4

    x1, x2, x3, x4 = func()
    assert x1 and x2 and not x3 and not x4


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_fallback_bool_str():
    """
    Feature: JIT Fallback
    Description: Test bool() in fallback runtime
    Expectation: No exception.
    """
    @jit
    def func():
        x = bool("")
        y = bool("123")
        return x, y

    x, y = func()
    assert not x and y


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_fallback_bool_none_and_complex():
    """
    Feature: JIT Fallback
    Description: Test bool() in fallback runtime
    Expectation: No exception.
    """
    @jit
    def func():
        x1 = bool(None)
        x2 = bool(complex(0, 0))
        x3 = bool(complex(1, 0))
        x4 = bool(complex(0, 1))
        return x1, x2, x3, x4

    x1, x2, x3, x4 = func()
    assert (not x1) and (not x2) and x3 and x4
