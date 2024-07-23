# Copyright 2023 Huawei Technologies Co., Ltd
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

import numpy as np
from collections import Iterator
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_runtime_zip_numpy():
    """
    Feature: JIT Fallback
    Description: Test zip() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo(x):
        ret = zip(np.array([1, 2]), x.asnumpy())
        return tuple(ret)

    x = Tensor(np.array([10, 20]))
    out = foo(x)
    assert out[0] == (1, 10)
    assert out[1] == (2, 20)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_runtime_zip_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test zip() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo(x):
        ret = zip(x.asnumpy(), x.asnumpy())
        return ret

    x = Tensor(np.array([10, 20]))
    out = foo(x)
    assert isinstance(out, Iterator)
    assert str(type(out)) == "<class 'zip'>"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_runtime_zip_asnumpy_tuple():
    """
    Feature: JIT Fallback
    Description: Test zip() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo(x):
        ret = zip(x.asnumpy(), x.asnumpy())
        return tuple(ret)

    x = Tensor(np.array([10, 20]))
    out = foo(x)
    assert isinstance(out, tuple)
    assert out == ((10, 10), (20, 20))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_runtime_zip_tensor():
    """
    Feature: JIT Fallback
    Description: Test zip() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo(x):
        out = 0
        for i in zip(x):
            out += i[0]
        return out

    x = Tensor(np.array([10, 20]))
    out = foo(x)
    assert out == 30


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_runtime_zip_string():
    """
    Feature: JIT Fallback
    Description: Test zip() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo():
        out = 0
        for _ in zip("abc"):
            out += 1
        return out

    out = foo()
    assert out == 3


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_runtime_zip_dict():
    """
    Feature: JIT Fallback
    Description: Test zip() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo():
        x = {'a': 1, 'b': 2}
        str_res = ""
        for i in zip(x):
            str_res += i[0]
        return str_res

    out = foo()
    assert out == "ab"
