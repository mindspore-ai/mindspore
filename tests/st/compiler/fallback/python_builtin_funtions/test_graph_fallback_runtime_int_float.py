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
""" int/float test """
import pytest
import mindspore as ms
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_int():
    """
    Feature: JIT Fallback
    Description: Test int() in fallback runtime
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return int(x)

    res = foo(Tensor(2))
    assert res == 2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_fallback_float():
    """
    Feature: JIT Fallback
    Description: Test float() in fallback runtime
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x):
        return float(x)

    res = foo(Tensor([-1.0]))
    assert res == -1.0


@pytest.mark.skip(reason="ScalarToRawMemory memcpy failed.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_fallback_int_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test int() in fallback runtime
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return int(x.asnumpy())

    x = Tensor([-1.0], ms.float32)
    res = foo(x)
    assert res == -1


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_fallback_int_str_base():
    """
    Feature: JIT Fallback
    Description: Test int() in fallback runtime
    Expectation: No exception.
    """
    @jit
    def func():
        x1 = int('12', 16)
        x2 = int('0xa', 16)
        x3 = int('10', 8)
        return x1, x2, x3

    x1, x2, x3 = func()
    assert x1 == 18 and x2 == 10 and x3 == 8
