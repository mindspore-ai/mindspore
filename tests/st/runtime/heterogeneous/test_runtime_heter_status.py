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
"""Test runtime heterogeneous scene."""
import os
from mindspore import Tensor, jit, ops
from mindspore.common import dtype as mstype
import mindspore.runtime as rt
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_single_if():
    """
    Feature: Heter status in kernel tensor.
    Description: Sync for one time.
    Expectation: Not throw exception.
    """

    @jit(backend="ms_backend")
    def foo(x, y):
        cond = x > y
        if cond:
            z = x + 1
        else:
            z = y * 3
        if cond:
            z = z + 1
        else:
            z = z * 3
        if cond:
            z = z + 1
        else:
            z = z * 3
        return x - 3, z - x

    x = Tensor(5, mstype.int32)
    y = Tensor(7, mstype.int32)
    ret = foo(x, y)
    assert ret == (Tensor(2, mstype.int32), Tensor(184, mstype.int32))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_single_if_twice():
    """
    Feature: Heter status in kernel tensor.
    Description: Sync for one time.
    Expectation: Not throw exception.
    """

    @jit(backend="ms_backend")
    def foo(x, y):
        cond = x > y
        if cond:
            z = x + 1
        else:
            z = y * 3
        if cond:
            z = z + 1
        else:
            z = z * 3
        if cond:
            z = z + 1
        else:
            z = z * 3
        return x - 3, z - x

    x = Tensor(5, mstype.int32)
    y = Tensor(7, mstype.int32)
    ret1 = foo(x=x, y=y)
    ret2 = foo(x=y, y=x)
    assert ret1 == (Tensor(2, mstype.int32), Tensor(184, mstype.int32))
    assert ret2 == (Tensor(4, mstype.int32), Tensor(3, mstype.int32))



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_single_if_twice_in_somas():
    """
    Feature: Heter status in kernel tensor.
    Description: Sync for one time.
    Expectation: Not throw exception.
    """
    rt.set_memory(optimize_level="O1")
    @jit(backend="ms_backend")
    def foo(x, y):
        cond = x > y
        if cond:
            z = x + 1
        else:
            z = y * 3
        if cond:
            z = z + 1
        else:
            z = z * 3
        if cond:
            z = z + 1
        else:
            z = z * 3
        return x - 3, z - x

    x = Tensor(5, mstype.int32)
    y = Tensor(7, mstype.int32)
    ret1 = foo(x=x, y=y)
    ret2 = foo(x=y, y=x)
    assert ret1 == (Tensor(2, mstype.int32), Tensor(184, mstype.int32))
    assert ret2 == (Tensor(4, mstype.int32), Tensor(3, mstype.int32))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_single_if_twice_inplace_condition():
    """
    Feature: Heter status in kernel tensor.
    Description: Sync for one time.
    Expectation: Not throw exception.
    """

    os.environ['MS_DEV_SIDE_EFFECT_LOAD_ELIM'] = '3'
    @jit(backend="ms_backend")
    def foo(x, y):
        cond = x > y
        if cond:
            z = x + 1
        else:
            z = y * 3
        cond2 = x < y
        ops.assign(cond, cond2)
        if cond:
            z = z + 1
        else:
            z = z * 3
        if cond:
            z = z + 1
        else:
            z = z * 3
        return x - 3, z - x

    x = Tensor(5, mstype.int32)
    y = Tensor(7, mstype.int32)
    ret1 = foo(x=x, y=y)
    ret2 = foo(x=y, y=x)
    os.unsetenv('MS_DEV_SIDE_EFFECT_LOAD_ELIM')
    assert ret1 == (Tensor(2, mstype.int32), Tensor(18, mstype.int32))
    assert ret2 == (Tensor(4, mstype.int32), Tensor(65, mstype.int32))
