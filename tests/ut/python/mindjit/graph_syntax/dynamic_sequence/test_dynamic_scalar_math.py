# Copyright 2022 Huawei Technologies Co., Ltd
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
"""test mul operation for dynamic sequence and variable integer in graph mode"""
import numpy as np
from mindspore import jit
from mindspore import context
from mindspore.common import mutable
from mindspore.ops import functional as F

context.set_context(mode=context.GRAPH_MODE)


def test_constant_scalar_div_and_mod():
    """
    Feature: Constant scalar div and mod operation.
    Description:
    Expectation: No exception.
    """

    @jit
    def foo():
        return 2/3, -2%3

    ret1, ret2 = foo()
    tol = 1e-6
    assert np.abs(ret1 - 0.666666) < tol
    assert np.abs(ret2 - 1) < tol


def test_constant_scalar_div_and_mod_bool():
    """
    Feature: Constant scalar div and mod operation.
    Description:
    Expectation: No exception.
    """

    @jit
    def foo():
        return False/True, True%True

    ret1, ret2 = foo()
    tol = 1e-6
    assert np.abs(ret1 - 0) < tol
    assert np.abs(ret2 - 0) < tol


def test_constant_scalar_bitwise():
    """
    Feature: Constant scalar bitwise operation.
    Description:
    Expectation: No exception.
    """

    @jit
    def foo():
        return 3 & 1, True & 4, 4 | 3

    ret1, ret2, ret3 = foo()
    assert ret1 == 1
    assert ret2 == 0
    assert ret3 == 7


def test_variable_scalar_div_and_mod():
    """
    Feature: Variable scalar div and mod operation.
    Description:
    Expectation: No exception.
    """

    @jit
    def foo():
        x = mutable(2)
        y = mutable(3)
        ret1 = x / y
        ret2 = x % y
        return isinstance(ret1, float), F.isconstant(ret2)

    ret1, ret2 = foo()
    assert ret1
    assert not ret2


def test_variable_scalar_bitwise():
    """
    Feature: Variable scalar bitwise operation.
    Description:
    Expectation: No exception.
    """

    @jit
    def foo():
        x = mutable(2)
        y = mutable(3)
        ret1 = x & y
        ret2 = x | y
        return isinstance(ret1, int), F.isconstant(ret2)

    ret1, ret2 = foo()
    assert ret1
    assert not ret2
