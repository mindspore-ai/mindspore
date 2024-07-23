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

#!/usr/bin/env python3

import numpy as np

from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from mindspore import jit
from mindspore import ops
from tests.mark_utils import arg_mark

context.set_context(mode=context.PYNATIVE_MODE)
input_x = Tensor(np.ones([1, 1, 120, 640]), dtype=mstype.float32)
input_y = Tensor(np.full((1, 1, 120, 640), 4), dtype=mstype.float32)
ret_output_2 = Tensor(np.full((1, 1, 120, 640), 3.125), dtype=mstype.float32)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_nested_local():
    """
    Feature: Jit graph
    Description: jit nested call
    Expectation: No exception.
    """
    @jit
    def function1(x, y):
        x = x ** y
        x /= y
        x += y
        x -= 1
        x %= 2
        return x

    @jit
    def function11(x, y):
        r = function1(x, y)
        out = r + r
        return out

    @jit
    def function2(x, y):
        r1 = function1(x, y)
        r2 = function11(x, y)
        z = r1 * r2
        return z

    output2 = function2(input_x, input_y)
    assert np.allclose(output2.asnumpy(), ret_output_2.asnumpy(), 0.0001, 0.0001)


@jit
def function1_g(x, y):
    x = x ** y
    x /= y
    x += y
    x -= 1
    x %= 2
    return x


@jit
def function11_g(x, y):
    r = function1_g(x, y)
    out = r + r
    return out


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_nested_global():
    """
    Feature: Jit graph
    Description: jit top call
    Expectation: No exception.
    """
    @jit
    def function2_g(x, y):
        r1 = function1_g(x, y)
        r2 = function11_g(x, y)
        z = r1 * r2
        return z

    output2 = function2_g(input_x, input_y)
    assert np.allclose(output2.asnumpy(), ret_output_2.asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_nested_grad():
    """
    Feature: Nested call of jit
    Description: test nested call of jit
    Expectation: First derivative 75, Second derivative 30
    """
    x = Tensor([5], dtype=mstype.float32)
    exp1 = Tensor([75], dtype=mstype.float32)
    exp2 = Tensor([30], dtype=mstype.float32)
    def f(x):
        return x**3

    # 一阶：3*x^2 = 75
    out = jit(ops.grad(f))(x)
    assert np.allclose(out[0].asnumpy(), exp1[0].asnumpy(), 0.0001, 0.0001)
    out = jit(jit(ops.grad(f)))(x)
    assert np.allclose(out[0].asnumpy(), exp1[0].asnumpy(), 0.0001, 0.0001)

    # 二阶：6*x = 30
    out = ops.grad(ops.grad(f))(x)
    assert np.allclose(out[0].asnumpy(), exp2[0].asnumpy(), 0.0001, 0.0001)
    out = jit(ops.grad(ops.grad(f)))(x)
    assert np.allclose(out[0].asnumpy(), exp2[0].asnumpy(), 0.0001, 0.0001)
    out = jit(jit(ops.grad(ops.grad(f))))(x)
    assert np.allclose(out[0].asnumpy(), exp2[0].asnumpy(), 0.0001, 0.0001)
