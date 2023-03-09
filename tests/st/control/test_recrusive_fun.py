# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from mindspore import Tensor, jit
from mindspore.common import dtype as mstype
from mindspore import ops
import mindspore.nn as nn
import numpy as np

ZERO = Tensor([0], mstype.int32)
ONE = Tensor([1], mstype.int32)


@jit
def f(x):
    y = ZERO
    if x < 0:
        y = f(x - 3)
    elif x < 3:
        y = x * f(x - 1)
    elif x < 5:
        y = x * f(x - 2)
    else:
        y = f(x - 4)
    z = y + 1
    return z


@jit
def fr(x):
    y = ZERO
    if x < 0:
        y = ONE
    elif x < 3:
        y = x * fr(x - 1)
    elif x < 5:
        y = x * fr(x - 2)
    else:
        y = fr(x - 4)
    z = y + 1
    return z


@jit
def f_pythonerr(x):
    if x > 0:
        return f_pythonerr(x - 1)
    return NOT_DEF


def test_python_error():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([5], mstype.int32)
    try:
        f_pythonerr(x)
    except NameError as e:
        assert 'not defined' in str(e)


@jit
def f_recrusive_endless(x):
    if x > 0:
        return f_recrusive_endless(x - 1)
    return f_recrusive_endless(x + 1)


def test_recrusive_endless():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([5], mstype.int32)
    try:
        f_recrusive_endless(x)
    except RuntimeError as e:
        assert 'loop' in str(e)


def test_endless():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([5], mstype.int32)
    try:
        f(x)
    except RuntimeError as e:
        assert 'loop' in str(e)


@jit
def f_ok(x):
    if x > 0:
        return f_ok(x - 1) + 1
    return ONE


def test_f_ok():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([3], mstype.int32)
    ret = f_ok(x)
    expect = Tensor([4], mstype.int32)
    assert ret == expect


def test_recrusive_fun():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([5], mstype.int32)
    ret = fr(x)
    expect = Tensor([3], mstype.int32)
    assert ret == expect


def test_branch_value_compatible():
    """
    Feature: control flow
    Description: test branch value must be compatible with the other branch.
    Expectation: Join Failed
    """
    class IfInWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.expand_dims = ops.ExpandDims()

        def construct(self, x, y, i):
            out = x
            while i < 3:
                if x + i < y:
                    out = out + x
                else:
                    out = out + y
                out = out + 1
                out = self.expand_dims(out, -1)
                i = i + 1
            return out

    forward_net = IfInWhileNet()
    i = Tensor(np.array(0), dtype=mstype.int32)
    x = Tensor(np.array(0), dtype=mstype.int32)
    y = Tensor(np.array(1), dtype=mstype.int32)

    try:
        forward_net(x, y, i)
    except RuntimeError as e:
        assert 'limit' in str(e)
    except ValueError as e:
        assert 'Join Failed' in str(e)


if __name__ == "__main__":
    test_endless()
