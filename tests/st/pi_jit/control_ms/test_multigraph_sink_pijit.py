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
"""test mutigraph sink in PIJit and pynative mode"""
import pytest

from mindspore.common import dtype as mstype
from mindspore.common import jit
from mindspore.common.tensor import Tensor
from mindspore import context
from tests.mark_utils import arg_mark


c1 = Tensor([2], mstype.int32)
c2 = Tensor([14], mstype.int32)
c3 = Tensor([1], mstype.int32)
c4 = Tensor([0], mstype.int32)
c5 = Tensor([14], mstype.int32)


#TODO: fix CODEHOOK
def simple_if(x, y):
    if x < y:
        x = x + 1
    else:
        x = x + 2
    x = x + 3
    return x


#TODO: fix CODEHOOK
def if_by_if(x, y):
    if x < y:
        x = x + 1
    if y > x:
        x = x + 2
    x = x + 3
    return x


#TODO: fix CODEHOOK
def if_in_if(x, y, z):
    out = c4
    if x < y:
        z = c4 + c4
        if z < y:
            z = z + 2
            out = out + z
        x = x + 3
    out = out + x
    return out


@jit(mode="PIJit")
def simple_while(x, y):
    y = y + 4
    while x < y:
        x = x + 1
    x = x + 3
    return x


@jit(mode="PIJit")
def while_by_while(x, y, z):
    while x < y:
        x = x + 1
    while z < c5:
        z = z + 1
        x = x + 1
    x = x + 1
    return x


@jit(mode="PIJit")
def while_in_while(x, y, z):
    out = c4
    while x < y:
        z = c4 + c4
        while z < y:
            z = z + 1
            out = out + z
        x = x + 1
    out = out + x
    return out


@jit(mode="PIJit")
def while_by_while_in_while(x, y, z):
    out = c4
    while x < c2:
        y = c4 + c4
        while y < c2:
            y = y + 1
        out = out + y
        z = c4 + c4
        while z < c2:
            z = z + 1
        out = out + z
        x = x + 1
    out = out + x
    return out


@jit(mode="PIJit")
def while_in_while_in_while(x, y, z):
    out = c4
    while x < c2:
        y = c4 + c4
        while y < c2:
            y = y + 1
            z = c4 + c4
            while z < c2:
                z = z + 1
            out = out + z
        out = out + y
        x = x + 1
    out = out + x
    return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_simple_if():
    """
    Feature: if
    Description: Test simple if else control flow
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    output = simple_if(c1, c2)
    expect = Tensor([6], mstype.int32)
    assert output == expect


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_if_by_if():
    """
    Feature: if by if
    Description: Test simple if else control flow
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    output = if_by_if(c1, c2)
    expect = Tensor([8], mstype.int32)
    assert output == expect


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_if_in_if():
    """
    Feature: if in if
    Description: Test simple if if control flow
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    output = if_in_if(c1, c2, c3)
    expect = Tensor([7], mstype.int32)
    assert output == expect


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_simple_while():
    """
    Feature: while
    Description: Test simple while control flow
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    output = simple_while(c1, c2)
    expect = Tensor([21], mstype.int32)
    assert output == expect


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_while_by_while():
    """
    Feature: while by while
    Description: Test while by while control flow
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    output = while_by_while(c1, c2, c3)
    expect = Tensor([28], mstype.int32)
    assert output == expect


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_while_in_while_pi():
    """
    Feature: while in while
    Description: Test while in while control flow
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    output = while_in_while(c1, c2, c3)
    expect = Tensor([1274], mstype.int32)
    assert output == expect


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_while_by_while_in_while():
    """
    Feature: while
    Description: Test while by while in while control flow
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    output = while_by_while_in_while(c1, c2, c3)
    expect = Tensor([350], mstype.int32)
    assert output == expect


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_while_in_while_in_while():
    """
    Feature: while
    Description: Test while in while in while control flow
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    output = while_in_while_in_while(c1, c2, c3)
    expect = Tensor([2534], mstype.int32)
    assert output == expect
