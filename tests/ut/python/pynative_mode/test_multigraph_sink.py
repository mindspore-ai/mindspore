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
""" test_multigraph_sink """
import mindspore.context as context
from mindspore.common import dtype as mstype
from mindspore.common import jit
from mindspore.common.tensor import Tensor


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


c1 = Tensor([2], mstype.int32)
c2 = Tensor([14], mstype.int32)
c3 = Tensor([1], mstype.int32)
c4 = Tensor([0], mstype.int32)
c5 = Tensor([14], mstype.int32)


@jit
def simple_if(x, y, z):
    if x < y:
        x = x + 1
    else:
        x = x + 2
    x = x + 3
    return x


@jit
def if_by_if(x, y, z):
    if x < y:
        x = x + 1
    if y > x:
        x = x + 2
    x = x + 3
    return x


@jit
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


@jit
def simple_while(x, y, z):
    y = y + 4
    while x < y:
        x = x + 1
    x = x + 3
    return x


@jit
def while_by_while(x, y, z):
    while x < y:
        x = x + 1
    while z < c5:
        z = z + 1
        x = x + 1
    x = x + 1
    return x


@jit
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


def test_simple_if():
    output = simple_if(c1, c2, c3)
    expect = Tensor([6], mstype.int32)
    assert output == expect


def test_if_by_if():
    output = if_by_if(c1, c2, c3)
    expect = Tensor([8], mstype.int32)
    assert output == expect


def test_if_in_if():
    output = if_in_if(c1, c2, c3)
    expect = Tensor([7], mstype.int32)
    assert output == expect


def test_simple_while():
    output = simple_while(c1, c2, c3)
    expect = Tensor([21], mstype.int32)
    assert output == expect


def test_while_by_while():
    output = while_by_while(c1, c2, c3)
    expect = Tensor([28], mstype.int32)
    assert output == expect


def test_while_in_while():
    output = while_in_while(c1, c2, c3)
    expect = Tensor([1274], mstype.int32)
    assert output == expect


@jit
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


def test_while_by_while_in_while():
    output = while_by_while_in_while(c1, c2, c3)
    expect = Tensor([350], mstype.int32)
    assert output == expect
