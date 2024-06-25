# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test basic operation with one stage"""
import pytest
import math
import numpy as np
import mindspore.nn as nn
from math import cos
from mindspore import Tensor, context
from mindspore.common.api import jit
from tests.mark_utils import arg_mark

cfg = {
    "replace_nncell_by_construct": True,
    "print_after_all": False,
    "compile_by_trace": True,
    "print_bb": False,
    "MAX_INLINE_DEPTH": 10,
    "allowed_inline_modules": ["mindspore"],  # buildsubgraph
}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_return_dict():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            m = {"1": x+1, "2": y+1}
            return m

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    b = Tensor([2])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a, b)
    assert ret == {"1": Tensor([2]), "2": Tensor([3])}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_return_dict_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = {"1": x+1}
            return m

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert ret == {"1": Tensor([2])}


@pytest.mark.skip(reason="CodeHook for one stage failed")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_break_in_subgraph():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, inner_net):
            super(Net, self).__init__()
            self.inner_net = inner_net

        def construct(self, x):
            a = self.inner_net(x, x)
            return isinstance(a, int)

    class InnerNet(nn.Cell):
        def construct(self, x, y):
            m = x + y
            return type(m)

    context.set_context(mode=context.PYNATIVE_MODE)
    inner_net = InnerNet()
    net = Net(inner_net)
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert not ret


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_break_in_subgraph_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def out(x, y):
        m = x + y
        n = inner(x, y)
        ret = m/n
        return ret

    def inner(a, b):
        c = a - b
        return cos(c)

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = out(Tensor([1]), Tensor([2]))
    assert np.allclose(ret.asnumpy(), 5.5524473)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_break_in_subgraph_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def out(x, y):
        m = x + y
        n = inner(x, y)
        ret = m/n
        return ret

    def inner(a, b):
        c = a - b
        return math.cos(c)

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = out(Tensor([1]), Tensor([2]))
    assert np.allclose(ret.asnumpy(), 5.5524473)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip
def test_break_with_control_flow():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def out():
        x = np.array([3, 2])
        if x[0] > 1:
            x += 3
        return x

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = out()
    assert np.all(ret == np.array([6, 5]))


@pytest.mark.skip(reason="Random error occurs when run whole files")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_break_with_control_flow_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def out(a):
        a = a + 1
        x = np.array([3, 2])
        if x[0] > 1:
            x += 3
        return x, a

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = out(Tensor([1, 2, 3]))
    assert len(ret) == 2
    assert np.all(ret[0] == np.array([6, 5]))
    assert np.all(ret[1].asnumpy() == np.array([2, 3, 4]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_break_with_same_value():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def out(x):
        a, b, c, d = x
        return type(a), type(b), type(c), type(d)

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = out((1, 1, 1, 2))
    assert isinstance(ret, tuple)
    assert len(ret) == 4
    assert ret[0] == int
    assert ret[1] == int
    assert ret[2] == int
    assert ret[3] == int


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ud_collect_capture_output():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def foo(x, y):
        m = ((x, x+1), x+2)
        n = ((y, y-1), y+2)
        return m < n, m <= n, m > n, m >= n

    context.set_context(mode=context.PYNATIVE_MODE)
    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4


@pytest.mark.skip # One-stage will fix it later
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_while_after_for_in_if_4():
    """
    Feature: PIJit
    Description: Test PIJit with control flow.
    Expectation: No exception.
    """

    @jit(mode="PIJit", jit_config=cfg)
    def foo():
        x = [3, 2]
        y = [1, 2, 3, 4]
        if x[0] > x[1]:
            x[0] += 3
            x[1] += 3
            for i in y:
                if not i == 1:
                    break
                x[1] += i
        x = np.array(x)
        z = int(x[1])
        while len(y) < 5:
            y.append(z)
        return Tensor(y)

    context.set_context(mode=context.PYNATIVE_MODE)
    res = foo()
    assert (res.asnumpy() == [1, 2, 3, 4, 6]).all()
