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
# ==============================================================================
import os
import re
import time
import pytest
import numpy as np
import mindspore as ms
import mindspore.ops.operations as P
import mindspore.nn as nn
from mindspore.nn import Cell
from mindspore.nn import ReLU, BatchNorm2d, Conv2d, Dense, PReLU, ParameterUpdate
from mindspore.nn import Momentum, SoftmaxCrossEntropyWithLogits
from mindspore import context, Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.ops.primitive import constexpr
from capture import Capture, capture, check_output

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


@pytest.fixture(name="pynative_save_graphs")
def _pynative_save_graphs():
    context.set_context(mode=context.PYNATIVE_MODE, save_graphs=True)
    yield
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)
    clean_all_ir_files('./')


@pytest.fixture(name="with_save_graphs")
def _with_save_graphs():
    context.set_context(save_graphs=True)
    yield
    context.set_context(save_graphs=False)
    clean_all_ir_files('./')


def test_print():
    class Print(Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, x, y):
            self.print("input_x:", x, "input_y:", y)
            return x

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        input_y = Tensor(4, dtype=ms.int32)
        net = Print()
        net(input_x, input_y)
        time.sleep(0.1)

    patterns = {'input_x:\nTensor(shape=[], dtype=Int32, value=3)\n'
                'input_y:\nTensor(shape=[], dtype=Int32, value=4)'}
    check_output(cap.output, patterns)


def test_print_add():
    class Print_Add(Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()
            self.add = P.Add()

        def construct(self, x, y):
            x = self.add(x, y)
            self.print("input_x:", x, "input_y:", y)
            return x

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        input_y = Tensor(4, dtype=ms.int32)
        expect = Tensor(7, dtype=ms.int32)
        net = Print_Add()
        out = net(input_x, input_y)
        time.sleep(0.1)
        np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())

    patterns = {'input_x:\nTensor(shape=[], dtype=Int32, value=7)\n'
                'input_y:\nTensor(shape=[], dtype=Int32, value=4)'}
    check_output(cap.output, patterns)


def test_print_assign():
    class Print_Assign(Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')

        def construct(self, x):
            self.print("before:", self.para)
            self.para = x
            self.print("after:", self.para)
            return self.para

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        expect = Tensor(3, dtype=ms.int32)
        net = Print_Assign()
        out = net(input_x)
        time.sleep(0.1)
        np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())

    patterns = {'before:\nTensor(shape=[], dtype=Int32, value=1)',
                'after:\nTensor(shape=[], dtype=Int32, value=3)'}
    check_output(cap.output, patterns)


def test_print_assign_add():
    class Print_Assign_Add(Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()
            self.add = P.Add()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')

        def construct(self, x, y):
            self.print("before:", self.para)
            self.para = x
            self.print("after:", self.para)
            x = self.add(self.para, y)
            return x

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        input_y = Tensor(4, dtype=ms.int32)
        expect = Tensor(7, dtype=ms.int32)
        net = Print_Assign_Add()
        out = net(input_x, input_y)
        time.sleep(0.1)
        np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())

    patterns = {'before:\nTensor(shape=[], dtype=Int32, value=1)',
                'after:\nTensor(shape=[], dtype=Int32, value=3)'}
    check_output(cap.output, patterns)


def test_print_while():
    class Print_While(Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, x, y):
            self.print("input_x before:", x, "input_y before:", y)
            while x < y:
                self.print("input_x after:", x, "input_y after:", y)
                x = x + 1
            return x

    cap = Capture()
    with capture(cap):
        input_x = Tensor(1, dtype=ms.int32)
        input_y = Tensor(4, dtype=ms.int32)
        expect = Tensor(4, dtype=ms.int32)
        net = Print_While()
        out = net(input_x, input_y)
        time.sleep(0.1)
        np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())

    patterns = {'input_x before:\nTensor(shape=[], dtype=Int32, value=1)\n'
                'input_y before:\nTensor(shape=[], dtype=Int32, value=4)',
                'input_x after:\nTensor(shape=[], dtype=Int32, value=1)\n'
                'input_y after:\nTensor(shape=[], dtype=Int32, value=4)',
                'input_x after:\nTensor(shape=[], dtype=Int32, value=2)\n'
                'input_y after:\nTensor(shape=[], dtype=Int32, value=4)',
                'input_x after:\nTensor(shape=[], dtype=Int32, value=3)\n'
                'input_y after:\nTensor(shape=[], dtype=Int32, value=4)'}
    check_output(cap.output, patterns)


def test_print_if():
    class Print_If(Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, x, y):
            self.print("input_x before:", x, "input_y before:", y)
            if x < y:
                self.print("input_x after:", x, "input_y after:", y)
                x = x + 1
            return x

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        input_y = Tensor(4, dtype=ms.int32)
        expect = Tensor(4, dtype=ms.int32)
        net = Print_If()
        out = net(input_x, input_y)
        time.sleep(0.1)
        np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())

    patterns = {'input_x before:\nTensor(shape=[], dtype=Int32, value=3)\n'
                'input_y before:\nTensor(shape=[], dtype=Int32, value=4)',
                'input_x after:\nTensor(shape=[], dtype=Int32, value=3)\n'
                'input_y after:\nTensor(shape=[], dtype=Int32, value=4)'}
    check_output(cap.output, patterns)


def test_print_assign_while():
    class Print_Assign_While(Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()
            self.para = Parameter(Tensor(0, dtype=ms.int32), name='para')

        def construct(self, x, y):
            self.print("input_x before:", x, "input_y before:",
                       y, "para before:", self.para)
            while x < y:
                self.para = x
                x = self.para + 1
                self.print("input_x after:", x, "input_y after:",
                           y, "para after:", self.para)
            return x

    cap = Capture()
    with capture(cap):
        input_x = Tensor(1, dtype=ms.int32)
        input_y = Tensor(4, dtype=ms.int32)
        expect = Tensor(4, dtype=ms.int32)
        net = Print_Assign_While()
        out = net(input_x, input_y)
        time.sleep(0.1)
        np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())

    patterns = {
        'input_x before:\nTensor(shape=[], dtype=Int32, value=1)\n'
        'input_y before:\nTensor(shape=[], dtype=Int32, value=4)\n'
        'para before:\nTensor(shape=[], dtype=Int32, value=0)',
        'input_x after:\nTensor(shape=[], dtype=Int32, value=2)\n'
        'input_y after:\nTensor(shape=[], dtype=Int32, value=4)\n'
        'para after:\nTensor(shape=[], dtype=Int32, value=1)',
        'input_x after:\nTensor(shape=[], dtype=Int32, value=3)\n'
        'input_y after:\nTensor(shape=[], dtype=Int32, value=4)\n'
        'para after:\nTensor(shape=[], dtype=Int32, value=2)',
        'input_x after:\nTensor(shape=[], dtype=Int32, value=4)\n'
        'input_y after:\nTensor(shape=[], dtype=Int32, value=4)\n'
        'para after:\nTensor(shape=[], dtype=Int32, value=3)'}
    check_output(cap.output, patterns)


def test_print_assign_if():
    class Print_Assign_If(Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')

        def construct(self, x, y):
            self.print("input_x before:", x, "input_y before:",
                       y, "para before:", self.para)
            self.para = x
            if x < y:
                x = self.para + 1
                self.print("input_x after:", x, "input_y after:",
                           y, "para after:", self.para)
            return x

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        input_y = Tensor(4, dtype=ms.int32)
        expect = Tensor(4, dtype=ms.int32)
        net = Print_Assign_If()
        out = net(input_x, input_y)
        time.sleep(0.1)
        np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())

    patterns = {
        'input_x before:\nTensor(shape=[], dtype=Int32, value=3)\n'
        'input_y before:\nTensor(shape=[], dtype=Int32, value=4)\n'
        'para before:\nTensor(shape=[], dtype=Int32, value=1)',
        'input_x after:\nTensor(shape=[], dtype=Int32, value=4)\n'
        'input_y after:\nTensor(shape=[], dtype=Int32, value=4)\n'
        'para after:\nTensor(shape=[], dtype=Int32, value=3)'}
    check_output(cap.output, patterns)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_assign():
    class Assign(Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')

        def construct(self, value):
            self.para = value
            return self.para

    input_x = Tensor(3, dtype=ms.int32)
    expect = Tensor(3, dtype=ms.int32)
    net = Assign()
    out = net(input_x)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_assign_implicit():
    class Assign_Implicit(Cell):
        def __init__(self):
            super(Assign_Implicit, self).__init__()
            self.b = Parameter(initializer(
                1, [5], ms.float32), name="global_step")

        def construct(self, w):
            self.b = w
            return self.b

    input_data = Tensor(np.ones([5]).astype(np.int32))
    net = Assign_Implicit()
    out = net(input_data)
    assert out.dtype == ms.float32


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_assign_write_after_read():
    class Assign_WAR(Cell):
        def __init__(self):
            super(Assign_WAR, self).__init__()
            self.assign = P.Assign()
            self.sub = P.Sub()
            self.add = P.Add()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')
            self.weight = Parameter(Tensor(5, dtype=ms.int32), name='weight')

        def construct(self, x, y):
            # without auto_monad, execute order is wrong: Add - Assign - Sub - Assign
            # expected execute order: Add - Assign - Assign - Sub
            self.para = self.add(y, x)
            self.assign(self.para, y)
            return self.sub(self.para, self.weight)

    input_x = Tensor(3, dtype=ms.int32)
    input_y = Tensor(4, dtype=ms.int32)
    expect = Tensor(-1, dtype=ms.int32)
    net = Assign_WAR()
    out = net(input_x, input_y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_assign_read_after_write():
    class Assign_RAW(Cell):
        def __init__(self):
            super(Assign_RAW, self).__init__()
            self.assign_add = P.AssignAdd()
            self.greater = P.Greater()
            self.add = P.Add()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')

        def construct(self, x, y):
            # without auto_monad, execute order is wrong: Add - Assign - Greater - AssignAdd
            # expected execute order: AssignAdd - Add - Assign
            self.greater(x, y)
            self.assign_add(self.para, x)
            self.para = self.add(x, y)
            return self.para

    input_x = Tensor(3, dtype=ms.int32)
    input_y = Tensor(4, dtype=ms.int32)
    expect = Tensor(7, dtype=ms.int32)
    net = Assign_RAW()
    out = net(input_x, input_y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_assign_if():
    class Assign_If(Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')

        def construct(self, x, y):
            if x < y:
                self.para = x
            else:
                self.para = y
            return self.para

    input_x = Tensor(3, dtype=ms.int32)
    input_y = Tensor(4, dtype=ms.int32)
    expect = Tensor(3, dtype=ms.int32)
    net = Assign_If()
    out = net(input_x, input_y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if():
    class If(Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.sub = P.Sub()

        def construct(self, x, y):
            if x > y:
                x = self.sub(x, y)
            else:
                x = self.add(x, y)
            return x

    input_x = Tensor(3, dtype=ms.int32)
    input_y = Tensor(4, dtype=ms.int32)
    expect = Tensor(7, dtype=ms.int32)
    net = If()
    out = net(input_x, input_y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while():
    class While(Cell):
        def construct(self, x, y):
            y = y + 4
            while x < y:
                x = x + 1
            x = x + 3
            return x

    input_x = Tensor(2, dtype=ms.int32)
    input_y = Tensor(14, dtype=ms.int32)
    expect = Tensor(21, dtype=ms.int32)
    net = While()
    out = net(input_x, input_y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_assign_while():
    class Assign_While(Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')

        def construct(self, x, y):
            y = y + 4
            while x < y:
                x = x + 1
                self.para = x
            self.para = x - 1
            return self.para

    input_x = Tensor(2, dtype=ms.int32)
    input_y = Tensor(14, dtype=ms.int32)
    expect = Tensor(17, dtype=ms.int32)
    net = Assign_While()
    out = net(input_x, input_y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for():
    class For(Cell):
        def construct(self, x, y):
            y = x + y
            for _ in range(20):
                y = y + 1
            return y

    input_x = Tensor(2, dtype=ms.int32)
    input_y = Tensor(4, dtype=ms.int32)
    expect = Tensor(26, dtype=ms.int32)
    net = For()
    out = net(input_x, input_y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


def test_print_for():
    class Print_For(Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, x, y):
            y = x + y
            self.print("input_x before:", x, "input_y before:", y)
            for _ in range(3):
                y = y + 1
                self.print("input_x after:", x, "input_y after:", y)
            return y

    cap = Capture()
    with capture(cap):
        input_x = Tensor(2, dtype=ms.int32)
        input_y = Tensor(4, dtype=ms.int32)
        expect = Tensor(9, dtype=ms.int32)
        net = Print_For()
        out = net(input_x, input_y)
        time.sleep(0.1)
        np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())

    patterns = {
        'input_x before:\nTensor(shape=[], dtype=Int32, value=2)\n'
        'input_y before:\nTensor(shape=[], dtype=Int32, value=6)',
        'input_x after:\nTensor(shape=[], dtype=Int32, value=2)\n'
        'input_y after:\nTensor(shape=[], dtype=Int32, value=7)',
        'input_x after:\nTensor(shape=[], dtype=Int32, value=2)\n'
        'input_y after:\nTensor(shape=[], dtype=Int32, value=8)',
        'input_x after:\nTensor(shape=[], dtype=Int32, value=2)\n'
        'input_y after:\nTensor(shape=[], dtype=Int32, value=9)'}
    check_output(cap.output, patterns)


def test_print_assign_for():
    class Print_Assign_For(Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')

        def construct(self, x, y):
            y = x + y
            self.print("input_x before:", x, "input_y before:",
                       y, "para before:", self.para)
            for _ in range(3):
                y = y + 1
                self.para = x + y
                self.print("input_x after:", x, "input_y after:",
                           y, "para after:", self.para)
            return y

    cap = Capture()
    with capture(cap):
        input_x = Tensor(2, dtype=ms.int32)
        input_y = Tensor(4, dtype=ms.int32)
        expect = Tensor(9, dtype=ms.int32)
        net = Print_Assign_For()
        out = net(input_x, input_y)
        time.sleep(0.1)
        np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())

    patterns = {
        'input_x before:\nTensor(shape=[], dtype=Int32, value=2)\n'
        'input_y before:\nTensor(shape=[], dtype=Int32, value=6)\n'
        'para before:\nTensor(shape=[], dtype=Int32, value=1)',
        'input_x after:\nTensor(shape=[], dtype=Int32, value=2)\n'
        'input_y after:\nTensor(shape=[], dtype=Int32, value=7)\n'
        'para after:\nTensor(shape=[], dtype=Int32, value=9)',
        'input_x after:\nTensor(shape=[], dtype=Int32, value=2)\n'
        'input_y after:\nTensor(shape=[], dtype=Int32, value=8)\n'
        'para after:\nTensor(shape=[], dtype=Int32, value=10)',
        'input_x after:\nTensor(shape=[], dtype=Int32, value=2)\n'
        'input_y after:\nTensor(shape=[], dtype=Int32, value=9)\n'
        'para after:\nTensor(shape=[], dtype=Int32, value=11)'}
    check_output(cap.output, patterns)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_assign_for():
    class Assign_For(Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')

        def construct(self, x, y):
            y = y + 4
            for _ in range(5):
                x = x + y
                self.para = x
            return self.para

    input_x = Tensor(2, dtype=ms.int32)
    input_y = Tensor(3, dtype=ms.int32)
    expect = Tensor(37, dtype=ms.int32)
    net = Assign_For()
    out = net(input_x, input_y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@constexpr
def _check_shape(shape):
    if len(shape) != 1:
        raise ValueError(f"Invalid shape {shape}")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_constexpr_check():
    class ConstexprCheck(Cell):
        def __init__(self):
            super(ConstexprCheck, self).__init__()
            self.shape = P.Shape()

        def construct(self, x, y):
            s = self.shape(x)
            _check_shape(s)
            x = x + y
            return x

    x = Tensor([2], dtype=ms.int32)
    y = Tensor([3], dtype=ms.int32)
    expect = Tensor(5, dtype=ms.int32)
    net = ConstexprCheck()
    # Input with valid shape.
    out = net(x, y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())
    # Input with wrong shape, exception is expected.
    with pytest.raises(ValueError):
        wrong_x = Tensor(np.ones((2, 2)), dtype=ms.int32)
        out = net(wrong_x, y)
        print(out)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_lambda():
    class If_Lambda(Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')

        def construct(self, x, y):
            out = x
            if x < y:
                x2 = (lambda a: a + a)
                out = x2(self.para)
                out = out + y
            return out

    input_x = Tensor(2, dtype=ms.int32)
    input_y = Tensor(3, dtype=ms.int32)
    expect = Tensor(5, dtype=ms.int32)
    net = If_Lambda()
    out = net(input_x, input_y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_multi_assign():
    class Multi_Assign(Cell):
        def __init__(self):
            super().__init__()
            self.assign = P.Assign()
            self.para1 = Parameter(Tensor(1, dtype=ms.int32), name='para1')
            self.para2 = Parameter(Tensor(2, dtype=ms.int32), name='para2')
            self.para3 = Parameter(Tensor(3, dtype=ms.int32), name='para3')

        def construct(self, x, y, z):
            a = self.assign(self.para1, x)
            a = self.assign(self.para2, y)
            a = self.assign(self.para3, z)
            return self.para1 + self.para2 + a

    x = Tensor(4, dtype=ms.int32)
    y = Tensor(5, dtype=ms.int32)
    z = Tensor(6, dtype=ms.int32)
    expect = Tensor(15, dtype=ms.int32)
    net = Multi_Assign()
    out = net(x, y, z)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_multi_assign_addn():
    class Multi_Assign_Addn(Cell):
        def __init__(self):
            super().__init__()
            self.addn = P.AddN()
            self.assign = P.Assign()
            self.para1 = Parameter(Tensor(1.0, dtype=ms.float32), name='para1')
            self.para2 = Parameter(Tensor(3.0, dtype=ms.float32), name='para2')

        def construct(self, inputs):
            self.assign(self.para1, inputs)
            out = self.addn((inputs, self.para1, self.para2))
            self.assign(self.para2, inputs)
            out = self.addn((out, self.para1, self.para2))
            return out

    x = Tensor(9.0, dtype=ms.float32)
    expect = Tensor(39.0, dtype=ms.float32)
    net = Multi_Assign_Addn()
    out = net(x)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


def test_multi_assign_print():
    class Multi_Assign_Print(Cell):
        def __init__(self):
            super().__init__()
            self.pow = P.Pow()
            self.print = P.Print()
            self.assign = P.Assign()
            self.exponent = Tensor([2.0], ms.float32)
            self.para1 = Parameter(Tensor(1.0, dtype=ms.float32), name='para1')
            self.para2 = Parameter(Tensor(3.0, dtype=ms.float32), name='para2')

        def construct(self, inputs):
            self.assign(self.para1, inputs)
            self.assign(self.para2, self.pow(inputs, self.exponent))
            self.print(inputs)
            self.print(self.para1)
            self.print(self.para2)
            return inputs

    x = Tensor(9.0, dtype=ms.float32)
    expect = Tensor(9.0, dtype=ms.float32)
    expect_para1 = Tensor(9.0, dtype=ms.float32)
    expect_para2 = Tensor(81.00001, dtype=ms.float32)
    net = Multi_Assign_Print()
    out = net(x)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())
    np.testing.assert_almost_equal(
        net.para1.data.asnumpy(), expect_para1.asnumpy())
    np.testing.assert_almost_equal(
        net.para2.data.asnumpy(), expect_para2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_assign_biasadd():
    class Matmul_Assign_Biasadd(Cell):
        def __init__(self):
            super().__init__()
            inputs = np.array([[1, 1], [1, 1]])
            self.parameter1 = Parameter(
                Tensor(inputs, ms.float32), name="parameter1")
            biasadd = np.array([0, -1])
            self.parameter2 = Parameter(
                Tensor(biasadd, ms.float32), name="biasadd")
            self.assign = P.Assign()
            self.matmul = P.MatMul()
            self.biasadd = P.BiasAdd()

        def construct(self, x):
            self.assign(self.parameter1, x)
            x = self.matmul(x, self.parameter1)
            self.assign(self.parameter1, x)
            x = self.biasadd(x, self.parameter2)
            return x

    net = Matmul_Assign_Biasadd()
    inputs = np.array([[1, 2], [3, 4]])
    out1 = net(Tensor(inputs, ms.float32))
    net = Matmul_Assign_Biasadd()
    try:
        context.set_context(mode=context.PYNATIVE_MODE)
        out2 = net(Tensor(inputs, ms.float32))
        np.testing.assert_almost_equal(out1.asnumpy(), out2.asnumpy())
    finally:
        context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_assign_while_if():
    class Assign_While_If(Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.addn = P.AddN()
            self.assign = P.Assign()
            self.assign_sub = P.AssignSub()
            self.para = Parameter(Tensor(1.0, dtype=ms.float32), name='para')

        def construct(self, x, y, z, w):
            self.assign(self.para, x)
            if self.para > y:
                self.assign(self.para, y)
                x = self.mul(x, x)
            while self.para > z:
                x = self.addn((x, self.para))
                self.assign_sub(self.para, w)
            return x

    x = Tensor(99.0, dtype=ms.float32)
    y = Tensor(44.0, dtype=ms.float32)
    z = Tensor(11.0, dtype=ms.float32)
    w = Tensor(1.0, dtype=ms.float32)
    expect = Tensor(10725.0, dtype=ms.float32)
    net = Assign_While_If()
    out = net(x, y, z, w)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_isolate_call():
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.para1 = Parameter(Tensor(1, dtype=ms.int32), name='para1')
            self.para2 = Parameter(Tensor(2, dtype=ms.int32), name='para2')

        def construct(self, x, y):
            self.setpara(x, y)
            return self.para1 + self.para2

        def setpara(self, x, y):
            self.para1 = x
            self.setpara2(y)
            return x

        def setpara2(self, y):
            self.para2 = y
            return y

    x = Tensor(4, dtype=ms.int32)
    y = Tensor(5, dtype=ms.int32)
    expect = Tensor(9, dtype=ms.int32)
    net = Net()
    out = net(x, y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_assign_return_true():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')

        def construct(self, x, y):
            if self.mycheck(x, y):
                out = x + y
            else:
                out = x - y
            out = self.para + out
            return out

        def mycheck(self, x, y):
            self.setpara(x, y)
            return True

        def setpara(self, x, y):
            self.para = x + y
            return True

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(3, dtype=ms.int32)
    expect = Tensor(10, dtype=ms.int32)
    net = Net()
    out = net(x, y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_unpack_call():
    class SetPara(Cell):
        def __init__(self, para):
            super(SetPara, self).__init__()
            self.para = para

        def construct(self, x, y):
            self.para = x + y
            return True

    class MyNet(Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')
            self.set_para = SetPara(self.para)

        def construct(self, *inputs):
            self.call_func(self.set_para, *inputs)
            out = self.para + 1
            return out

        def call_func(self, func, *inputs):
            func(*inputs)
            return True

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(3, dtype=ms.int32)
    expect = Tensor(6, dtype=ms.int32)
    net = MyNet()
    out = net(x, y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tuple_of_tuple():
    class SetPara(Cell):
        def __init__(self, para):
            super(SetPara, self).__init__()
            self.para = para

        def construct(self, x, y):
            self.para = x + y
            return True

    class MyNet(Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')
            self.set_para = SetPara(self.para)

        def construct(self, x, y):
            t1 = (self.set_para, x)
            t2 = (t1, y)
            t2[0][0](t2[1], t1[1])
            out = self.para + 1
            return out

        def call_func(self, func, *inputs):
            func(*inputs)
            return True

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(3, dtype=ms.int32)
    expect = Tensor(6, dtype=ms.int32)
    net = MyNet()
    out = net(x, y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_write_read_write():
    class MyNet(Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.para1 = Parameter(Tensor(1, dtype=ms.int32), name='para1')
            self.para2 = Parameter(Tensor(2, dtype=ms.int32), name='para2')

        def construct(self, x, y, x1, y1):
            self.para1 = x
            self.para2 = y
            a = self.para1 + self.para2
            self.para1 = x1
            self.para2 = y1
            return a + self.para1 + self.para2

    x = Tensor(3, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    x1 = Tensor(5, dtype=ms.int32)
    y1 = Tensor(6, dtype=ms.int32)
    expect = Tensor(18, dtype=ms.int32)
    net = MyNet()
    out = net(x, y, x1, y1)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_variable_from_outer_graph():
    class MyNet(Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.cond = False
            self.add = P.Add()
            self.para = Parameter(Tensor(1, dtype=ms.int32), name='para')

        def construct(self, x, y):
            b = self.para + x
            a = self.para + b
            if self.cond:
                a = self.add(a, x)
            else:
                a = self.add(a, y)
            return a + b

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(3, dtype=ms.int32)
    expect = Tensor(10, dtype=ms.int32)
    net = MyNet()
    out = net(x, y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ctrl_while_by_while_and_if_in_first_while():
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.sigmoid = P.Sigmoid()
            self.tanh = P.Tanh()
            self.add = P.Add()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            out = x
            while self.a < 7:
                if self.a < self.c:
                    out = self.relu(x)
                self.a += 1
            while self.c > 5:
                out = self.add(out, out)
                self.c -= 1
            return out

    context.set_context(mode=context.GRAPH_MODE)
    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_a = Tensor(input_np_a)
    net = Net()
    net(input_me_a)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ctrl_if_by_while_and_while_in_first_if():
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.sigmoid = P.Sigmoid()
            self.tanh = P.Tanh()
            self.add = P.Add()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            out = x
            if self.a < self.c:
                out = self.relu(x)
                while self.a < 7:
                    self.a += 1

            while self.c > 5:
                out = self.add(out, out)
                self.c -= 1
            return out

    context.set_context(mode=context.GRAPH_MODE)
    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_a = Tensor(input_np_a)
    net = Net()
    net(input_me_a)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ctrl_while_by_while_and_while_in_first_while():
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.sigmoid = P.Sigmoid()
            self.tanh = P.Tanh()
            self.add = P.Add()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            out = x
            while self.a < self.c:
                out = self.relu(x)
                while self.b > 1:
                    self.b -= 1
                self.a += 1

            while self.c > 5:
                out = self.add(out, out)
                self.c -= 1
            return out

    context.set_context(mode=context.GRAPH_MODE)
    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_a = Tensor(input_np_a)
    net = Net()
    net(input_me_a)


def clear_json_info():
    os.system("rm -rf ./kernel_meta/*.json")
    os.system("rm -rf ./kernel_meta/*.info")


def find_json_info(file):
    result = os.system("ls -al ./kernel_meta/%s" % (file))
    return result


class MultiOutReluBywaySqrt(Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sqrt = P.Sqrt()

    def construct(self, x):
        x = self.relu(x)
        x = self.relu(x)
        x1 = self.relu(x)
        x = self.relu(x1)
        y = self.sqrt(x1)
        return x, y


class MultiOutReluSqrtBywaySqrt(Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sqrt = P.Sqrt()
        self.sin = P.Sin()

    def construct(self, x):
        x = self.relu(x)
        x = self.sqrt(x)
        x1 = self.relu(x)
        x = self.sin(x1)
        y = self.sqrt(x1)
        return x, y


def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat') or file_name.endswith('.pb') or \
                    file_name.startswith('trace_code_graph'):
                os.remove(os.path.join(folder_path, file_name))


def find_newest_validateir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'\d+_validate_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def read_file():
    filename = find_newest_validateir_file('./')
    with open((os.path.join(filename)), 'r') as f:
        content = f.read()
    return content


# Net contain Prelu,BN,Conv,Dense which have weight value
class NetRrelu(Cell):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = PReLU(channel=in_channel, w=0.25)
        self.bn = BatchNorm2d(num_features=in_channel)
        self.conv = Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=1, has_bias=False,
                           weight_init='ones', pad_mode='same')
        self.mean = P.ReduceMean(keep_dims=False)
        self.fc = Dense(in_channels=out_channel, out_channels=out_channel,
                        weight_init='ones', bias_init='zeros', has_bias=True)

    def construct(self, x):
        x = self.relu(x)
        x = self.bn(x)
        x = self.conv(x)
        x = self.mean(x, (2, 3))
        x = self.fc(x)
        return x


def check_keep_batchnorm_fp32_false(kwargs, level):
    if ms.context.get_context("device_target") == "GPU":
        if level == "O2":
            if "keep_batchnorm_fp32" in kwargs.keys() and (not kwargs["keep_batchnorm_fp32"]):
                if "cast_model_type" not in kwargs.keys() or kwargs["cast_model_type"] == ms.float16:
                    return True
        else:
            if "cast_model_type" in kwargs.keys() and kwargs["cast_model_type"] == ms.float16:
                if "keep_batchnorm_fp32" not in kwargs.keys() or (not kwargs["keep_batchnorm_fp32"]):
                    return True
    return False


def use_build_train_network_check_cast_num(network, level, inputs, label, cast_num, loss_flag=True, **kwargs):
    diff_cast = 0
    if check_keep_batchnorm_fp32_false(kwargs, level):
        diff_cast += 8
    opt = Momentum(learning_rate=0.0001, momentum=0.009,
                   params=network.trainable_params())
    loss = None
    if loss_flag:
        loss = SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')

    train_network = ms.amp.build_train_network(
        network, opt, loss, level=level, **kwargs)
    out_me = train_network(inputs, label)
    if context.get_context("mode") == 0:
        content = read_file()
        castnum = re.findall('Cast', content)
        assert len(castnum) == max(cast_num - diff_cast, 0)
    return out_me


def test_auto_mixed_precision_train_prelunet(with_save_graphs):
    net2 = NetRrelu(3, 12)
    input32 = Tensor(np.ones([1, 3, 2, 2]).astype(np.float32))
    label32 = Tensor(np.zeros([1, 12]).astype(np.float32))
    use_build_train_network_check_cast_num(net2, "O2", input32, label32, 16)


class AssignNet(Cell):
    def __init__(self):
        super().__init__()
        #self._save_graphs(save_graph_flag=True, save_graph_path=".")
        self.relu = ReLU()
        self.mean = P.ReduceMean(keep_dims=False)
        self.assign_sub = P.AssignSub()
        self.input_data = Parameter(initializer(
            1, [1, 3, 2, 2], ms.float32), name='value')

    def construct(self, x):
        x = self.assign_sub(self.input_data, x)
        x = self.relu(x)
        x = self.mean(x, (2, 3))
        return x


def test_auto_mixed_precision_train_1(pynative_save_graphs):
    net = AssignNet()
    input32 = Tensor(np.ones([1, 3, 2, 2]).astype(np.float32))
    label32 = Tensor(np.zeros([1, 3]).astype(np.float32))
    use_build_train_network_check_cast_num(net, "O0", input32, label32, 0)


def test_auto_mixed_precision_train_2(pynative_save_graphs):
    net = AssignNet()
    input32 = Tensor(np.ones([1, 3, 2, 2]).astype(np.float32))
    label32 = Tensor(np.zeros([1, 3]).astype(np.float32))
    use_build_train_network_check_cast_num(net, "O2", input32, label32, 2)


class MixControlNet(Cell):
    def __init__(self, in_channel, x):
        super().__init__()
        #self._save_graphs(save_graph_flag=True, save_graph_path=".")
        self.biasadd = P.BiasAdd()
        self.equal = P.Equal()
        self.addn = P.AddN()
        self.conv = Conv2d(in_channels=in_channel, out_channels=in_channel,
                           kernel_size=1, stride=1, has_bias=False,
                           weight_init='ones', pad_mode='same')
        self.bn = BatchNorm2d(num_features=in_channel)
        self.assignadd = P.AssignAdd()
        self.assign = P.Assign()
        self.relu = ReLU()
        self.mean = P.ReduceMean(keep_dims=False)
        self.bias = Parameter(
            Tensor(np.random.randint(2, size=(3,)).astype((np.float32))),
            name="bias")
        self.bias2 = Parameter(Tensor(np.ones([3]).astype(np.float32)),
                               name="bias2")
        self.parameterupdate = ParameterUpdate(self.bias)
        self.value = Tensor(np.random.randn(*(3,)), ms.float32)
        self.x = x

    def construct(self, input_x):
        x = self.x
        z = self.x
        out = self.biasadd(input_x, self.bias)
        while x < 20:
            update = self.parameterupdate(self.bias2)
            out = self.biasadd(out, update)
            if x < 10:
                out = self.addn((input_x, out))
                while z < 20:
                    out = self.conv(out)
                    z = z + 1
            if x < 20:
                out = self.biasadd(out, self.bias)
                if x % 2 == 0:
                    out = self.biasadd(out, self.bias)
                    self.assignadd(self.bias, self.value)
                    out = self.bn(out)
                else:
                    out = self.conv(out)
            x = x + 1
        out = self.addn((out, out))
        out = self.mean(out, (2, 3))
        return out


def use_build_train_network_controlflow_check_cast_num(network, level, input_x,
                                                       label, cast_num,
                                                       sparse=False,
                                                       loss_flag=True,
                                                       **kwargs):
    opt = Momentum(learning_rate=0.0001, momentum=0.009,
                   params=network.trainable_params())
    loss = None
    if loss_flag:
        loss = SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction='mean')

    train_network = ms.amp.build_train_network(network, opt, loss, level=level,
                                               **kwargs)
    out_me = train_network(input_x, label)
    if context.get_context("mode") == 0:
        content = read_file()
        castnum = re.findall('Cast', content)
        assert len(castnum) == cast_num
    return out_me


def test_auto_mixed_precision_controlflow_auto(pynative_save_graphs):
    net = MixControlNet(3, 5)
    input_x = Tensor(
        np.random.randint(2, size=(1, 3, 2, 2)).astype((np.float32)))
    label = Tensor(np.zeros([1, 3]).astype(np.float32))
    if ms.context.get_context("device_target") == "Ascend":
        cast_num = 77
    if ms.context.get_context("device_target") == "GPU":
        cast_num = 73
    use_build_train_network_controlflow_check_cast_num(net, "auto", input_x,
                                                       label, cast_num)


# op_cast should be located in order_list after abstract_specialize.
# Besides Ascend, it can work on CPU.
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_cast():
    class Net(nn.Cell):
        def __init__(self, cond1):
            super().__init__()
            self.cond1 = cond1
            self.op_cast = P.Cast()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, beta1, beta2):
            z_local = self.op_cast(self.z, ms.float16)
            self.z = beta2
            if self.cond1:
                out = z_local + beta1
            else:
                out = z_local - beta1

            return out

    context.set_context(save_graphs=False)
    net = Net(True)
    beta1 = Tensor(np.array([2]).astype(np.float32))
    beta2 = Tensor(np.array([10]).astype(np.float32))
    r1 = net(beta1, beta2)
    expect = Tensor(np.array([3]).astype(np.float32))
    np.testing.assert_array_equal(r1.asnumpy(), expect.asnumpy())


@pytest.mark.skip(reason="not supported yet")
def test_multi_add_assign():
    class Net(Cell):
        def __init__(self, i1):
            super(Net, self).__init__()
            self.add = P.Add()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.assign = P.Assign()
            self.p = Parameter(i1, name='para')

        def construct(self, a, d, e):
            res1 = self.add(self.add(self.add(self.p, a), a), a)
            mul = self.mul(d, e)
            self.assign(self.p, mul)
            res2 = self.sub(self.p, e)
            return res2, res1

    def numpy_out(p, a, d, e):
        res1 = p + a + a + a
        res_as = d * e
        res2 = d * e - e
        return res2, res1, res_as

    p = (np.abs(np.random.normal(0, 1, [3])) + 1).astype(np.float32)
    i0 = (np.abs(np.random.normal(0, 1, [3])) + 1).astype(np.float32)
    i1 = (np.abs(np.random.normal(0, 1, [3])) + 1).astype(np.float32)
    i2 = (np.abs(np.random.normal(0, 1, [3])) + 1).astype(np.float32)

    net = Net(Tensor(p))
    r2, r1 = net(Tensor(i0), Tensor(i1), Tensor(i2))

    outputs = [r2.asnumpy(), r1.asnumpy(), net.p.data.asnumpy()]
    expects = numpy_out(p, i0, i1, i2)
    np.testing.assert_array_equal(outputs, expects)


@pytest.mark.skip(reason="not supported yet")
def test_multi_abs_add_assign():
    class Net(Cell):
        def __init__(self, para):
            super(Net, self).__init__()
            self.add = P.Add()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.abs = P.Abs()
            self.assign = P.Assign()
            self.p = Parameter(para, name='para')

        def construct(self, a, d, e):
            tmp = self.abs(self.add(self.abs(a), self.abs(self.mul(a, a))))
            res1 = self.add(self.p, tmp)
            mul = self.mul(d, e)
            self.assign(self.p, mul)
            res2 = self.sub(self.p, e)
            return res2, res1, tmp

    def numpy_out(p, a, d, e):
        tmp = np.abs(np.abs(a) + np.abs(a * a))
        res1 = p + tmp
        res_as = d * e
        res2 = d * e - e
        return res2, res1, res_as, tmp

    p = -(np.abs(np.random.normal(0, 1, [3])) + 1).astype(np.float32)
    i0 = -(np.abs(np.random.normal(0, 1, [3])) + 1).astype(np.float32)
    i1 = -(np.abs(np.random.normal(0, 1, [3])) + 1).astype(np.float32)
    i2 = -(np.abs(np.random.normal(0, 1, [3])) + 1).astype(np.float32)

    net = Net(Tensor(p))
    r2, r1, tmp = net(Tensor(i0), Tensor(i1), Tensor(i2))

    outputs = [r2.asnumpy(), r1.asnumpy(), net.p.data.asnumpy(), tmp.asnumpy()]
    expects = numpy_out(p, i0, i1, i2)
    np.testing.assert_array_equal(outputs, expects)
