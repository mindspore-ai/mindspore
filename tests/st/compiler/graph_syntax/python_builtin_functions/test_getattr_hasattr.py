# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""test graph getattr, hasattr"""
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, jit, context, jit_class
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_getattr_tensor():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        abs_func = getattr(x, "abs")
        return abs_func()

    out = foo(Tensor([-1, -2, -3]))
    assert np.all(out.asnumpy() == np.array([1, 2, 3]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_getattr_tensor_with_concate_string():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input and concate string.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        attr_str = "a" + "bs"
        abs_func = getattr(x, attr_str)
        return abs_func()

    out = foo(Tensor([-1, -2, -3]))
    assert np.all(out.asnumpy() == np.array([1, 2, 3]))


@jit_class
class MSClass1:
    def __init__(self):
        self.num0 = Tensor(0)
        self.num1 = Tensor(1)
        self.num2 = Tensor(2)
        self.num3 = Tensor(3)
        self.none = None


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_getattr_ms_class_with_default():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support ms_class input.
    Expectation: No exception.
    """
    ms_obj = MSClass1()

    @jit
    def foo():
        return getattr(ms_obj, "none", 10)

    out = foo()
    assert out is None


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.a0 = Tensor([0])
        self.a1 = Tensor([1])
        self.a2 = Tensor([2])
        self.a3 = Tensor([3])
        self.none = None

    def construct(self):
        return self.a0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_getattr_cell_obj_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support cell object input.
    Expectation: No exception.
    """
    cell_obj = Net()

    @jit
    def foo():
        return getattr(cell_obj, "none")

    out = foo()
    assert out is None


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_getattr_for_classtype_object():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support class type object.
    Expectation: No Exception
    """

    class User:
        x = 1

    class UserNet(nn.Cell):
        def construct(self):
            return User.x, getattr(User, "y", 10)

    context.set_context(mode=context.GRAPH_MODE)
    net = UserNet()
    out1, out2 = net()
    assert out1 == 1
    assert out2 == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_getattr_for_other_cell_method():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support nn.Cell.
    Expectation: No Exception
    """

    class NetBase(nn.Cell):
        def construct(self, x):
            return x * 2

    class UserNet(NetBase):
        def construct(self, x):
            return NetBase.construct(self, x), getattr(NetBase, "y", 20)

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(5)
    net = UserNet()
    out1, out2 = net(x)
    assert out1 == 10
    assert out2 == 20


class SubCellClass1(nn.Cell):
    def __init__(self):
        super(SubCellClass1, self).__init__()
        self.a = 1

    def construct(self, x):
        @jit
        def add(x):
            return x + self.a

        return add(x)


class SubCellClass2(nn.Cell):
    def __init__(self):
        super(SubCellClass2, self).__init__()
        self.a = 1

    def construct(self, x):
        b = self

        @jit
        def add(x):
            return x + b.a

        return add(x)


class PlainClass():
    def __init__(self):
        self.a = 1

    def construct(self, x):
        @jit
        def add(x):
            return x + self.a

        return add(x)


def test_getattr_from_self_in_function_closure():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr from self in function.
    Expectation: No Exception
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    SubCellClass1()(1)
    SubCellClass2()(1)
    PlainClass().construct(1)


def test_getattr_from_no_called_cell():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr from a cell instance which will not be called.
    Expectation: No Exception
    """

    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.attr = 1111

        def construct(self):
            global undefined_symbol  # pylint: disable=global-variable-not-assigned
            return undefined_symbol

    net = MyNet()

    @jit
    def func(a, b):
        print(net.attr)
        return a + b

    out = func(Tensor(1), Tensor(1))
    assert out == 2


def test_getattr_from_called_cell():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr from a cell instance which will be called.
    Expectation: No Exception
    """

    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.attr = 1111

        def construct(self):
            return 2

    net = MyNet()

    @jit
    def func():
        print(net.attr)
        return net()

    out = func()
    assert out == 2
