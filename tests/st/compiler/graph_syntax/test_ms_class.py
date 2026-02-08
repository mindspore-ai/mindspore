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
""" test jit_class """
import numpy as np
import mindspore as ms
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, jit_class, jit
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_class_method():
    """
    Feature: JIT Fallback
    Description: Access the methods of user-defined classes decorated with jit_class.
    Expectation: No exception.
    """
    @jit_class
    class InnerNet:
        def __init__(self):
            self.val = Tensor(2, dtype=mstype.int32)

        def act(self, x, y):
            return self.val * (x + y)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.inner_net = InnerNet()

        def construct(self, x, y):
            out = self.inner_net.act(x, y)
            return out

    x = Tensor(2, dtype=mstype.int32)
    y = Tensor(3, dtype=mstype.int32)
    net = Net()
    out = net(x, y)
    assert out.asnumpy() == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_ms_class_call():
    """
    Feature: JIT Fallback
    Description: Call the __call__ function of user-defined classes decorated with jit_class.
    Expectation: No exception.
    """
    @jit_class
    class InnerNet:
        def __init__(self, val):
            self.val = val

        def __call__(self, x, y):
            return self.val * (x + y)

    class Net(nn.Cell):
        def __init__(self, val):
            super().__init__()
            self.inner_net = InnerNet(val)

        def construct(self, x, y):
            out = self.inner_net(x, y)
            return out

    val = Tensor(2, dtype=mstype.int32)
    x = Tensor(3, dtype=mstype.int32)
    y = Tensor(4, dtype=mstype.int32)
    net = Net(val)
    out = net(x, y)
    assert out.asnumpy() == 14


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_class_create_instance_method():
    """
    Feature: JIT Fallback
    Description: Access the methods of the created class instance.
    Expectation: No exception.
    """
    @jit_class
    class InnerNet:
        def __init__(self, val):
            self.number = val

        def act(self, x, y):
            return self.number * (x + y)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.inner_net = InnerNet

        def construct(self, x, y, z):
            net = self.inner_net(x)
            return net.act(y, z)

    x = 2
    y = Tensor(2, dtype=mstype.int32)
    z = Tensor(3, dtype=mstype.int32)
    net = Net()
    out = net(x, y, z)
    assert out.asnumpy() == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_class_type_method():
    """
    Feature: JIT Fallback
    Description: Access the methods of the created class instance.
    Expectation: No exception.
    """
    @jit_class
    class InnerNet:
        number = 2

        def act(self, x, y):
            return self.number * (x + y)

    class Net(nn.Cell):
        def construct(self, x, y):
            return InnerNet.act(InnerNet, x, y)

    x = Tensor(2, dtype=mstype.int32)
    y = Tensor(3, dtype=mstype.int32)
    net = Net()
    out = net(x, y)
    assert out.asnumpy() == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_class_create_instance_call():
    """
    Feature: JIT Fallback
    Description: Call the __call__ function of the created class instance.
    Expectation: No exception.
    """
    @jit_class
    class InnerNet:
        def __init__(self, number):
            self.number = number

        def __call__(self, x, y):
            return self.number * (x + y)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.inner_net = InnerNet

        def construct(self, x, y, z):
            net = self.inner_net(x)
            out = net(y, z)
            return out

    x = 2
    y = Tensor(2, dtype=mstype.int32)
    z = Tensor(3, dtype=mstype.int32)
    net = Net()
    out = net(x, y, z)
    assert out == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_class_call_twice():
    """
    Feature: JIT Fallback
    Description: Call class object twice.
    Expectation: No exception.
    """
    @ms.jit_class
    class Save:
        def __init__(self):
            self.num = ms.Parameter(0, name="num", requires_grad=False)

        def __call__(self, x):
            self.num = self.num + 1
            return x

    save = Save()

    class Net(nn.Cell):
        def construct(self, x):
            x = save(x)
            x = save(x + 1)
            return x + 1, save.num

    x = ms.Tensor([1, 2, 3])
    net = Net()
    out, num = net(x)
    assert np.all(out.asnumpy() == np.array([3, 4, 5]))
    assert num == 2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_class_attr():
    """
    Feature: JIT Class.
    Description: Access the attr of user-defined classes decorated with jit_class.
    Expectation: Expected result.
    """
    @jit_class
    class MyClass:
        def __init__(self):
            self.val = Tensor(2, dtype=mstype.int32)

    @jit
    def func():
        net = MyClass()
        return net.val

    out = func()
    assert out.asnumpy() == 2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_class_property():
    """
    Feature: JIT Class.
    Description: Access the property decorator attr of user-defined classes decorated with jit_class.
    Expectation: Expected result.
    """
    @jit_class
    class MyClass:
        def __init__(self):
            self.x = 1

        @property
        def double(self):
            return 2 * self.x

    @jit
    def func():
        net = MyClass()
        return net.double

    out = func()
    assert out == 2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_class_inherit():
    """
    Feature: JIT Class
    Description: Access the parent class attr of user-defined classes decorated with jit_class.
    Expectation: Expected result.
    """
    class People:
        def __init__(self, a):
            self.age = a

    @jit_class
    class Student(People):
        def __init__(self, a, g):
            super().__init__(a)
            self.grade = g
    @jit
    def func():
        net = Student(20, 95)
        return net.age * net.grade

    out = func()
    assert out == 1900


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_class_static_method():
    """
    Feature: JIT Class
    Description: Access the staticmethod decorator function of user-defined classes decorated with jit_class.
    Expectation: Expected result.
    """
    @jit_class
    class MyClass:
        @staticmethod
        def mul(x, y):
            return x * y

    @jit
    def func(x, y):
        return MyClass.mul(x, y)

    out = func(1, 2)
    assert out == 2
