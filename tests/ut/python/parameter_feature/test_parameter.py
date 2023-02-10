# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore.context as context
import mindspore.ops.composite as C
from mindspore import Tensor, Parameter, jit
from mindspore import nn
from mindspore.nn import Cell
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)

grad_all = C.GradOperation(get_all=True)
grad_all_with_sens = C.GradOperation(sens_param=True)


def test_parser_three_default_mixed_args_subnet():
    class SubNetDefaultMixedArgs(Cell):
        def construct(self, y, x=3, x1=None, x2=(1, 2)):
            if x == 3:
                if x1 == None:
                    return y
            return -y

    class NetOut(Cell):
        def __init__(self):
            super(NetOut, self).__init__()
            self.net_inside = SubNetDefaultMixedArgs()

        def construct(self, x, y=3):
            z = self.net_inside(x)

            return z

    tensor1 = Tensor(np.full((2, 3), 2).astype(np.float32))
    tensor2 = Tensor(np.full((3, 2), 4).astype(np.float32))
    net = NetOut()
    assert np.all(net(tensor1, tensor2).asnumpy() == tensor1.asnumpy())


# pylint: disable=keyword-arg-before-vararg
def test_net_vararg_kwonlyarg_kwarg():
    class FirstNet(Cell):
        def __init__(self):
            super(FirstNet, self).__init__()
            self.net = SecondNet()

        def construct(self, x=1, z=2 + 2 + 4, y=3):
            c = self.net(22, 33, x, y, z, 2, 3, 4, 5, key1=10, key2=20, key3=30, key4=40)
            return c

    class SecondNet(Cell):
        def construct(self, x, y=2, p=5, q=40, *var, key1=1, key2=3, **kwargs):
            a = x - y
            b = p * q
            c = a / b
            d = var[0] * var[1] * var[2] * var[3]
            e = key1 - key2 - kwargs["key3"] + kwargs["key4"]
            return a + b + c + d + e

    net = FirstNet()
    net()


# pylint: disable=keyword-arg-before-vararg
def test_net_vararg_normal_input():
    class FirstNet(Cell):
        def __init__(self):
            super(FirstNet, self).__init__()
            self.net = SecondNet()

        def construct(self, x=1, z=2 + 2 + 4, y=3):
            c = self.net(22, 33, x, y, z, 2, 3, 4, 5, key1=10, key2=20, key3=30, key4=40)
            return c

    class SecondNet(Cell):
        def construct(self, x, y=2, p=5, q=40, *var, key1=1, key2=3, **kwargs):
            a = x - y
            b = p * q
            c = a / b
            d = var[0] * var[1] * var[2] * var[3]
            e = key1 - key2 - kwargs["key3"] + kwargs["key4"]
            return a + b + c + d + e

    x = Tensor(np.ones((2, 3, 4), np.int32))
    net = FirstNet()
    net(x, x, x)


def test_prim_vararg_kwonlyarg():
    class FirstNet(Cell):
        def __init__(self):
            super(FirstNet, self).__init__()
            self.max = P.Maximum()
            self.min = P.Minimum()
            self.net = SecondNet()
            self.x = Tensor(np.ones((2, 3, 4), np.float32))
            self.y = Tensor(np.ones((2, 3, 4), np.float32))

        def construct(self):
            a = self.max(self.x, self.y)
            b = self.min(self.x, self.y)
            t = {"x": a, "y": b}
            c = self.net(t["x"], t["y"], a, b, z=a, r=b)
            return c

    class SecondNet(Cell):
        def __init__(self):
            super(SecondNet, self).__init__()
            self.addN = P.AddN()
            self.max = P.Maximum()
            self.add = P.Add()

        def construct(self, x, y, *args, z=0, r=1):
            c = self.max(args[0], args[1])
            d = self.addN(args)
            e = self.max(*args)
            ret = x + y + c + d + e + z + r
            return ret

    net = FirstNet()
    net()


def test_no_vararg():
    class FirstNet(Cell):
        def __init__(self):
            super(FirstNet, self).__init__()
            self.max = P.Maximum()
            self.min = P.Minimum()
            self.net = SecondNet()
            self.x = Tensor(np.ones((2, 3, 4), np.float32))
            self.y = Tensor(np.ones((2, 3, 4), np.float32))

        def construct(self):
            t = {"x": self.x, "y": self.y}
            a = self.max(self.x, self.y)
            b = self.min(self.x, self.y)
            c = self.net(a, b, z=a, r=b)
            return c

    class SecondNet(Cell):
        def construct(self, x, y, *, z=0, r=1):
            ret = x + y + z + r
            return ret

    net = FirstNet()
    net()


def test_net_variable_and_weights():
    class FirstNet(Cell):
        def __init__(self):
            super(FirstNet, self).__init__()
            self.max = P.Maximum()
            self.min = P.Minimum()
            self.net = SecondNet()
            self.x = Tensor(np.ones((3, 4), np.float32))
            self.y = Tensor(np.ones((3, 4), np.float32))
            self.weight = Parameter(Tensor(np.ones((2, 3, 4)).astype(np.float32)), "w1", requires_grad=True)

        def construct(self, *args):
            t = (self.x, self.y)
            a = self.max(self.x, self.weight)
            b = self.min(self.weight, args[0])
            c = self.net(a, b, *t)
            return c

    class SecondNet(Cell):
        def __init__(self):
            super(SecondNet, self).__init__()
            self.addN = P.AddN()
            self.max = P.Maximum()
            self.add = P.Add()
            self.weight = Parameter(Tensor(np.ones((2, 3, 4), np.float32)), "w2", requires_grad=True)

        def construct(self, a, b, *args):
            c = self.max(args[0], a)
            d = self.addN(args)
            ret = a + b + c + d + self.weight
            return ret

    net = FirstNet()
    x = Tensor(np.ones((4,), np.float32))
    y = Tensor(np.ones((4,), np.float32))
    z = Tensor(np.ones((4,), np.float32))
    net(x, y, z)


def test_net_vargs_expand():
    class InputBackward(Cell):
        """ InputBackward definition """

        def __init__(self, network, c1=None, c2=None):
            super(InputBackward, self).__init__()
            self.network = network
            self.network.set_train()
            self.grad = grad_all_with_sens
            self.c1 = c1
            self.c2 = c2

        def construct(self, *inputs):
            return self.grad(self.network)(*inputs)

    class AddNet(Cell):
        def construct(self, x, y):
            return x + y

    net = InputBackward(AddNet())
    x = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    y = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    sens = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))

    net.set_train()
    net(x, y, sens)


def test_mixed_precision_const_parameter():
    class NetLoss(Cell):
        def __init__(self):
            super(NetLoss, self).__init__()
            self.shape = P.Shape()
            self.up_sample1 = P.ResizeBilinear((14, 14))
            self.up_sample2 = P.ResizeBilinear((28, 28))
            self.up_sample3 = P.ResizeBilinear((36, 36))

        def construct(self, x, y, z, *args):
            ret = 0
            if args[0] == self.shape(z)[2]:
                if args[0] == 14:
                    ret = self.up_sample1(y) + x
                elif args[0] == 28:
                    ret = self.up_sample2(y) - x
                else:
                    ret = x / y
            else:
                ret = x * y
            ret = ret * z
            return ret

    class NetMain(Cell):
        def __init__(self, loss_fn):
            super(NetMain, self).__init__()
            self.loss_fn = loss_fn
            self.shape = P.Shape()

        def construct(self, x, y, z):
            size_x = self.shape(x)[2]
            size_y = self.shape(y)[2]
            ret = self.loss_fn(x, y, z, size_x, size_y)
            return ret

    loss_fn = NetLoss()
    net = NetMain(loss_fn)
    net.add_flags_recursive(fp32=True)
    x = Tensor(np.ones((1, 3, 28, 28), np.float32))
    y = Tensor(np.ones((1, 3, 14, 14), np.float32))
    z = Tensor(np.ones((1, 3, 28, 28), np.float32))
    _ = net(x, y, z)


def test_pass_args_by_key_ward_way():
    class KeyWardNet(Cell):
        def construct(self, x, y, z):
            return x + y - z

    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.grad = C.GradOperation(get_all=True, sens_param=True)
            self.net = net
            self.sens = Tensor(np.ones((3, 3, 4), np.float32))

        def construct(self, x, y, z, sens):
            return self.grad(self.net)(x, y, z, sens)

    x = Tensor(np.ones((1, 3, 4), np.float32))
    y = Tensor(np.ones((1, 3, 4), np.float32))
    z = Tensor(np.ones((3, 3, 4), np.float32))
    net = KeyWardNet()
    net(x, z=z, y=y)
    grad_net = GradNet(net)
    sens = Tensor(np.ones((3, 3, 4), np.float32))
    grad_net(x, y=y, z=z, sens=sens)


def test_none_input():
    """
    Feature: Net's inputs
    Description: Support none input for the outermost net
    Expectation: no error
    """

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.op = nn.ResizeBilinear()

        def construct(self, a, b, c, d):
            return self.op(a, b, c, d)

    x = Tensor(np.array([1, 2, 3, 4]).astype(np.float32).reshape((1, 1, 2, 2,)))
    net = Net()
    net(x, (4, 4), None, True)


def test_args_kwarg_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are varargs and kwargs properly.
    Description: Function with unused parameters which are varargs and kwargs.
    Expectation: compile success and result == 0
    """

    class Net(Cell):
        def trivial(self, *args, **kwargs):
            return 0

        def construct(self, x, y):
            ret = self.trivial(x, y)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == 0


def test_args_kwonlyargs_1_kwarg_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are varargs, kwonlyargs and
    kwargs properly.
    Description: Function with unused parameters which are varargs, 1 kwonlyargs and kwargs.
    Expectation: compile success and result == 0
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, **kwargs):
            return 0

        def construct(self, x, y):
            ret = self.trivial(x, y)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == 0


def test_args_kwonlyargs_2_kwarg_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are varargs, kwonlyargs and
    kwargs properly.
    Description: Function with unused parameters which are varargs, 2 kwonlyargs and kwargs.
    Expectation: compile success and result == 0
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, only2=4, **kwargs):
            return 0

        def construct(self, x, y):
            ret = self.trivial(x, y)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == 0


def test_args_1_used_kwonlyargs_kwarg_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are kwonlyargs and
    kwargs properly.
    Description: Function with unused parameters which are 1 kwonlyargs and kwargs.
    Expectation: compile success and result == x
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, **kwargs):
            return args[0]

        def construct(self, x, y):
            ret = self.trivial(x, y)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == x


def test_args_2_used_kwonlyargs_kwarg_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are kwonlyargs and
    kwargs properly.
    Description: Function with unused parameters which are 1 kwonlyargs and kwargs.
    Expectation: compile success and result == y
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, **kwargs):
            return args[1]

        def construct(self, x, y):
            ret = self.trivial(x, y)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == y


def test_kwonlyargs_1_used_args_kwarg_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are varargs and
    kwargs properly.
    Description: Function with unused parameters which are varargs and kwargs.
    Expectation: compile success and result == only1
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, **kwargs):
            return only1

        def construct(self, x, y):
            ret = self.trivial(x, y)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == 3


def test_kwonlyargs_2_used_args_kwarg_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are varargs and
    kwargs properly.
    Description: Function with unused parameters which are varargs and kwargs.
    Expectation: compile success and result == only2
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, only2=4, **kwargs):
            return only2

        def construct(self, x, y):
            ret = self.trivial(x, y)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == 4


def test_kwarg_used_args_kwonlyargs_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are varargs and
    kwonlyargs properly.
    Description: Function with unused parameters which are varargs and kwonlyargs.
    Expectation: compile success and result == kw1
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, only2=4, **kwargs):
            return kwargs["kw1"]

        def construct(self, x, y):
            ret = self.trivial(x, y, kw1=5)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == 5


def test_args_1_kwonlyargs_1_used_kwarg_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are kwargs properly.
    Description: Function with unused parameters which are kwargs.
    Expectation: compile success and result == (x, 3)
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, only2=4, **kwargs):
            return (args[0], only1)

        def construct(self, x, y):
            ret = self.trivial(x, y)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == (x, 3)


def test_args_2_kwonlyargs_1_used_kwarg_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are kwargs properly.
    Description: Function with unused parameters which are kwargs.
    Expectation: compile success and result == (x, y, 3)
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, only2=4, **kwargs):
            return (args[0], args[1], only1)

        def construct(self, x, y):
            ret = self.trivial(x, y)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == (x, y, 3)


def test_args_2_kwonlyargs_2_used_kwarg_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are kwargs properly.
    Description: Function with unused parameters which are kwargs.
    Expectation: compile success and result == (x, y, only1, only2)
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, only2=4, **kwargs):
            return (args[0], args[1], only1, only2)

        def construct(self, x, y):
            ret = self.trivial(x, y)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == (x, y, 3, 4)


def test_kwonlyargs_1_kwarg_used_args_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are varargs properly.
    Description: Function with unused parameters which are varargs.
    Expectation: compile success and result == (y, kw1)
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, only2=4, **kwargs):
            return (only1, kwargs["kw1"])

        def construct(self, x, y):
            ret = self.trivial(x, y, kw1=5)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == (3, 5)


def test_kwonlyargs_2_kwarg_used_args_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are varargs properly.
    Description: Function with unused parameters which are varargs.
    Expectation: compile success and result == (only1, only2, kw1)
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, only2=4, **kwargs):
            return (only1, only2, kwargs["kw1"])

        def construct(self, x, y):
            ret = self.trivial(x, y, kw1=5)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == (3, 4, 5)


def test_args_1_kwarg_used_kwonlyargs_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are kwonlyargs properly.
    Description: Function with unused parameters which are kwonlyargs.
    Expectation: compile success and result == (x, kw1)
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, only2=4, **kwargs):
            return (args[0], kwargs["kw1"])

        def construct(self, x, y):
            ret = self.trivial(x, y, kw1=5)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == (x, 5)


def test_args_2_kwarg_used_kwonlyargs_not_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which are kwonlyargs properly.
    Description: Function with unused parameters which are kwonlyargs.
    Expectation: compile success and result == (x, y, kw1)
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, only2=4, **kwargs):
            return (args[0], args[1], kwargs["kw1"])

        def construct(self, x, y):
            ret = self.trivial(x, y, kw1=5)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == (x, y, 5)


def test_args_1_kwonlyargs_1_kwarg_used():
    """
    Feature: Eliminate Parameter pass can remove unused parameters and arguments which is kwonlyarg properly.
    Description: Function with unused parameters which is kwonlyarg.
    Expectation: compile success and result == (x, only1, kw1)
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, only2=4, **kwargs):
            return (args[0], only1, kwargs["kw1"])

        def construct(self, x, y):
            ret = self.trivial(x, y, kw1=5)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == (x, 3, 5)


def test_args_2_kwonlyargs_2_kwarg_used():
    """
    Feature: Eliminate Parameter pass should not remove parameters and arguments all used.
    Description: Function without unused parameters.
    Expectation: compile success and result == (x, y, only1, only2, kw1)
    """

    class Net(Cell):
        def trivial(self, *args, only1=3, only2=4, **kwargs):
            return (args[0], args[1], only1, only2, kwargs["kw1"])

        def construct(self, x, y):
            ret = self.trivial(x, y, kw1=5)
            return ret

    net = Net()
    x = 1
    y = 2
    assert net(x, y) == (x, y, 3, 4, 5)


def test_cell_keyword_argument():
    """
    Feature: Support kwargs for top graph.
    Description: Only positional arguments.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, a, b):
            return a * b

    net = Net()
    assert net(2, b=3) == 6
    assert net(a=2, b=3) == 6
    assert net(b=3, a=2) == 6


def test_cell_default_argument():
    """
    Feature: Support kwargs for top graph.
    Description: Positional arguments with default values.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y=3, z=4):
            return x ** y + z

    net = Net()
    assert net(2) == 12
    assert net(2, 1) == 6
    assert net(2, 3, 2) == 10
    assert net(y=1, z=3, x=2) == 5


def test_cell_args1():
    """
    Feature: Support kwargs for top graph.
    Description: Only varargs.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, *args):
            x = 0
            for arg in args:
                x = x + arg
            return x

    net = Net()
    assert net(1, 2, 3) == 6


def test_cell_args2():
    """
    Feature: Support kwargs for top graph.
    Description: Positional arguments and varargs.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, *args):
            for arg in args:
                x = x + arg
            return x

    net = Net()
    assert net(1, 2, 3) == 6


def test_cell_kwargs1():
    """
    Feature: Support kwargs for top graph.
    Description: Only kwarg.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, **kwargs):
            return kwargs["a"] + kwargs["b"]

    net = Net()
    assert net(a=1, b=2, c=3) == 3


def test_cell_kwargs2():
    """
    Feature: Support kwargs for top graph.
    Description: Positional arguments and kwarg.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, **kwargs):
            return x + kwargs["a"] - kwargs["b"]

    net = Net()
    assert net(1, a=2, b=3) == 0


def test_cell_args_kwargs():
    """
    Feature: Support kwargs for top graph.
    Description: Vararg and kwarg.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, *args, **kwargs):
            x = args[0] + args[1] - kwargs["c"] + kwargs["d"]
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    assert net(1, 2, c=3, d=4) == 4


def test_cell_kwonly1():
    """
    Feature: Support kwargs for top graph.
    Description: Only kwonly arguments.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, *, a, b):
            x = a + b
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    assert net(a=1, b=2) == 3


def test_cell_kwonly2():
    """
    Feature: Support kwargs for top graph.
    Description: Positional args and kwonly arguments with default values.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, a, *, b, c=3):
            x = a + b - c
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    assert net(1, b=2, c=3) == 0
    assert net(1, b=2) == 0


def test_cell_mixed_arguments1():
    """
    Feature: Support kwargs for top graph.
    Description: Mixed arguments.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, a, b, c=3, *args, **kwargs):
            x = a + b - c + args[0] - args[1] + kwargs["d"]
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    assert net(1, 2, 3, 4, 5, d=6, e=7) == 5


def test_cell_mixed_arguments2():
    """
    Feature: Support kwargs for top graph.
    Description: Mixed arguments.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, a, *args, b, c=1, **kwargs):
            x = a + args[0] - args[1] + b - c + kwargs["d"]
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    assert net(1, 2, 3, b=4, c=5, d=6) == 5
    assert net(1, 2, 3, b=4, d=6) == 9


def test_cell_mixed_arguments3():
    """
    Feature: Support kwargs for top graph.
    Description: Mixed arguments.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, a, *, b, c=1, **kwargs):
            x = a + b - c + kwargs["d"]
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    assert net(1, b=2, c=3, d=4) == 4
    assert net(1, b=4, d=6) == 10


def test_cell_mixed_arguments_with_dict_input():
    """
    Feature: Support kwargs for top graph.
    Description: Mixed arguments with dictionary.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, a, *args, b, c=1, **kwargs):
            x = a["item0"] + args[0] + args[1]["item1"] + b["item2"] + c + kwargs["d"]["item3"]
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    assert net({"item0": 1}, 2, {"item1": 3}, b={"item2": 4}, c=5, d={"item3": 6}) == 21


def test_cell_mixed_arguments_with_sub_cell():
    """
    Feature: Support kwargs for top graph.
    Description: Mixed arguments with sub cell.
    Expectation: No exception.
    """

    class SubNet(nn.Cell):
        def construct(self, a, *args, b, c=1, **kwargs):
            x = a + args[0] + args[1] + b + c + kwargs["d"]
            return x

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.subnet = SubNet()

        def construct(self, a, arg0, arg1, b, c, d):
            x = self.subnet(a, arg0, arg1, b=b, c=c, d=d)
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    assert net(1, 2, 3, 4, 5, 6) == 21


def test_jit_kwargs():
    """
    Feature: Support kwargs for top graph.
    Description: Vararg and kwarg for jit function.
    Expectation: No exception.
    """

    @jit
    def func(*args, **kwargs):
        x = args[0] + args[1] - kwargs["d"]
        return x

    context.set_context(mode=context.GRAPH_MODE)
    assert func(1, 2, c=3, d=4) == -1
    assert func(1, 2, d=4) == -1
    assert func(1, 2, d=4, e=5) == -1
    assert func(1, 2, 2.1, 2.2, d=4, e=5, f=6) == -1

    context.set_context(mode=context.PYNATIVE_MODE)
    assert func(1, 2, c=3, d=4) == -1
    assert func(1, 2, d=4) == -1
    assert func(1, 2, d=4, e=5) == -1
    assert func(1, 2, 2.1, 2.2, d=4, e=5, f=6) == -1


def test_jit_mixed_arguments():
    """
    Feature: Support kwargs for top graph.
    Description: Vararg and kwarg for jit function.
    Expectation: No exception.
    """

    @jit
    def func(a, *args, b, c=5, **kwargs):
        x = a + args[0] - args[1] + b - c + kwargs["d"]
        return x

    context.set_context(mode=context.GRAPH_MODE)
    assert func(1, 2, 3, b=4, c=5, d=6) == 5
    assert func(1, 2, 3, b=4, d=6) == 5
    assert func(1, 2, 3, 4, b=5, c=6, d=7) == 6

    context.set_context(mode=context.PYNATIVE_MODE)
    assert func(1, 2, 3, b=4, c=5, d=6) == 5
    assert func(1, 2, 3, b=4, d=6) == 5
    assert func(1, 2, 3, 4, b=5, c=6, d=7) == 6
