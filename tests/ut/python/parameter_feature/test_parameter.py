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
import numpy as np

import mindspore.context as context
import mindspore.ops.composite as C
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, save_graphs=True)

grad_all = C.GradOperation(get_all=True)
grad_all_with_sens = C.GradOperation(sens_param=True)


def test_parser_three_default_mixed_args_subnet():
    class SubNetDefaultMixedArgs(Cell):
        def __init__(self):
            super().__init__()

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
        def __init__(self):
            super(SecondNet, self).__init__()

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
        def __init__(self):
            super(SecondNet, self).__init__()

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
        def __init__(self):
            super(SecondNet, self).__init__()

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
        def __init__(self):
            super(AddNet, self).__init__()

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
        def __init__(self):
            super(KeyWardNet, self).__init__()

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
