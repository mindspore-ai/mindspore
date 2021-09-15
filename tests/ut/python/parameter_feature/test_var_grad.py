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

import mindspore.ops.composite as C
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.parameter import ParameterTuple
from mindspore.nn import Cell
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)


grad_by_list = C.GradOperation(get_by_list=True)
grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)
grad_by_list_with_sens = C.GradOperation(get_by_list=True, sens_param=True)
grad_all = C.GradOperation(get_all=True)
grad_with_sens = C.GradOperation(sens_param=True)


def test_net_vargs_expand():
    class AddNet(Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.w = Parameter(
                Tensor(np.ones((3, 4, 5), np.float32)), "w2", requires_grad=True)

        def construct(self, x, y):
            return x + y

    x = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    y = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    sens = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    net = AddNet()
    _ = grad_all_with_sens(net, net.trainable_params())(x, y, sens)


class VarNet(Cell):
    def __init__(self, net):
        super(VarNet, self).__init__()
        self.b = Parameter(
            Tensor(np.ones([3, 4, 5]), dtype=mstype.float32), "b", requires_grad=True)
        self.w = Parameter(
            Tensor(np.ones([3, 4, 5]), dtype=mstype.float32), "w", requires_grad=True)
        self.net = net

    def construct(self, *args):
        return self.net(*args) * self.w + self.b


class SecondNet(Cell):
    def __init__(self):
        super(SecondNet, self).__init__()
        self.b2 = Parameter(
            Tensor(np.ones([3, 4, 5]), dtype=mstype.float32), "b2", requires_grad=True)

    def construct(self, *args):
        res = args[0] + args[1]
        return res + self.b2


class Bprop(Cell):
    def __init__(self, func, wrt_params, params, grad_op, sens=None):
        super(Bprop, self).__init__(auto_prefix=False)
        self.func = func
        self.wrt_params = wrt_params
        self.params = None
        if self.wrt_params and params:
            self.params = ParameterTuple(params)
        self.grad = grad_op
        self.with_sens = False
        self.sens = sens
        if not sens is None:
            self.sens = sens if isinstance(sens, Tensor) else Tensor(sens, dtype=mstype.float32)
            self.with_sens = True

    def construct(self, *inputs):
        # pylint: disable=no-else-return
        if self.wrt_params:
            if self.with_sens:
                return self.grad(self.func, self.params)(*inputs, self.sens)
            else:
                return self.grad(self.func, self.params)(*inputs)
        elif self.with_sens:
            return self.grad(self.func)(*inputs, self.sens)
        else:
            return self.grad(self.func)(*inputs)


def test_all_var_args_grad_with_sens():
    """"test grad_by_list_with_sens with all var args input"""

    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net

        def construct(self, *inputs):
            return grad_by_list_with_sens(self.net, self.weights)(*inputs)

    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    sens = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    _ = grad_net(x, y, sens)


def test_grad_list_var_args():
    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net

        def construct(self, *inputs):
            return grad_by_list(self.net, self.weights)(*inputs)

    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    _ = grad_net(x, y)


def test_grad_all_var_args():
    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    _ = grad_net(x, y)


def test_grad_all_var_args_with_sens():
    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net

        def construct(self, *inputs):
            return grad_all_with_sens(self.net)(*inputs)

    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    sens = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    _ = grad_net(x, y, sens)


def test_grad_var_args_with_sens():
    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net

        def construct(self, *inputs):
            return grad_with_sens(self.net)(*inputs)

    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    sens = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    _ = grad_net(x, y, sens)


def test_grad_with_param_sens():
    """"test grad_with_sens parameter"""

    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net
            self.sens = Parameter(Tensor(np.ones([3, 4, 5]), dtype=mstype.float32), name='sens', requires_grad=False)
            self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        def construct(self, x, y):
            return self.grad(self.net, self.weights)(x, y, self.sens)

    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    net = SecondNet()
    grad_net = GradNet(net)
    _ = grad_net(x, y)


def test_var_args_grad():
    class VarNet(Cell):
        def __init__(self, net):
            super(VarNet, self).__init__()
            self.b = Parameter(
                Tensor(np.ones([3, 4, 5]), dtype=mstype.float32), "b", requires_grad=True)
            self.net = net

        def construct(self, *args):
            return self.net(*args) + self.b

    class SecondNet(Cell):
        def __init__(self):
            super(SecondNet, self).__init__()
            self.b2 = Parameter(
                Tensor(np.ones([3, 4, 5]), dtype=mstype.float32), "b2", requires_grad=True)

        def construct(self, *args):
            res = args[0] + args[1]
            return res + self.b2

    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        def construct(self, x, y, sens):
            return grad_by_list_with_sens(self.net, self.weights)(x, y, sens)

    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    sens = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    _ = grad_net(x, y, sens)


def test_var_args_positional():
    """"test grad_all with var args in inner graph"""

    class VarNet(Cell):
        def __init__(self, net):
            super(VarNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            return self.net(x, y) * x

    class SecondNet(Cell):
        def __init__(self):
            super(SecondNet, self).__init__()

        def construct(self, *args):
            return args[0] + args[1]

    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        def construct(self, x, y):
            return grad_all(self.net)(x, y)

    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    _ = grad_net(x, y)


def test_grad_within_if_else():
    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net
            grad_op = C.GradOperation(get_all=False, get_by_list=True, sens_param=True)
            sens = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
            self.grad = Bprop(self.net, True, self.weights, grad_op, sens)

        def construct(self, *inputs):
            return self.grad(*inputs)

    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    out = grad_net(x, y)
    print("test_grad_var_args_with_sens out=", out)


def test_grad_for_concat():
    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net
            grad_op = C.GradOperation(get_all=True, get_by_list=False, sens_param=True)
            self.grad = Bprop(self.net, False, self.weights, grad_op)

        def construct(self, *inputs):
            return self.grad(*inputs)

    class Concat(Cell):
        def __init__(self, axis):
            super().__init__()
            self.concat = P.Concat(axis=axis)

        def construct(self, *input1):
            return self.concat(input1)

    class ConcatFactory:
        def __init__(self, input_shape, axis, dtype=np.float32):
            super(ConcatFactory, self).__init__()
            self.inputs_np = []
            for s in input_shape:
                self.inputs_np.append(np.random.randn(*s).astype(dtype))
            self.axis = axis
            self.out_numpy = np.concatenate(self.inputs_np, axis=self.axis)
            self.out_grad_np = self.out_numpy

        def grad_mindspore_impl(self):
            inputs = []
            for i in self.inputs_np:
                inputs.append(Tensor(i))
            net = Concat(axis=self.axis)
            grad_net = GradNet(net)
            grad_net.set_train()
            _ = grad_net(*inputs, Tensor(self.out_grad_np))

        def grad_cmp(self):
            self.grad_mindspore_impl()

    fact = ConcatFactory(input_shape=(
        (2, 184320, 1), (2, 46080, 1), (2, 11520, 1), (2, 2880, 1), (2, 720, 1)), axis=1)
    fact.grad_cmp()
