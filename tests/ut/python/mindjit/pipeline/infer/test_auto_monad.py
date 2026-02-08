# Copyright 2021-2024 Huawei Technologies Co., Ltd
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
import pytest

import mindspore as ms
import mindspore.ops.composite as C
from mindspore import context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
from tests.security_utils import security_off_wrap

grad_all_list = C.GradOperation(get_all=True, get_by_list=True)
grad_by_list = C.GradOperation(get_by_list=True)

context.set_context(mode=context.GRAPH_MODE)


def test_load_grad():
    class LoadNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * y * self.z
            return x

    x = Tensor(np.array([2.0], np.float32))
    y = Tensor(np.array([3.0], np.float32))
    load_net = LoadNet()
    grad_net = grad_all_list(
        load_net, ParameterTuple(load_net.trainable_params()))
    print(grad_net(x, y))


def test_assign_only_grad():
    class AssignOnlyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            self.z = x
            x = x * y
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.parameter_tuple = ParameterTuple(self.trainable_params())

        def construct(self, x, y):
            return grad_all_list(self.net, self.parameter_tuple)(x, y)


    assign_net = AssignOnlyNet()
    net = GradNet(assign_net)
    x = Tensor(np.array([2.0], np.float32))
    y = Tensor(np.array([3.0], np.float32))
    print(net(x, y))


def test_load_assign_grad():
    class AssignNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')
            self.assign = P.Assign()

        def construct(self, x, y):
            x = x * self.z
            self.assign(self.z, x)
            out = y * self.z
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.parameter_tuple = ParameterTuple(net.trainable_params())

        def construct(self, x, y):
            return grad_all_list(self.net, self.parameter_tuple)(x, y)

    assign_net = AssignNet()
    net = GradNet(assign_net)
    x = Tensor(np.array([2.0], np.float32))
    y = Tensor(np.array([3.0], np.float32))
    print(net(x, y))


def test_insert_gradient_of():
    class InsertGradientNet(nn.Cell):
        def __init__(self):
            super(InsertGradientNet, self).__init__()
            self.gather = P.Gather()
            self.damping = Tensor(np.array([0.03, 0.03], np.float32))
            self.cov_step = Parameter(0, name="cov_step", requires_grad=False)
            self.freq = Tensor(278, ms.int32)
            self.getG = P.InsertGradientOf(self.save_gradient)

        def save_gradient(self, dout):
            self.cov_step = self.cov_step + self.freq
            return dout

        def construct(self, x):
            self.gather(self.damping, self.cov_step, 0)
            out = P.ReLU()(x)
            out = self.getG(out)
            out = self.getG(out)
            return out

    net = InsertGradientNet()
    input_data = np.array([[1.2, 2.1], [2.2, 3.2]]).astype(np.float32)
    grad_net = grad_all_list(net, ParameterTuple(net.trainable_params()))
    print(grad_net(Tensor(input_data)))


@security_off_wrap
def test_user_defined_bprop():
    class UserDefinedNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, x, y):
            out = x * y
            return out

        def bprop(self, x, y, out, dout):
            self.print(out)
            out = x * y
            self.print(out)
            self.print(dout)
            return y, x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.parameter_tuple = ParameterTuple(net.trainable_params())

        def construct(self, x, y):
            return grad_all_list(self.net, self.parameter_tuple)(x, y)

    user_defined_net = UserDefinedNet()
    net = GradNet(user_defined_net)
    x = Tensor(np.array([2.0], np.float32))
    y = Tensor(np.array([3.0], np.float32))
    print(net(x, y))


# user defined bprop don't have the same size of parameters with primal's
@security_off_wrap
def test_user_defined_bad_bprop():
    class UserDefinedNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, x, y):
            out = x * y
            return out

        def bprop(self, x, out, dout):
            self.print(out)
            out = x
            self.print(out)
            self.print(dout)
            return x, x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.parameter_tuple = ParameterTuple(net.trainable_params())

        def construct(self, x, y):
            return grad_all_list(self.net, self.parameter_tuple)(x, y)

    user_defined_net = UserDefinedNet()
    net = GradNet(user_defined_net)
    x = Tensor(np.array([2.0], np.float32))
    y = Tensor(np.array([3.0], np.float32))
    with pytest.raises(TypeError):
        net(x, y)


# shoul compile success and Print in presented in the final function graph.
@security_off_wrap
@pytest.mark.skip(reason="isolated nodes exception")
def test_unused_var():
    class UnusedVar(nn.Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, x, y):
            shape1 = self.get_shape(x)
            out = x
            for _ in range(shape1):
                out = out + y
            return out

        def get_shape(self, x):
            self.print(x)
            _, c, _, _ = F.shape(x)
            return c

    net = UnusedVar()
    x = Tensor(np.ones(shape=[3, 2, 1, 2]), ms.float32)
    y = Tensor(np.ones(shape=[3, 2, 1, 2]), ms.float32)
    print(net(x, y))


# shoul compile success and Print in presented in the final function graph.
@security_off_wrap
@pytest.mark.skip(reason="isolated nodes exception")
def test_hof_unused_var():
    class UnusedVar(nn.Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, x, y):
            shape1 = self.hof_get_shape(self.get_shape, x)
            out = x
            for _ in range(shape1):
                out = out + y
            return out

        def hof_get_shape(self, hof, x):
            return hof(x)

        def get_shape(self, x):
            self.print(x)
            _, c, _, _ = F.shape(x)
            return c

    net = UnusedVar()
    x = Tensor(np.ones(shape=[3, 2, 1, 2]), ms.float32)
    y = Tensor(np.ones(shape=[3, 2, 1, 2]), ms.float32)
    print(net(x, y))


# shoul compile success and Print in presented in the final function graph.
@security_off_wrap
@pytest.mark.skip(reason="isolated nodes exception")
def test_partial_hof_unused_var():
    class UnusedVar(nn.Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, x, y):
            shape1 = self.hof_get_shape(x)()
            out = x
            for _ in range(shape1):
                out = out + y
            return out

        def hof_get_shape(self, x):
            return F.partial(self.get_shape, x)

        def get_shape(self, x):
            self.print(x)
            _, c, _, _ = F.shape(x)
            return c

    net = UnusedVar()
    x = Tensor(np.ones(shape=[3, 2, 1, 2]), ms.float32)
    y = Tensor(np.ones(shape=[3, 2, 1, 2]), ms.float32)
    print(net(x, y))


# should compile success without endless loop.
def test_while_if():
    class WhileIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.zero = Tensor(np.zeros([1]).astype(np.float32))
            self.param = Parameter(Tensor(np.zeros([1]).astype(np.float32)))

        def construct(self, idx, end, x):
            out = self.zero
            while idx < end:
                if x < end:
                    out = out + self.param * 2
                else:
                    out = out + self.param
                idx = idx + 1
            return out

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(5), dtype=ms.int32)
    x = Tensor(np.zeros([1]).astype(np.float32))
    m = WhileIfNet()
    m(idx, end, x)


# should compile success without zeros_like_tensor args mismatch, the generated graph files
# should not contain EnvironGet or EnvironSet.
# InsertGradientOf primitive will make func_graph bprop_construct had BackPropAutoMonad flag set,
# so all graph it used will be checked if any side effect it has, so the hyper_map_zeros_like
# will have U as parameter, but the call site zeros_like(fv) don't have U argument.
def test_grad_fv_and_insert_gradient_of():
    class FvAndInsertGradientNet(nn.Cell):
        def __init__(self):
            super(FvAndInsertGradientNet, self).__init__()
            self.gather = P.Gather()
            self.damping = Tensor(np.array([0.03, 0.03], np.float32))
            self.cov_step = Parameter(0, name="cov_step", requires_grad=False)
            self.freq = Tensor(278, ms.int32)
            self.getG = P.InsertGradientOf(self.save_gradient)

            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def save_gradient(self, dout):
            self.cov_step = self.cov_step + self.freq
            return dout

        def construct(self, *inputs):
            # fv self.z from construct_wrapper
            x, = inputs
            self.z = x

            # insert_gradient_of
            self.gather(self.damping, self.cov_step, 0)
            out = self.getG(x)
            return out

    net = FvAndInsertGradientNet()
    input_data = Tensor(np.array([1.0], np.float32))
    # if use grad_all_list, the generated graph will have EnvironSet
    # as gradient for inputs is constant zero, so it will depend on result of grad.
    grad_net = grad_by_list(net, ParameterTuple(net.trainable_params()))
    print(grad_net(input_data))

# should compile success as cnode with Partial primitive will not bind an additional U monad.
def test_partial_parameter():
    z = Parameter(Tensor(np.array([True], np.bool_)), name='z')

    class PartialNet(nn.Cell):
        def __init__(self, input_z):
            super().__init__()
            self.input = input_z

        def construct(self):
            # getattr of all will be convert to Partial
            out = self.input.all(axis=())
            return out

    net = PartialNet(z)
    print(net())


# Better to run in ST and check stdout.
def test_return_const_value_with_side_effect_io():
    """
    Feature: Side effect
    Description: Test side effect with returned const value.
    Expectation: Not throw exception.
    """
    class Demo(nn.Cell):
        def construct(self, x):
            print('print here...')
            y = x[1]
            y[1] = 9
            return y

    x = [[1, 2, 3, 4], [5, 6, 7, 8]]
    net = Demo()
    output = net(x)
    print(output)


def test_return_const_value_with_side_effect_mem():
    """
    Feature: Side effect
    Description: Test side effect with returned const value.
    Expectation: Not throw exception.
    """
    y = Parameter(Tensor([1]))

    class Demo(nn.Cell):
        def construct(self, x):
            P.Assign()(x, Tensor([0]))
            P.Assign()(y, Tensor([0]))
            return True

    x = Parameter(Tensor([1]))
    net = Demo()
    output = net(x)
    print(output)
    print(Tensor(x))
    print(Tensor(y))


def test_grad_with_high_order_cnode():
    """
    Feature: Side effect
    Description: Test side effect detecting with high order cnode.
    Expectation: Not parameter size miss match error.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.array([2, 2], np.float32)))

        def construct(self, x, y):
            return x * y * self.w

    class Grad(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            # pylint: disable=E1102
            grad_net = ms.ops.grad(self.net)
            return grad_net(x, y)

    net = Net()
    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 4]).astype(np.float32))
    grad_net = Grad(net)
    grad = grad_net(x, y)
    print(grad)
