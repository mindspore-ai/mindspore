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

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.api import jit, jit_class

grad_all = C.GradOperation(get_all=True)
grad_by_list = C.GradOperation(get_by_list=True)


class CropAndResizeNet(nn.Cell):
    def __init__(self, crop_size):
        super(CropAndResizeNet, self).__init__()
        self.crop_and_resize = P.CropAndResize()
        self.crop_size = crop_size

    def construct(self, x, boxes, box_indices):
        return self.crop_and_resize(x, boxes, box_indices, self.crop_size)

    def bprop(self, x, boxes, box_indices, out, dout):
        return x, boxes, box_indices


class TestUserDefinedBpropNet(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(TestUserDefinedBpropNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=1, has_bias=False,
                              weight_init='ones', pad_mode='same')
        self.crop = CropAndResizeNet((10, 10))
        self.boxes = Tensor(np.ones((128, 4)).astype(np.float32))
        self.box_indices = Tensor(np.ones((128,)).astype(np.int32))

    def construct(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.crop(x, self.boxes, self.box_indices)
        return x


class TestUserDefinedBpropGradNet(nn.Cell):
    def __init__(self, net):
        super(TestUserDefinedBpropGradNet, self).__init__()
        self.net = net

    def construct(self, x):
        return grad_all(self.net)(x)


def test_user_defined_bprop():
    context.set_context(mode=context.GRAPH_MODE)
    net = TestUserDefinedBpropNet(3, 10)
    grad_net = TestUserDefinedBpropGradNet(net)
    x = Tensor(np.ones((128, 3, 12, 12)).astype(np.float32))
    grad_net(x)


class TwoInputBPropOperator(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = P.Mul()
        self.add = P.Add()

    def construct(self, x, y):
        return self.op(x, y)

    def bprop(self, x, y, out, dout):
        return self.add(5, x), self.add(y, 9)


class BPropOperatatorNet(nn.Cell):
    def __init__(self, mul_size):
        super().__init__()
        mul_np = np.full(mul_size, 0.1, dtype=np.float32)
        floordiv_np = np.full(mul_size, 0.1, dtype=np.float32)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        self.floordiv_weight = Parameter(Tensor(floordiv_np), name="floordiv_weight")
        self.mul = TwoInputBPropOperator()
        self.floor_div = P.FloorDiv()
        self.bn = nn.BatchNorm1d(num_features=96)

    def construct(self, inputs):
        x = self.mul(inputs, self.mul_weight)
        x = self.floor_div(x, self.floordiv_weight)
        x = self.bn(x)
        return x


def test_user_defined_bprop_with_u():
    context.set_context(mode=context.GRAPH_MODE)
    net = BPropOperatatorNet(mul_size=(128, 96))
    grad_net = TestUserDefinedBpropGradNet(net)
    x = Tensor(np.random.randn(128, 96).astype(np.float32))
    grad_net(x)


class SinNet(nn.Cell):
    def __init__(self):
        super(SinNet, self).__init__()
        self.sin = ops.Sin()

    def construct(self, x):
        out = self.sin(x)
        return out


class SinGrad(nn.Cell):
    def __init__(self, network):
        super(SinGrad, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network

    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout


class SinGradSec(nn.Cell):
    def __init__(self, network):
        super(SinGradSec, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network

    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout


def test_second_grad_with_j_primitive():
    context.set_context(mode=context.GRAPH_MODE)
    net = SinNet()
    first_grad = SinGrad(net)
    second_grad = SinGradSec(first_grad)
    x = Tensor(np.array([1.0], dtype=np.float32))
    second_grad(x)


# A CNode being used as FV is MapMorphism after MapMorphism of call-site CNode;
def test_ad_fv_cnode_order():
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        # cnode xay is not being MapMorphism when cnode second_level() is being MapMorphism and
        # BackPropagateFv as MapMorphism is started from output node and from left to right order.
        def construct(self, x, y):
            def first_level():
                xay = x + y

                def second_level():
                    return xay

                return second_level() + xay

            return first_level()

    input_x = Tensor(np.array([1.0], dtype=np.float32))
    input_y = Tensor(np.array([2.0], dtype=np.float32))

    net = Net()
    net.add_flags_recursive(defer_inline=True)
    grad_net = grad_all(net)
    grad_net(input_x, input_y)


# True and False branch of switch have different number of parameters.
def test_if_branch_with_different_params():
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.weight1 = Parameter(Tensor(np.array([1.0], dtype=np.float32)), name="weight1")
            self.weight2 = Parameter(Tensor(np.array([2.0], dtype=np.float32)), name="weight2")

        def construct(self, idx, end, x):
            out = x
            if idx < end:
                out = out + self.weight1 * self.weight2
            else:
                out = out + self.weight1
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        def construct(self, idx, end, x):
            return grad_by_list(self.net, self.weights)(idx, end, x)

    idx = Tensor(np.array((0), dtype=np.int32))
    end = Tensor(np.array((3), dtype=np.int32))
    x = Tensor(np.array([2.0], dtype=np.float32))

    net = Net()
    grad_net = GradNet(net)
    grad_net(idx, end, x)


# Only lift fv in scope of lift_top_func_graph other than all func_graphs inside manager.
# Otherwise, "Illegal AnfNode for evaluating" may be reported
# because weight1 in Net may use old_parameter other than replicated one.
def test_limit_lift_fv_scope():
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.weight1 = Parameter(Tensor(np.array([1.0], dtype=np.float32)), name="weight1")

        def construct(self, x, y):
            def inner_add(a, b):
                return a + b

            out = inner_add(x, y) + self.weight1
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        def construct(self, x, y):
            def inner_grad_add(a, b):
                return a + b

            d_weight = grad_by_list(self.net, self.weights)(x, y)[0]
            d_out = inner_grad_add(d_weight, y)
            return d_out

    x = Tensor(np.array([2.0], dtype=np.float32))
    y = Tensor(np.array([2.0], dtype=np.float32))

    net = Net()
    net.add_flags_recursive(defer_inline=True)
    grad_net = GradNet(net)
    grad_net.add_flags_recursive(defer_inline=True)
    grad_net(x, y)


def test_same_primal_used_by_multi_j():
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x):
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad = ops.GradOperation()

        def construct(self, x):
            out = self.net(x)
            gout = self.grad(self.net)(x)
            gout1 = self.grad(self.net)(x)
            return out, gout, gout1

    x = Tensor(np.array([1.0], dtype=np.float32))
    net = Net()
    grad = GradNet(net)
    grad(x)


def test_same_primal_used_by_multi_j_with_monad1():
    context.set_context(mode=context.GRAPH_MODE)

    class AdamNet(nn.Cell):
        def __init__(self, var, m, v):
            super(AdamNet, self).__init__()
            self.apply_adam = P.Adam()
            self.var = Parameter(var, name="var")
            self.m = Parameter(m, name="m")
            self.v = Parameter(v, name="v")

        def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
            self.apply_adam(self.var, self.m, self.v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
            return self.var

    class AdamGradNet(nn.Cell):
        def __init__(self, network):
            super(AdamGradNet, self).__init__()
            self.grad_fn = ops.GradOperation(sens_param=True)
            self.sens = [Tensor(np.ones([3, 3, 3]).astype(np.float32)), Tensor(np.ones([3, 3, 3]).astype(np.float32))]
            self.network = network

        def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
            out = self.network(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
            gout1 = self.grad_fn(self.network)(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, self.sens[0])
            gout2 = self.grad_fn(self.network)(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, self.sens[1])
            return out, gout1, gout2

    var = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    m = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    v = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    beta1_power = Tensor(np.array([0.9], dtype=np.float32))
    beta2_power = Tensor(np.array([0.999], dtype=np.float32))
    lr = Tensor(np.array([0.001], dtype=np.float32))
    beta1 = Tensor(np.array([0.9], dtype=np.float32))
    beta2 = Tensor(np.array([0.999], dtype=np.float32))
    epsilon = Tensor(np.array([1e-8], dtype=np.float32))
    grad = Tensor(np.random.rand(3, 3, 3).astype(np.float32))
    net = AdamNet(var, m, v)
    grad_net = AdamGradNet(net)
    grad_net(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)


def test_same_primal_used_by_multi_j_with_monad2():
    context.set_context(mode=context.GRAPH_MODE)

    class AdamNet(nn.Cell):
        def __init__(self, var, m, v):
            super(AdamNet, self).__init__()
            self.apply_adam = P.Adam()
            self.var = Parameter(var, name="var")
            self.m = Parameter(m, name="m")
            self.v = Parameter(v, name="v")

        def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
            self.apply_adam(self.var, self.m, self.v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
            return self.var

    class AdamGradNet(nn.Cell):
        def __init__(self, network):
            super(AdamGradNet, self).__init__()
            self.grad = ops.GradOperation(sens_param=True)
            self.sens = [Tensor(np.ones([3, 3, 3]).astype(np.float32)), Tensor(np.ones([3, 3, 3]).astype(np.float32))]
            self.network = network

        def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
            out = self.network(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
            grad_fn = self.grad(self.network)
            gout1 = grad_fn(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, self.sens[0])
            gout2 = grad_fn(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, self.sens[1])
            return out, gout1, gout2

    var = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    m = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    v = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    beta1_power = Tensor(np.array([0.9], dtype=np.float32))
    beta2_power = Tensor(np.array([0.999], dtype=np.float32))
    lr = Tensor(np.array([0.001], dtype=np.float32))
    beta1 = Tensor(np.array([0.9], dtype=np.float32))
    beta2 = Tensor(np.array([0.999], dtype=np.float32))
    epsilon = Tensor(np.array([1e-8], dtype=np.float32))
    grad = Tensor(np.random.rand(3, 3, 3).astype(np.float32))
    net = AdamNet(var, m, v)
    grad_net = AdamGradNet(net)
    grad_net(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)


def test_grad_args_type_error1():
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = ops.GradOperation(get_all=2)

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = Tensor(np.array([2.0], dtype=np.float32))
    y = Tensor(np.array([2.0], dtype=np.float32))
    try:
        GradNetWrtX(Net())(x, y)
    except TypeError as e:
        assert "For 'GradOperation', the 'get_all' should be bool, but got" in str(e)


def test_grad_args_type_error2():
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = ops.GradOperation(get_by_list=2)

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = Tensor(np.array([2.0], dtype=np.float32))
    y = Tensor(np.array([2.0], dtype=np.float32))
    try:
        GradNetWrtX(Net())(x, y)
    except TypeError as e:
        assert "For 'GradOperation', the 'get_by_list' should be bool, but got" in str(e)


def test_grad_args_type_error3():
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = ops.GradOperation(sens_param=2)

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = Tensor(np.array([2.0], dtype=np.float32))
    y = Tensor(np.array([2.0], dtype=np.float32))
    try:
        GradNetWrtX(Net())(x, y)
    except TypeError as e:
        assert "For 'GradOperation', the 'sens_param' should be bool, but got" in str(e)


def test_grad_net_is_none():
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = P.Add()
            self.grad_op = ops.GradOperation()

        def construct(self, x, y):
            gradient_function = self.grad_op(None)
            return gradient_function(x, y)

    x = Tensor(np.array([2.0], dtype=np.float32))
    y = Tensor(np.array([2.0], dtype=np.float32))
    try:
        GradNetWrtX(Net())(x, y)
    except Exception as e:
        assert "For 'GradOperation', the first argument must be a 'Function' or 'Cell', but got" in str(e)


def test_grad_call_self_net():
    """
    Feature: Custom cell use GradOperation.
    Description: GradOperation does not support __call__ magic methods as object.
    Expectation: Raise an error.
    """
    context.set_context(mode=context.GRAPH_MODE)

    @jit_class
    class Net:
        def __init__(self):
            self.weight = Parameter([10, 10], name='v')

        @jit
        def __call__(self, x):
            a = self.func(x)
            out = self.func(a)
            return out

        def func(self, x):
            self.weight = 2 * self.weight
            return self.weight * x

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.grad_op = ops.GradOperation()
            self.net = net

        def construct(self, x):
            grad_net = self.grad_op(self.net)
            grad = grad_net(x)
            return grad

    x = Tensor(np.array([1.0, 1.0], dtype=np.float32))
    try:
        GradNetWrtX(Net())(x)
    except Exception as e:
        assert "For 'GradOperation', the first argument must be a 'Function' or 'Cell' type object, " \
               "but got object with jit_class type 'Net'." in str(e)


def test_grad_missing_net():
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = ops.GradOperation()

        def construct(self, x, y):
            gradient_function = self.grad_op()
            return gradient_function(x, y)

    x = Tensor(np.array([2.0], dtype=np.float32))
    y = Tensor(np.array([2.0], dtype=np.float32))
    try:
        GradNetWrtX(Net())(x, y)
    except Exception as e:
        assert "'GradOperation' requires a forward network or function as an input, while the input is empty." in str(e)


def test_user_defined_bprop_inputs_size_error():
    context.set_context(mode=context.GRAPH_MODE)

    class BpropUserDefinedNet(nn.Cell):
        def __init__(self):
            super(BpropUserDefinedNet, self).__init__()
            self.zeros_like = P.ZerosLike()

        def construct(self, x, y):
            return x + y

        def bprop(self, out):
            return self.zeros_like(out), self.zeros_like(out)

    class BpropUserDefinedGradNet(nn.Cell):
        def __init__(self, net):
            super(BpropUserDefinedGradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            return grad_all(self.net)(x, y)

    net = BpropUserDefinedNet()
    grad_net = BpropUserDefinedGradNet(net)
    x = Tensor(np.array([2.0], dtype=np.float32))
    y = Tensor(np.array([2.0], dtype=np.float32))
    try:
        grad_net(x, y)
    except Exception as e:
        assert "The function 'bprop' of Primitive or Cell requires at least 2 params 'out' and 'dout', but got only" \
               in str(e)


def test_user_defined_bprop_net_has_parameter():
    context.set_context(mode=context.GRAPH_MODE)

    class BpropUserDefinedNet(nn.Cell):
        def __init__(self):
            super(BpropUserDefinedNet, self).__init__()
            self.zeros_like = P.ZerosLike()
            self.x = Parameter(Tensor(np.array([2.0], dtype=np.float32)), name="x")

        def construct(self, y):
            return self.x + y

        def bprop(self, y, out, dout):
            return (self.zeros_like(out),)

    class BpropUserDefinedGradNet(nn.Cell):
        def __init__(self, net):
            super(BpropUserDefinedGradNet, self).__init__()
            self.net = net

        def construct(self, y):
            return grad_all(self.net)(y)

    net = BpropUserDefinedNet()
    grad_net = BpropUserDefinedGradNet(net)
    y = Tensor(np.array([2.0], dtype=np.float32))
    try:
        grad_net(y)
    except Exception as e:
        assert "The Cell with user defined 'bprop' function in scope" in str(e)
        assert "does not support Parameter data type." in str(e)


def test_user_defined_bprop_inputs_size_error1():
    context.set_context(mode=context.GRAPH_MODE)

    class BpropUserDefinedNet(nn.Cell):
        def __init__(self):
            super(BpropUserDefinedNet, self).__init__()
            self.zeros_like = P.ZerosLike()

        def construct(self, x, y):
            return x + y

        def bprop(self, x, y, out):
            return self.zeros_like(out), self.zeros_like(out)

    class BpropUserDefinedGradNet(nn.Cell):
        def __init__(self, net):
            super(BpropUserDefinedGradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            return grad_all(self.net)(x, y)

    net = BpropUserDefinedNet()
    grad_net = BpropUserDefinedGradNet(net)
    x = Tensor(np.array([2.0], dtype=np.float32))
    y = Tensor(np.array([2.0], dtype=np.float32))
    try:
        grad_net(x, y)
    except TypeError as e:
        assert "The params of function 'bprop' of Primitive or Cell requires the forward inputs as well as the 'out' " \
               "and 'dout'." in str(e)


def test_grad_hook():
    context.set_context(mode=context.GRAPH_MODE)

    def var_hook_function(grad_out):
        assert grad_out[0].asnumpy().shape == (32, 120)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.Add()
            self.hook = P.HookBackward(var_hook_function)

        def construct(self, x, y):
            x = self.hook(x)
            out = self.add(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = ops.GradOperation()

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = Tensor(np.array([2.0], dtype=np.float32))
    y = Tensor(np.array([2.0], dtype=np.float32))
    try:
        GradNetWrtX(Net())(x, y)
    except Exception as e:
        assert "The Primitive 'HookBackward' is not supported in graph mode, which is only supported in pynative " \
               "mode." in str(e)


def test_custom_cell_bprop_with_parameter():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs when the custom cell bprop use Parameter.
    Expectation: Raise an error
    """

    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

        def bprop(self, x, y, out, dout):
            dx = x * self.z
            dy = y + y
            return dx, dy

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            grad_f = grad_all(self.net)
            return grad_f(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    try:
        GradNet(Net())(x, y)
    except Exception as e:
        assert "The user defined 'bprop' function in scope" in str(e)
        assert "does not support using Parameter" in str(e)


def test_custom_cell_bprop_with_parameter_in_sub_cell():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs when the custom cell bprop use Parameter in sub-cell.
    Expectation: Raise an error
    """

    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.net = Net1()

        def construct(self, x, y):
            return self.net(x, y)

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

        def bprop(self, x, y, out, dout):
            dx = x * self.z
            dy = y + y
            return dx, dy

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            grad_f = grad_all(self.net)
            return grad_f(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    try:
        GradNet(Net())(x, y)
    except Exception as e:
        assert "The user defined 'bprop' function in scope" in str(e)
        assert "does not support using Parameter" in str(e)


def test_pynative_custom_cell_bprop_with_parameter():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs when the custom cell bprop use Parameter.
    Expectation: Raise an error
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

        def bprop(self, x, y, out, dout):
            dx = x * self.z
            dy = y + y
            return dx, dy

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            grad_f = grad_all(self.net)
            return grad_f(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    try:
        GradNet(Net())(x, y)
    except Exception as e:
        assert "The user defined 'bprop' function does not support using Parameter" in str(e)


def test_pynative_custom_cell_bprop_with_parameter_in_sub_cell():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs when the custom cell bprop use Parameter in sub-cell.
    Expectation: Raise an error
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.net = Net1()

        def construct(self, x, y):
            return self.net(x, y)

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

        def bprop(self, x, y, out, dout):
            dx = x * self.z
            dy = y + y
            return dx, dy

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            grad_f = grad_all(self.net)
            return grad_f(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    try:
        GradNet(Net())(x, y)
    except Exception as e:
        assert "The user defined 'bprop' function does not support using Parameter" in str(e)


def test_multiple_second_grad_with_same_forward():
    """
    Feature: Second order gradient.
    Description: Get multiple second order gradients with the same forward.
    Expectation: Compile successfully
    """

    @jit
    def f(x):
        return ops.relu(x)

    @jit
    def ff(x):
        return ops.grad(f)(x)

    @jit
    def fff(x, y):
        out1 = ops.grad(ff)(x)
        out2 = ops.grad(ff)(y)
        return out1, out2

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.5, 0.6, 0.3], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    fff(x, y)
