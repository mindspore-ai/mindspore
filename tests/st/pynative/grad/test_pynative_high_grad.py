# Copyright 2023 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import ops
from mindspore.ops import operations as P
from mindspore import jit
from mindspore.graph.api import _pynative_executor
from tests.st.pynative.utils import GradOfFirstInput, GradOfAllInputs, HighGrad
from tests.mark_utils import arg_mark


class OneInputBprop(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = P.ReLU()

    def construct(self, x):
        return self.op(x)

    def bprop(self, x, out, dout):
        return (5 * x,)


class OneInputBpropWithJit(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = P.ReLU()

    @jit
    def neg(self, x):
        fun = P.Neg()(x)
        return fun

    def construct(self, x):
        x = self.neg(x)
        x = self.op(x)
        return x

    def bprop(self, x, out, dout):
        return (5 * x,)


class TwoInputBprop(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = P.Mul()

    def construct(self, x, y):
        return self.op(x, y)

    def bprop(self, x, y, out, dout):
        return x * 5, y * 8


class NestedNet(nn.Cell):
    def construct(self, x):
        y = x * x
        return y


class HighGradNet1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.nested_net = NestedNet()

    def construct(self, x):
        x = x + x
        dx = mindspore.grad(self.nested_net)(x)
        z = dx * dx
        return z


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_nested_highgrad_net1():
    """
    Feature: Test nest high grad feature
    Description: high grad, nest grad
    Expectation: Success
    """

    net = HighGradNet1()
    x = Tensor(np.array([2]).astype(np.float32))
    dx = mindspore.grad(net)(x)
    assert (dx.asnumpy() == np.array([64.]).astype(np.float32)).all()


class NestedNetWithBprop(nn.Cell):
    def construct(self, x):
        y = x * x
        return y

    def bprop(self, x, out, dout):
        return 2 * x * dout


class HighGradNet2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.nested_net = NestedNetWithBprop()

    def construct(self, x):
        x = x + x
        dx = mindspore.grad(self.nested_net)(x)
        z = dx * dx
        return z


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_nested_highgrad_net2():
    """
    Feature: Test nest high grad feature
    Description: high grad, nest grad
    Expectation: Success
    """

    net = HighGradNet2()
    x = Tensor(np.array([2]).astype(np.float32))
    dx = mindspore.grad(net)(x)
    assert (dx.asnumpy() == np.array([64.]).astype(np.float32)).all()


class OpsNet(nn.Cell):
    def construct(self, x):
        bessel_io = ops.BesselI0()
        y = bessel_io(x)
        return y


class HighGradNet3(nn.Cell):
    def __init__(self):
        super().__init__()
        self.nested_net = OpsNet()

    def construct(self, x):
        x = x + x
        dx = mindspore.grad(self.nested_net)(x)
        z = dx * dx
        return z


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_nested_highgrad_net3():
    """
    Feature: Test nest high grad feature
    Description: high grad, nest grad
    Expectation: Success
    """

    net = HighGradNet3()
    x = Tensor(np.array([2]).astype(np.float32))
    dx = mindspore.grad(net)(x)
    assert (dx.asnumpy() == np.array([345.9557]).astype(np.float32)).all()


class NestedNetJit(nn.Cell):
    @jit
    def construct(self, x):
        y = x * x
        return y


class HighGradNetJit(nn.Cell):
    def __init__(self):
        super().__init__()
        self.nested_net = NestedNetJit()

    def construct(self, x):
        x = x + x
        dx = mindspore.grad(self.nested_net)(x)
        z = dx * dx
        return z


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_nested_highgrad_jit_net():
    """
    Feature: Test nest high grad feature
    Description: high grad, nest grad
    Expectation: Success
    """

    net = HighGradNetJit()
    x = Tensor(np.array([2]).astype(np.float32))
    dx = mindspore.grad(net)(x)
    assert (dx.asnumpy() == np.array([64.]).astype(np.float32)).all()


class NestedNetJitWithParam(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

    @jit
    def construct(self, x):
        y = self.p1 * self.p1
        y = y * x
        return y


class HighGradNetJitWithParam(nn.Cell):
    def __init__(self):
        super().__init__()
        self.nested_net = NestedNetJitWithParam()

    def construct(self, x):
        x = x + x
        grads = mindspore.grad(self.nested_net, weights=self.nested_net.trainable_params())(x)
        z = grads[1][0] * grads[1][0]
        return z


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_nested_highgrad_jit_net_param():
    """
    Feature: Test nest high grad feature
    Description: high grad, nest grad
    Expectation: Success
    """

    net = HighGradNetJitWithParam()
    x = Tensor(np.array([2]).astype(np.float32))
    dx = mindspore.grad(net)(x)
    assert (dx.asnumpy() == np.array([64.]).astype(np.float32)).all()


class NetWithInplace(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

    def construct(self, x):
        y = self.p1 * self.p1
        y *= x
        return y


class HighGradNetWithInplace(nn.Cell):
    def __init__(self):
        super().__init__()
        self.nested_net = NetWithInplace()

    def construct(self, x):
        x = x + x
        grads = mindspore.grad(self.nested_net, weights=self.nested_net.trainable_params())(x)
        z = grads[1][0] * grads[1][0]
        return z


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_nested_highgrad_with_inplace():
    """
    Feature: Test nest high grad feature
    Description: high grad with inplace
    Expectation: Success
    """

    net = HighGradNetWithInplace()
    x = Tensor(np.array([2]).astype(np.float32))
    dx = mindspore.grad(net)(x)
    assert (dx.asnumpy() == np.array([64.]).astype(np.float32)).all()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_highgrad_one_input_sec_grad():
    """
    Feature: Test high grad feature
    Description: est high grad, bprop two input, second grad
    Expectation: Success
    """

    net = OneInputBprop()
    x = Tensor(np.array([2, 2]).astype(np.float32))
    grad_net = HighGrad(net, [GradOfFirstInput, GradOfFirstInput])
    dxdx = grad_net(x)
    assert (dxdx.asnumpy() == np.array([5, 5]).astype(np.float32)).all()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_highgrad_one_input_third_grad():
    """
    Feature: Test high grad feature
    Description: test high grad, bprop one input, third grad
    Expectation: Success
    """
    net = OneInputBprop()
    x = Tensor(np.array([2, 2]).astype(np.float32))
    grad_net = HighGrad(net, [GradOfFirstInput, GradOfFirstInput, GradOfFirstInput])
    third_grad = grad_net(x)
    assert (third_grad.asnumpy() == np.array([0, 0]).astype(np.float32)).all()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_highgrad_two_input_sec_grad():
    """
    Feature: Test high grad feature
    Description: est high grad, bprop two input, second grad
    Expectation: Success
    """

    net = TwoInputBprop()
    input_x = Tensor(np.array([1, 1]).astype(np.float32))
    input_y = Tensor(np.array([1, 1]).astype(np.float32))
    grad_net = HighGrad(net, [GradOfAllInputs, GradOfAllInputs],
                        sens_param=True,
                        real_inputs_count=2)
    sens_0 = Tensor(np.array([0, 0]).astype(np.float32))
    sens_1 = Tensor(np.array([1, 1]).astype(np.float32))
    dxdx, dxdy = grad_net(Tensor(input_x), Tensor(input_y), sens_1, sens_0)
    assert (dxdx.asnumpy() == np.array([5, 5]).astype(np.float32)).all()
    assert (dxdy.asnumpy() == np.array([0, 0]).astype(np.float32)).all()
    dydx, dydy = grad_net(Tensor(input_x), Tensor(input_y), sens_0, sens_1)
    assert (dydx.asnumpy() == np.array([0, 0]).astype(np.float32)).all()
    assert (dydy.asnumpy() == np.array([8, 8]).astype(np.float32)).all()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_pynative_ms_function_highgrad_one_input_sec_grad():
    """
    Feature: Test ms_function high grad feature
    Description: test ms_function highgrad one_input_sec_grad
    Expectation: Success
    """

    net = OneInputBpropWithJit()
    x = Tensor(np.array([2, 2]).astype(np.float32))
    grad_net = HighGrad(net, [GradOfFirstInput, GradOfFirstInput])
    dxdx = grad_net(x)
    assert (dxdx.asnumpy() == np.array([5, 5]).astype(np.float32)).all()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_pynative_ms_function_highgrad_one_input_third_grad():
    """
    Feature: Test ms_function high grad feature
    Description: test ms_function highgrad one_input_sec_grad
    Expectation: Success
    """
    class OneInputWithJit(nn.Cell):
        @jit
        def construct(self, x):
            z = x * x * x
            h = x + z
            return h
    net = OneInputWithJit()
    x = Tensor(np.array([2, 2]).astype(np.float32))
    grad_net = HighGrad(net, [GradOfFirstInput, GradOfFirstInput, GradOfFirstInput])
    dxdxdx = grad_net(x)
    assert (dxdxdx.asnumpy() == np.array([6, 6]).astype(np.float32)).all()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_pynative_ms_function_highgrad_outer_jit():
    """
    Feature: Test ms_function high grad feature
    Description: test ms_function highgrad one_input_sec_grad
    Expectation: Success
    """
    class OneInputOuterJit(nn.Cell):
        def neg(self, x):
            fun = P.Neg()(x)
            return fun

        @jit
        def construct(self, x):
            x = self.neg(x)
            y = x * x
            return y

    net = OneInputOuterJit()
    x = Tensor(np.array([2, 2]).astype(np.float32))
    grad_net = HighGrad(net, [GradOfFirstInput, GradOfFirstInput])
    dxdxdx = grad_net(x)
    assert (dxdxdx.asnumpy() == np.array([2, 2]).astype(np.float32)).all()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_exception_handler():
    """
    Feature: Test exception handle correctly after raise exception
    Description: An exception followed a test case that should run normally, with no errors
    Expectation: Success
    """

    class OneInputJitBprop(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = P.ReLU()

        def construct(self, x):
            return self.op(x)

        @jit
        def bprop(self, x, out, dout):
            return 5 * x

    net = OneInputJitBprop()
    x = Tensor(np.array([2, 2]).astype(np.float32))
    grad_net = HighGrad(net, [GradOfFirstInput, GradOfFirstInput])
    with pytest.raises(RuntimeError):
        grad_net(x)
        _pynative_executor.sync()
