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

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import jit
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


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
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
          level_mark='level1',
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
          level_mark='level1',
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
          essential_mark='essential')
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
