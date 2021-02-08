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
import pytest
import numpy as np
from mindspore import context, nn, Tensor, Parameter, ParameterTuple
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    yield
    context.set_context(mode=context.GRAPH_MODE)


class _Grad(nn.Cell):
    def __init__(self, grad, network, wrt_params=False, real_inputs_count=None):
        super().__init__()
        self.network = network
        self.grad = grad
        self.sens_param = self.grad.sens_param
        self.wrt_params = wrt_params
        self.real_inputs_count = real_inputs_count
        if self.wrt_params:
            self.params = ParameterTuple(self.network.trainable_params())

    def construct(self, *inputs):
        if self.wrt_params:
            if self.real_inputs_count is None or self.sens_param is False:
                return self.grad(self.network, self.params)(*inputs)
            real_inputs = inputs[:self.real_inputs_count]
            sense_param_inputs = inputs[self.real_inputs_count:]
            return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)

        if self.real_inputs_count is None or self.sens_param is False:
            return self.grad(self.network)(*inputs)
        real_inputs = inputs[:self.real_inputs_count]
        sense_param_inputs = inputs[self.real_inputs_count:]
        return self.grad(self.network)(*real_inputs, sense_param_inputs)


class GradOfFirstInput(_Grad):
    """
    get grad of first input
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=C.GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class GradOfAllInputs(_Grad):
    """
    get grad of first input
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=C.GradOperation(get_all=True, sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


def test_multi_grad():
    class ForwardNetMul(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x, y):
            a = x * x
            b = y * y
            return a * b

    class ForwardNetAdd(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x, y):
            a = x + x + x
            b = y + y
            return a * b
    mulnet = ForwardNetMul()
    addnet = ForwardNetAdd()
    x = Tensor(np.ones([32]), dtype=mstype.float32)
    y = Tensor(np.ones([32])*2, dtype=mstype.float32)
    sens = Tensor(np.ones([32]), dtype=mstype.float32)
    mulnet.set_grad()
    addnet.set_grad()
    out1 = mulnet(x, y)
    out2 = addnet(x, y)
    grad_mul = GradOfAllInputs(mulnet)
    grad_add = GradOfAllInputs(addnet)
    grad_mul(x, y, sens)
    grad_add(x, y, sens)


def test_multi_same_grad():
    class ForwardNetMul(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x, y):
            a = x * x
            b = y * y
            return a * b

    class ForwardNetAdd(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x, y):
            a = x*3
            b = y*2
            return a + b
    mulnet = ForwardNetMul()
    addnet = ForwardNetAdd()
    x = Tensor(np.ones([32]), dtype=mstype.float32)
    y = Tensor(np.ones([32]), dtype=mstype.float32)
    sens = Tensor(np.ones([32]), dtype=mstype.float32)
    mulnet.set_grad()
    addnet.set_grad()
    out1 = mulnet(x, y)
    out2 = addnet(x, y)
    grad_mul = GradOfAllInputs(mulnet)
    grad_add = GradOfFirstInput(mulnet)
    grad_mul(x, y, sens)
    grad_add(x, y, sens)


def test_net_inner_grad():
    class ForwardNetMul(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x, y):
            a = x * x
            b = y * y
            return a * b

    class ForwardNetAdd(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            a = x + x
            b = y + y
            res = self.net(a, b)
            return res
    mulnet = ForwardNetMul()
    addnet = ForwardNetAdd(mulnet)
    x = Tensor(np.ones([32]), dtype=mstype.float32)
    y = Tensor(np.ones([32]), dtype=mstype.float32)
    sens = Tensor(np.ones([32]), dtype=mstype.float32)
    mulnet.set_grad()
    addnet.set_grad()
    out1 = mulnet(x, y)
    out2 = addnet(x, y)
    grad_mul = GradOfAllInputs(addnet)
    grad_add = GradOfAllInputs(mulnet)
    grad_mul(x, y, sens)
    grad_add(x, y, sens)


def test_net_inner_first_run_grad():
    class ForwardNetMul(nn.Cell):
        def __init__(self):
            super().__init__()
            self.z1 = Parameter(Tensor(np.ones([32])*2, dtype=mstype.float32), name='z1')

        def construct(self, x, y):
            a = x * self.z1
            b = y * y
            return a * b

    class ForwardNetAdd(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.z2 = Parameter(Tensor(np.ones([32]), dtype=mstype.float32), name='z2')
            self.z3 = Parameter(Tensor(np.ones([32]), dtype=mstype.float32), name='z2')

        def construct(self, x, y):
            a = x + x*self.z3
            b = y + y*self.z2
            res = self.net(a, b)
            return res
    mulnet = ForwardNetMul()
    addnet = ForwardNetAdd(mulnet)
    x = Tensor(np.ones([32]), dtype=mstype.float32)
    y = Tensor(np.ones([32]), dtype=mstype.float32)
    sens = Tensor(np.ones([32]), dtype=mstype.float32)
    mulnet.set_grad()
    addnet.set_grad()
    out1 = mulnet(x, y)
    out2 = addnet(x, y)
    grad_mul = GradOfAllInputs(addnet)
    grad_add = GradOfFirstInput(mulnet)
    grad_mul(x, y, sens)
    grad_add(x, y, sens)
