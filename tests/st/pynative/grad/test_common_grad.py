# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
# pylint: disable=consider-using-from-import,super-with-arguments
""" test_bprop """
import numpy as np
import pytest
import mindspore as ms
from mindspore import grad
import mindspore.nn as nn
from mindspore import context, ops
from mindspore.common import Tensor
from mindspore.common.api import jit, _pynative_executor
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import GradOperation
from mindspore import dtype as mstype
from tests.mindspore_test_framework.utils.bprop_util import bprop
from tests.st.pynative.utils import GradOfFirstInput, GradOfAllInputs, GradOfAllInputsAndParams
from tests.mark_utils import arg_mark


def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE)


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.matmul = P.MatMul()
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

    @jit
    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return x, out


def test_bprop_no_sens():
    grads = bprop(Net(), Tensor(np.ones([2, 3]).astype(np.float32)),
                  Tensor(np.ones([3, 2]).astype(np.float32)), wrt=['inputs'])
    print(grads)


def test_bprop_sens():
    grads = bprop(Net(), Tensor(np.ones([2, 3]).astype(np.float32)), Tensor(np.ones([3, 2]).astype(np.float32)),
                  grads_wrt_outputs=(Tensor(np.ones([2, 3]).astype(np.float32)),
                                     Tensor(np.ones([2, 2]).astype(np.float32))), wrt=['inputs'])
    print(grads)


def test_bprop_first_only():
    grads = bprop(Net(), Tensor(np.ones([2, 3]).astype(np.float32)), Tensor(np.ones([3, 2]).astype(np.float32)),
                  grads_wrt_outputs=(Tensor(np.ones([2, 3]).astype(np.float32)),
                                     Tensor(np.ones([2, 2]).astype(np.float32))))
    print(grads)


def test_bprop_wrt_params():
    net = Net()
    grads = bprop(net, Tensor(np.ones([2, 3]).astype(np.float32)), Tensor(np.ones([3, 2]).astype(np.float32)),
                  grads_wrt_outputs=(Tensor(np.ones([2, 3]).astype(np.float32)),
                                     Tensor(np.ones([2, 2]).astype(np.float32))),
                  wrt=['params'],
                  params=net.trainable_params())
    print(grads)


def test_bprop_wrt_params_no_sens():
    net = Net()
    grads = bprop(net, Tensor(np.ones([2, 3]).astype(np.float32)), Tensor(np.ones([3, 2]).astype(np.float32)),
                  wrt=['params'],
                  params=net.trainable_params())
    print(grads)


def test_bprop_wrt_inputs_and_params():
    net = Net()
    grads = bprop(net, Tensor(np.ones([2, 3]).astype(np.float32)), Tensor(np.ones([3, 2]).astype(np.float32)),
                  grads_wrt_outputs=(Tensor(np.ones([2, 3]).astype(np.float32)),
                                     Tensor(np.ones([2, 2]).astype(np.float32))),
                  wrt=['inputs', 'params'],
                  params=net.trainable_params())
    print(grads)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_with_dict_output():
    """
    Feature: Test sens dict
    Description: Net out is dict
    Expectation: Success
    """

    class DicNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        def construct(self, x):
            y = self.relu(x)
            out = {Tensor(True): y}
            return out

    x = np.array([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]])
    ms_net = DicNet()
    # No sens
    ms_grad = GradOfFirstInput(ms_net, False)
    grad_out = ms_grad(Tensor(x))
    assert np.allclose(np.ones_like(x), grad_out.asnumpy())

    # Have sens
    out = ms_net(Tensor(x))
    ms_grad = GradOfFirstInput(ms_net, True)
    grad_out = ms_grad(Tensor(x), out)
    assert np.allclose(x, grad_out.asnumpy())


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_network_with_dict_output():
    """
    Feature: Test sens dict in jit
    Description: Net out is dict in jit
    Expectation: Success
    """

    class DicNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        @jit
        def construct(self, x):
            y = self.relu(x)
            out = {'a': y}
            return out

    x = np.array([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]])
    ms_net = DicNet()
    # No sens
    ms_grad = GradOfFirstInput(ms_net, False)
    grad_out = ms_grad(Tensor(x))
    assert np.allclose(np.ones_like(x), grad_out.asnumpy())

    # Have sens
    ms_net = DicNet()
    out = ms_net(Tensor(x))
    ms_grad = GradOfFirstInput(ms_net, True)
    grad_out = ms_grad(Tensor(x), out)
    assert np.allclose(x, grad_out.asnumpy())


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_network_with_multi_output_contain_dict():
    """
    Feature: Test pynative jit with multi output contain dict
    Description: Net in jit has multi out, and one element is a dict
    Expectation: Success
    """

    class DicNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        @jit
        def construct(self, x):
            y = self.relu(x)
            out = {'a': y}
            return out, y

    x = np.array([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]])
    ms_net = DicNet()
    # No sens
    ms_grad = GradOfFirstInput(ms_net, False)
    grad_out = ms_grad(Tensor(x))
    assert np.allclose(2 * np.ones_like(x), grad_out.asnumpy())

    # Have sens
    ms_net = DicNet()
    out = ms_net(Tensor(x))
    ms_grad = GradOfFirstInput(ms_net, True)
    grad_out = ms_grad(Tensor(x), out)
    assert np.allclose(2 * x, grad_out.asnumpy())


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_network_with_dict_output_has_constant_value():
    """
    Feature: Test pynative jit with dict output has constant value
    Description: Net in jit has dict out, one of the element pair has constant value
    Expectation: Success
    """

    class DicNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        @jit
        def construct(self, x):
            y = self.relu(x)
            out = {'a': y, 'b': 2}
            return out

    x = np.array([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]])
    ms_net = DicNet()
    # No sens
    ms_grad = GradOfFirstInput(ms_net, False)
    grad_out = ms_grad(Tensor(x))
    assert np.allclose(np.ones_like(x), grad_out.asnumpy())

    # Have sens
    ms_net = DicNet()
    out = ms_net(Tensor(x))
    ms_grad = GradOfFirstInput(ms_net, True)
    grad_out = ms_grad(Tensor(x), out)
    assert np.allclose(x, grad_out.asnumpy())


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_network_with_list_output():
    """
    Feature: Test list in jit
    Description: Net out is list in jit
    Expectation: Success
    """

    class GradCell(nn.Cell):
        def __init__(self, network, get_all=False, get_by_list=False, sens_param=False):
            super().__init__()
            self.network = network
            self.grad = C.GradOperation(get_all, get_by_list, sens_param)

        def construct(self, *inputs):
            grads = self.grad(self.network)(*inputs)
            return grads

    class ListNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.tensor_add = P.Add()

        @jit
        def construct(self, x):
            t = (l * l for l in range(10) if l > 5)
            return t

    input_x = Tensor(np.full((2, 3), 50).astype(np.float32))
    input_y = Tensor(np.full((2, 3), 5).astype(np.float32))
    output_x = [36, 49, 64, 81]
    output_y = np.array([0, 0, 0])
    list_net = ListNet()
    output_net = list_net(input_x)
    assert output_net == output_x
    grad_net = GradCell(list_net)
    output_grad = grad_net(input_y)
    assert np.allclose(output_grad.asnumpy(), output_y, 0.0001, 0.0001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_synchronize():
    """
    Feature: Test pynative synchronize
    Description: Test the code for the synchronous branch.
    Expectation: success
    """
    try:
        context.set_context(pynative_synchronize=True)

        # Cell object to be differentiated
        class MulNet(nn.Cell):
            def construct(self, x, y, z):
                return x * y * z

        x = Tensor([1, 2], ms.float32)
        y = Tensor([-2, 3], ms.float32)
        z = Tensor([0, 3], ms.float32)
        net = MulNet()
        net.set_inputs(Tensor(shape=[None], dtype=ms.float32), y, z)
        output = grad(net, grad_position=(1, 2))(x, y, z)
        assert (output[0].asnumpy() == np.array([0, 6], dtype=np.float32)).all()
        assert (output[1].asnumpy() == np.array([-2, 6], dtype=np.float32)).all()
    finally:
        context.set_context(pynative_synchronize=False)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_multi_grad():
    """
    Feature: Test pynative multi grad
    Description: Test the code for PyNative multi grad.
    Expectation: success
    """

    class ForwardNetMul(nn.Cell):
        def construct(self, x, y):
            a = x * x
            b = y * y
            return a * b

    class ForwardNetAdd(nn.Cell):
        def construct(self, x, y):
            a = x + x + x
            b = y + y
            return a * b

    mulnet = ForwardNetMul()
    addnet = ForwardNetAdd()
    x = Tensor(np.ones([32]), dtype=ms.float32)
    y = Tensor(np.ones([32]) * 2, dtype=ms.float32)
    sens = Tensor(np.ones([32]), dtype=ms.float32)
    mulnet.set_grad()
    addnet.set_grad()
    mulnet(x, y)
    addnet(x, y)
    grad_mul = GradOfAllInputs(mulnet)
    grad_add = GradOfAllInputs(addnet)
    grad_mul(x, y, sens)
    grad_add(x, y, sens)


class GradFactory:
    def __init__(self, net_me, get_all, get_by_list, sens_param, net_params=None,
                 defalut_para=False):
        self.net_me = net_me
        self.get_all = get_all
        self.get_by_list = get_by_list
        self.sens_param = sens_param
        self.net_params = net_params
        self.default_para = defalut_para

    def get_grad(self, ms_input):
        output_grad_me = []
        out = self.net_me(*ms_input)
        if isinstance(out, tuple):
            for it in out:
                if self.sens_param:
                    grad_np = np.random.randn(*it.shape).astype(np.float32)
                else:
                    grad_np = np.ones(it.shape).astype(np.float32)
                output_grad_me.append(Tensor(grad_np))
            output_grad_me = tuple(output_grad_me)
        else:
            if self.sens_param:
                grad_np = np.random.randn(*out.shape).astype(np.float32)
            else:
                grad_np = np.ones(out.shape).astype(np.float32)
            output_grad_me = Tensor(grad_np)
        return output_grad_me

    def one_backnet_call_twice(self, first_ms_input, second_ms_input, loss=0.001):
        grad_input = self.get_grad(first_ms_input)
        if self.default_para:
            back_net = nn.ForwardValueAndGrad(self.net_me)
            back_net(*first_ms_input)
        else:
            if self.get_by_list:
                weight = self.net_params
            else:
                weight = None
            back_net = nn.ForwardValueAndGrad(self.net_me,
                                              weights=weight, get_all=self.get_all,
                                              get_by_list=self.get_by_list,
                                              sens_param=self.sens_param)
            if self.sens_param:
                back_net(*first_ms_input, grad_input[0])
            else:
                back_net(*first_ms_input)

        # second call
        grad_input = self.get_grad(second_ms_input)
        if self.default_para:
            back_net(*second_ms_input)
        else:
            if self.sens_param:
                back_net(*second_ms_input, grad_input[0])
            else:
                back_net(*second_ms_input)

    def two_backnet_call_twice(self, first_ms_input, second_ms_input, loss=0.001):
        grad_input = self.get_grad(first_ms_input)
        if self.default_para:
            back_net = nn.ForwardValueAndGrad(self.net_me)
            back_net(*first_ms_input)
        else:
            if self.get_by_list:
                weight = self.net_params
            else:
                weight = None
            back_net = nn.ForwardValueAndGrad(self.net_me,
                                              weights=weight, get_all=self.get_all,
                                              get_by_list=self.get_by_list,
                                              sens_param=self.sens_param)
            if self.sens_param:
                back_net(*first_ms_input, grad_input[0])
            else:
                back_net(*first_ms_input)

        # second call
        grad_input = self.get_grad(second_ms_input)
        if self.default_para:
            back_net2 = nn.ForwardValueAndGrad(self.net_me)
            back_net2(*second_ms_input)
        else:
            back_net2 = nn.ForwardValueAndGrad(self.net_me,
                                               weights=weight, get_all=self.get_all,
                                               get_by_list=self.get_by_list,
                                               sens_param=self.sens_param)
            if self.sens_param:
                back_net2(*second_ms_input, grad_input[0])
            else:
                back_net2(*second_ms_input)

    def first_forward_second_backnet(self, first_ms_input, second_ms_input, loss=0.001):
        # second call
        grad_input = self.get_grad(second_ms_input)
        if self.default_para:
            back_net2 = nn.ForwardValueAndGrad(self.net_me)
            back_net2(*second_ms_input)
        else:
            if self.get_by_list:
                weight = self.net_params
            else:
                weight = None
            back_net2 = nn.ForwardValueAndGrad(self.net_me,
                                               weights=weight, get_all=self.get_all,
                                               get_by_list=self.get_by_list,
                                               sens_param=self.sens_param)
            if self.sens_param:
                back_net2(*second_ms_input, grad_input[0])
            else:
                back_net2(*second_ms_input)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_forward_value_and_grad_0():
    """
    Feature: Test pynative value and grad
    Description: Test the code for pynative value and grad.
    Expectation: success
    """

    class Net0(nn.Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor([2, 3, 4], ms.float32), name="para")

        def construct(self):
            x = self.para * self.para
            return x

    net_me = Net0()
    fact = GradFactory(net_me=net_me,
                       get_all=True,
                       get_by_list=True,
                       sens_param=False,
                       net_params=ParameterTuple(net_me.trainable_params()))

    first_input = ()
    second_input = ()
    fact.one_backnet_call_twice(first_input, second_input)
    fact.two_backnet_call_twice(first_input, second_input)
    fact.first_forward_second_backnet(first_input, second_input)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_forward_value_and_grad_1():
    """
    Feature: Test pynative value and grad
    Description: Test the code for pynative value and grad.
    Expectation: success
    """

    class Net1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor([1], ms.float32), name="para")

        def construct(self, x):
            y = x + self.para
            return y

    net_me = Net1()
    fact = GradFactory(net_me=net_me,
                       get_all=False,
                       get_by_list=False,
                       sens_param=False,
                       defalut_para=True)

    input_1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    first_input = (input_1,)

    input_1 = Tensor(np.random.randn(1, 2, 3, 4).astype(np.float32))
    second_input = (input_1,)
    fact.one_backnet_call_twice(first_input, second_input)
    fact.two_backnet_call_twice(first_input, second_input)
    fact.first_forward_second_backnet(first_input, second_input)


class CustomNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(Tensor(np.array([1.0], np.float32)), name='p1')
        self.p2 = Parameter(Tensor(np.array([1.0], np.float32)), name='p2')
        self.p3 = Parameter(Tensor(np.array([1.0], np.float32)), name='p3')
        self.p1.requires_grad = False
        self.p2.requires_grad = False
        self.p3.requires_grad = True

    def construct(self, x):
        out = self.p1 * x
        out = out * self.p2
        out = out + self.p3
        return out


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_requires_grad():
    """
    Feature: Test pynative requires grad
    Description: Test the code for requires grad
    Expectation: success
    """
    x = Tensor([1], ms.float32)
    net = CustomNet()
    output = GradOfAllInputsAndParams(net, sens_param=False)(x)
    assert (output[1][0].asnumpy() == np.array([1.0], dtype=np.float32)).all()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_requires_grad_use_grad_operation():
    """
    Feature: Test pynative requires grad use grad operation
    Description: Test the code for requires grad
    Expectation: success
    """

    # Cell object to be differentiated
    x = Tensor([1], ms.float32)
    net = CustomNet()
    output = GradOperation(get_all=True, get_by_list=True)(net, [net.p1, net.p2, net.p3])(x)
    assert (output[1][0].asnumpy() == np.array([0.0], dtype=np.float32)).all()
    assert (output[1][1].asnumpy() == np.array([0.0], dtype=np.float32)).all()
    assert (output[1][2].asnumpy() == np.array([1.0], dtype=np.float32)).all()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_requires_grad_without_params():
    """
    Feature: Test pynative requires grad without params
    Description: Test the code for requires grad
    Expectation: success
    """

    # Cell object to be differentiated
    x = Tensor([1], ms.float32)
    net = CustomNet()
    output = GradOperation(get_all=True, get_by_list=True)(net)(x)
    assert len(output[1]) == 1
    assert (output[1][0].asnumpy() == np.array([1.0], dtype=np.float32)).all()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_requires_grad_case2():
    """
    Feature: Test pynative requires grad case2
    Description: Test the code for requires grad
    Expectation: success
    """

    # Cell object to be differentiated
    x = Tensor([1], ms.float32)
    net = CustomNet()
    output = GradOperation(get_all=True, get_by_list=True)(net, [net.p1])(x)
    assert (output[1][0].asnumpy() == np.array([0.0], dtype=np.float32)).all()
    assert len(output[1]) == 1


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_kwargs_with_no_sens():
    """
    Feature: Test kwargs with no sens.
    Description: Run kwargs with no sens.
    Expectation: No exception.
    """
    inputs = Tensor([1., 2., 3.])
    kwargs = {"approximate": "tanh"}
    grad_fn = GradOperation(get_all=True, sens_param=False)(ops.gelu)
    grad_fn(inputs, **kwargs)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_kwargs_with_sens_not_in_kwargs():
    """
    Feature: Test kwargs with no sens.
    Description: Run kwargs with no sens.
    Expectation: No exception.
    """
    inputs = Tensor([1., 2., 3.])
    gradiente_inputs = Tensor([1., 2., 3.])
    kwargs = {"approximate": "tanh"}
    grad_fn = GradOperation(get_all=True, sens_param=True)(ops.gelu)
    grad_fn(inputs, gradiente_inputs, **kwargs)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_kwargs_with_sens_in_kwargs():
    """
    Feature: Test kwargs with sens.
    Description: Run kwargs with sens.
    Expectation: No exception.
    """
    inputs = Tensor([1., 2., 3.])
    kwargs = {'sens': Tensor([1., 2., 3.]), "approximate": "tanh"}
    grad_fn = GradOperation(get_all=True, sens_param=True)(ops.gelu)
    grad_fn(inputs, **kwargs)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_jit_with_while():
    """
    Feature: Control flow
    Description: Test control flow in jit function under pynative mode.
    Expectation: No exception.
    """

    class InnerNet(nn.Cell):
        @ms.jit
        def construct(self, x, y):
            while x < y:
                x = x * x + 1
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.grad_op = C.GradOperation(get_all=True)

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = Tensor([2.0], dtype=mstype.float32)
    y = Tensor([2.0], dtype=mstype.float32)
    grads = GradNet(InnerNet())(x, y)
    assert np.allclose(grads[0].asnumpy(), 1.0, 0.001, 0.001)
    assert np.allclose(grads[1].asnumpy(), 0.0, 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_jit_with_dict_input():
    """
    Feature: Grad jit function has dict input
    Description: Test calculate grad of jit function has dict input under pynative mode.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net(input_str):
        x = input_str["a"]
        m = 2 * x + 1
        return m

    x = Tensor(2)
    out = GradOperation()(dict_net)({"a": x})
    assert np.allclose(out, 2.0, 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_jit_with_scalar_output():
    """
    Feature: Grad jit function has multiple output contain scalar
    Description: Test jit function has multiple output contain scalar under pynative mode.
    Expectation: No exception.
    """

    @jit
    def fn(x):
        m = x + 1
        z = x * (m + 2) + 2 * m
        return z, 1

    x = Tensor([1.0, 2.0])
    grad1 = GradOperation()(fn)(x)
    assert (grad1.asnumpy() == [7.0, 9.0]).all()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_jit_bprop_net():
    """
    Feature: Test jit grad custom bprop construct func
    Description: Test jit grad custom bprop construct func
    Expectation: Success.
    """

    class CustomBpropNet(nn.Cell):
        @jit
        def construct(self, x):
            y = x * x
            z = y + y
            return z

        def bprop(self, *args):
            return (args[0] * 4,)

    x = Tensor([2], ms.float32)
    net = CustomBpropNet()
    grads = GradOperation()(net)(x)
    assert np.allclose(grads.asnumpy(), np.array([8], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=ms.float32))
    grads = GradOperation()(net)(x)
    assert np.allclose(grads.asnumpy(), np.array([8], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_jit_stop_gradient():
    """
    Feature: Test jit grad stop gradient.
    Description: Test jit grad stop gradient.
    Expectation: Success.
    """

    class StopGradientNet(nn.Cell):
        def __init__(self):
            super(StopGradientNet, self).__init__()
            self.p1 = Parameter(Tensor([2], dtype=ms.float32))

        @jit
        def construct(self, x):
            y = x * x
            y = ops.stop_gradient(y)
            z = y * self.p1
            return z

    x = Tensor([2], ms.float32)
    net = StopGradientNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net)(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=ms.float32))
    grads = grad_net(net)(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


class StopGradientInplaceNet(nn.Cell):
    def construct(self, x):
        y = x * x
        ops.stop_gradient_(y)
        z = y * x
        return z


class StopGradientInplaceViewNet(nn.Cell):
    def construct(self, x):
        y = x * x
        y = y[0]
        ops.stop_gradient_(y)
        z = y * x
        return z


class StopGradientInplaceParameterNet(nn.Cell):
    def __init__(self):
        super(StopGradientInplaceParameterNet, self).__init__()
        self.p1 = Parameter(Tensor([2.0, 3.0], dtype=ms.float32))

    def construct(self, x):
        ops.stop_gradient_(self.p1)
        return x * self.p1


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_stop_gradient_inplace():
    """
    Feature: Test stop gradient inplace.
    Description: Test stop gradient inplace.
    Expectation: Success.
    """
    x = Tensor([2.0, 3.0], ms.float32)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)

    net = StopGradientInplaceNet()
    grads = grad_op(net)(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([4.0, 9.0], dtype=np.float32), 0.00001, 0.00001)

    net = StopGradientInplaceParameterNet()
    grads = grad_op(net, net.trainable_params())(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2.0, 3.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([0.0, 0.0], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_stop_gradient_inplace_input():
    """
    Feature: Test stop gradient inplace.
    Description: The input is applied by inplace stop_gradient.
    Expectation: Success.
    """
    x = Tensor([2.0], ms.float32)
    y = Tensor([3.0], ms.float32)

    def fn(x, y):
        ops.stop_gradient_(x)
        return x * y
    grad_op = ops.GradOperation(get_all=True)
    grads = grad_op(fn)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([0.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([2.0], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_no_grad_stop_gradient_inplace_view():
    """
    Feature: Test stop gradient inplace view in no grad mode.
    Description: Test stop gradient inplace view in no grad mode.
    Expectation: Raise Runtime Error.
    """
    x = Tensor([2.0, 3.0], ms.float32)
    net = StopGradientInplaceViewNet()
    with pytest.raises(RuntimeError, match="Cannot stop_gradient view inplace"):
        net(x)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_stop_gradient_inplace_view_error():
    """
    Feature: Test stop gradient inplace view exception.
    Description: The operation will raise error.
    Expectation: Success.
    """
    x = Tensor([2.0, 3.0], ms.float32)
    net = StopGradientInplaceViewNet()
    grad_op = ops.GradOperation(get_all=True)
    with pytest.raises(RuntimeError, match="Cannot stop_gradient view inplace"):
        grad_op(net)(x)


class AuxNet(nn.Cell):
    def construct(self, x):
        y = x * x
        z = y + y
        h = x * x
        return y, z, h


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_has_aux():
    """
    Feature: Test hax aux.
    Description: Test value_and_grad has aux.
    Expectation: Success.
    """
    x = Tensor([2.0], ms.float32)
    net = AuxNet()
    grad_op = ops.value_and_grad(net, 0, None, True)
    _, grads = grad_op(x)
    assert np.allclose(grads.asnumpy(), np.array([4.0], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionAutoReduceNet(nn.Cell):
    def construct(self, x, y):
        x2 = x + y
        return x2

    def bprop(self, *args):
        return Tensor([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]]), Tensor([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]])


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_auto_reduce():
    """
    Feature: Custom bprop function.
    Description: Test auto reduce.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], ms.float32)
    net = CustomFunctionAutoReduceNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([4., 4., 4.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]], dtype=np.float32),
                       0.00001, 0.00001)


class CustomFunctionAutoCastNet(nn.Cell):
    def construct(self, x, y):
        x2 = x + y
        return x2

    def bprop(self, *args):
        return Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=ms.int64), Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]],
                                                                                 dtype=ms.int64)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_auto_cast():
    """
    Feature: Custom bprop function.
    Description: Test auto cast.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], ms.float32)
    net = CustomFunctionAutoCastNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert grads[0].dtype == ms.float32
    assert grads[1].dtype == ms.float32
    assert np.allclose(grads[0].asnumpy(), np.array([4., 4., 4.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]], dtype=np.float32),
                       0.00001, 0.00001)


class CustomFunctionBroadcastExecptionNet(nn.Cell):
    def construct(self, x, y):
        x2 = x + y
        return x2

    def bprop(self, *args):
        return Tensor([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=ms.int64), \
               Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=ms.int64)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_reduce_exception():
    """
    Feature: Custom bprop function.
    Description: Test auto reduce.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], ms.float32)
    net = CustomFunctionBroadcastExecptionNet()
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "For custom function, grad tensor should be broadcast to" in str(err.value)


class CustomFunctionReturnSelfNet(nn.Cell):
    def construct(self, x):
        return x

    def bprop(self, *args):
        return Tensor([1, 1, 1], dtype=ms.float32)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_return_self_net():
    """
    Feature: Custom bprop function.
    Description: Test bprop function return self.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    net = CustomFunctionReturnSelfNet()
    net.set_grad()
    output = net(x)
    grad_net = C.GradOperation(get_all=True)
    grad_net(net)(x)
    assert id(output) != id(x)


class CustomFunctionMultiOutputReturnSelfNet(nn.Cell):
    def construct(self, x):
        return x, Tensor([3, 3, 3], ms.float32)

    def bprop(self, *args):
        return Tensor([1, 1, 1], dtype=ms.float32)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_output_return_self_net():
    """
    Feature: Custom bprop function.
    Description: Test bprop function return self.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    net = CustomFunctionMultiOutputReturnSelfNet()
    net.set_grad()
    output = net(x)
    grad_net = C.GradOperation(get_all=True)
    grad_net(net)(x)
    assert id(output[0]) != id(x)
