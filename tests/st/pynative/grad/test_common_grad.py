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
""" test_bprop """
import numpy as np
import mindspore as ms
from mindspore import grad
import mindspore.nn as nn
from mindspore import context
from mindspore.common import Tensor
from mindspore.common.api import jit
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import GradOperation
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
          level_mark='level1',
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


@arg_mark(plat_marks=['platform_gpu'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@arg_mark(plat_marks=['platform_gpu'],
          level_mark='level0',
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
        self.p3 = Parameter(Tensor(np.array([1.0], np.float32)), name='p2')
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
    assert (output[1][0].asnumpy() == np.array([0.0], dtype=np.float32)).all()
    assert (output[1][1].asnumpy() == np.array([0.0], dtype=np.float32)).all()
    assert (output[1][2].asnumpy() == np.array([1.0], dtype=np.float32)).all()


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
