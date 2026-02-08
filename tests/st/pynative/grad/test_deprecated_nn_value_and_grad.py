# Copyright 2025 Huawei Technologies Co., Ltd
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
""" test_auto_grad """

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore import nn
from mindspore.common.parameter import Parameter, ParameterTuple
from tests.mark_utils import arg_mark


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
            self.para = Parameter(Tensor([2, 3, 4], mindspore.float32), name="para")

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
            self.para = Parameter(Tensor([1], mindspore.float32), name="para")

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
