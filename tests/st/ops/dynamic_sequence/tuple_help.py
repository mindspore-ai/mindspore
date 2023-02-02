# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore.common import mutable
from mindspore.nn import Cell
from mindspore.ops.composite import GradOperation
from mindspore.common import ParameterTuple


class _Grad(Cell):
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
        return self.grad(self.network)(*inputs)


class GradOfFirstInput(_Grad):
    """
    get grad of first input
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class GradOfAllInputs(_Grad):
    """
    get grads of all inputs
    """
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class TupleFactory():
    def __init__(self, net_x, func_x, input_x, const_value_idx=None):
        self.input_num = len(input_x)
        self.input = []
        if const_value_idx is None:
            const_value_idx = []
        for i, item in enumerate(input_x):
            if i not in const_value_idx:
                if isinstance(item, int):
                    self.input.append(mutable(item))
                else:
                    self.input.append(mutable(item, True))
            else:
                self.input.append(item)
        self.input_func = input_x
        self.net = net_x
        self.func = func_x
        self.grad_input = None

    def forward_cmp(self):
        out_func = self.func(*self.input_func)
        self.grad_input = out_func
        out_mindspore = self.net(*self.input)
        assert out_func == out_mindspore

    def grad_impl(self):
        grad_net = GradOfFirstInput(self.net) if self.input_num == 1 else GradOfAllInputs(self.net)
        grad_net.set_train()
        input_grad = grad_net(*self.input, self.grad_input)
        return input_grad
