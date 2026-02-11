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

from mindspore.nn import Cell
from mindspore.ops.composite import GradOperation
from mindspore.common import ParameterTuple
import numpy as np


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
        super().__init__(grad=GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class GradOfAllInputs(_Grad):
    """
    get grads of all inputs
    """
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class GradOfAllParams(_Grad):
    """
    get grads of all params
    """
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_by_list=True, sens_param=sens_param),
                         network=network, wrt_params=True, real_inputs_count=real_inputs_count)


class GradOfAllInputsAndParams(_Grad):
    """
    get grads of all inputs and params
    """
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, get_by_list=True,
                                            sens_param=sens_param),
                         network=network, wrt_params=True, real_inputs_count=real_inputs_count)


class GradOfDefault(_Grad):
    """
    get default grad
    """

    def __init__(self, network, sens_param=False, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=False, get_by_list=False,
                                            sens_param=sens_param),
                         network=network, wrt_params=False, real_inputs_count=real_inputs_count)


class HighGrad(Cell):
    """
    get any order of grad
    """
    def __init__(self, network, grad_list, sens_param=False, real_inputs_count=None):
        super().__init__()
        self.grads = [network,]
        for i in range(len(grad_list)-1):
            _grad = grad_list[i](self.grads[i], sens_param=False)
            self.grads.append(_grad)
        self.final_grad = grad_list[-1](self.grads[-1],
                                        sens_param=sens_param, real_inputs_count=real_inputs_count)

    def construct(self, *inputs):
        return self.final_grad(*inputs)


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)) or np.any(np.isnan(data_me)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape
