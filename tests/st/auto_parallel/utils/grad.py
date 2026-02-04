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
"""Gradient computation utilities for testing automatic differentiation in MindSpore.

This module provides wrapper classes for computing gradients with respect to inputs and/or
parameters in neural networks. It supports various gradient computation modes including:
- Gradients of first input only
- Gradients of all inputs
- Gradients of all parameters
- Gradients of both inputs and parameters
- Higher-order gradients (e.g., second derivatives)

The module also includes performance profiling capabilities for measuring gradient computation time,
and support for dynamic shape inference during gradient computation.

Key Features:
- Flexible gradient computation with GradOperation wrapper
- Support for sensitivity parameter (loss scaling)
- Performance timing for profiling
- Higher-order derivative computation
- Real inputs count tracking for mixed input types
- Dynamic shape handling during gradient operations

Classes:
- _Grad: Base gradient computation wrapper cell
- GradOfFirstInput: Computes gradients of network output w.r.t. first input
- GradOfAllInputs: Computes gradients w.r.t. all inputs
- GradOfAllParams: Computes gradients w.r.t. all trainable parameters
- GradOfAllInputsAndParams: Computes gradients w.r.t. inputs and parameters
- HighGrad: Computes higher-order gradients (nested gradient operations)

Usage:
    This module is primarily used in distributed training tests to validate gradient
    computation correctness across different parallelism strategies and configurations.
"""
import time
import stat
import os
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

    def __call__(self, *inputs):
        if os.getenv("MS_DEV_AUTO_DYNAMIC_SHAPE") == "on":
            # auto dynamic convert all inputs and sens in __call__
            # here we do convert in advance
            # sens will be set static in the following code
            self._get_compile_args(inputs)
        if self.sens_param and self._dynamic_shape_inputs is not None:
            # not support dynamic shape sens
            if self.real_inputs_count is None:
                dyn_inputs = self._dynamic_shape_inputs[:-1]
                real_sens = inputs[-1:]
            else:
                idx = self.real_inputs_count
                dyn_inputs = self._dynamic_shape_inputs[:idx]
                real_sens = inputs[idx:]
            static_sens = list(dyn_inputs) + list(real_sens)
            super().set_inputs(*static_sens)

        a = time.perf_counter()
        out = super().__call__(*inputs)
        b = time.perf_counter()
        if os.environ.get("perf") == '1':
            phase = os.environ.get("PHASE")
            flags = os.O_WRONLY | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(phase, flags, modes), 'w') as f:
                f.write(str(b - a))
        return out

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


class HighGrad(Cell):
    """
    get any order of grad
    """
    def __init__(self, network, grad_list, sens_param=False, real_inputs_count=None):
        super().__init__()
        self.grads = [network, ]
        for i in range(len(grad_list)-1):
            _grad = grad_list[i](self.grads[i], sens_param=False)
            self.grads.append(_grad)
        self.final_grad = grad_list[-1](self.grads[-1],
                sens_param=sens_param, real_inputs_count=real_inputs_count)

    def construct(self, *inputs):
        return self.final_grad(*inputs)
