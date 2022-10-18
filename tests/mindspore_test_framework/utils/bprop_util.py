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

"""Utils for computing gradients."""

from mindspore import context
from mindspore.common import ParameterTuple
from mindspore.common.api import jit
from mindspore.nn import Cell
from mindspore.ops.composite.base import GradOperation


class Bprop(Cell):
    """
    The gradient wrapper.
    """

    def __init__(self, func, wrt_params, params, grad_op, sens):
        super(Bprop, self).__init__(auto_prefix=False)
        self.func = func
        self.wrt_params = wrt_params
        self.params = None
        if self.wrt_params and params:
            self.params = ParameterTuple(params)
        self.grad = grad_op
        self.sens = sens
        self.with_sens = False
        if sens is not None:
            self.with_sens = True

    def construct(self, *inputs):
        # pylint: disable=no-else-return
        if self.wrt_params:
            if self.with_sens:
                return self.grad(self.func, self.params)(*inputs, self.sens)
            else:
                return self.grad(self.func, self.params)(*inputs)
        elif self.with_sens:
            return self.grad(self.func)(*inputs, self.sens)
        else:
            return self.grad(self.func)(*inputs)


def bprop(func, *inputs, grads_wrt_outputs=None, wrt: list = None, params: list = None):
    """
    Compute gradients of function.

    Args:
        func (Function): The target function.
        inputs (Variable argument): Inputs of the func.
        grads_wrt_outputs (List): Gradients of the loss wrt outputs of func, default [1.0].
        wrt (List): Compute gradients wrt ['inputs' | 'params'].
        params (List): Specify the params to compute gradients wrt, default all trainable_params.

    Returns:
        Tuple, gradients of function.
    """
    assert isinstance(func, Cell)
    func.set_train()

    with_sens_param = False
    if grads_wrt_outputs is not None:
        with_sens_param = True

    if wrt is None:
        wrt = []
    wrt_inputs = False
    if 'inputs' in wrt:
        wrt_inputs = True
    wrt_params = False
    if 'params' in wrt:
        wrt_params = True
        if not params:
            params = func.trainable_params()

    grad_op = GradOperation(get_all=wrt_inputs, get_by_list=wrt_params, sens_param=with_sens_param)
    grad = Bprop(func, wrt_params, params, grad_op, grads_wrt_outputs)

    if context.get_context("mode") == context.PYNATIVE_MODE:
        def func_pynative(*inputs):
            @jit
            def _func_pynative(*inputs):
                return grad(*inputs)

            return _func_pynative(*inputs)

        return func_pynative(*inputs)
    return grad(*inputs)
