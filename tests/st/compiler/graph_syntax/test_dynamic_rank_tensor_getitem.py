# coding=utf-8

import numpy as np
import os

import mindspore as ms
from mindspore import Tensor, nn, mutable, ops, ParameterTuple, context

from tests.mark_utils import arg_mark
from tests.st.compiler.utils import match_array


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

    def __call__(self, *inputs):
        if os.getenv('MS_DEV_AUTO_DYNAMIC_SHAPE') == 'on':
            self._get_compile_args(inputs)
        if self.sens_param and self._dynamic_shape_inputs is not None:
            # not support dynamic shape
            if self.real_inputs_count is None:
                dyn_inputs = self._dynamic_shape_inputs[:-1]
                real_sens = inputs[-1:]
            else:
                idx = self.real_inputs_count
                dyn_inputs = self._dynamic_shape_inputs[:idx]
                real_sens = inputs[idx:]
            static_sens = list(dyn_inputs) + list(real_sens)
            super().set_inputs(*static_sens)

        out = super().__call__(*inputs)
        return out

    def construct(self, *inputs):
        # pylint: disable=no-else-return
        if self.wrt_params:
            if self.real_inputs_count is None or not self.sens_param:
                return self.grad(self.network, self.params)(*inputs)
            else:
                real_inputs = inputs[:self.real_inputs_count]
                sense_param_inputs = inputs[self.real_inputs_count:]
                return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)
        else:
            if self.real_inputs_count is None or not self.sens_param:
                return self.grad(self.network)(*inputs)
            else:
                real_inputs = inputs[:self.real_inputs_count]
                sense_param_inputs = inputs[self.real_inputs_count:]
                return self.grad(self.network)(*real_inputs, sense_param_inputs)


class GradOfAllInputs(_Grad):
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=ops.GradOperation(get_all=True, sens_param=sens_param), network=network,
                         real_inputs_count=real_inputs_count)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_list_mutable():
    """
    Features: Function grad.
    Description: Test ones_like_for_grad in multitype_funcgraph.
    Expectation: No exception.
    """

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.idx = mutable([2, 1, 0])

        def construct(self, x: Tensor):
            out = x[self.idx]
            return out

    x = Tensor(np.random.rand(3, 3, 2), dtype=ms.float32)
    d = Tensor(None, dtype=ms.float32)

    pynative_model = Model()
    pynative_model.set_inputs(d)
    jit_model = Model()
    jit_model.set_inputs(d)

    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_out = pynative_model(x)
    pynative_grad = GradOfAllInputs(pynative_model, sens_param=False)(x)

    context.set_context(mode=context.GRAPH_MODE)
    jit_out = jit_model(x)
    jit_grad = GradOfAllInputs(jit_model, sens_param=False)(x)

    match_array(pynative_out, jit_out, error=5)
    for g1, g2 in zip(pynative_grad, jit_grad):
        if g1 is None:
            continue
        match_array(g1, g2, error=5)
