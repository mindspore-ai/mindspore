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
""" test_pynative_layernorm_input_and_argmaxwithvalue """
import numpy as np
import mindspore.ops.operations as op
from mindspore import Tensor, context
from mindspore.nn import LayerNorm, Cell
from mindspore.common import ParameterTuple
from mindspore.ops.composite import GradOperation
from mindspore.train import Model
from tests.mark_utils import arg_mark

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

class GradOfAllInputsAndParams(_Grad):
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, get_by_list=True, sens_param=sens_param),
                         network=network, wrt_params=True, real_inputs_count=real_inputs_count)

class MetaFactory:
    def __init__(self):
        self.device_target = context.get_context('device_target')
        self.rank_size = None
        self.device_id = None
        self.global_rank_id = None

class OpsFactory(MetaFactory):
    def __init__(self, dtype=np.float16):
        super().__init__()
        self.dtype = dtype
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype == np.float32:
            self.loss = 1e-4
        elif self.dtype == np.float64:
            self.loss = 1e-5
        else:
            self.loss = 0

def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me)*rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count/total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])

def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True

class LayerNormFactory(OpsFactory):
    def __init__(self, input_shape, norm_shape, gamma_shape, beta_shape, gamma_init=None, beta_init=None,
                 norm_axis=-1, params_axis=-1, dtype=np.float32):
        super().__init__(dtype=dtype)
        np.random.seed(1)
        self.input_np = np.random.randn(*input_shape).astype(dtype=dtype)
        self.gamma_np = np.ones(shape=gamma_shape, dtype=dtype)
        self.gamma_init = gamma_init
        self.beta_np = np.zeros(shape=beta_shape, dtype=dtype)
        self.beta_init = beta_init
        self.output_grad_np = np.random.randn(*input_shape).astype(dtype=dtype)
        self.begin_norm_axis = norm_axis
        self.begin_params_axis = params_axis
        self.input_shape = norm_shape

    def forward_mindspore_impl(self):
        input_ms = Tensor(self.input_np)
        gamma = Tensor(self.gamma_np)
        beta = Tensor(self.beta_np)
        net = LayerNorm(self.input_shape, self.begin_norm_axis, self.begin_params_axis, gamma, beta)
        net.set_train()
        model = Model(net)
        out_me = model.predict(Tensor(input_ms))
        return out_me.asnumpy()

    def grad_mindspore_impl(self):
        input_nn = Tensor(self.input_np)
        output_grad = Tensor(self.output_grad_np)
        net = LayerNorm(self.input_shape, self.begin_norm_axis, self.begin_params_axis,
                        Tensor(self.gamma_np), Tensor(self.beta_np))
        grad_net = GradOfAllInputsAndParams(net)
        grad_net.set_train()
        input_grad = grad_net(input_nn, output_grad)
        return input_grad[0][0].asnumpy(), input_grad[1][1].asnumpy(), input_grad[1][0].asnumpy()

    def forward_cmp(self):
        context.set_context(mode=context.GRAPH_MODE, device_target=context.get_context('device_target'))
        graph_out = self.forward_mindspore_impl()

        context.set_context(mode=context.PYNATIVE_MODE, device_target=context.get_context('device_target'))
        pynative_out = self.forward_mindspore_impl()

        allclose_nparray(graph_out[0], pynative_out[0], self.loss, self.loss)

    def grad_cmp(self):
        context.set_context(mode=context.GRAPH_MODE, device_target=context.get_context('device_target'))
        graph_grad1, graph_grad2, graph_grad3 = self.grad_mindspore_impl()

        context.set_context(mode=context.PYNATIVE_MODE, device_target=context.get_context('device_target'))
        pynative_grad1, pynative_grad2, pynative_grad3 = self.grad_mindspore_impl()

        allclose_nparray(graph_grad1, pynative_grad1, self.loss, self.loss)
        allclose_nparray(graph_grad2, pynative_grad2, self.loss, self.loss)
        allclose_nparray(graph_grad3, pynative_grad3, self.loss, self.loss)

class ArgMaxWithValue(Cell):
    def __init__(self, axis, keep_dims):
        super().__init__()
        self.op = op.ArgMaxWithValue(axis=axis, keep_dims=keep_dims)

    def construct(self, input_value):
        return self.op(input_value)

class GradOfFirstInput(_Grad):
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)

class ArgMaxWithValueFactory(OpsFactory):
    def __init__(self, input_shape, axis, keep_dims, dtype=np.float32):
        super().__init__(dtype=dtype)
        np.random.seed(1)
        self.input_np = np.random.rand(*input_shape).astype(dtype)
        self.output_grad_np = None
        self.axis = axis
        self.keep_dims = keep_dims
        self.index_dtype = None

    def forward_mindspore_impl(self):
        input_forward = Tensor(self.input_np)
        net = ArgMaxWithValue(axis=self.axis, keep_dims=self.keep_dims)
        index, value = net(input_forward)
        if self.index_dtype is None:
            self.index_dtype = index.dtype
        return index.asnumpy().reshape(1, -1), value.asnumpy()

    def forward_numpy_impl(self):
        index = np.argmax(self.input_np, axis=self.axis)
        value = np.amax(self.input_np, axis=self.axis, keepdims=self.keep_dims)
        return index.reshape(1, -1), value.astype(self.dtype)

    def grad_mindspore_impl(self):
        input_back = Tensor(self.input_np)
        np.random.seed(1)
        self.output_grad_np = np.random.randn(*input_back[0].shape).astype(self.dtype)
        output_grad = Tensor(self.output_grad_np, self.index_dtype)
        output_grad_2 = Tensor(self.output_grad_np)
        net = ArgMaxWithValue(axis=self.axis, keep_dims=self.keep_dims)
        grad_net = GradOfFirstInput(net, real_inputs_count=1)
        grad_net.set_train()
        input_grad = grad_net(input_back, output_grad, output_grad_2)
        return input_grad.asnumpy()

    def forward_cmp(self):
        context.set_context(mode=context.PYNATIVE_MODE, device_target=context.get_context('device_target'))
        out_numpy = self.forward_numpy_impl()
        out_mindspore = self.forward_mindspore_impl()
        allclose_nparray(out_numpy[0], out_mindspore[0], self.loss, self.loss)
        allclose_nparray(out_numpy[1], out_mindspore[1], self.loss, self.loss)

    def grad_cmp(self):
        context.set_context(mode=context.GRAPH_MODE, device_target=context.get_context('device_target'))
        graph_grad = self.grad_mindspore_impl()

        context.set_context(mode=context.PYNATIVE_MODE, device_target=context.get_context('device_target'))
        pynative_grad = self.grad_mindspore_impl()

        allclose_nparray(graph_grad, pynative_grad, self.loss, self.loss)

def layernorm_input():
    fact = LayerNormFactory(input_shape=(1, 128, 1024), norm_shape=(1024,), gamma_shape=(1024,), beta_shape=(1024,),
                            norm_axis=2, params_axis=2, dtype=np.float16)
    fact.forward_cmp()
    fact.loss = 5e-3
    fact.grad_cmp()

def argmaxwithvalue_input():
    fact = ArgMaxWithValueFactory(input_shape=[1024, 1024], axis=-1, keep_dims=False)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_layernorm_input_ascend():
    context.set_context(device_target="Ascend")
    layernorm_input()


@arg_mark(plat_marks=['platform_gpu'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_layernorm_input_gpu():
    context.set_context(device_target="GPU")
    layernorm_input()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_argmaxwithvalue_input_ascend():
    context.set_context(device_target="Ascend")
    argmaxwithvalue_input()


@arg_mark(plat_marks=['platform_gpu'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_argmaxwithvalue_input_gpu():
    context.set_context(device_target="GPU")
    argmaxwithvalue_input()
