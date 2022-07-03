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
import numpy as np
import pytest

import mindspore.context as context
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.ops.composite import GradOperation
from mindspore.common import ParameterTuple

TF_INSTALL_FLG = 1
try:
    import tensorflow as tf
except ImportError:
    TF_INSTALL_FLG = 0

context.set_context(device_target="Ascend")


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


class GradOfAllInputs(_Grad):
    """
    get grads of all inputs
    """
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class HighGrad(Cell):
    """
    get any order of grad
    """
    def __init__(self, network, grad_list, sens_param=False, real_inputs_count=None):
        super().__init__()
        self.grads = [network]
        for i in range(len(grad_list)-1):
            _grad = grad_list[i](self.grads[i], sens_param=False)
            self.grads.append(_grad)
        self.final_grad = grad_list[-1](self.grads[-1],
                                        sens_param=sens_param, real_inputs_count=real_inputs_count)

    def construct(self, *inputs):
        return self.final_grad(*inputs)


class BiasAdd(nn.Cell):
    def __init__(self, ms_format):
        super().__init__()
        self.op = P.BiasAdd(ms_format)

    def construct(self, x, b):
        return self.op(x, b)


def count_unequal_element(data_expected, data_me, rtol, atol):
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
        count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


class TestEntry:
    def __init__(self, input_x_np, dtype, ms_format, tf_format):
        self.input_x_np = input_x_np
        self.dtype = dtype
        self.ms_format = ms_format
        self.tf_format = tf_format

        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype == np.float32:
            self.loss = 1e-4
        elif self.dtype == np.float64:
            self.loss = 1e-5
        elif self.dtype == np.complex64:
            self.loss = 2e-6
        elif self.dtype == np.complex128:
            self.loss = 2e-10
        else:
            self.loss = 0

    def highgrad_mindspore_impl(self):
        x = Tensor(self.input_x_np[0].copy().astype(self.dtype))
        b = Tensor(self.input_x_np[1].copy().astype(self.dtype))

        net = BiasAdd(ms_format=self.ms_format)
        grad_net = HighGrad(net, grad_list=[GradOfAllInputs, GradOfAllInputs])
        y = grad_net(x, b)
        return y

    def highgrad_tensorflow_impl(self):
        x = tf.Variable(self.input_x_np[0].copy().astype(self.dtype))
        b = tf.Variable(self.input_x_np[1].copy().astype(self.dtype))

        with tf.GradientTape(persistent=True) as tape:
            y = tf.nn.bias_add(x, b, data_format=self.tf_format)
            dydx, dydb = tape.gradient(y, [x, b])
        ddx, ddb = tape.gradient([dydx, dydb], [x, b], unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return ddx, ddb

    def highgrad_cmp(self):
        out_ms = self.highgrad_mindspore_impl()
        if TF_INSTALL_FLG == 1:
            out_tf = self.highgrad_tensorflow_impl()
        else:
            out_tf = []
            out_tf.append(np.zeros_like(self.input_x_np[0]))
            out_tf.append(np.zeros_like(self.input_x_np[1]))
        allclose_nparray(out_tf[0], out_ms[0].asnumpy(), self.loss, self.loss)
        allclose_nparray(out_tf[1], out_ms[1].asnumpy(), self.loss, self.loss)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_biasadd_high_grad_dim2_float16():
    """
    Feature: Biasadd Grad Grad operation
    Description: test the high grad of Rsqrt. Input tensor has 2 dims, float16 type.
    Expectation: the output is same with tensorflow
    """
    test = TestEntry(input_x_np=[np.arange(1, 7).reshape((2, 3)), np.ones(shape=(3,))],
                     dtype=np.float16, ms_format="NCHW", tf_format="NCHW")
    test.highgrad_cmp()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_biasadd_high_grad_dim4_float32():
    """
    Feature: Biasadd Grad Grad operation
    Description: test the high grad of Rsqrt. Input tensor has 4 dims, float32 type.
    Expectation: the output is same with tensorflow
    """
    test = TestEntry(input_x_np=[np.random.randn(3, 2, 3, 3), np.ones(shape=(2,))],
                     dtype=np.float32, ms_format="NCHW", tf_format="NCHW")
    test.highgrad_cmp()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_biasadd_high_grad_dim5_float64():
    """
    Feature: Biasadd Grad Grad operation
    Description: test the high grad of Rsqrt. Input tensor has 5 dims, float64 type.
    Expectation: the output is same with tensorflow
    """
    test = TestEntry(input_x_np=[np.random.randn(1, 5, 2, 2, 2), np.ones(shape=(5,))],
                     dtype=np.float64, ms_format="NCDHW", tf_format="NCDHW")
    test.highgrad_cmp()
