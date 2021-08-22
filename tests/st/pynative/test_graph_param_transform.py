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
import pytest
import numpy as np
from mindspore import RowTensor
from mindspore import context, nn, Tensor, ParameterTuple
from mindspore.common import dtype as mstype
from mindspore.common import ms_function
from mindspore.ops import operations as P
from mindspore.ops import composite as C


def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE, enable_sparse=False)


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
        super().__init__(grad=C.GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class GradOfAllInputs(_Grad):
    """
    get grad of first input
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=C.GradOperation(get_all=True, sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_row_tensor_in_while():
    class RowTensorValuesDouble(nn.Cell):

        def construct(self, x):
            indices = x.indices
            values = x.values * 2
            dense_shape = x.dense_shape
            return RowTensor(indices, values, dense_shape)

    class RowTensorValuesAdd2(nn.Cell):

        def construct(self, x):
            indices = x.indices
            values = x.values + 2
            dense_shape = x.dense_shape
            return RowTensor(indices, values, dense_shape)

    class RowTensorWithControlWhile(nn.Cell):
        def __init__(self, dense_shape):
            super().__init__()
            self.op1 = RowTensorValuesDouble()
            self.op2 = RowTensorValuesAdd2()
            self.dense_shape = dense_shape

        @ms_function
        def construct(self, a, b, indices, values):
            x = RowTensor(indices, values, self.dense_shape)
            x = self.op2(x)
            while a > b:
                x = self.op1(x)
                b = b + 1
            return x.indices, x.values, x.dense_shape
    a = Tensor(np.array(3).astype(np.int32))
    b = Tensor(np.array(0).astype(np.int32))
    indices = Tensor(np.array([0, 2]).astype(np.int32))
    values = Tensor(np.ones([2, 2]).astype(np.float32))
    dense_shape = (5, 2)
    net = RowTensorWithControlWhile(dense_shape)
    out = net(a, b, indices, values)
    assert np.allclose(indices.asnumpy(), out[0].asnumpy(), .0, .0)
    assert np.allclose(values.asnumpy()*24, out[1].asnumpy(), .0, .0)
    assert dense_shape == out[2]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parser_switch_layer_inputs_tuple():
    class Add(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = P.Add()

        def construct(self, x):
            y = self.op(x[0], x[1])
            return self.op(x[0], y)

    class Mul(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = P.Mul()

        def construct(self, x):
            y = self.op(x[0], x[1])
            return self.op(x[0], y)

    class MulTwoInput(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = P.Mul()

        @ms_function
        def construct(self, x, y):
            y = self.op(x, y)
            return self.op(x, y)

    class TwoInputTupleFinalNet(nn.Cell):
        def __init__(self, funcs):
            super().__init__()
            self.funcs = funcs

        @ms_function
        def construct(self, i, inputa, inputb):
            inputs = (inputa, inputb)
            x = self.funcs[i](inputs)
            return x

    func1 = Add()
    func2 = Mul()

    funcs = (func1, func2)
    net = TwoInputTupleFinalNet(funcs)

    input_data = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    input2 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    i = Tensor(1, mstype.int32)
    netout = net(i, input_data, input2)
    net_good = MulTwoInput()
    goodout = net_good(input_data, input2)
    assert np.allclose(goodout.asnumpy(), netout.asnumpy(), 0, 0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_imagenet():
    class ImageGradients(nn.Cell):
        def __init__(self):
            super().__init__()
            self.imagegradients = nn.ImageGradients()

        def construct(self, inputs):
            return self.imagegradients(inputs)

    net = ImageGradients()
    net_me = GradOfFirstInput(net, real_inputs_count=1)
    net_me.set_train()
    input_data = Tensor(np.ones([32, 16, 8, 8]), dtype=mstype.float32)
    output_grad = (Tensor(np.ones([32, 16, 8, 8]), dtype=mstype.float32),
                   Tensor(np.ones([32, 16, 8, 8]), dtype=mstype.float32))
    net_me(input_data, *output_grad)
