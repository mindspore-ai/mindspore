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
from mindspore.ops import composite as C


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    yield
    context.set_context(mode=context.GRAPH_MODE)


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


def test_row_tensor_in_while():
    class RowTensorValuesDouble(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x):
            indices = x.indices
            values = x.values * 2
            dense_shape = x.dense_shape
            return RowTensor(indices, values, dense_shape)

    class RowTensorValuesAdd2(nn.Cell):
        def __init__(self):
            super().__init__()

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
            while (a > b):
                x = self.op1(x)
                b = b + 1
            return x.indices, x.values, x.dense_shape
    a = Tensor(np.array(3).astype(np.int32))
    b = Tensor(np.array(0).astype(np.int32))
    indices = Tensor(np.array([0, 2]).astype(np.int32))
    values = Tensor(np.ones([2, 2]).astype(np.float32))
    dense_shape = (5, 2)

    net = RowTensorWithControlWhile(dense_shape)
    net(a, b, indices, values)


def test_multi_out_sens():
    class ImageGradients(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x, y, z):
            resa = x * y
            resb = y * z
            resc = x * z
            return resa, (resb, resc)

    net = ImageGradients()
    net_me = GradOfAllInputs(net, real_inputs_count=3)
    net_me.set_train()
    input_data = Tensor(np.ones([32]), dtype=mstype.float32)
    output_grad = (Tensor(np.ones([32]), dtype=mstype.float32),
                   (Tensor(np.ones([32]), dtype=mstype.float32), Tensor(np.ones([32]), dtype=mstype.float32)))
    net_me(input_data, input_data, input_data, *output_grad)
