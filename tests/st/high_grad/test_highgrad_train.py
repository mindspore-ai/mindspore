# Copyright 2021 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor, Parameter
from mindspore.nn import TrainOneStepCell
from mindspore.nn.optim import Momentum
from mindspore.ops.composite import GradOperation
from mindspore.common import ParameterTuple

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
        if self.real_inputs_count is None or self.sens_param is False:
            if self.wrt_params:
                return self.grad(self.network, self.params)(*inputs)
            return self.grad(self.network)(*inputs)

        real_inputs = inputs[:self.real_inputs_count]
        sense_param_inputs = inputs[self.real_inputs_count:]
        if self.wrt_params:
            return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)
        return self.grad(self.network)(*real_inputs, sense_param_inputs)


class GradOfFirstInput(_Grad):
    """
    get grad of first input
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()
        self.add = ops.TensorAdd()
        weight_np = np.array([2]).astype(np.float32)
        bias_np = np.array([1]).astype(np.float32)
        self.weight = Parameter(Tensor(weight_np),
                                name='weight', requires_grad=True)
        self.bias = Parameter(Tensor(bias_np),
                              name="bias", requires_grad=True)

    def construct(self, x):
        xw = self.mul(x, self.weight)
        output = self.add(xw, self.bias)
        return output


class WithLossCellLocal(nn.Cell):
    def __init__(self, grad, loss):
        super(WithLossCellLocal, self).__init__(auto_prefix=False)
        self.grad = grad
        self.loss = loss

    def construct(self, data, label):
        out = self.grad(data)
        return self.loss(out, label)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_high_grad_train():
    x_pure = np.random.randint(-10, 100, 32)
    x_train = x_pure.astype(np.float32)
    y_noise = 3 * x_pure + 2 + np.random.randn(32) / 10
    y_train = y_noise.astype(np.float32)
    net = Net()
    grad_net = GradOfFirstInput(net, sens_param=False)
    epoch = 2
    momentum = 0.0
    learning_rate = 0.001
    optimizer = Momentum(filter(lambda x: x.requires_grad,
                                grad_net.get_parameters()), learning_rate, momentum)
    criterion = nn.loss.MSELoss()
    net_with_criterion = WithLossCellLocal(grad_net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    for i in range(epoch):
        train_network(Tensor([x_train[i]]), Tensor([y_train[i]]))
