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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetCenteredRMSProp(nn.Cell):
    def __init__(self, lr, decay, momentum, epsilon, var, g, mg, rms, mom):
        super(NetCenteredRMSProp, self).__init__()
        self.rms_opt = P.ApplyCenteredRMSProp()
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.var = var
        self.g = g
        self.mg = mg
        self.rms = rms
        self.mom = mom

    def construct(self):
        return self.rms_opt(self.var, self.mg, self.rms, self.mom, self.g, self.lr, self.decay, self.momentum,
                            self.epsilon)


class NetRMSProp(nn.Cell):
    def __init__(self, lr, decay, momentum, epsilon, var, g, mg, rms, mom):
        super(NetRMSProp, self).__init__()
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.var = var
        self.g = g
        self.mg = mg
        self.rms = rms
        self.mom = mom
        self.rms_opt = P.ApplyRMSProp()

    def construct(self):
        return self.rms_opt(self.var, self.rms, self.mom, self.lr, self.g, self.decay, self.momentum, self.epsilon)


def rmsprop_numpy(variable, gradients, mean_square, moment,
                  learning_rate, decay, momentum, epsilon):
    mean_square = mean_square * decay + (1.0 - decay) * gradients * gradients
    moment = momentum * moment + learning_rate / np.sqrt(mean_square + epsilon) * gradients
    variable = variable - moment
    return variable, gradients, mean_square, moment


def rmspropcented_numpy(variable, gradients, mean_gradients, mean_square, moment,
                        learning_rate, decay, momentum, epsilon):
    mean_gradients = mean_gradients * decay + (1.0 - decay) * gradients
    mean_square = mean_square * decay + (1.0 - decay) * gradients * gradients
    moment = momentum * moment + learning_rate / np.sqrt(
        mean_square - mean_gradients * mean_gradients + epsilon) * gradients
    variable = variable - moment
    return variable, gradients, mean_gradients, mean_square, moment


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rmsprop():
    learning_rate, decay, momentum, epsilon, centered = [0.5, 0.8, 0.9, 1e-3, True]

    variable_np = np.array([1.0, 2.0], dtype=np.float32)
    gradients_np = np.array([0.1, 0.2], dtype=np.float32)
    mean_gradients_np = np.array([0.0, 0.0], dtype=np.float32)
    mean_square_np = np.array([epsilon, epsilon], dtype=np.float32)
    moment_np = np.array([0.0, 0.0], dtype=np.float32)

    variable = Tensor(variable_np)
    gradients = Tensor(gradients_np)
    mean_gradients = Tensor(mean_gradients_np)
    mean_square = Tensor(mean_square_np)
    moment = Tensor(moment_np)

    variable_ms = Parameter(initializer(variable, variable.shape), name='var')
    gradients_ms = Parameter(initializer(gradients, gradients.shape), name='grad')
    mean_gradients_ms = Parameter(initializer(mean_gradients, mean_gradients.shape), name='mg')
    mean_square_ms = Parameter(initializer(mean_square, mean_square.shape), name='msr')
    moment_ms = Parameter(initializer(moment, moment.shape), name='mom')

    if centered:
        variable_np, gradients_np, mean_gradients_np, mean_square_np, moment_np = \
            rmspropcented_numpy(variable_np, gradients_np, mean_gradients_np, mean_square_np, moment_np,
                                learning_rate, decay, momentum, epsilon)
        net = NetCenteredRMSProp(learning_rate, decay, momentum, epsilon, variable_ms, gradients_ms, mean_gradients_ms,
                                 mean_square_ms, moment_ms)
        _ = net()

    else:
        variable_np, gradients_np, mean_square_np, moment_np = \
            rmsprop_numpy(variable_np, gradients_np, mean_square_np, moment_np,
                          learning_rate, decay, momentum, epsilon)
        net = NetRMSProp(learning_rate, decay, momentum, epsilon, variable_ms, gradients_ms, mean_gradients_ms,
                         mean_square_ms, moment_ms)
        _ = net()

    error = np.ones(shape=variable_np.shape) * 10e-6
    diff = variable_ms.asnumpy() - variable_np
    assert np.all(diff < error)

    error = np.ones(shape=gradients_np.shape) * 10e-6
    diff = gradients_ms.asnumpy() - gradients_np
    assert np.all(diff < error)

    error = np.ones(shape=mean_gradients_np.shape) * 10e-6
    diff = mean_gradients_ms.asnumpy() - mean_gradients_np
    assert np.all(diff < error)

    error = np.ones(shape=mean_square_np.shape) * 10e-6
    diff = mean_square_ms.asnumpy() - mean_square_np
    assert np.all(diff < error)

    error = np.ones(shape=moment_np.shape) * 10e-6
    diff = moment_ms.asnumpy() - moment_np
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rmspropcenter():
    learning_rate, decay, momentum, epsilon, centered = [0.1, 0.3, 0.9, 1.0, False]

    variable_np = np.array([1.0, 2.0], dtype=np.float32)
    gradients_np = np.array([0.1, 0.2], dtype=np.float32)
    mean_gradients_np = np.array([0.0, 0.0], dtype=np.float32)
    mean_square_np = np.array([epsilon, epsilon], dtype=np.float32)
    moment_np = np.array([0.0, 0.0], dtype=np.float32)

    variable = Tensor(variable_np)
    gradients = Tensor(gradients_np)
    mean_gradients = Tensor(mean_gradients_np)
    mean_square = Tensor(mean_square_np)
    moment = Tensor(moment_np)

    variable_ms = Parameter(initializer(variable, variable.shape), name='var')
    gradients_ms = Parameter(initializer(gradients, gradients.shape), name='grad')
    mean_gradients_ms = Parameter(initializer(mean_gradients, mean_gradients.shape), name='mg')
    mean_square_ms = Parameter(initializer(mean_square, mean_square.shape), name='msr')
    moment_ms = Parameter(initializer(moment, moment.shape), name='mom')

    if centered:
        variable_np, gradients_np, mean_gradients_np, mean_square_np, moment_np = \
            rmspropcented_numpy(variable_np, gradients_np, mean_gradients_np, mean_square_np, moment_np,
                                learning_rate, decay, momentum, epsilon)
        net = NetCenteredRMSProp(learning_rate, decay, momentum, epsilon, variable_ms, gradients_ms, mean_gradients_ms,
                                 mean_square_ms, moment_ms)
        _ = net()
    else:
        variable_np, gradients_np, mean_square_np, moment_np = \
            rmsprop_numpy(variable_np, gradients_np, mean_square_np, moment_np,
                          learning_rate, decay, momentum, epsilon)
        net = NetRMSProp(learning_rate, decay, momentum, epsilon, variable_ms, gradients_ms, mean_gradients_ms,
                         mean_square_ms, moment_ms)
        _ = net()

    error = np.ones(shape=variable_np.shape) * 10e-6
    diff = variable_ms.asnumpy() - variable_np
    assert np.all(diff < error)

    error = np.ones(shape=gradients_np.shape) * 10e-6
    diff = gradients_ms.asnumpy() - gradients_np
    assert np.all(diff < error)

    error = np.ones(shape=mean_gradients_np.shape) * 10e-6
    diff = mean_gradients_ms.asnumpy() - mean_gradients_np
    assert np.all(diff < error)

    error = np.ones(shape=mean_square_np.shape) * 10e-6
    diff = mean_square_ms.asnumpy() - mean_square_np
    assert np.all(diff < error)

    error = np.ones(shape=moment_np.shape) * 10e-6
    diff = moment_ms.asnumpy() - moment_np
    assert np.all(diff < error)
