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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetRMSProp(nn.Cell):
    def __init__(self, use_centered):
        super(NetRMSProp, self).__init__()
        self.use_centered = use_centered
        if use_centered:
            self.rms_opt = P.ApplyCenteredRMSProp()
        else:
            self.rms_opt = P.ApplyRMSProp()

    def construct(self, var, g, mg, rms, mom, lr, decay, momentum, epsilon):
        if self.use_centered:
            return self.rms_opt(var, mg, rms, mom, g, lr, decay, momentum, epsilon)
        else:
            return self.rms_opt(var, rms, mom, g, lr, decay, momentum, epsilon)


def rmsprop_numpy(variable, gradients, mean_square, moment,
                  learning_rate, decay, momentum, epsilon):
    mean_square = mean_square * decay + (1.0 - decay) * gradients * gradients
    moment = momentum * moment + learning_rate / np.sqrt(mean_square + epsilon) * gradients
    variable = variable - moment


def rmspropcented_numpy(variable, gradients, mean_gradients, mean_square, moment,
                        learning_rate, decay, momentum, epsilon):
    mean_gradients = mean_gradients * decay + (1.0 - decay) * gradients
    mean_square = mean_square * decay + (1.0 - decay) * gradients * gradients
    moment = momentum * moment + learning_rate / np.sqrt(
        mean_square - mean_gradients * mean_gradients + epsilon) * gradients
    variable = variable - moment


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_rmsprop():
    learning_rate, decay, momentum, epsilon, centered = [0.5, 0.8, 0.9, 1e-3, True]

    variable_np = np.array([1.0, 2.0], dtype=np.float32)
    gradients_np = np.array([0.1, 0.2], dtype=np.float32)
    mean_gradients_np = np.array([0.0, 0.0], dtype=np.float32)
    mean_square_np = np.array([epsilon, epsilon], dtype=np.float32)
    moment_np = np.array([0.0, 0.0], dtype=np.float32)

    variable_ms = Tensor(variable_np)
    gradients_ms = Tensor(gradients_np)
    mean_gradients_ms = Tensor(mean_gradients_np)
    mean_square_ms = Tensor(mean_square_np)
    moment_ms = Tensor(moment_np)

    if centered:
        rmspropcented_numpy(variable_np, gradients_np, mean_gradients_np, mean_square_np, moment_np,
                            learning_rate, decay, momentum, epsilon)
    else:
        rmsprop_numpy(variable_np, gradients_np, mean_square_np, moment_np,
                      learning_rate, decay, momentum, epsilon)

    net = NetRMSProp(centered)
    _ = net(variable_ms, gradients_ms, mean_gradients_ms, mean_square_ms,
            moment_ms, learning_rate, decay, momentum, epsilon)

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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_rmspropcenter():
    learning_rate, decay, momentum, epsilon, centered = [0.1, 0.3, 0.9, 1.0, False]

    variable_np = np.array([1.0, 2.0], dtype=np.float32)
    gradients_np = np.array([0.1, 0.2], dtype=np.float32)
    mean_gradients_np = np.array([0.0, 0.0], dtype=np.float32)
    mean_square_np = np.array([epsilon, epsilon], dtype=np.float32)
    moment_np = np.array([0.0, 0.0], dtype=np.float32)

    variable_ms = Tensor(variable_np)
    gradients_ms = Tensor(gradients_np)
    mean_gradients_ms = Tensor(mean_gradients_np)
    mean_square_ms = Tensor(mean_square_np)
    moment_ms = Tensor(moment_np)

    if centered:
        rmspropcented_numpy(variable_np, gradients_np, mean_gradients_np, mean_square_np, moment_np,
                            learning_rate, decay, momentum, epsilon)
    else:
        rmsprop_numpy(variable_np, gradients_np, mean_square_np, moment_np,
                      learning_rate, decay, momentum, epsilon)

    net = NetRMSProp(centered)
    _ = net(variable_ms, gradients_ms, mean_gradients_ms, mean_square_ms, moment_ms,
            learning_rate, decay, momentum, epsilon)

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
