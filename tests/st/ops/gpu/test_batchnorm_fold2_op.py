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
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops.operations import _quant_ops as Q

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = Q.BatchNormFold2(100000)

    @jit
    def construct(self, x, beta, gamma, batch_std, batch_mean, running_std, running_mean, current_step):
        return self.op(x, beta, gamma, batch_std, batch_mean, running_std, running_mean, current_step)


class Net_gnd(nn.Cell):
    def __init__(self):
        super(Net_gnd, self).__init__()
        self.conv_mul = P.ConvMul(freeze_bn=100000)
        self.correct_add = P.CorrectionAdd(freeze_bn=100000)
        self.add_fold = P.AddFold()

    @jit
    def construct(self, x, beta, gamma, batch_std, batch_mean, running_std, running_mean, current_step):
        out = self.conv_mul(x, batch_std, running_std, current_step)
        out = self.correct_add(out, gamma, batch_std, batch_mean,
                               running_std, running_mean, current_step)
        out = self.add_fold(out, beta, gamma, batch_std, batch_mean)
        return out


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnrom_fold2():
    net = Net()
    c = 64
    freeze_bn = 100000
    x = np.random.uniform(-1, 1, size=[3, c, 32, 32]).astype('float32')
    beta = np.random.uniform(1, 2, size=[c]).astype('float32')
    gamma = np.random.uniform(1, 2, size=[c]).astype('float32')
    batch_std = np.random.uniform(1, 2, size=[c]).astype('float32')
    batch_mean = np.random.uniform(1, 2, size=[c]).astype('float32')
    running_std = np.random.uniform(1, 2, size=[c]).astype('float32')
    running_mean = np.random.uniform(1, 2, size=[c]).astype('float32')
    current_step = np.array([0]).astype('int32')
    output = net(Tensor(x), Tensor(beta), Tensor(gamma), Tensor(batch_std), Tensor(batch_mean),
                 Tensor(running_std), Tensor(running_mean), Tensor(current_step))
    expect = (x + beta.reshape(-1, 1,
                               1) - (gamma * running_mean / running_std).reshape(-1, 1,
                                                                                 1) if current_step >= freeze_bn else
              x * (running_std / batch_std).reshape(-1, 1, 1) + (beta - gamma * batch_mean / batch_std).reshape(-1, 1,
                                                                                                                1))
    error = np.ones(shape=expect.shape) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(diff > error * -1)

    current_step = np.array([100000]).astype('int32')
    output = net(Tensor(x), Tensor(beta), Tensor(gamma), Tensor(batch_std), Tensor(batch_mean), Tensor(running_std),
                 Tensor(running_mean), Tensor(current_step))
    expect = (x + beta.reshape(-1, 1,
                               1) - (gamma * running_mean / running_std).reshape(-1, 1,
                                                                                 1) if current_step >= freeze_bn else
              x * (batch_std / running_std).reshape(-1, 1, 1) + (beta - gamma * batch_mean / batch_std).reshape(-1, 1,
                                                                                                                1))
    error = np.ones(shape=expect.shape) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(diff > error * -1)
