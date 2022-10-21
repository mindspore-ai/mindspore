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
from mindspore.ops.operations import _quant_ops as Q

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = Q.BatchNormFoldGrad(freeze_bn=10)

    @jit
    def construct(self, d_batch_mean, d_batch_std, x, batch_mean, batch_std, current_step):
        dx = self.op(d_batch_mean, d_batch_std, x, batch_mean, batch_std, current_step)
        return dx


def np_result(d_batch_mean, d_batch_std, x, batch_mean, batch_std):
    n = x.shape[0] * x.shape[2] * x.shape[3]
    dx = d_batch_mean.reshape(1, -1, 1, 1) / n + d_batch_std.reshape(1, -1, 1, 1) * (
        x - batch_mean.reshape(1, -1, 1, 1)) / batch_std.reshape(1, -1, 1, 1) / n
    return dx


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_fold_grad1():
    net = Net()
    c = 64
    x = np.random.uniform(1, 10, size=[3, c, 32, 32]).astype('float32')
    d_batch_mean = np.random.uniform(1, 10, size=[c]).astype('float32')
    d_batch_std = np.random.uniform(1, 10, size=[c]).astype('float32')
    batch_mean = np.random.uniform(1, 10, size=[c]).astype('float32')
    batch_std = np.random.uniform(1, 10, size=[c]).astype('float32')
    current_step = np.array([0]).astype('int32')
    dx = net(Tensor(d_batch_mean), Tensor(d_batch_std), Tensor(x), Tensor(batch_mean), Tensor(batch_std),
             Tensor(current_step))
    expect = np_result(d_batch_mean, d_batch_std, x, batch_mean, batch_std)
    assert np.allclose(dx.asnumpy(), expect, rtol=1.e-7, atol=1.e-7)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_fold_grad2():
    net = Net()
    c = 64
    x = np.random.uniform(1, 10, size=[1, c, 256, 256]).astype('float32')
    d_batch_mean = np.random.uniform(1, 10, size=[c]).astype('float32')
    d_batch_std = np.random.uniform(1, 10, size=[c]).astype('float32')
    batch_mean = np.random.uniform(1, 10, size=[c]).astype('float32')
    batch_std = np.random.uniform(1, 10, size=[c]).astype('float32')
    current_step = np.array([0]).astype('int32')
    dx = net(Tensor(d_batch_mean), Tensor(d_batch_std), Tensor(x), Tensor(batch_mean), Tensor(batch_std),
             Tensor(current_step))
    expect = np_result(d_batch_mean, d_batch_std, x, batch_mean, batch_std)
    assert np.allclose(dx.asnumpy(), expect, rtol=1.e-7, atol=1.e-7)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_fold_grad_freeze():
    net = Net()
    c = 64
    x = np.random.uniform(1, 10, size=[3, c, 32, 32]).astype('float32')
    d_batch_mean = np.random.uniform(1, 10, size=[c]).astype('float32')
    d_batch_std = np.random.uniform(1, 10, size=[c]).astype('float32')
    batch_mean = np.random.uniform(1, 10, size=[c]).astype('float32')
    batch_std = np.random.uniform(1, 10, size=[c]).astype('float32')
    current_step = np.array([10]).astype('int32')
    dx = net(Tensor(d_batch_mean), Tensor(d_batch_std), Tensor(x), Tensor(batch_mean), Tensor(batch_std),
             Tensor(current_step))
    expect = np.zeros_like(x)
    assert np.allclose(dx.asnumpy(), expect, rtol=1.e-7, atol=1.e-7)
