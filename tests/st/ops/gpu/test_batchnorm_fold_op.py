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
from mindspore.common.api import jit
from mindspore.ops.operations import _quant_ops as Q

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class Net(nn.Cell):
    def __init__(self, mean, variance):
        super(Net, self).__init__()
        self.mean = mean
        self.variance = variance
        self.op = Q.BatchNormFold(momentum=0.9, freeze_bn=10)

    @jit
    def construct(self, x, current_step):
        a, b, c, d = self.op(x, self.mean, self.variance, current_step)
        return a, b, c, d


def np_result(x, mean, var, momentum, epsilon):
    np_mean = x.mean(axis=(0, 2, 3))
    np_var = x.var(axis=(0, 2, 3))
    n = x.shape[0] * x.shape[2] * x.shape[3]
    mean_update = (1 - momentum) * np_mean + momentum * mean
    var_update = (1 - momentum) * np_var * n / (n - 1) + momentum * var
    np_var = np.sqrt(np_var + epsilon)
    delay_mean = mean.copy()
    delay_std = np.sqrt(var + epsilon)
    return np_mean, np_var, mean_update, var_update, delay_mean, delay_std


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_batchnorm_fold():
    c = 64
    x = np.random.uniform(1, 10, size=[3, c, 32, 32]).astype('float32')
    mean = np.random.uniform(1, 10, size=[c]).astype('float32')
    variance = np.random.uniform(1, 10, size=[c]).astype('float32')
    current_step = np.array([0]).astype('int32')
    ms_mean_t = Tensor(mean)
    ms_var_t = Tensor(variance)
    ms_mean = Parameter(initializer(ms_mean_t, ms_mean_t.shape), name='mean')
    ms_var = Parameter(initializer(ms_var_t, ms_var_t.shape), name='var')
    net = Net(ms_mean, ms_var)
    batch_mean, batch_var, delay_mean, delay_std = net(Tensor(x), Tensor(current_step))

    expect1, expect2, expect3, expect4, expect5, expect6 = np_result(x, mean, variance, 0.9, 1e-12)
    assert np.allclose(batch_mean.asnumpy(), expect1, rtol=1.e-7, atol=1.e-5)
    assert np.allclose(batch_var.asnumpy(), expect2, rtol=1.e-7, atol=1.e-5)
    assert np.allclose(ms_mean.asnumpy(), expect3, rtol=1.e-7, atol=1.e-5)
    assert np.allclose(ms_var.asnumpy(), expect4, rtol=1.e-7, atol=1.e-5)
    assert np.allclose(delay_mean.asnumpy(), expect5, rtol=1.e-7, atol=1.e-5)
    assert np.allclose(delay_std.asnumpy(), expect6, rtol=1.e-7, atol=1.e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_batchnorm_fold2():
    c = 64
    x = np.random.uniform(1, 10, size=[3, c, 512, 512]).astype('float32')
    mean = np.random.uniform(1, 10, size=[c]).astype('float32')
    variance = np.random.uniform(1, 10, size=[c]).astype('float32')
    current_step = np.array([0]).astype('int32')
    ms_mean_t = Tensor(mean)
    ms_var_t = Tensor(variance)
    ms_mean = Parameter(initializer(ms_mean_t, ms_mean_t.shape), name='mean')
    ms_var = Parameter(initializer(ms_var_t, ms_var_t.shape), name='var')
    net = Net(ms_mean, ms_var)
    batch_mean, batch_var, delay_mean, delay_std = net(Tensor(x), Tensor(current_step))
    expect1, expect2, expect3, _, expect5, expect6 = np_result(x, mean, variance, 0.9, 1e-12)
    assert np.allclose(batch_mean.asnumpy(), expect1, rtol=1.e-7, atol=1.e-5)
    assert np.allclose(batch_var.asnumpy(), expect2, rtol=1.e-7, atol=1.e-5)
    assert np.allclose(ms_mean.asnumpy(), expect3, rtol=1.e-7, atol=1.e-5)
    assert np.allclose(delay_mean.asnumpy(), expect5, rtol=1.e-7, atol=1.e-5)
    assert np.allclose(delay_std.asnumpy(), expect6, rtol=1.e-7, atol=1.e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_batchnorm_fold_freeze():
    c = 64
    x = np.random.uniform(1, 10, size=[3, c, 32, 32]).astype('float32')
    mean = np.random.uniform(1, 10, size=[c]).astype('float32')
    variance = np.random.uniform(1, 10, size=[c]).astype('float32')
    current_step = np.array([10]).astype('int32')
    ms_mean_t = Tensor(mean)
    ms_var_t = Tensor(variance)
    ms_mean = Parameter(initializer(ms_mean_t, ms_mean_t.shape), name='mean')
    ms_var = Parameter(initializer(ms_var_t, ms_var_t.shape), name='var')
    net = Net(ms_mean, ms_var)
    batch_mean, batch_var, delay_mean, delay_std = net(Tensor(x), Tensor(current_step))
    _, _, _, _, expect5, expect6 = np_result(x, mean, variance, 0.9, 1e-12)
    assert np.allclose(batch_mean.asnumpy(), np.zeros_like(mean), rtol=1.e-7, atol=1.e-5)
    assert np.allclose(batch_var.asnumpy(), np.ones_like(mean), rtol=1.e-7, atol=1.e-5)
    assert np.allclose(ms_mean.asnumpy(), mean, rtol=1.e-7, atol=1.e-5)
    assert np.allclose(ms_var.asnumpy(), variance, rtol=1.e-7, atol=1.e-5)
    assert np.allclose(delay_mean.asnumpy(), expect5, rtol=1.e-7, atol=1.e-5)
    assert np.allclose(delay_std.asnumpy(), expect6, rtol=1.e-7, atol=1.e-5)
