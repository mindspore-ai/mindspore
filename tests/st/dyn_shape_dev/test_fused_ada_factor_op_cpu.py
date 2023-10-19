# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
import pytest
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype

param_shape = [2, 3, 2]


class Net(nn.Cell):
    def __init__(self, epsilon, clip_threshold, beta1, beta2, weight_decay, lr):
        super(Net, self).__init__()
        self.epsilon = epsilon
        self.clip_threshold = clip_threshold
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.lr = lr
        self.opt = ops.FusedAdaFactor()
        self.param = Parameter(Tensor(np.ones(param_shape), mstype.float32), name="param")
        self.exp_avg = Parameter(Tensor(np.zeros(param_shape), mstype.float32), name="exp_avg")
        self.exp_avg_sq = Parameter(Tensor(np.zeros(param_shape), mstype.float32), name="exp_avg_sq")
        self.exp_avg_sq_row = Parameter(Tensor(np.zeros([2, 3]), mstype.float32), name="exp_avg_sq_row")
        self.exp_avg_sq_col = Parameter(Tensor(np.zeros([2, 2]), mstype.float32), name="exp_avg_sq_col")

    def construct(self, grad):
        out = self.opt(self.epsilon, self.clip_threshold, self.beta1, self.beta2, self.weight_decay, self.lr, grad,
                       self.param, self.exp_avg,
                       self.exp_avg_sq_row, self.exp_avg_sq_col, self.exp_avg_sq)
        return out


class NetWithGlobalNorm(Net):
    def __init__(self, epsilon, clip_threshold, beta1, beta2, weight_decay, lr):
        super(NetWithGlobalNorm, self).__init__(epsilon, clip_threshold, beta1, beta2, weight_decay, lr)
        self.opt = ops.FusedAdaFactorWithGlobalNorm()

    def construct(self, grad, global_norm):
        out = self.opt(self.epsilon, self.clip_threshold, self.beta1, self.beta2, self.weight_decay, self.lr, grad,
                       self.param, self.exp_avg,
                       self.exp_avg_sq_row, self.exp_avg_sq_col, self.exp_avg_sq, global_norm)
        return out


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adafactor():
    '''
    Feature: AdaFactor
    Description: Test AdaFactor
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = Net((1e-30, 1e-3), 1.0, 0.9, 0.8, 1e-2, 0.03)
    gradient = Tensor(np.ones(param_shape), mstype.float32)
    net(gradient)
    diff = net.param.asnumpy() - np.ones(param_shape) * 0.97
    assert np.all(diff < 1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adafactor_with_global_norm():
    '''
    Feature: AdaFactor
    Description: Test AdaFactorWithGlobalNorm
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = NetWithGlobalNorm((1e-30, 1e-3), 1.0, 0.9, 0.8, 1e-2, 0.03)
    gradient = Tensor(np.ones(param_shape), mstype.float32)
    net(gradient, 10.0)
    diff = net.param.asnumpy() - np.ones(param_shape) * 0.97
    assert np.all(diff < 1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adafactor_dynamic_shape():
    '''
    Feature: AdaFactor
    Description: Test AdaFactor with dynamic shape
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = Net((1e-30, 1e-3), 1.0, 0.9, 0.8, 1e-2, 0.03)
    gradient = Tensor(np.ones(param_shape), mstype.float32)
    x_dynamic = Tensor(shape=[None for _ in param_shape], dtype=mstype.float32)
    net.set_inputs(x_dynamic)
    net(gradient)
    diff = net.param.asnumpy() - np.ones(param_shape) * 0.97
    assert np.all(diff < 1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adafactor_with_global_norm_dynamic_shape():
    '''
    Feature: AdaFactor
    Description: Test AdaFactorWithGlobalNorm with dynamic shape
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = NetWithGlobalNorm((1e-30, 1e-3), 1.0, 0.9, 0.8, 1e-2, 0.03)
    gradient = Tensor(np.ones(param_shape), mstype.float32)
    x_dynamic = Tensor(shape=[None for _ in param_shape], dtype=mstype.float32)
    net.set_inputs(x_dynamic, 10.0)
    net(gradient, 10.0)
    diff = net.param.asnumpy() - np.ones(param_shape) * 0.97
    assert np.all(diff < 1e-3)
