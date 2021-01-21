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

import numpy as np
import pytest

import mindspore.context as context
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.nn import Cell
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops import operations as P


class NetFusedBatchNormEx(Cell):
    def __init__(self, num_features, gamma_init, beta_init, mean_init, var_init, use_batch_statistics=None):
        super(NetFusedBatchNormEx, self).__init__()
        self.bn = P.FusedBatchNormEx(mode=1, epsilon=0.00001, momentum=0.1)
        self.moving_mean = Parameter(initializer(
            mean_init, num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(initializer(
            var_init, num_features), name="variance", requires_grad=False)
        self.gamma = Parameter(initializer(
            gamma_init, num_features), name="gamma", requires_grad=True)
        self.beta = Parameter(initializer(
            beta_init, num_features), name="beta", requires_grad=True)
        self.dynshape = inner.GpuConvertToDynamicShape()

    def construct(self, x):
        x = self.bn(x, self.gamma, self.beta, self.moving_mean, self.moving_variance)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_fused_bn_ex():
    x = np.array([[
        [[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
        [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    expect_output = np.array([[[[-0.6059, 0.3118, 0.3118, 1.2294],
                                [-0.1471, 0.7706, 1.6882, 2.6059],
                                [0.3118, 1.6882, 2.1471, 2.1471],
                                [0.7706, 0.3118, 2.6059, -0.1471]],

                               [[0.9119, 1.8518, 1.3819, -0.0281],
                                [-0.0281, 0.9119, 1.3819, 1.8518],
                                [2.7918, 0.4419, -0.4981, 0.9119],
                                [1.8518, 0.9119, 2.3218, -0.9680]]]]).astype(np.float32)

    weight = np.ones(2).astype(np.float32)
    bias = np.ones(2).astype(np.float32)
    moving_mean = np.ones(2).astype(np.float32)
    moving_var = np.ones(2).astype(np.float32)
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-4

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = NetFusedBatchNormEx(2, Tensor(weight), Tensor(bias), Tensor(moving_mean), Tensor(moving_var))
    output_list = bn_net(Tensor(x))
    output = output_list[0]
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)


class NetFusedBatchNormExDynamic(Cell):
    def __init__(self, num_features, gamma_init, beta_init, mean_init, var_init, use_batch_statistics=None):
        super(NetFusedBatchNormExDynamic, self).__init__()
        self.bn = P.FusedBatchNormEx(mode=1, epsilon=0.00001, momentum=0.1)
        self.moving_mean = Parameter(initializer(
            mean_init, num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(initializer(
            var_init, num_features), name="variance", requires_grad=False)
        self.gamma = Parameter(initializer(
            gamma_init, num_features), name="gamma", requires_grad=True)
        self.beta = Parameter(initializer(
            beta_init, num_features), name="beta", requires_grad=True)
        self.dynshape = inner.GpuConvertToDynamicShape()

    def construct(self, x):
        x = self.dynshape(x)
        x = self.bn(x, self.gamma, self.beta, self.moving_mean, self.moving_variance)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_fused_bn_ex_dynamic():
    x = np.array([[
        [[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
        [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    expect_output = np.array([[[[-0.6059, 0.3118, 0.3118, 1.2294],
                                [-0.1471, 0.7706, 1.6882, 2.6059],
                                [0.3118, 1.6882, 2.1471, 2.1471],
                                [0.7706, 0.3118, 2.6059, -0.1471]],

                               [[0.9119, 1.8518, 1.3819, -0.0281],
                                [-0.0281, 0.9119, 1.3819, 1.8518],
                                [2.7918, 0.4419, -0.4981, 0.9119],
                                [1.8518, 0.9119, 2.3218, -0.9680]]]]).astype(np.float32)

    weight = np.ones(2).astype(np.float32)
    bias = np.ones(2).astype(np.float32)
    moving_mean = np.ones(2).astype(np.float32)
    moving_var = np.ones(2).astype(np.float32)
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-4

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = NetFusedBatchNormExDynamic(2, Tensor(weight), Tensor(bias), Tensor(moving_mean), Tensor(moving_var))
    output_list = bn_net(Tensor(x))
    output = output_list[0]
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)
