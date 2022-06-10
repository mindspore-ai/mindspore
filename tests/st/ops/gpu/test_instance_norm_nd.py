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
import mindspore
import mindspore.context as context
from mindspore import Tensor, Parameter, nn
from mindspore.ops.operations.nn_ops import InstanceNorm

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class InstanceNormNet(nn.Cell):
    def __init__(self, channel, epsilon=1e-5):
        super(InstanceNormNet, self).__init__()
        self.instance_norm = InstanceNorm(epsilon=epsilon)
        self.gamma = Parameter(Tensor(np.ones([channel]), mindspore.float32), name="gamma")
        self.beta = Parameter(Tensor(np.zeros([channel]), mindspore.float32), name="beta")
        self.mean = Parameter(Tensor(np.zeros([channel]), mindspore.float32), name="mean")
        self.variance = Parameter(Tensor(np.ones([channel]), mindspore.float32), name="variance")

    def construct(self, input_x):
        out = self.instance_norm(input_x, self.gamma, self.beta, self.mean, self.variance)
        return out[0]


def instance_norm_np(x, eps=1e-5):
    shape = x.shape
    b = shape[0]
    c = shape[1]
    x = x.reshape((b, c, -1))
    mu = np.expand_dims(np.mean(x, axis=-1), axis=-1)
    std = np.expand_dims(np.std(x, axis=-1), axis=-1)
    result = (x - mu) / (std + eps)
    return result.reshape(shape)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("shape", [(8, 4, 5)])
@pytest.mark.parametrize("data_type, err", [(np.float16, 1e-3), (np.float32, 1e-4)])
def test_instancenorm_1d(shape, data_type, err):
    """
    Feature: InstanceNorm 1D operator.
    Description: Compatible with instance_norm_np.
    Expectation: The result matches numpy implementation.
    """
    np.random.seed(0)
    input_x_np = np.random.randn(np.prod(shape)).reshape(shape).astype(data_type)
    input_x = Tensor(input_x_np)
    net = InstanceNormNet(shape[1])
    output = net(input_x)
    expected = instance_norm_np(input_x_np)
    assert np.allclose(output.asnumpy(), expected, atol=err, rtol=err)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("shape", [(8, 4, 3, 4)])
@pytest.mark.parametrize("data_type, err", [(np.float16, 1e-3), (np.float32, 1e-4)])
def test_instancenorm_2d(shape, data_type, err):
    """
    Feature: InstanceNorm 2D operator.
    Description: Compatible with instance_norm_np.
    Expectation: The result matches numpy implementation.
    """
    np.random.seed(0)
    input_x_np = np.random.randn(np.prod(shape)).reshape(shape).astype(data_type)
    input_x = Tensor(input_x_np)
    net = InstanceNormNet(shape[1])
    output = net(input_x)
    expected = instance_norm_np(input_x_np)
    assert np.allclose(output.asnumpy(), expected, atol=err, rtol=err)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("shape", [(8, 4, 3, 4, 7)])
@pytest.mark.parametrize("data_type, err", [(np.float16, 1e-3), (np.float32, 1e-4)])
def test_instancenorm_3d(shape, data_type, err):
    """
    Feature: InstanceNorm 3D operator.
    Description: Compatible with instance_norm_np.
    Expectation: The result matches numpy implementation.
    """
    np.random.seed(0)
    input_x_np = np.random.randn(np.prod(shape)).reshape(shape).astype(data_type)
    input_x = Tensor(input_x_np)
    net = InstanceNormNet(shape[1])
    output = net(input_x)
    expected = instance_norm_np(input_x_np)
    assert np.allclose(output.asnumpy(), expected, atol=err, rtol=err)
