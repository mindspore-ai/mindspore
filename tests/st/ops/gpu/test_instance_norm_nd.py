# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, nn, context, ms_function
from mindspore import dtype as mstype
from mindspore.ops.composite import GradOperation
from mindspore.ops import functional as F


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @ms_function
    def construct(self, input_x, grad):
        return self.grad(self.network)(input_x, grad)


class Expected1d(nn.Cell):
    def __init__(self, n, gamma_init=0.5, beta_init=0.5):
        super(Expected1d, self).__init__()
        self.ops = nn.BatchNorm2d(n, use_batch_statistics=True, gamma_init=gamma_init, beta_init=beta_init)

    def construct(self, x):
        shape = F.shape(x)
        return F.reshape(self.ops(F.reshape(x, (1, -1, 1, shape[2]))), shape)


class Expected2d(nn.Cell):
    def __init__(self, n, gamma_init=0.5, beta_init=0.5):
        super(Expected2d, self).__init__()
        self.ops = nn.BatchNorm2d(n, use_batch_statistics=True, gamma_init=gamma_init, beta_init=beta_init)

    def construct(self, x):
        shape = F.shape(x)
        return F.reshape(self.ops(F.reshape(x, (1, -1, shape[2], shape[3]))), shape)


class Expected3d(nn.Cell):
    def __init__(self, n, gamma_init=0.5, beta_init=0.5):
        super(Expected3d, self).__init__()
        self.ops = nn.BatchNorm3d(n, use_batch_statistics=True, gamma_init=gamma_init, beta_init=beta_init)

    def construct(self, x):
        shape = F.shape(x)
        return F.reshape(self.ops(F.reshape(x, (1, -1, shape[2], shape[3], shape[4]))), shape)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("shape", [(8, 4, 5)])
@pytest.mark.parametrize("data_type", [np.float16, np.float32])
def test_instancenorm_1d(shape, data_type):
    """
    Feature: InstanceNorm 1D operator.
    Description: Compatible with instance_norm_np.
    Expectation: The result matches numpy implementation.
    """
    np.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_np = Tensor(np.random.randn(*shape).astype(data_type))
    grad = Tensor(np.random.randn(*shape).astype(data_type))

    instance_op = nn.InstanceNorm1d(shape[1], gamma_init=0.5, beta_init=0.5)
    expected_net = Expected1d(shape[0] * shape[1], gamma_init=0.5, beta_init=0.5)

    result = instance_op(Tensor(x_np))
    expected = expected_net(Tensor(x_np))
    assert np.allclose(result.asnumpy(), expected.asnumpy())

    instance_backward_net = Grad(instance_op)
    expected_backward_net = Grad(expected_net)

    result = instance_backward_net(Tensor(x_np), Tensor(grad))
    expected = expected_backward_net(Tensor(x_np), Tensor(grad))
    assert np.allclose(result[0].asnumpy(), expected[0].asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("shape", [(8, 4, 3, 4)])
@pytest.mark.parametrize("data_type", [np.float16, np.float32])
def test_instancenorm_2d(shape, data_type):
    """
    Feature: InstanceNorm 2D operator.
    Description: Compatible with instance_norm_np.
    Expectation: The result matches numpy implementation.
    """
    np.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_np = Tensor(np.random.randn(*shape).astype(data_type))
    grad = Tensor(np.random.randn(*shape).astype(data_type))

    instance_op = nn.InstanceNorm2d(shape[1], gamma_init=0.5, beta_init=0.5)
    expected_net = Expected2d(shape[0] * shape[1], gamma_init=0.5, beta_init=0.5)

    result = instance_op(Tensor(x_np))
    expected = expected_net(Tensor(x_np))
    assert np.allclose(result.asnumpy(), expected.asnumpy())

    instance_backward_net = Grad(instance_op)
    expected_backward_net = Grad(expected_net)

    result = instance_backward_net(Tensor(x_np), Tensor(grad))
    expected = expected_backward_net(Tensor(x_np), Tensor(grad))
    assert np.allclose(result[0].asnumpy(), expected[0].asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("shape", [(8, 4, 3, 4, 7)])
@pytest.mark.parametrize("data_type", [np.float16, np.float32])
def test_instancenorm_3d(shape, data_type):
    """
    Feature: InstanceNorm 3D operator.
    Description: Compatible with instance_norm_np.
    Expectation: The result matches numpy implementation.
    """
    np.random.seed(0)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    x_np = Tensor(np.random.randn(*shape).astype(data_type))
    grad = Tensor(np.random.randn(*shape).astype(data_type))

    instance_op = nn.InstanceNorm3d(shape[1], gamma_init=0.5, beta_init=0.5)
    expected_net = Expected3d(shape[0] * shape[1], gamma_init=0.5, beta_init=0.5)

    result = instance_op(Tensor(x_np))
    expected = expected_net(Tensor(x_np))
    assert np.allclose(result.asnumpy(), expected.asnumpy())

    instance_backward_net = Grad(instance_op)
    expected_backward_net = Grad(expected_net)

    result = instance_backward_net(Tensor(x_np), Tensor(grad))
    expected = expected_backward_net(Tensor(x_np), Tensor(grad))
    assert np.allclose(result[0].asnumpy(), expected[0].asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_instancenorm_2d_dynamic_shape():
    """
    Feature: InstanceNorm 2D operator with dynamic shape.
    Description: Compatible with instance_norm_np.
    Expectation: The result matches numpy implementation.
    """
    shape = (8, 4, 3, 4)
    data_type = np.float32

    np.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_np = Tensor(np.random.randn(*shape).astype(data_type))
    grad = Tensor(np.random.randn(*shape).astype(data_type))

    dynamic_instance_op = nn.InstanceNorm2d(shape[1], gamma_init=0.5, beta_init=0.5)
    expected_net = Expected2d(shape[0] * shape[1], gamma_init=0.5, beta_init=0.5)

    place_holder = Tensor(shape=[8, 4, None, 4], dtype=mstype.float32)
    dynamic_instance_op.set_inputs(place_holder)

    result = dynamic_instance_op(Tensor(x_np))
    expected = expected_net(Tensor(x_np))
    assert np.allclose(result.asnumpy(), expected.asnumpy())

    instance_backward_net = Grad(dynamic_instance_op)
    expected_backward_net = Grad(expected_net)

    result = instance_backward_net(Tensor(x_np), Tensor(grad))
    expected = expected_backward_net(Tensor(x_np), Tensor(grad))
    assert np.allclose(result[0].asnumpy(), expected[0].asnumpy())
