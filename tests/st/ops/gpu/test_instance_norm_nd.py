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
from tests.mark_utils import arg_mark

import pytest
import numpy as np
from mindspore import Tensor, Parameter, nn, context, jit, ops
from mindspore import dtype as mstype
from mindspore.ops.composite import GradOperation
from mindspore.ops import functional as F
from mindspore.ops.operations.nn_ops import InstanceNorm
from mindspore.ops.operations._grad_ops import InstanceNormGrad


class VmapInstanceNet(nn.Cell):
    def __init__(self):
        super(VmapInstanceNet, self).__init__()
        self.instance = InstanceNorm()

    def construct(self, x, gamma, beta, mean, variance):
        return self.instance(x, gamma, beta, mean, variance)


class VmapInstanceGradNet(nn.Cell):
    def __init__(self):
        super(VmapInstanceGradNet, self).__init__()
        self.instance = InstanceNormGrad()

    def construct(self, dy, x, gamma, mean, variance):
        return self.instance(dy, x, gamma, mean, variance)


class VMapNet(nn.Cell):
    def __init__(self, net, gamma, beta, mean, variance, in_axes, out_axes):
        super(VMapNet, self).__init__()
        self.gamma = Parameter(gamma, name="gamma")
        self.beta = Parameter(beta, name="beta")
        self.mean = Parameter(mean, name="mean")
        self.variance = Parameter(variance, name="variance")
        self.net = net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, x):
        return ops.vmap(self.net, self.in_axes, self.out_axes)(x, self.gamma, self.beta, self.mean, self.variance)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @jit
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_instancenorm_vmap():
    """
    Feature: test InstanceNorm operator with vmap.
    Description: Compatible with instance_norm_np.
    Expectation: The result matches numpy implementation.
    """
    vmap_batch = 8
    batch = 4
    channel = 3
    shape = (vmap_batch, batch, channel, 4, 5)
    parameter_shape = (vmap_batch, channel)
    data_type = np.float32

    np.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.random.randn(*shape).astype(data_type)
    gamma_np = np.random.randn(*parameter_shape).astype(data_type)
    beta_np = np.random.randn(*parameter_shape).astype(data_type)
    moving_mean_np = np.random.randn(*parameter_shape).astype(data_type)
    moving_variance_np = np.random.randn(*parameter_shape).astype(data_type)

    # vmap
    in_axes = (0, 0, 0, 0, 0)
    out_axes = (0, 0, 0)
    vmap_output_x, vmap_updated_moving_mean, vmap_updated_moving_variance = \
        VMapNet(VmapInstanceNet(), Tensor(gamma_np), Tensor(beta_np), Tensor(moving_mean_np),
                Tensor(moving_variance_np), in_axes, out_axes)(Tensor(x_np))

    # for loop
    instance_norm = InstanceNorm()
    output_x_list = []
    updated_moving_mean_list = []
    updated_moving_variance_list = []
    for i in range(vmap_batch):
        output_x, updated_moving_mean, updated_moving_variance = instance_norm(
            Tensor(x_np[i, ...]),
            Parameter(Tensor(gamma_np[i, ...].copy())),
            Parameter(Tensor(beta_np[i, ...].copy())),
            Parameter(Tensor(moving_mean_np[i, ...].copy())),
            Parameter(Tensor(moving_variance_np[i, ...].copy())),
        )
        output_x_list.append(output_x)
        updated_moving_mean_list.append(updated_moving_mean)
        updated_moving_variance_list.append(updated_moving_variance)

    for_output_x = ops.Stack()(output_x_list)
    for_updated_moving_mean = ops.Stack()(updated_moving_mean_list)
    for_updated_moving_variance = ops.Stack()(updated_moving_variance_list)

    assert np.allclose(vmap_output_x.asnumpy(), for_output_x.asnumpy())
    assert np.allclose(vmap_updated_moving_mean.asnumpy(), for_updated_moving_mean.asnumpy())
    assert np.allclose(vmap_updated_moving_variance.asnumpy(), for_updated_moving_variance.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_instancenorm_grad_vmap():
    """
    Feature: test InstanceNormGrad operator with vmap.
    Description: Compatible with instance_norm_np.
    Expectation: The result matches numpy implementation.
    """
    vmap_batch = 8
    batch = 4
    channel = 3
    shape = (vmap_batch, batch, channel, 4, 5)
    parameter_shape = (vmap_batch, channel)
    save_parameter_shape = (vmap_batch, batch * channel)
    data_type = np.float32

    np.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    dy_np = np.random.randn(*shape).astype(data_type)
    x_np = np.random.randn(*shape).astype(data_type)
    gamma_np = np.random.randn(*parameter_shape).astype(data_type)
    moving_mean_np = np.random.randn(*save_parameter_shape).astype(data_type)
    moving_variance_np = np.random.randn(*save_parameter_shape).astype(data_type)

    # vmap
    in_axes = (0, 0, 0, 0, 0)
    out_axes = (0, 0, 0)
    vmap_output_x, vmap_updated_moving_mean, vmap_updated_moving_variance = \
        ops.vmap(VmapInstanceGradNet(), in_axes, out_axes)(Tensor(dy_np), Tensor(x_np), Tensor(gamma_np),
                                                           Tensor(moving_mean_np), Tensor(moving_variance_np))

    # for loop
    instance_norm_grad = InstanceNormGrad()
    output_x_list = []
    updated_moving_mean_list = []
    updated_moving_variance_list = []
    for i in range(vmap_batch):
        output_x, updated_moving_mean, updated_moving_variance = instance_norm_grad(
            Tensor(dy_np[i, ...]),
            Tensor(x_np[i, ...]),
            Tensor(gamma_np[i, ...].copy()),
            Tensor(moving_mean_np[i, ...].copy()),
            Tensor(moving_variance_np[i, ...].copy()),
        )
        output_x_list.append(output_x)
        updated_moving_mean_list.append(updated_moving_mean)
        updated_moving_variance_list.append(updated_moving_variance)

    for_output_x = ops.Stack()(output_x_list)
    for_updated_moving_mean = ops.Stack()(updated_moving_mean_list)
    for_updated_moving_variance = ops.Stack()(updated_moving_variance_list)

    assert np.allclose(vmap_output_x.asnumpy(), for_output_x.asnumpy())
    assert np.allclose(vmap_updated_moving_mean.asnumpy(), for_updated_moving_mean.asnumpy())
    assert np.allclose(vmap_updated_moving_variance.asnumpy(), for_updated_moving_variance.asnumpy())
