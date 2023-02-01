# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)


class AdamNet(nn.Cell):
    def __init__(self, var, m, v):
        super(AdamNet, self).__init__()
        self.apply_adam = P.Adam()
        self.var = Parameter(var, name="var")
        self.m = Parameter(m, name="m")
        self.v = Parameter(v, name="v")

    def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
        self.apply_adam(self.var, self.m, self.v, beta1_power,
                        beta2_power, lr, beta1, beta2, epsilon, grad)
        return self.var, self.m, self.v


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_adam():
    """
    Feature: Auto monad feature.
    Description: Verify Adam operator.
    Expectation: No exception.
    """
    var = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    m = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    v = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    net = AdamNet(var, m, v)

    beta1_power = Tensor(0.9, mstype.float32)
    beta2_power = Tensor(0.999, mstype.float32)
    lr = Tensor(0.001, mstype.float32)
    beta1 = Tensor(0.9, mstype.float32)
    beta2 = Tensor(0.999, mstype.float32)
    epsilon = Tensor(1e-8, mstype.float32)
    grad = Tensor(np.random.rand(3, 3, 3).astype(np.float32))
    new_var, new_m, new_v = net(
        beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
    assert ((new_var != var).any() and (new_m != m).any() and (new_v != v).any()), \
        "The results should be different!"


class ApplyAdaMaxNet(nn.Cell):
    def __init__(self, val, m, v):
        super(ApplyAdaMaxNet, self).__init__()
        self.apply_ada_max = P.ApplyAdaMax()
        self.var = Parameter(val, name="var")
        self.m = Parameter(m, name="m")
        self.v = Parameter(v, name="v")

    def construct(self, beta1_power, lr, beta1, beta2, epsilon, grad):
        self.apply_ada_max(self.var, self.m, self.v,
                           beta1_power, lr, beta1, beta2, epsilon, grad)
        return self.var, self.m, self.v


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_ada_max():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyAdaMax operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    m = Tensor(np.random.rand(3, 3).astype(np.float32))
    v = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = ApplyAdaMaxNet(var, m, v)

    beta1_power = Tensor(0.9, mstype.float32)
    lr = Tensor(0.001, mstype.float32)
    beta1 = Tensor(0.9, mstype.float32)
    beta2 = Tensor(0.99, mstype.float32)
    epsilon = Tensor(1e-10, mstype.float32)
    grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    new_var, new_m, new_v = net(beta1_power, lr, beta1, beta2, epsilon, grad)
    assert ((new_var != var).any() and (new_m != m).any() and (new_v != v).any()), \
        "The results should be different!"


class ApplyAdadeltaNet(nn.Cell):
    def __init__(self, var, accum, accum_update):
        super(ApplyAdadeltaNet, self).__init__()
        self.apply_adadelta = P.ApplyAdadelta()
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")
        self.accum_update = Parameter(accum_update, name="accum_update")

    def construct(self, lr, rho, epsilon, grad):
        self.apply_adadelta(self.var, self.accum,
                            self.accum_update, lr, rho, epsilon, grad)
        return self.var, self.accum, self.accum_update


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_adadelta():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyAdadelta operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    accum = Tensor(np.random.rand(3, 3).astype(np.float32))
    accum_update = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = ApplyAdadeltaNet(var, accum, accum_update)

    lr = Tensor(0.001, mstype.float32)
    rho = Tensor(0.0, mstype.float32)
    epsilon = Tensor(1e-6, mstype.float32)
    grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    new_var, new_accum, new_accum_update = net(lr, rho, epsilon, grad)
    assert ((new_var != var).any() and (new_accum != accum).any() and (new_accum_update != accum_update).any()), \
        "The results should be different!"


class ApplyAdagrad(nn.Cell):
    def __init__(self, var, accum):
        super(ApplyAdagrad, self).__init__()
        self.apply_adagrad = P.ApplyAdagrad()
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")

    def construct(self, lr, grad):
        self.apply_adagrad(self.var, self.accum, lr, grad)
        return self.var, self.accum


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_adagrad():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyAdagrad operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    accum = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = ApplyAdagrad(var, accum)

    lr = Tensor(0.001, mstype.float32)
    grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    new_var, new_accum = net(lr, grad)
    assert ((new_var != var).any() and (new_accum != accum).any()), \
        "The results should be different!"


class ApplyAdagradV2Net(nn.Cell):
    def __init__(self, var, accum):
        super(ApplyAdagradV2Net, self).__init__()
        self.apply_adagrad_v2 = P.ApplyAdagradV2(epsilon=1e-6)
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")

    def construct(self, lr, grad):
        self.apply_adagrad_v2(self.var, self.accum, lr, grad)
        return self.var, self.accum


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_adagrad_v2():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyAdagradV2 operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    accum = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = ApplyAdagradV2Net(var, accum)

    lr = Tensor(0.001, mstype.float32)
    grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    new_var, new_accum = net(lr, grad)
    assert ((new_var != var).any() and (new_accum != accum).any()), \
        "The results should be different!"


class ApplyAddSignNet(nn.Cell):
    def __init__(self, var, m):
        super(ApplyAddSignNet, self).__init__()
        self.apply_add_sign = P.ApplyAddSign()
        self.var = Parameter(var, name="var")
        self.m = Parameter(m, name="m")

    def construct(self, lr, alpha, sign_decay, beta, grad):
        self.apply_add_sign(self.var, self.m, lr, alpha,
                            sign_decay, beta, grad)
        return self.var, self.m


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_add_sign():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyAddSign operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    m = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = ApplyAddSignNet(var, m)

    lr = Tensor(0.001, mstype.float32)
    alpha = Tensor(1.0, mstype.float32)
    sign_decay = Tensor(0.99, mstype.float32)
    beta = Tensor(0.9, mstype.float32)
    grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    new_var, new_m = net(lr, alpha, sign_decay, beta, grad)
    assert ((new_var != var).any() and (new_m != m).any()), \
        "The results should be different!"


class ApplyCenteredRMSPropNet(nn.Cell):
    def __init__(self, var):
        super(ApplyCenteredRMSPropNet, self).__init__()
        self.apply_centered_rms_prop = P.ApplyCenteredRMSProp()
        self.var = Parameter(var, name="var")

    def construct(self, mean_grad, mean_square, moment, grad, learning_rate):
        self.apply_centered_rms_prop(self.var, mean_grad, mean_square, moment, grad,
                                     learning_rate, 0.0, 1e-10, 0.05)
        return self.var


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_centered_rms_prop():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyCenteredRMSProp operator.
    Expectation: No exception.
    """
    var = Tensor(
        np.arange(-6, 6).astype(np.float32).reshape(2, 3, 2), mstype.float32)
    net = ApplyCenteredRMSPropNet(var)

    mean_grad = Tensor(np.arange(12).astype(
        np.float32).reshape(2, 3, 2), mstype.float32)
    mean_square = Tensor(
        np.arange(-8, 4).astype(np.float32).reshape(2, 3, 2), mstype.float32)
    moment = Tensor(np.arange(12).astype(
        np.float32).reshape(2, 3, 2), mstype.float32)
    grad = Tensor(np.arange(12).astype(
        np.float32).reshape(2, 3, 2), mstype.float32)
    learning_rate = Tensor(0.9, mstype.float32)
    new_var = net(mean_grad, mean_square, moment, grad, learning_rate)
    assert (new_var != var).any(), "The results should be different!"


class ApplyFtrlNet(nn.Cell):
    def __init__(self, var, accum, linear):
        super(ApplyFtrlNet, self).__init__()
        self.apply_ftrl = P.ApplyFtrl()
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")
        self.linear = Parameter(linear, name="linear")

    def construct(self, grad, lr, l1, l2, lr_power):
        self.apply_ftrl(self.var, self.accum, self.linear,
                        grad, lr, l1, l2, lr_power)
        return self.var, self.accum, self.linear


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_ftrl():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyFtrl operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    accum = Tensor(np.random.rand(3, 3).astype(np.float32))
    linear = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = ApplyFtrlNet(var, accum, linear)

    grad = Tensor(np.random.randint(-4, 4, (3, 3)), mstype.float32)
    lr = Tensor(0.001, mstype.float32)
    l1 = Tensor(0.0, mstype.float32)
    l2 = Tensor(0.0, mstype.float32)
    lr_power = Tensor(-0.5, mstype.float32)
    new_var, new_accum, new_linear = net(grad, lr, l1, l2, lr_power)
    assert ((new_var != var).any() and (new_accum != accum).any() and (new_linear != linear).any()), \
        "The results should be different!"


class ApplyGradientDescentNet(nn.Cell):
    def __init__(self, var):
        super(ApplyGradientDescentNet, self).__init__()
        self.apply_gradient_descent = P.ApplyGradientDescent()
        self.var = Parameter(var, name="var")

    def construct(self, alpha, delta):
        self.apply_gradient_descent(self.var, alpha, delta)
        return self.var


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_gradient_descent():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyGradientDescent operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = ApplyGradientDescentNet(var)

    alpha = Tensor(0.001, mstype.float32)
    delta = Tensor(np.random.rand(3, 3).astype(np.float32))
    new_var = net(alpha, delta)
    assert (new_var != var).any(), "The results should be different!"


class ApplyMomentumNet(nn.Cell):
    def __init__(self, var, accum):
        super(ApplyMomentumNet, self).__init__()
        self.apply_momentum = P.ApplyMomentum(gradient_scale=1024.0)
        self.var = Parameter(var, name='var')
        self.accum = Parameter(accum, name='accum')

    def construct(self, lr, grad, momentum):
        self.apply_momentum(self.var, self.accum, lr, grad, momentum)
        return self.var, self.accum


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_momentum():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyMomentum operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.normal(size=(2, 3, 3, 4)).astype(np.float32))
    accum = Tensor(np.random.normal(size=(2, 3, 3, 4)).astype(np.float32))
    net = ApplyMomentumNet(var, accum)

    lr = Tensor(np.random.normal(size=(1,)).astype(np.float32))
    grad = Tensor(np.random.normal(size=(2, 3, 3, 4)).astype(np.float32))
    momentum = Tensor(np.random.normal(size=(1,)).astype(np.float32))
    new_var, new_accum = net(lr, grad, momentum)
    assert ((new_var != var).any() and (new_accum != accum).any()), \
        "The results should be different!"


class ApplyPowerSignNet(nn.Cell):
    def __init__(self, var, m):
        super(ApplyPowerSignNet, self).__init__()
        self.apply_power_sign = P.ApplyPowerSign()
        self.var = Parameter(var, name="var")
        self.m = Parameter(m, name="m")

    def construct(self, lr, logbase, sign_decay, beta, grad):
        self.apply_power_sign(self.var, self.m, lr,
                              logbase, sign_decay, beta, grad)
        return self.var, self.m


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_power_sign():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyPowerSign operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    m = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = ApplyPowerSignNet(var, m)

    lr = Tensor(0.001, mstype.float32)
    logbase = Tensor(np.e, mstype.float32)
    sign_decay = Tensor(0.99, mstype.float32)
    beta = Tensor(0.9, mstype.float32)
    grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    new_var, new_m = net(lr, logbase, sign_decay, beta, grad)
    assert ((new_var != var).any() and (new_m != m).any()), \
        "The results should be different!"


class ApplyProximalAdagradNet(nn.Cell):
    def __init__(self, var, accum):
        super(ApplyProximalAdagradNet, self).__init__()
        self.apply_proximal_adagrad = P.ApplyProximalAdagrad()
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name='accum')

    def construct(self, lr, l1, l2, grad):
        self.apply_proximal_adagrad(self.var, self.accum, lr, l1, l2, grad)
        return self.var, self.accum


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_proximal_adagrad():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyProximalAdagrad operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    accum = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = ApplyProximalAdagradNet(var, accum)

    lr = Tensor(0.01, mstype.float32)
    l1 = Tensor(0.0, mstype.float32)
    l2 = Tensor(0.0, mstype.float32)
    grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    new_var, new_accum = net(lr, l1, l2, grad)
    assert ((new_var != var).any() and (new_accum != accum).any()), \
        "The results should be different!"


class ApplyProximalGradientDescentNet(nn.Cell):
    def __init__(self, var):
        super(ApplyProximalGradientDescentNet, self).__init__()
        self.apply_proximal_gradient_descent = P.ApplyProximalGradientDescent()
        self.var = Parameter(var, name="var")

    def construct(self, alpha, l1, l2, delta):
        self.apply_proximal_gradient_descent(self.var, alpha, l1, l2, delta)
        return self.var


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_proximal_gradient_descent():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyProximalGradientDescent operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = ApplyProximalGradientDescentNet(var)

    alpha = Tensor(0.001, mstype.float32)
    l1 = Tensor(0.0, mstype.float32)
    l2 = Tensor(0.0, mstype.float32)
    delta = Tensor(np.random.rand(3, 3).astype(np.float32))
    new_var = net(alpha, l1, l2, delta)
    assert (new_var != var).any(), "The results should be different!"


class ApplyRMSPropNet(nn.Cell):
    def __init__(self, var):
        super(ApplyRMSPropNet, self).__init__()
        self.apply_rms_prop = P.ApplyRMSProp()
        self.var = Parameter(var, name="var")

    def construct(self, mean_square, moment, learning_rate, grad):
        self.apply_rms_prop(self.var, mean_square, moment,
                            learning_rate, grad, 0.0, 1e-10, 0.001)
        return self.var


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_apply_rms_prop():
    """
    Feature: Auto monad feature.
    Description: Verify ApplyRMSProp operator.
    Expectation: No exception.
    """
    var = Tensor(1., mstype.float32)
    net = ApplyRMSPropNet(var)

    mean_square = Tensor(2., mstype.float32)
    moment = Tensor(1., mstype.float32)
    learning_rate = Tensor(0.9, mstype.float32)
    grad = Tensor(2., mstype.float32)
    new_var = net(mean_square, moment, learning_rate, grad)
    assert (new_var != var).any(), "The results should be different!"


class FusedSparseAdamNet(nn.Cell):
    def __init__(self, var, m, v):
        super(FusedSparseAdamNet, self).__init__()
        self.fused_sparse_adam = P.FusedSparseAdam()
        self.var = Parameter(var, name="var")
        self.m = Parameter(m, name="m")
        self.v = Parameter(v, name="v")

    def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, indices):
        self.fused_sparse_adam(self.var, self.m, self.v, beta1_power, beta2_power, lr, beta1, beta2,
                               epsilon, grad, indices)
        return self.var, self.m, self.v


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fused_sparse_adam():
    """
    Feature: Auto monad feature.
    Description: Verify FusedSparseAdam operator.
    Expectation: No exception.
    """
    var = Tensor(np.ones([3, 1, 2]).astype(np.float32))
    m = Tensor(np.ones([3, 1, 2]).astype(np.float32))
    v = Tensor(np.ones([3, 1, 2]).astype(np.float32))
    net = FusedSparseAdamNet(var, m, v)

    beta1_power = Tensor(0.9, mstype.float32)
    beta2_power = Tensor(0.999, mstype.float32)
    lr = Tensor(0.001, mstype.float32)
    beta1 = Tensor(0.9, mstype.float32)
    beta2 = Tensor(0.999, mstype.float32)
    epsilon = Tensor(1e-8, mstype.float32)
    gradient = Tensor(np.random.rand(2, 1, 2), mstype.float32)
    indices = Tensor([0, 1], mstype.int32)
    new_var, new_m, new_v = net(
        beta1_power, beta2_power, lr, beta1, beta2, epsilon, gradient, indices)
    assert ((new_var != var).any() and (new_m != m).any() and (new_v != v).any()), \
        "The results should be different!"


class FusedSparseFtrlNet(nn.Cell):
    def __init__(self, var, accum, linear):
        super(FusedSparseFtrlNet, self).__init__()
        self.fused_sparse_ftrl = P.FusedSparseFtrl(
            lr=0.01, l1=0.0, l2=0.0, lr_power=-0.5)
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")
        self.linear = Parameter(linear, name="linear")

    def construct(self, grad, indices):
        self.fused_sparse_ftrl(self.var, self.accum,
                               self.linear, grad, indices)
        return self.var, self.accum, self.linear


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fused_sparse_ftrl():
    """
    Feature: Auto monad feature.
    Description: Verify FusedSparseFtrl operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 1, 2).astype(np.float32))
    accum = Tensor(np.random.rand(3, 1, 2).astype(np.float32))
    linear = Tensor(np.random.rand(3, 1, 2).astype(np.float32))
    net = FusedSparseFtrlNet(var, accum, linear)

    grad = Tensor(np.random.rand(2, 1, 2).astype(np.float32))
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    new_var, new_accum, new_linear = net(grad, indices)
    assert ((new_var != var).any() and (new_accum != accum).any() and (new_linear != linear).any()), \
        "The results should be different!"


class FusedSparseLazyAdamNet(nn.Cell):
    def __init__(self, var, m, v):
        super(FusedSparseLazyAdamNet, self).__init__()
        self.fused_sparse_lazyadam = P.FusedSparseLazyAdam()
        self.var = Parameter(var, name="var")
        self.m = Parameter(m, name="m")
        self.v = Parameter(v, name="v")

    def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, indices):
        self.fused_sparse_lazyadam(self.var, self.m, self.v, beta1_power, beta2_power, lr, beta1,
                                   beta2, epsilon, grad, indices)
        return self.var, self.m, self.v


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fused_sparse_lazyadam():
    """
    Feature: Auto monad feature.
    Description: Verify FusedSparseLazyAdam operator.
    Expectation: No exception.
    """
    var = Tensor(np.ones([3, 1, 2]).astype(np.float32))
    m = Tensor(np.ones([3, 1, 2]).astype(np.float32))
    v = Tensor(np.ones([3, 1, 2]).astype(np.float32))
    net = FusedSparseLazyAdamNet(var, m, v)

    beta1_power = Tensor(0.9, mstype.float32)
    beta2_power = Tensor(0.999, mstype.float32)
    lr = Tensor(0.001, mstype.float32)
    beta1 = Tensor(0.9, mstype.float32)
    beta2 = Tensor(0.999, mstype.float32)
    epsilon = Tensor(1e-8, mstype.float32)
    gradient = Tensor(np.random.rand(2, 1, 2), mstype.float32)
    indices = Tensor([0, 1], mstype.int32)
    new_var, new_m, new_v = net(
        beta1_power, beta2_power, lr, beta1, beta2, epsilon, gradient, indices)
    assert ((new_var != var).any() and (new_m != m).any() and (new_v != v).any()), \
        "The results should be different!"


class FusedSparseProximalAdagradNet(nn.Cell):
    def __init__(self, var, accum):
        super(FusedSparseProximalAdagradNet, self).__init__()
        self.fused_sparse_proximal_adagrad = P.FusedSparseProximalAdagrad()
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")

    def construct(self, lr, l1, l2, grad, indices):
        self.fused_sparse_proximal_adagrad(
            self.var, self.accum, lr, l1, l2, grad, indices)
        return self.var, self.accum


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fused_sparse_proximal_adagrad():
    """
    Feature: Auto monad feature.
    Description: Verify FusedSparseProximalAdagrad operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 1, 2).astype(np.float32))
    accum = Tensor(np.random.rand(3, 1, 2).astype(np.float32))
    net = FusedSparseProximalAdagradNet(var, accum)

    lr = Tensor(0.01, mstype.float32)
    l1 = Tensor(0.0, mstype.float32)
    l2 = Tensor(0.0, mstype.float32)
    grad = Tensor(np.random.rand(2, 1, 2).astype(np.float32))
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    new_var, new_accum = net(lr, l1, l2, grad, indices)
    assert ((new_var != var).any() and (new_accum != accum).any()), \
        "The results should be different!"


class SparseApplyAdagradNet(nn.Cell):
    def __init__(self, var, accum):
        super(SparseApplyAdagradNet, self).__init__()
        self.sparse_apply_adagrad = P.SparseApplyAdagrad(lr=0.01)
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")

    def construct(self, grad, indices):
        self.sparse_apply_adagrad(self.var, self.accum, grad, indices)
        return self.var, self.accum


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sparse_apply_adagrad():
    """
    Feature: Auto monad feature.
    Description: Verify SparseApplyAdagrad operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    accum = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = SparseApplyAdagradNet(var, accum)

    grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    indices = Tensor(np.ones((3,), np.int32))
    new_var, _ = net(grad, indices)
    # new_accum is equal to accum.
    assert (new_var != var).any(), "The results should be different!"


class SparseApplyAdagradV2Net(nn.Cell):
    def __init__(self, var, accum):
        super(SparseApplyAdagradV2Net, self).__init__()
        self.sparse_apply_adagrad_v2 = P.SparseApplyAdagradV2(
            lr=0.01, epsilon=0.001)
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")

    def construct(self, grad, indices):
        self.sparse_apply_adagrad_v2(self.var, self.accum, grad, indices)
        return self.var, self.accum


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sparse_apply_adagrad_v2():
    """
    Feature: Auto monad feature.
    Description: Verify SparseApplyAdagradV2 operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    accum = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = SparseApplyAdagradV2Net(var, accum)

    grad = grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    indices = Tensor(np.ones((3,), np.int32))
    new_var, new_accum = net(grad, indices)
    assert ((new_var != var).any() and (new_accum != accum).any()), \
        "The results should be different!"


class SparseApplyFtrlNet(nn.Cell):
    def __init__(self, var, accum, linear):
        super(SparseApplyFtrlNet, self).__init__()
        self.sparse_apply_ftrl = P.SparseApplyFtrl(
            lr=0.01, l1=0.0, l2=0.0, lr_power=-0.5)
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")
        self.linear = Parameter(linear, name="linear")

    def construct(self, grad, indices):
        self.sparse_apply_ftrl(self.var, self.accum,
                               self.linear, grad, indices)
        return self.var, self.accum, self.linear


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sparse_apply_ftrl():
    """
    Feature: Auto monad feature.
    Description: Verify SparseApplyFtrl operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    accum = Tensor(np.random.rand(3, 3).astype(np.float32))
    linear = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = SparseApplyFtrlNet(var, accum, linear)

    grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    indices = Tensor(np.ones((3,), np.int32))
    new_var, new_accum, new_linear = net(grad, indices)
    assert ((new_var != var).any() and (new_accum != accum).any() and (new_linear != linear).any()), \
        "The results should be different!"


class SparseApplyFtrlV2Net(nn.Cell):
    def __init__(self, var, accum, linear):
        super(SparseApplyFtrlV2Net, self).__init__()
        self.sparse_apply_ftrl_v2 = P.SparseApplyFtrlV2(
            lr=0.01, l1=0.0, l2=0.0, l2_shrinkage=0.0, lr_power=-0.5)
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")
        self.linear = Parameter(linear, name="linear")

    def construct(self, grad, indices):
        self.sparse_apply_ftrl_v2(
            self.var, self.accum, self.linear, grad, indices)
        return self.var, self.accum, self.linear


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sparse_apply_ftrl_v2():
    """
    Feature: Auto monad feature.
    Description: Verify SparseApplyFtrlV2 operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    accum = Tensor(np.random.rand(3, 3).astype(np.float32))
    linear = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = SparseApplyFtrlV2Net(var, accum, linear)

    grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    indices = Tensor(np.ones((3,), np.int32))
    new_var, new_accum, new_linear = net(grad, indices)
    assert ((new_var != var).any() and (new_accum != accum).any() and (new_linear != linear).any()), \
        "The results should be different!"


class SparseApplyProximalAdagradNet(nn.Cell):
    def __init__(self, var, accum):
        super(SparseApplyProximalAdagradNet, self).__init__()
        self.sparse_apply_proximal_adagrad = P.SparseApplyProximalAdagrad()
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")

    def construct(self, lr, l1, l2, grad, indices):
        self.sparse_apply_proximal_adagrad(
            self.var, self.accum, lr, l1, l2, grad, indices)
        return self.var, self.accum


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sparse_apply_proximal_adagrad():
    """
    Feature: Auto monad feature.
    Description: Verify SparseApplyProximalAdagrad operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    accum = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = SparseApplyProximalAdagradNet(var, accum)

    lr = Tensor(0.01, mstype.float32)
    l1 = Tensor(0.0, mstype.float32)
    l2 = Tensor(0.0, mstype.float32)
    grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    indices = Tensor(np.ones((3,), np.int32))
    new_var, new_accum = net(lr, l1, l2, grad, indices)
    assert ((new_var != var).any() and (new_accum != accum).any()), \
        "The results should be different!"


class SGDNet(nn.Cell):
    def __init__(self, var):
        super(SGDNet, self).__init__()
        self.sgd = P.SGD()
        self.var = Parameter(var, name="var")

    def construct(self, gradient, learning_rate, accum, momentum, stat):
        self.sgd(self.var, gradient, learning_rate, accum, momentum, stat)
        return self.var


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sgd():
    """
    Feature: Auto monad feature.
    Description: Verify SGD operator.
    Expectation: No exception.
    """
    var = Tensor(np.array([2, -0.5, 1.7, 4]), mstype.float32)
    net = SGDNet(var)

    gradient = Tensor(np.array([1, -1, 0.5, 2]), mstype.float32)
    learning_rate = Tensor(0.01, mstype.float32)
    accum = Tensor(np.array([0.1, 0.3, -0.2, -0.1]), mstype.float32)
    momentum = Tensor(0.1, mstype.float32)
    stat = Tensor(np.array([1.5, -0.3, 0.2, -0.7]), mstype.float32)
    new_var = net(gradient, learning_rate, accum, momentum, stat)
    assert (new_var != var).any(), "The results should be different!"


class ApplyProximalAdagradConstantNet(nn.Cell):
    def __init__(self, var, accum):
        super().__init__()
        self.depend = P.Depend()
        self.sparse_apply_proximal_adagrad = P.SparseApplyProximalAdagrad()
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")
        self.const = Tensor(9999, mstype.float32)

    def construct(self, lr, l1, l2, grad, indices):
        self.sparse_apply_proximal_adagrad(self.var, self.accum, lr, l1, l2, grad, indices)
        return self.const


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sparse_apply_proximal_adagrad_constant():
    """
    Feature: Auto monad feature.
    Description: Verify SparseApplyProximalAdagrad operator.
    Expectation: No exception.
    """
    var = Tensor(np.random.rand(3, 3).astype(np.float32))
    accum = Tensor(np.random.rand(3, 3).astype(np.float32))
    net = ApplyProximalAdagradConstantNet(var, accum)
    lr = Tensor(0.01, mstype.float32)
    l1 = Tensor(0.1, mstype.float32)
    l2 = Tensor(0.2, mstype.float32)
    grad = Tensor(np.random.rand(3, 3).astype(np.float32))
    indices = Tensor(np.ones((3,), np.int32))
    net(lr, l1, l2, grad, indices)
    assert (net.parameters_dict()['var'].data != var).any()
    assert (net.parameters_dict()['accum'].data != accum).any()


class MulSGDNet(nn.Cell):
    def __init__(self, var):
        super().__init__()
        self.sgd = P.SGD()
        self.var = Parameter(var, name="var")
        self.mul = P.Mul()

    def construct(self, gradient, learning_rate, accum, momentum, stat):
        out = self.mul(self.var, self.var)
        self.sgd(self.var, gradient, learning_rate, accum, momentum, stat)
        return out


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mul_sgd():
    """
    Feature: Auto monad feature.
    Description: Verify sgd operator.
    Expectation: No exception.
    """
    var = Tensor(np.array([2, -0.5, 1.7, 4]), mstype.float32)
    net = MulSGDNet(var)
    gradient = Tensor(np.array([1, -1, 0.5, 2]), mstype.float32)
    learning_rate = Tensor(0.01, mstype.float32)
    accum = Tensor(np.array([0.1, 0.3, -0.2, -0.1]), mstype.float32)
    momentum = Tensor(0.1, mstype.float32)
    stat = Tensor(np.array([1.5, -0.3, 0.2, -0.7]), mstype.float32)
    net(gradient, learning_rate, accum, momentum, stat)
    assert (net.parameters_dict()['var'].data != var).any()
