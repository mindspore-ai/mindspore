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
import math
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, context, Parameter
from mindspore.ops.functional import vmap

ms.set_seed(2022)
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class ApplyAdamWithAmsgradTEST(nn.Cell):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False):
        super(ApplyAdamWithAmsgradTEST, self).__init__()
        shape = (8, 9, 6, 10, 5)
        self.apply_adam_with_amsgrad = P.ApplyAdamWithAmsgrad(beta1, beta2, epsilon, use_locking)
        self.var_np = np.random.randn(*shape).astype(np.float32)
        self.m_np = np.random.randn(*shape).astype(np.float32)
        self.v_np = np.random.randn(*shape).astype(np.float32)
        self.vhat_np = np.random.randn(*shape).astype(np.float32)
        self.var = Parameter(Tensor(self.var_np), name="var")
        self.m = Parameter(Tensor(self.m_np), name="m")
        self.v = Parameter(Tensor(self.v_np), name="v")
        self.vhat = Parameter(Tensor(self.vhat_np), name="vhat")

    def construct(self, beta1_power, beta2_power, lr, grad):
        return self.apply_adam_with_amsgrad(self.var, self.m, self.v, self.vhat, beta1_power, beta2_power, lr, grad)


def numpy_apply_adam_with_amsgrad(var, m, v, vhat, grad, beta1=0.9, beta2=0.999, eps=1e-8, lr=0.01):
    new_lr = lr * math.sqrt(1 - beta2) / (1 - beta1)
    m = m * beta1 + grad * (1 - beta1)
    v = v * beta2 + grad * grad * (1 - beta2)
    vhat = np.maximum(vhat, v)
    var = var - new_lr * m / (np.sqrt(vhat) + eps)
    return var


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_apply_adam_with_amsgrad_op(data_type):
    """
    Feature: ApplyAdamWithAmsgrad gpu kernel
    Description: test the ApplyAdamWithAmsgrad.
    Expectation: match to np benchmark.
    """
    shape = (8, 9, 6, 10, 5)
    amsgrad = ApplyAdamWithAmsgradTEST()
    error = 1e-4
    if data_type == np.float16:
        error = 1e-3

    grad_np = np.random.randn(*shape).astype(np.float32)
    grad = Tensor(grad_np)

    output = amsgrad(Tensor(0.9), Tensor(0.999), Tensor(0.01), grad)
    ms_var = output[0].asnumpy()
    np_var = numpy_apply_adam_with_amsgrad(amsgrad.var_np, amsgrad.m_np,
                                           amsgrad.v_np, amsgrad.vhat_np, grad_np)

    np.testing.assert_allclose(ms_var, np_var, rtol=error, atol=error)


class AmsgradNetVmap(nn.Cell):
    def __init__(self, net):
        super(AmsgradNetVmap, self).__init__()
        shape = (8, 9, 6, 10, 5)
        self.net = net
        self.var_np = np.random.randn(*shape).astype(np.float32)
        self.m_np = np.random.randn(*shape).astype(np.float32)
        self.v_np = np.random.randn(*shape).astype(np.float32)
        self.vhat_np = np.random.randn(*shape).astype(np.float32)
        self.var = Parameter(Tensor(self.var_np), name="var")
        self.m = Parameter(Tensor(self.m_np), name="m")
        self.v = Parameter(Tensor(self.v_np), name="v")
        self.vhat = Parameter(Tensor(self.vhat_np), name="vhat")
        self.vmap_amsgrad = vmap(self.net, in_axes=(
            0, 0, 0, 0, None, None, None, 0), out_axes=0)

    def construct(self, beta1_power, beta2_power, lr, grad):
        return self.vmap_amsgrad(self.var, self.m, self.v, self.vhat, beta1_power, beta2_power, lr, grad)


@pytest.mark.level1
@pytest.mark.env_onecard
def test_apply_adam_witm_amsgrad_op_vmap():
    """
    Feature: ApplyAdamWithAmsgrad gpu kernel
    Description: test the ApplyAdamWithAmsgrad vmap.
    Expectation: match to np benchmark.
    """
    shape = (8, 9, 6, 10, 5)
    def cal_amsgrad(var, m, v, vhat, beta1_power, beta2_power, lr, grad):
        return P.ApplyAdamWithAmsgrad()(var, m, v, vhat, beta1_power, beta2_power, lr, grad)

    error = 1e-4
    grad_np = np.random.randn(*shape).astype(np.float32)
    grad = Tensor(grad_np)

    vmap_amsgrad = AmsgradNetVmap(cal_amsgrad)
    _ = vmap_amsgrad(Tensor(0.9), Tensor(0.999), Tensor(0.01), grad)
    ms_var = vmap_amsgrad.var.asnumpy()
    np_var = numpy_apply_adam_with_amsgrad(vmap_amsgrad.var_np, vmap_amsgrad.m_np,
                                           vmap_amsgrad.v_np, vmap_amsgrad.vhat_np, grad_np)

    np.testing.assert_allclose(ms_var, np_var, rtol=error, atol=error)


class AmsgradNetVmap2(nn.Cell):
    def __init__(self, net):
        super(AmsgradNetVmap2, self).__init__()
        shape = (8, 9, 6, 10, 5)
        self.net = net
        self.var_np = np.random.randn(*shape).astype(np.float32)
        self.m_np = np.random.randn(*shape).astype(np.float32)
        self.v_np = np.random.randn(*shape).astype(np.float32)
        self.vhat_np = np.random.randn(*shape).astype(np.float32)
        self.var = Parameter(Tensor(self.var_np), name="var")
        self.m = Parameter(Tensor(self.m_np), name="m")
        self.v = Parameter(Tensor(self.v_np), name="v")
        self.vhat = Parameter(Tensor(self.vhat_np), name="vhat")

        self.vmap_amsgrad = vmap(vmap(self.net, in_axes=(0, 0, 0, 0, None, None, None, 0), out_axes=0),
                                 in_axes=(0, 0, 0, 0, None, None, None, 0), out_axes=0)

    def construct(self, beta1_power, beta2_power, lr, grad):
        return self.vmap_amsgrad(self.var, self.m, self.v, self.vhat, beta1_power, beta2_power, lr, grad)


@pytest.mark.level1
@pytest.mark.env_onecard
def test_apply_adam_with_amsgrad_grad_op_vmap2():
    """
    Feature: ApplyAdamWithAmsgrad gpu kernel
    Description: test the ApplyAdamWithAmsgrad vmap.
    Expectation: match to np benchmark.
    """
    shape = (8, 9, 6, 10, 5)
    def cal_amsgrad(var, m, v, vhat, beta1_power, beta2_power, lr, grad):
        return P.ApplyAdamWithAmsgrad()(var, m, v, vhat, beta1_power, beta2_power, lr, grad)

    error = 1e-4
    grad_np = np.random.randn(*shape).astype(np.float32)
    grad = Tensor(grad_np)

    vmap_amsgrad = AmsgradNetVmap2(cal_amsgrad)
    _ = vmap_amsgrad(Tensor(0.9), Tensor(0.999), Tensor(0.01), grad)
    ms_var = vmap_amsgrad.var.asnumpy()
    np_var = numpy_apply_adam_with_amsgrad(vmap_amsgrad.var_np, vmap_amsgrad.m_np,
                                           vmap_amsgrad.v_np, vmap_amsgrad.vhat_np, grad_np)
    np.testing.assert_allclose(ms_var, np_var, rtol=error, atol=error)
