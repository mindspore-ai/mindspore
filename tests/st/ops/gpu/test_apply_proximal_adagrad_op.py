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
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, context, Parameter
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class ApplyProximalAdagradTEST(nn.Cell):
    def __init__(self):
        super(ApplyProximalAdagradTEST, self).__init__()
        self.apply_proximal_adagrad = P.ApplyProximalAdagrad()
        self.var = Parameter(
            Tensor(np.array([[0.6, 0.4], [0.1, 0.5]]).astype(np.float32)), name="var")
        self.accum = Parameter(
            Tensor(np.array([[0.6, 0.5], [0.2, 0.6]]).astype(np.float32)), name="accum")

    def construct(self, lr, l1, l2, grad):
        return self.apply_proximal_adagrad(self.var, self.accum, lr, l1, l2, grad)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_apply_proximal_adagrad_op(data_type):
    """
    Feature: ApplyProximalAdagrad GPU kernel
    Description: test the ApplyProximalAdagrad.
    Expectation: match to np benchmark.
    """
    adgrad = ApplyProximalAdagradTEST()
    error = 1e-6
    if data_type == np.float16:
        error = 1e-3

    lr = 0.01
    l1 = 0.0
    l2 = 0.0
    grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))

    context.set_context(mode=context.GRAPH_MODE)
    output = adgrad(lr, l1, l2, grad)
    mindspore_var_out = output[0].asnumpy()
    mindspore_accum_out = output[1].asnumpy()
    print(mindspore_var_out)
    print(mindspore_accum_out)

    expect_var = np.array([[5.96388459e-01, 3.92964751e-01],
                           [9.78178233e-02, 4.92815793e-01]]).astype(data_type)
    expect_accum = np.array([[6.90000057e-01, 9.90000010e-01],
                             [2.10000008e-01, 1.24000001e+00]]).astype(data_type)

    np.testing.assert_allclose(mindspore_var_out, expect_var, rtol=error)
    np.testing.assert_allclose(mindspore_accum_out, expect_accum, rtol=error)


class AdgradNetVmap(nn.Cell):
    def __init__(self, net):
        super(AdgradNetVmap, self).__init__()
        self.net = net
        self.var = Parameter(
            Tensor(np.array([[[0.6, 0.4], [0.1, 0.5]], [[0.6, 0.4], [0.1, 0.5]]]).astype(np.float32)), name="var")
        self.accum = Parameter(
            Tensor(np.array([[[0.6, 0.5], [0.2, 0.6]], [[0.6, 0.5], [0.2, 0.6]]]).astype(np.float32)), name="accum")
        self.vmap_adagrad = vmap(self.net, in_axes=(
            0, 0, 0, None, None, 0), out_axes=0)

    def construct(self, lr, l1, l2, grad):
        return self.vmap_adagrad(self.var, self.accum, lr, l1, l2, grad)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_apply_proximal_adagrad_op_vmap():
    """
    Feature: ApplyProximalAdagrad GPU kernel
    Description: test the ApplyProximalAdagrad vmap.
    Expectation: match to np benchmark.
    """
    def cal_adagrad(var, accum, lr, l1, l2, grad):
        return P.ApplyProximalAdagrad()(var, accum, lr, l1, l2, grad)
    error = 1e-3
    grad = Tensor(np.array([[[0.3, 0.7], [0.1, 0.8]],
                            [[0.3, 0.7], [0.1, 0.8]]]).astype(np.float32))

    lr = Tensor(np.array([0.01, 0.01]).astype(np.float32))
    l1 = 0.
    l2 = Tensor(0.0)

    vmap_agrad = AdgradNetVmap(cal_adagrad)
    output = vmap_agrad(lr, l1, l2, grad)
    mindspore_var_out = output[0].asnumpy()
    mindspore_accum_out = output[1].asnumpy()
    print(mindspore_var_out)
    print(mindspore_accum_out)

    expect_var = np.array([[[5.96388459e-01, 3.92964751e-01], [9.78178233e-02, 4.92815793e-01]],
                           [[5.96388459e-01, 3.92964751e-01], [9.78178233e-02, 4.92815793e-01]]]).astype(np.float32)

    expect_accum = np.array([[[6.90000057e-01, 9.90000010e-01], [2.10000008e-01, 1.24000001e+00]],
                             [[6.90000057e-01, 9.90000010e-01], [2.10000008e-01, 1.24000001e+00]]]).astype(np.float32)

    np.testing.assert_allclose(mindspore_var_out, expect_var, rtol=error)
    np.testing.assert_allclose(mindspore_accum_out, expect_accum, rtol=error)


class AdgradNetVmap2(nn.Cell):
    def __init__(self, net):
        super(AdgradNetVmap2, self).__init__()
        self.net = net
        self.var = Parameter(
            Tensor(np.array([[[[0.6, 0.4], [0.1, 0.5]], [[0.7, 0.4], [0.1, 0.5]]],
                             [[[0.8, 0.4], [0.1, 0.5]], [[0.9, 0.4], [0.1, 0.5]]]]).astype(np.float32)), name="var")
        self.accum = Parameter(
            Tensor(np.array([[[[0.6, 0.5], [0.2, 0.6]], [[0.7, 0.5], [0.2, 0.6]]],
                             [[[0.8, 0.5], [0.2, 0.6]], [[0.9, 0.5], [0.2, 0.6]]]]).astype(np.float32)), name="accum")
        self.vmap_adagrad = vmap(vmap(self.net, in_axes=(
            0, 0, 0, 0, None, 0), out_axes=0), in_axes=(0, 0, 0, None, None, 0), out_axes=0)

    def construct(self, lr, l1, l2, grad):
        return self.vmap_adagrad(self.var, self.accum, lr, l1, l2, grad)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_apply_proximal_adagrad_op_vmap2():
    """
    Feature: ApplyProximalAdagrad GPU kernel
    Description: test the ApplyProximalAdagrad vmap.
    Expectation: match to np benchmark.
    """
    def cal_adagrad(var, accum, lr, l1, l2, grad):
        return P.ApplyProximalAdagrad()(var, accum, lr, l1, l2, grad)
    error = 1e-3
    grad = Tensor(np.array([[[[0.3, 0.7], [0.1, 0.8]], [[0.3, 0.7], [0.1, 0.8]]],
                            [[[0.3, 0.7], [0.1, 0.8]], [[0.3, 0.7], [0.1, 0.8]]]]).astype(np.float32))
    lr = Tensor(np.array([[0.01, 0.02], [0.03, 0.04]]).astype(np.float32))
    l1 = Tensor(np.array([0.0, 0.1]).astype(np.float32))
    l2 = Tensor(0.0)

    vmap_agrad = AdgradNetVmap2(cal_adagrad)
    output = vmap_agrad(lr, l1, l2, grad)

    mindspore_var_out = output[0].asnumpy()
    mindspore_accum_out = output[1].asnumpy()
    print(mindspore_var_out)
    print(mindspore_accum_out)

    expect_var = np.array([[[[5.96388459e-01, 3.92964751e-01], [9.78178233e-02, 4.92815793e-01]],
                            [[0.69099927, 0.3839194], [0.09127129, 0.48383552]]],
                           [[[0.79046005, 0.3788942], [0.09345347, 0.47844738]],
                            [[0.88391936, 0.3678388], [0.08254258, 0.46767104]]]]).astype(np.float32)

    expect_accum = np.array([[[[6.90000057e-01, 9.90000010e-01], [2.10000008e-01, 1.24000001e+00]],
                              [[0.78999996, 0.99], [0.21000001, 1.24]]],
                             [[[0.89, 0.99], [0.21000001, 1.24]],
                              [[0.99, 0.99], [0.21000001, 1.24]]]]).astype(np.float32)

    np.testing.assert_allclose(mindspore_var_out, expect_var, rtol=error)
    np.testing.assert_allclose(mindspore_accum_out, expect_accum, rtol=error)
