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

import numpy as np
import pytest

from mindspore import Tensor, Parameter, nn, context
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    """ NetWithApplyAdadelta definition """

    def __init__(self, var_np, accum_np, accum_update_np):
        super(Net, self).__init__()
        self.apply_adadelta = P.ApplyAdadelta()
        self.var = Parameter(Tensor(var_np), name="var")
        self.accum = Parameter(Tensor(accum_np), name="accum")
        self.accum_update = Parameter(Tensor(accum_update_np), name="accum_update")

    def construct(self, lr, rho, epsilon, grad):
        return self.apply_adadelta(self.var, self.accum, self.accum_update, lr, rho, epsilon, grad)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_adadelta():
    """
    Feature: Adadelta cpu kernel
    Description: test the Adadelta.
    Expectation: match to np benchmark.
    """
    var_np = np.array([[0.1, 0.2]], dtype=np.float32)
    accum_np = np.array([[0.2, 0.3]], dtype=np.float32)
    accum_update_np = np.array([[0.3, 0.4]], dtype=np.float32)
    grident_np = np.array([[0.4, 0.5]], dtype=np.float32)
    lr_tensor = Tensor(0.001)
    rho_tensor = Tensor(0.95)
    epsilon_tensor = Tensor(1e-6)
    grad_tensor = Tensor(grident_np)

    net = Net(var_np, accum_np, accum_update_np)
    out = net(lr_tensor, rho_tensor, epsilon_tensor, grad_tensor)
    output_var = out[0].asnumpy()
    output_accum = out[1].asnumpy()
    output_accum_update = out[2].asnumpy()

    error = 1e-6
    expect_var = np.array([[0.09950764, 0.19942023]], dtype=np.float32)
    expect_accum = np.array([[0.198, 0.29749998]], dtype=np.float32)
    expect_accum_update = np.array([[0.2971212, 0.39680672]], dtype=np.float32)
    np.testing.assert_allclose(output_var, expect_var, rtol=error)
    np.testing.assert_allclose(output_accum, expect_accum, rtol=error)
    np.testing.assert_allclose(output_accum_update, expect_accum_update, rtol=error)


class AdadeltaNetVmap(nn.Cell):
    """ NetVmapWithApplyAdadelta definition """

    def __init__(self, net, var_np, accum_np, accum_update_np):
        super(AdadeltaNetVmap, self).__init__()
        self.net = net
        self.var = Parameter(Tensor(var_np), name="var")
        self.accum = Parameter(Tensor(accum_np), name="accum")
        self.accum_update = Parameter(Tensor(accum_update_np), name="accum_update")

        self.vmap_adagrad = vmap(self.net, in_axes=(
            0, 0, 0, 0, None, None, 0), out_axes=0)

    def construct(self, lr, rho, epsilon, grad):
        return self.vmap_adagrad(self.var, self.accum, self.accum_update, lr, rho, epsilon, grad)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_adadelta_vmap():
    """
    Feature: Adadelta cpu kernel
    Description: test the Adadelta vmap.
    Expectation: match to np benchmark.
    """

    def cal_adagrad(var, accum, accum_update, lr, rho, epsilon, grad):
        return P.ApplyAdadelta()(var, accum, accum_update, lr, rho, epsilon, grad)

    var_np = np.array([[0.1, 0.2],
                       [0.1, 0.2]], dtype=np.float32)
    accum_np = np.array([[0.2, 0.3],
                         [0.2, 0.3]], dtype=np.float32)
    accum_update_np = np.array([[0.3, 0.4],
                                [0.3, 0.4]], dtype=np.float32)
    grident_np = np.array([[0.4, 0.5],
                           [0.4, 0.5]], dtype=np.float32)

    lr_tensor = Tensor(np.array([0.001, 0.001]).astype(np.float32))
    rho_tensor = 0.95
    epsilon_tensor = 1e-6
    grad_tensor = Tensor(grident_np)

    vmap_agrad = AdadeltaNetVmap(cal_adagrad, var_np, accum_np, accum_update_np)
    output = vmap_agrad(lr_tensor, rho_tensor, epsilon_tensor, grad_tensor)
    output_var = output[0].asnumpy()
    output_accum = output[1].asnumpy()
    output_accum_update = output[2].asnumpy()

    error = 1e-6
    expect_var = np.array([[0.09950764, 0.19942023],
                           [0.09950764, 0.19942023]], dtype=np.float32)
    expect_accum = np.array([[0.198, 0.29749998],
                             [0.198, 0.29749998]], dtype=np.float32)
    expect_accum_update = np.array([[0.2971212, 0.39680672],
                                    [0.2971212, 0.39680672]], dtype=np.float32)
    np.testing.assert_allclose(output_var, expect_var, rtol=error)
    np.testing.assert_allclose(output_accum, expect_accum, rtol=error)
    np.testing.assert_allclose(output_accum_update, expect_accum_update, rtol=error)


class AdadeltaNetVmap2(nn.Cell):
    """ NetVmapWithApplyAdadelta definition """

    def __init__(self, net, var_np, accum_np, accum_update_np):
        super(AdadeltaNetVmap2, self).__init__()
        self.net = net
        self.var = Parameter(Tensor(var_np), name="var")
        self.accum = Parameter(Tensor(accum_np), name="accum")
        self.accum_update = Parameter(Tensor(accum_update_np), name="accum_update")

        self.vmap_adagrad = vmap(vmap(self.net, in_axes=(
            0, 0, 0, 0, 0, None, 0), out_axes=0), in_axes=(0, 0, 0, 0, None, None, 0), out_axes=0)

    def construct(self, lr, rho, epsilon, grad):
        return self.vmap_adagrad(self.var, self.accum, self.accum_update, lr, rho, epsilon, grad)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_apply_adadelta_vmap2():
    """
    Feature: Adadelta cpu kernel
    Description: test the Adadelta vmap.
    Expectation: match to np benchmark.
    """

    def cal_adagrad(var, accum, accum_update, lr, rho, epsilon, grad):
        return P.ApplyAdadelta()(var, accum, accum_update, lr, rho, epsilon, grad)

    var_np = np.array([[[0.1, 0.2], [0.1, 0.2]],
                       [[0.1, 0.2], [0.1, 0.2]]], dtype=np.float32)
    accum_np = np.array([[[0.2, 0.3], [0.2, 0.3]],
                         [[0.2, 0.3], [0.2, 0.3]]], dtype=np.float32)
    accum_update_np = np.array([[[0.3, 0.4], [0.3, 0.4]],
                                [[0.3, 0.4], [0.3, 0.4]]], dtype=np.float32)
    grident_np = np.array([[[0.4, 0.5], [0.4, 0.5]],
                           [[0.4, 0.5], [0.4, 0.5]]], dtype=np.float32)

    lr_tensor = Tensor(np.array([[0.01, 0.02], [0.03, 0.04]]).astype(np.float32))
    rho_tensor = Tensor(np.array([0.9, 0.95]).astype(np.float32))
    epsilon_tensor = Tensor(1e-6)
    grad_tensor = Tensor(grident_np)

    vmap_agrad = AdadeltaNetVmap2(cal_adagrad, var_np, accum_np, accum_update_np)
    output = vmap_agrad(lr_tensor, rho_tensor, epsilon_tensor, grad_tensor)
    output_var = output[0].asnumpy()
    output_accum = output[1].asnumpy()
    output_accum_update = output[2].asnumpy()

    expect_var = np.array([[[0.09505129, 0.19417778], [0.09015269, 0.18840459]],
                           [[0.08515386, 0.18253334], [0.08030538, 0.17680918]]], dtype=np.float32)
    expect_accum = np.array([[[0.196, 0.29500002], [0.198, 0.2975]],
                             [[0.196, 0.29500002], [0.198, 0.2975]]], dtype=np.float32)
    expect_accum_update = np.array([[[0.29448977, 0.39389828], [0.2971212, 0.39680672]],
                                    [[0.29448977, 0.39389828], [0.2971212, 0.39680672]]], dtype=np.float32)

    error = 1e-6
    np.testing.assert_allclose(output_var, expect_var, rtol=error)
    np.testing.assert_allclose(output_accum, expect_accum, rtol=error)
    np.testing.assert_allclose(output_accum_update, expect_accum_update, rtol=error)
