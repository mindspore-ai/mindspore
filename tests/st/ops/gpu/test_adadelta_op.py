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
# ============================================================================

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

eps_f64 = np.array([1e-5 for i in range(4)]).reshape(2, 2)
eps_f32 = np.array([1e-4 for i in range(4)]).reshape(2, 2)


class Net(nn.Cell):
    def __init__(self, var_np, accum_np, accum_update_np):
        super(Net, self).__init__()
        self.apply_adadelta = P.ApplyAdadelta()
        self.var = Parameter(Tensor(var_np), name="var")
        self.accum = Parameter(Tensor(accum_np), name="accum")
        self.accum_update = Parameter(Tensor(accum_update_np), name="accum_update")

    def construct(self, lr, rho, epsilon, grad):
        z = self.apply_adadelta(self.var, self.accum, self.accum_update, lr, rho, epsilon, grad)
        return z


def main_test(var_np, accum_np, accum_update_np, lr_np, rho_np, epsilon_np, grident_np):
    lr = Tensor(lr_np)
    rho = Tensor(rho_np)
    epsilon = Tensor(epsilon_np)
    grad = Tensor(grident_np)

    # expect
    expect_accum_np = rho_np * accum_np + (1-rho_np) * grident_np * grident_np
    update = np.sqrt(accum_update_np + epsilon_np) * grident_np / np.sqrt(expect_accum_np + epsilon_np)
    expect_accum_update_np = rho_np * accum_update_np + (1-rho_np) * update * update
    expect_var_np = var_np - lr_np * update


    net = Net(var_np, accum_np, accum_update_np)
    out = net(lr, rho, epsilon, grad)
    res_var_mindspore = out[0].asnumpy()
    res_accum_mindspore = out[1].asnumpy()
    res_accum_update_mindspore = out[2].asnumpy()

    return (expect_accum_np, res_accum_mindspore), (expect_accum_update_np, res_accum_update_mindspore),\
           (expect_var_np, res_var_mindspore)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_apply_adadelta_fff():
    """
    Feature: None
    Description: basic test fff
    Expectation: just test
    """
    var_np = np.array([[0.6, 0.4], [0.1, 0.5]]).astype(np.float32)
    accum_np = np.array([[0.6, 0.5], [0.2, 0.6]]).astype(np.float32)
    accum_update_np = np.array([[0.9, 0.1], [0.7, 0.8]]).astype(np.float32)

    lr_np = np.array(0.001).astype(np.float32)
    rho_np = np.array(0.0).astype(np.float32)
    epsilon_np = np.array(1e-6).astype(np.float32)

    grident_np = np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32)

    accum, accum_update, var,\
        = main_test(var_np, accum_np, accum_update_np, lr_np, rho_np, epsilon_np, grident_np)

    assert np.all(abs(accum[0] - accum[1]) < eps_f32)
    assert np.all(abs(accum_update[0] - accum_update[1]) < eps_f32)
    assert np.all(abs(var[0] - var[1]) < eps_f32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_apply_adadelta_ddd():
    """
    Feature: None
    Description: basic test ddd
    Expectation: just test
    """

    var_np = np.array([[0.6, 0.4], [0.1, 0.5]]).astype(np.float64)
    accum_np = np.array([[0.6, 0.5], [0.2, 0.6]]).astype(np.float64)
    accum_update_np = np.array([[0.9, 0.1], [0.7, 0.8]]).astype(np.float64)

    lr_np = np.array(0.001).astype(np.float64)
    rho_np = np.array(0.0).astype(np.float64)
    epsilon_np = np.array(1e-6).astype(np.float64)

    grident_np = np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float64)

    accum, accum_update, var,\
        = main_test(var_np, accum_np, accum_update_np, lr_np, rho_np, epsilon_np, grident_np)

    assert np.all(abs(accum[0] - accum[1]) < eps_f64)
    assert np.all(abs(accum_update[0] - accum_update[1]) < eps_f64)
    assert np.all(abs(var[0] - var[1]) < eps_f64)
