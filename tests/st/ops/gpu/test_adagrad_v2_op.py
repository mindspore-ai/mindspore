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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

eps_f64 = np.array([1e-5 for i in range(4)]).reshape(2, 2)
eps_f32 = np.array([1e-4 for i in range(4)]).reshape(2, 2)


class Net(nn.Cell):
    def __init__(self, var_np, accum_np, epsilon=1e-6, update_slots=True):
        super(Net, self).__init__()
        self.apply_adagrad_v2 = P.ApplyAdagradV2(epsilon=epsilon, update_slots=update_slots)
        self.var = Parameter(Tensor(var_np), name="var")
        self.accum = Parameter(Tensor(accum_np), name="accum")

    def construct(self, lr, grad):
        z = self.apply_adagrad_v2(self.var, self.accum, lr, grad)
        return z


def main_test(var_np, accum_np, lr_np, grident_np, epsilon_np, update_slots):
    lr = Tensor(lr_np)
    grad = Tensor(grident_np)

    # expect
    if update_slots:
        expect_accum_np = accum_np + grident_np * grident_np
    else:
        expect_accum_np = accum_np
    expect_var_np = var_np - lr_np * grident_np / np.sqrt(expect_accum_np + epsilon_np)

    net = Net(var_np, accum_np, epsilon_np, update_slots)
    out = net(lr, grad)
    res_var_mindspore = out[0].asnumpy()
    res_accum_mindspore = out[1].asnumpy()

    return (expect_var_np, res_var_mindspore), (expect_accum_np, res_accum_mindspore)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_apply_adagradv2_fff():
    """
    Feature: None
    Description: basic test fff
    Expectation: just test
    """
    var_np = np.array([[0.6, 0.4], [0.1, 0.5]]).astype(np.float32)
    accum_np = np.array([[0.6, 0.5], [0.2, 0.6]]).astype(np.float32)

    lr_np = np.array(0.001).astype(np.float32)
    epsilon_np = 1e-6

    update_slots = True

    grident_np = np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32)

    var, accum = main_test(var_np, accum_np, lr_np, grident_np, epsilon_np, update_slots)

    assert np.all(abs(accum[0] - accum[1]) < eps_f32)
    assert np.all(abs(var[0] - var[1]) < eps_f32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_apply_adagradv2_ddd():
    """
    Feature: None
    Description: basic test ddd
    Expectation: just test
    """

    var_np = np.array([[0.6, 0.4], [0.1, 0.5]]).astype(np.float64)
    accum_np = np.array([[0.6, 0.5], [0.2, 0.6]]).astype(np.float64)

    lr_np = np.array(0.001).astype(np.float64)
    epsilon_np = 1e-6
    update_slots = True

    grident_np = np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float64)

    var, accum = main_test(var_np, accum_np, lr_np, grident_np, epsilon_np, update_slots)
    assert np.all(abs(accum[0] - accum[1]) < eps_f64)
    assert np.all(abs(var[0] - var[1]) < eps_f64)
