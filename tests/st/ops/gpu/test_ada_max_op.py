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

import numpy
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

eps_f32 = numpy.array([1e-4 for i in range(4)]).reshape(2, 2)
eps_f64 = numpy.array([1e-5 for i in range(4)]).reshape(2, 2)


class Net(nn.Cell):
    def __init__(self, var_numpy, m_numpy, v_numpy):
        super(Net, self).__init__()
        self.apply_adamax = P.ApplyAdaMax()
        self.var = Parameter(Tensor(var_numpy), name="var")
        self.m = Parameter(Tensor(m_numpy), name="accum")
        self.v = Parameter(Tensor(v_numpy), name="accum_update")

    def construct(self, beta1_power, lr, beta1, beta2, epsilon, grad):
        z = self.apply_adamax(self.var, self.m, self.v, beta1_power, lr, beta1, beta2, epsilon, grad)
        return z


def main_test(var_numpy, m_numpy, v_numpy, beta1_power_numpy, lr_numpy, beta1_numpy, beta2_numpy, epsilon_numpy,
              grident_numpy):
    beta1_power = Tensor(beta1_power_numpy)
    lr = Tensor(lr_numpy)
    beta1 = Tensor(beta1_numpy)
    beta2 = Tensor(beta2_numpy)
    epsilon = Tensor(epsilon_numpy)
    grad = Tensor(grident_numpy)

    expect_m = beta1_numpy * m_numpy + (1-beta1_numpy) * grident_numpy
    expect_v = numpy.maximum(abs(grident_numpy), beta2_numpy * v_numpy)
    expect_var = var_numpy - lr_numpy * expect_m / (1-beta1_power_numpy) / (expect_v + epsilon_numpy)

    net = Net(var_numpy, m_numpy, v_numpy)

    out = net(beta1_power, lr, beta1, beta2, epsilon, grad)
    res_var_mindspore = out[0].asnumpy()
    res_m_mindspore = out[1].asnumpy()
    res_v_mindspore = out[2].asnumpy()

    return (expect_m, res_m_mindspore), (expect_v, res_v_mindspore), (expect_var, res_var_mindspore)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_apply_adamax_fff():
    """
    Feature: None
    Description: basic test fff
    Expectation: just test
    """

    m_numpy = numpy.array([[0.6, 0.4], [0.1, 0.5]]).astype(numpy.float32)
    v_numpy = numpy.array([[0.6, 0.5], [0.2, 0.6]]).astype(numpy.float32)
    var_numpy = numpy.array([[0.9, 0.1], [0.7, 0.8]]).astype(numpy.float32)

    beta1_power_numpy = numpy.array(0.9).astype(numpy.float32)
    lr_numpy = numpy.array(0.001).astype(numpy.float32)
    beta1_numpy = numpy.array(0.9).astype(numpy.float32)
    beta2_numpy = numpy.array(0.99).astype(numpy.float32)
    epsilon_numpy = numpy.array(1e-10).astype(numpy.float32)

    grident_numpy = numpy.array([[0.3, 0.7], [0.1, 0.8]]).astype(numpy.float32)

    m, v, var = main_test(var_numpy, m_numpy, v_numpy, beta1_power_numpy, lr_numpy, beta1_numpy, beta2_numpy,
                          epsilon_numpy, grident_numpy)

    assert numpy.all(m[0] - m[1] < eps_f32)
    assert numpy.all(v[0] - v[1] < eps_f32)
    assert numpy.all(var[0] - var[1] < eps_f32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_apply_adamax_ddd():
    """
    Feature: None
    Description: basic test ddd
    Expectation: just test
    """

    m_numpy = numpy.array([[0.6, 0.4], [0.1, 0.5]]).astype(numpy.float64)
    v_numpy = numpy.array([[0.6, 0.5], [0.2, 0.6]]).astype(numpy.float64)
    var_numpy = numpy.array([[0.9, 0.1], [0.7, 0.8]]).astype(numpy.float64)

    beta1_power_numpy = numpy.array(0.9).astype(numpy.float64)
    lr_numpy = numpy.array(0.001).astype(numpy.float64)
    beta1_numpy = numpy.array(0.9).astype(numpy.float64)
    beta2_numpy = numpy.array(0.99).astype(numpy.float64)
    epsilon_numpy = numpy.array(1e-10).astype(numpy.float64)

    grident_numpy = numpy.array([[0.3, 0.7], [0.1, 0.8]]).astype(numpy.float64)

    m, v, var = main_test(var_numpy, m_numpy, v_numpy, beta1_power_numpy, lr_numpy, beta1_numpy, beta2_numpy,
                          epsilon_numpy, grident_numpy)

    assert numpy.all(m[0] - m[1] < eps_f64)
    assert numpy.all(v[0] - v[1] < eps_f64)
    assert numpy.all(var[0] - var[1] < eps_f64)
