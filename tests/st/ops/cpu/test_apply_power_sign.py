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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
import mindspore.ops as ops
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.apply_power_sign = ops.ApplyPowerSign()
        self.var = Parameter(Tensor(np.array([[0.6, 0.4],
                                              [0.1, 0.5]]).astype(np.float32)), name="var")
        self.m = Parameter(Tensor(np.array([[0.6, 0.5],
                                            [0.2, 0.6]]).astype(np.float32)), name="m")
        self.lr = 0.001
        self.logbase = np.e
        self.sign_decay = 0.99
        self.beta = 0.9

    def construct(self, grad):
        out = self.apply_power_sign(self.var, self.m, self.lr, self.logbase,
                                    self.sign_decay, self.beta, grad)
        return out


def test_apply_power_assign():
    """
    Feature: test ops ApplyPowerSign.
    Description: Update var and m by ApplyPowerSign op.
    Expectation: match to expected benchmark output.
    """
    grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
    net = Net()
    net(grad)
    expect_var = [[5.95575690e-01, 3.89676481e-01],
                  [9.85252112e-02, 4.88201708e-01]]
    expect_m = [[5.70000052e-01, 5.19999981e-01],
                [1.89999998e-01, 6.20000064e-01]]
    assert np.allclose(net.var.asnumpy(), expect_var, atol=0.0001, rtol=0.0001, equal_nan=True)
    assert np.allclose(net.m.asnumpy(), expect_m, atol=0.0001, rtol=0.0001, equal_nan=True)


class PowerSignNetVmap(nn.Cell):
    def __init__(self, net):
        super(PowerSignNetVmap, self).__init__()
        self.net = net
        self.var = Parameter(
            Tensor(np.array([[[0.6, 0.4], [0.1, 0.5]], [[0.6, 0.4], [0.1, 0.5]]]).astype(np.float32)), name="var")
        self.m = Parameter(
            Tensor(np.array([[[0.6, 0.5], [0.2, 0.6]], [[0.6, 0.5], [0.2, 0.6]]]).astype(np.float32)), name="m")
        self.vmap_grad = vmap(self.net, in_axes=(0, 0, 0, None, None, None, 0), out_axes=0)

    def construct(self, lr, logbase, sign_decay, beta, grad):
        return self.vmap_grad(self.var, self.m, lr, logbase, sign_decay, beta, grad)


def test_apply_power_sign_op_vmap():
    """
    Feature: ApplyPowerSign cpu kernel
    Description: test the ApplyPowerSign vmap.
    Expectation: match to expected benchmark output.
    """
    def cal_grad(var, m, lr, logbase, sign_decay, beta, grad):
        return ops.ApplyPowerSign()(var, m, lr, logbase, sign_decay, beta, grad)
    error = 1e-3
    grad = Tensor(np.array([[[0.3, 0.7], [0.1, 0.8]],
                            [[0.3, 0.7], [0.1, 0.8]]]).astype(np.float32))

    lr = Tensor(np.array([0.01, 0.01]).astype(np.float32))
    logbase = np.e
    sign_decay = 0.99
    beta = 0.9

    vmap_agrad = PowerSignNetVmap(cal_grad)
    output = vmap_agrad(lr, logbase, sign_decay, beta, grad)
    mindspore_var_out = output[0].asnumpy()
    mindspore_m_out = output[1].asnumpy()

    expect_var = np.array([[[0.5557564, 0.29676488], [0.08525213, 0.38201702]],
                           [[0.5557564, 0.29676488], [0.08525213, 0.38201702]]]).astype(np.float32)

    expect_m = np.array([[[0.57, 0.52], [0.19, 0.62]],
                         [[0.57, 0.52], [0.19, 0.62]]]).astype(np.float32)

    np.testing.assert_allclose(mindspore_var_out, expect_var, rtol=error)
    np.testing.assert_allclose(mindspore_m_out, expect_m, rtol=error)


class PowerSignNetVmap2(nn.Cell):
    def __init__(self, net):
        super(PowerSignNetVmap2, self).__init__()
        self.net = net
        self.var = Parameter(
            Tensor(np.array([[[[0.6, 0.4], [0.1, 0.5]], [[0.7, 0.4], [0.1, 0.5]]],
                             [[[0.8, 0.4], [0.1, 0.5]], [[0.9, 0.4], [0.1, 0.5]]]]).astype(np.float32)), name="var")
        self.m = Parameter(
            Tensor(np.array([[[[0.6, 0.5], [0.2, 0.6]], [[0.7, 0.5], [0.2, 0.6]]],
                             [[[0.8, 0.5], [0.2, 0.6]], [[0.9, 0.5], [0.2, 0.6]]]]).astype(np.float32)), name="m")
        self.vmap_grad = vmap(vmap(self.net, in_axes=(
            0, 0, 0, None, 0, None, 0), out_axes=0), in_axes=(0, 0, 0, None, None, None, 0), out_axes=0)

    def construct(self, lr, logbase, sign_decay, beta, grad):
        return self.vmap_grad(self.var, self.m, lr, logbase, sign_decay, beta, grad)


def test_apply_power_sign_op_vmap2():
    """
    Feature: ApplyPowerSign cpu kernel
    Description: test the ApplyPowerSign vmap.
    Expectation: match to expected benchmark output.
    """
    def cal_grad(var, m, lr, logbase, sign_decay, beta, grad):
        return ops.ApplyPowerSign()(var, m, lr, logbase, sign_decay, beta, grad)
    error = 1e-3
    grad = Tensor(np.array([[[[0.3, 0.7], [0.1, 0.8]], [[0.3, 0.7], [0.1, 0.8]]],
                            [[[0.3, 0.7], [0.1, 0.8]], [[0.3, 0.7], [0.1, 0.8]]]]).astype(np.float32))
    lr = Tensor(np.array([[0.01, 0.02], [0.03, 0.04]]).astype(np.float32))
    logbase = np.e
    sign_decay = Tensor(np.array([0.99, 0.9]).astype(np.float32))
    beta = 0.9

    vmap_agrad = PowerSignNetVmap2(cal_grad)
    output = vmap_agrad(lr, logbase, sign_decay, beta, grad)

    mindspore_var_out = output[0].asnumpy()
    mindspore_m_out = output[1].asnumpy()

    expect_var = np.array([[[[0.5557564, 0.29676488], [0.08525213, 0.38201702]],
                            [[0.630716, 0.2383375], [0.07690535, 0.31524283]]],
                           [[[0.6672691, 0.09029466], [0.05575638, 0.14605102]],
                            [[0.7614321, 0.076675], [0.05381071, 0.13048568]]]]).astype(np.float32)

    expect_m = np.array([[[[0.57, 0.52], [0.19, 0.62]],
                          [[0.66, 0.52], [0.19, 0.62]]],
                         [[[0.75, 0.52], [0.19, 0.62]],
                          [[0.84, 0.52], [0.19, 0.62]]]]).astype(np.float32)

    np.testing.assert_allclose(mindspore_var_out, expect_var, rtol=error)
    np.testing.assert_allclose(mindspore_m_out, expect_m, rtol=error)
