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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

var_np = np.random.rand(3, 3).astype(np.float32)
accum_np = np.random.rand(3, 3).astype(np.float32)


class Net(nn.Cell):
    def __init__(self, user_eps):
        super(Net, self).__init__()
        self.apply_adagrad_v2 = P.ApplyAdagradV2(epsilon=user_eps)
        self.var = Parameter(Tensor(var_np), name="var")
        self.accum = Parameter(Tensor(accum_np), name="accum")

    def construct(self, lr, grad):
        return self.apply_adagrad_v2(self.var, self.accum, lr, grad)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_apply_adagrad_v2():
    """
    Feature: Test the ApplyAdagradV2 CPU operation
    Description: Recreate the operation in numpy, and compare the result with the mindspore version.
    Expectation: Both the numpy and mindspore version of the operation should produce the same result.
    """
    # numpy op
    v2_eps = 1e-6
    gradient_np = np.random.rand(3, 3).astype(np.float32)
    expect_accum_np = accum_np + gradient_np * gradient_np
    np_dividend = (np.sqrt(expect_accum_np) + v2_eps)
    #update zero values to avoid division-by-zero
    np_dividend[np_dividend == 0] = 1e-6
    expect_var_np = var_np - (0.001 * gradient_np * (1 / np_dividend))

    net = Net(user_eps=v2_eps)
    lr = Tensor(0.001, mstype.float32)
    grad = Tensor(gradient_np)
    out = net(lr, grad)
    res_var_mindspore = out[0].asnumpy()
    res_accum_mindspore = out[1].asnumpy()
    eps = np.array([1e-6 for i in range(9)]).reshape(3, 3)

    assert np.all(np.abs(expect_var_np - res_var_mindspore) < eps)
    assert np.all(np.abs(expect_accum_np - res_accum_mindspore) < eps)


class VmapNet(nn.Cell):
    def __init__(self):
        super(VmapNet, self).__init__()
        self.apply_gradient_descent = P.ApplyAdagradV2(1e-6)

    def construct(self, var, accum, lr, grad):
        return self.apply_gradient_descent(var, accum, lr, grad)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_vmap_apply_adagrad_v2():
    """
    Feature: ApplyAdagradV2 cpu op vmap.
    Description: test vmap feature for ApplyAdagradV2 cpu op.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_array = np.array([[0.59021956, 0.6402413, 0.26803252],
                          [0.64904916, 0.5892827, 0.6982026],
                          [0.4472826, 0.4589515, 0.29412633]])
    accum_array = np.array([[0.59021956, 0.6402413, 0.26803252],
                            [0.64904916, 0.5892827, 0.6982026],
                            [0.4472826, 0.4589515, 0.29412633]])
    var = Parameter(var_array, name="var")
    accum = Parameter(accum_array, name="accum")
    lr = Tensor([0.001, 0.1, 3], mstype.float32)
    grad = Tensor(np.arange(9).reshape(3, 3).astype(np.float16))
    net = VmapNet()
    expect_var = np.array([[0.59021956, 0.6394605, 0.26706442],
                           [0.55247104, 0.49107486, 0.59957045],
                           [-2.53425217, -2.52709651, -2.69900346]])
    expect_accum = np.array([[0.59021956, 1.64024138, 4.26803255],
                             [9.64904881, 16.58928299, 25.69820213],
                             [36.44728088, 49.45895004, 64.29412842]])
    [vmap_var, vmap_accum] = F.vmap(net, in_axes=(0, 0, 0, 0))(var, accum, lr, grad)
    error_var = np.ones(shape=expect_var.shape) * 1.0e-6
    error_accum = np.ones(shape=expect_accum.shape) * 1.0e-6
    assert np.all(abs(vmap_var.asnumpy() - expect_var) < error_var)
    assert np.all(abs(vmap_accum.asnumpy() - expect_accum) < error_accum)
