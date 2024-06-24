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
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class ApplyFtrlTEST(nn.Cell):
    def __init__(self, data_type):
        super(ApplyFtrlTEST, self).__init__()
        self.apply_ftrl = P.ApplyFtrl()
        self.var_np = np.array([[0.6, 0.4], [0.1, 0.5]]).astype(data_type)
        self.var = Parameter(Tensor(self.var_np), name="var")
        self.accum_np = np.array([[0.6, 0.5], [0.2, 0.6]]).astype(data_type)
        self.accum = Parameter(Tensor(self.accum_np), name="accum")
        self.linear_np = np.array([[0.9, 0.1], [0.7, 0.8]]).astype(data_type)
        self.linear = Parameter(Tensor(self.linear_np), name="linear")

    def construct(self, grad, lr, l1, l2, lr_power):
        return self.apply_ftrl(self.var, self.accum, self.linear, grad, lr, l1, l2, lr_power)


def numpy_apply_ftrl(var, accum, linear, grad, lr=0.001, l1=0.0, l2=0.0, lr_power=-0.5):
    accum_update = accum + grad * grad
    accum_update_power = accum_update ** (-lr_power)
    accum_power = accum ** (-lr_power)
    sigma = (accum_update_power - accum_power) / lr
    linear_update = linear + grad - sigma * var
    member = np.sign(linear_update) * l1 - linear_update
    denominator = accum_update_power / lr + 2 * l2
    expect_var_np_tmp = member / denominator
    expected_out = np.where(np.abs(linear_update) > l1, expect_var_np_tmp, 0.0)
    return expected_out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_apply_ftrl(data_type):
    """
    Feature: ApplyFtrl cpu kernel.
    Description: test the ApplyFtrl.
    Expectation: match to np benchmark.
    """
    net = ApplyFtrlTEST(data_type)
    error = 1e-4
    if data_type == np.float16:
        error = 1e-3

    lr = 0.001
    l1 = 0.0
    l2 = 0.0
    lr_power = -0.5
    grad_np = np.array([[0.3, 0.7], [0.1, 0.8]]).astype(data_type)
    grad = Tensor(grad_np)

    context.set_context(mode=context.GRAPH_MODE)
    output = net(grad, lr, l1, l2, lr_power)
    mindspore_var_out = net.var.asnumpy()
    expect_var = numpy_apply_ftrl(net.var_np, net.accum_np, net.linear_np, grad_np)

    np.testing.assert_allclose(
        mindspore_var_out, expect_var, rtol=error, atol=error)
    np.testing.assert_allclose(
        output.asnumpy(), expect_var, rtol=error, atol=error)


class FtrlNetVmap(nn.Cell):
    def __init__(self, net):
        super(FtrlNetVmap, self).__init__()
        self.net = net
        self.var_np = np.array([[[0.6, 0.4], [0.1, 0.5]], [[1.6, 1.4], [1.1, 1.5]]]).astype(np.float32)
        self.accum_np = np.array([[[0.6, 0.5], [0.2, 0.6]], [[1.6, 1.5], [1.2, 1.6]]]).astype(np.float32)
        self.linear_np = np.array([[[0.9, 0.1], [0.7, 0.8]], [[1.9, 1.1], [1.7, 1.8]]]).astype(np.float32)
        self.var = Parameter(Tensor(self.var_np), name="var")
        self.accum = Parameter(Tensor(self.accum_np), name="accum")
        self.linear = Parameter(Tensor(self.linear_np), name="linear")
        self.vmap_ftrl = vmap(self.net, in_axes=(
            0, 0, 0, 0, None, None, None, None), out_axes=0)

    def construct(self, grad, lr, l1, l2, lr_power):
        return self.vmap_ftrl(self.var, self.accum, self.linear, grad, lr, l1, l2, lr_power)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_apply_ftrl_op_vmap():
    """
    Feature: ApplyFtrl cpu kernel
    Description: test the ApplyFtrl vmap.
    Expectation: match to np benchmark.
    """

    def cal_ftrl(var, accum, linear, grad, lr, l1, l2, lr_power):
        return P.ApplyFtrl()(var, accum, linear, grad, lr, l1, l2, lr_power)

    error = 1e-4
    grad_np = np.array([[[0.3, 0.7], [0.1, 0.8]], [[0.3, 0.7], [0.1, 0.8]]]).astype(np.float32)
    grad = Tensor(grad_np)

    vmap_ftrl = FtrlNetVmap(cal_ftrl)
    lr = 0.001
    l1 = 0.0
    l2 = 0.0
    lr_power = -0.5
    _ = vmap_ftrl(grad, Tensor(lr), Tensor(l1), Tensor(l2), Tensor(lr_power))
    ms_var = vmap_ftrl.var.asnumpy()
    np_var = numpy_apply_ftrl(
        vmap_ftrl.var_np, vmap_ftrl.accum_np, vmap_ftrl.linear_np, grad_np)

    np.testing.assert_allclose(ms_var, np_var, rtol=error, atol=error)


class FtrlNetVmap2(nn.Cell):
    def __init__(self, net):
        super(FtrlNetVmap2, self).__init__()
        self.net = net
        self.var_np = np.array([[[[0.6, 0.4], [0.1, 0.5]], [[1.6, 0.4], [0.1, 1.5]]],
                                [[[0.6, 1.4], [1.1, 0.5]], [[2.6, 0.4], [0.1, 3.5]]]]).astype(np.float32)
        self.accum_np = np.array([[[[0.6, 0.5], [0.2, 0.6]], [[1.6, 0.5], [0.2, 1.6]]],
                                  [[[0.6, 1.5], [1.2, 0.6]], [[2.6, 0.5], [0.2, 3.6]]]]).astype(np.float32)
        self.linear_np = np.array([[[[0.9, 0.1], [0.7, 0.8]], [[1.9, 0.1], [0.7, 1.8]]],
                                   [[[0.9, 1.1], [1.7, 0.8]], [[3.9, 0.1], [0.7, 3.8]]]]).astype(np.float32)
        self.var = Parameter(Tensor(self.var_np), name="var")
        self.accum = Parameter(Tensor(self.accum_np), name="accum")
        self.linear = Parameter(Tensor(self.linear_np), name="linear")
        self.vmap_ftrl = vmap(vmap(self.net, in_axes=(0, 0, 0, 0, None, None, None, None), out_axes=0),
                              in_axes=(0, 0, 0, 0, None, None, None, None), out_axes=0)

    def construct(self, grad, lr, l1, l2, lr_power):
        return self.vmap_ftrl(self.var, self.accum, self.linear, grad, lr, l1, l2, lr_power)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_apply_ftrl_op_vmap2():
    """
    Feature: ApplyFtrl cpu kernel
    Description: test the ApplyFtrl vmap.
    Expectation: match to np benchmark.
    """

    def cal_ftrl(var, accum, linear, grad, lr, l1, l2, lr_power):
        return P.ApplyFtrl()(var, accum, linear, grad, lr, l1, l2, lr_power)

    error = 1e-4
    grad_np = np.array([[[[0.3, 0.7], [0.1, 0.8]], [[0.3, 0.7], [0.1, 0.8]]],
                        [[[0.3, 0.7], [0.1, 0.8]], [[0.3, 0.7], [0.1, 0.8]]]]).astype(np.float32)
    grad = Tensor(grad_np)

    vmap_ftrl = FtrlNetVmap2(cal_ftrl)
    lr = 0.001
    l1 = 0.0
    l2 = 0.0
    lr_power = -0.5
    _ = vmap_ftrl(grad, Tensor(lr), Tensor(l1), Tensor(l2), Tensor(lr_power))
    ms_var = vmap_ftrl.var.asnumpy()
    np_var = numpy_apply_ftrl(
        vmap_ftrl.var_np, vmap_ftrl.accum_np, vmap_ftrl.linear_np, grad_np)

    np.testing.assert_allclose(ms_var, np_var, rtol=error, atol=error)
