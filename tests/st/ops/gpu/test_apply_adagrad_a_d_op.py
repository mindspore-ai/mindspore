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
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap


class ApplyAdagradDANet(nn.Cell):
    def __init__(self, data_type, use_locking=False):
        super(ApplyAdagradDANet, self).__init__()
        self.apply_adagrad_d_a = P.ApplyAdagradDA(use_locking)
        self.var = Parameter(Tensor(np.array([[0.6, 0.4], [0.1, 0.5]]).astype(data_type)), name="var")
        self.gradient_accumulator = Parameter(Tensor(np.array([[0.1, 0.3],
                                                               [0.1, 0.5]]).astype(data_type)),
                                              name="gradient_accumulator")
        self.gradient_squared_accumulator = Parameter(Tensor(np.array([[0.2, 0.1],
                                                                       [0.1, 0.2]]).astype(data_type)),
                                                      name="gradient_squared_accumulator")

    def construct(self, grad, lr, l1, l2, global_step):
        out = self.apply_adagrad_d_a(self.var, self.gradient_accumulator,
                                     self.gradient_squared_accumulator, grad, lr, l1, l2, global_step)
        return out


class ApplyAdagradDANetVmap(nn.Cell):
    """ NetVmapWithApplyAdadelta definition """

    def __init__(self, net, data_type):
        super(ApplyAdagradDANetVmap, self).__init__()
        self.net = net
        self.var = Parameter(Tensor(np.array([[0.6, 0.4], [0.1, 0.5]]).astype(data_type)), name="var")
        self.gradient_accumulator = Parameter(Tensor(np.array([[0.1, 0.3],
                                                               [0.1, 0.5]]).astype(data_type)),
                                              name="gradient_accumulator")
        self.gradient_squared_accumulator = Parameter(Tensor(np.array([[0.2, 0.1],
                                                                       [0.1, 0.2]]).astype(data_type)),
                                                      name="gradient_squared_accumulator")

        self.vmap_adagrad_da = vmap(self.net, in_axes=(
            0, 0, 0, 0, None, None, None, None), out_axes=(0, 0, 0))

    def construct(self, grad, lr, l1, l2, global_step):
        return self.vmap_adagrad_da(self.var, self.gradient_accumulator,
                                    self.gradient_squared_accumulator, grad, lr, l1, l2, global_step)


class ApplyAdagradDANetVmap2(nn.Cell):
    """ NetVmapWithApplyAdadelta definition """

    def __init__(self, net, data_type):
        super(ApplyAdagradDANetVmap2, self).__init__()
        self.net = net
        self.var = Parameter(Tensor(np.array([[0.6, 0.4], [0.1, 0.5]]).astype(data_type)), name="var")
        self.gradient_accumulator = Parameter(Tensor(np.array([[0.1, 0.3],
                                                               [0.1, 0.5]]).astype(data_type)),
                                              name="gradient_accumulator")
        self.gradient_squared_accumulator = Parameter(Tensor(np.array([[0.2, 0.1],
                                                                       [0.1, 0.2]]).astype(data_type)),
                                                      name="gradient_squared_accumulator")
        self.vmap2_adagrad_da = vmap(vmap(self.net, in_axes=(0, 0, 0, 0, None, None, None, None), out_axes=(0, 0, 0)),
                                     in_axes=(0, 0, 0, 0, None, None, None, None), out_axes=(0, 0, 0))

    def construct(self, grad, lr, l1, l2, global_step):
        return self.vmap2_adagrad_da(self.var, self.gradient_accumulator,
                                     self.gradient_squared_accumulator, grad, lr, l1, l2, global_step)


def numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type):
    np_var = np.array([[0.6, 0.4], [0.1, 0.5]], dtype=data_type)
    np_grad_accum = np.array([[0.1, 0.3], [0.1, 0.5]], dtype=data_type)
    np_grad_squared_accum = np.array([[0.2, 0.1], [0.1, 0.2]], dtype=data_type)
    np_grad = np.array([[0.3, 0.4], [0.1, 0.2]])

    np_grad_accum += np_grad
    np_grad_squared_accum += np_grad * np_grad
    tmp_val = np.sign(np_grad_accum) * np.maximum(np.abs(np_grad_accum) - np_l1 * np_global_step,
                                                  0) if np_l1 > 0 else np_grad_accum
    x_value = -1 * np_lr * tmp_val
    y_value = np_l2 * np_global_step * np_lr + np.sqrt(np_grad_squared_accum)
    np_var = x_value / y_value

    return np_var, np_grad_accum, np_grad_squared_accum


def ms_forward_impl(grad, np_lr, np_l1, np_l2, np_global_step, data_type):
    grad_ms = Tensor(grad)
    lr_ms = Tensor(np_lr)
    l1_ms = Tensor(np_l1)
    l2_ms = Tensor(np_l2)
    global_step_ms = Tensor(np_global_step)

    adagrad_da = ApplyAdagradDANet(data_type=data_type)
    output = adagrad_da(grad_ms, lr_ms, l1_ms, l2_ms, global_step_ms)
    return output


def ms_forward_impl_vmap(grad, np_lr, np_l1, np_l2, np_global_step, data_type):
    def cal_apply_adagrad_a_d(var, gradient_accumulator, gradient_squared_accumulator, grad, lr, l1, l2, global_step):
        return P.ApplyAdagradDA(use_locking=False)(var, gradient_accumulator, gradient_squared_accumulator,
                                                   grad, lr, l1, l2, global_step)
    grad_ms = Tensor(grad)
    lr_ms = Tensor(np_lr)
    l1_ms = Tensor(np_l1)
    l2_ms = Tensor(np_l2)
    global_step_ms = Tensor(np_global_step)

    vmap_adagrad_da = ApplyAdagradDANetVmap(net=cal_apply_adagrad_a_d, data_type=data_type)
    output = vmap_adagrad_da(grad_ms, lr_ms, l1_ms, l2_ms, global_step_ms)
    return output


def ms_forward_impl_vmap2(grad, np_lr, np_l1, np_l2, np_global_step, data_type):
    def cal_apply_adagrad_a_d(var, gradient_accumulator, gradient_squared_accumulator, grad, lr, l1, l2, global_step):
        return P.ApplyAdagradDA(use_locking=False)(var, gradient_accumulator, gradient_squared_accumulator,
                                                   grad, lr, l1, l2, global_step)
    grad_ms = Tensor(grad)
    lr_ms = Tensor(np_lr)
    l1_ms = Tensor(np_l1)
    l2_ms = Tensor(np_l2)
    global_step_ms = Tensor(np_global_step)

    vmap_adagrad_da = ApplyAdagradDANetVmap2(net=cal_apply_adagrad_a_d, data_type=data_type)
    output = vmap_adagrad_da(grad_ms, lr_ms, l1_ms, l2_ms, global_step_ms)
    return output


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_apply_adagrad_da_float():
    """
    Feature: ApplyAdagradDA GPU kernel
    Description: test the calculation difference between numpy and mindscore in ApplyAdagradDA
    Expectation: success
    """
    # numpy
    error = 1e-6
    np_lr = np.float32(0.001)
    np_l1 = np.float32(0.001)
    np_l2 = np.float32(0.001)
    np_global_step = np.int32(2)
    grad = np.array([[0.3, 0.4], [0.1, 0.2]], dtype=np.float32)
    np_out = numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type=np.float32)
    ms_out = ms_forward_impl(grad, np_lr, np_l1, np_l2, np_global_step, data_type=np.float32)

    for i in range(3):
        np.testing.assert_allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)

    np_global_step = np.int64(2)
    np_out = numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type=np.float32)
    ms_out = ms_forward_impl(grad, np_lr, np_l1, np_l2, np_global_step, data_type=np.float32)
    for i in range(3):
        np.testing.assert_allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_apply_adagrad_da_float16():
    """
    Feature: ApplyAdagradDA GPU kernel
    Description: test the calculation difference between numpy and mindscore in ApplyAdagradDA
    Expectation: success
    """
    # numpy
    error = 1e-6
    np_lr = np.float16(0.001)
    np_l1 = np.float16(0.001)
    np_l2 = np.float16(0.001)
    np_global_step = np.int32(2)
    grad = np.array([[0.3, 0.4], [0.1, 0.2]], dtype=np.float16)

    np_out = numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type=np.float16)
    ms_out = ms_forward_impl(grad, np_lr, np_l1, np_l2, np_global_step, data_type=np.float16)

    for i in range(3):
        np.allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)
        np.testing.assert_allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)

    np_global_step = np.int64(2)
    np_out = numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type=np.float16)
    ms_out = ms_forward_impl(grad, np_lr, np_l1, np_l2, np_global_step, data_type=np.float16)
    for i in range(3):
        np.testing.assert_allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_apply_adagrad_da_float16_vmap():
    """
    Feature: ApplyAdagradDA vmap test on GPU
    Description: test the rightness of basic ApplyAdagradDA vmap
    Expectation: success
    """
    # numpy
    error = 1e-6
    np_lr = np.float16(0.001)
    np_l1 = np.float16(0.001)
    np_l2 = np.float16(0.001)
    np_global_step = np.int32(2)
    grad = np.array([[0.3, 0.4], [0.1, 0.2]], dtype=np.float16)

    np_out = numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type=np.float16)
    #MindSpore
    ms_out = ms_forward_impl_vmap(grad, np_lr, np_l1, np_l2, np_global_step, data_type=np.float16)
    for i in range(3):
        np.testing.assert_allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)

    np_global_step = np.int64(2)
    np_out = numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type=np.float16)
    ms_out = ms_forward_impl_vmap(grad, np_lr, np_l1, np_l2, np_global_step, data_type=np.float16)
    for i in range(3):
        np.testing.assert_allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_apply_adagrad_da_float_vmap():
    """
    Feature: ApplyAdagradDA vmap test on GPU
    Description: test the rightness of basic ApplyAdagradDA vmap
    Expectation: success
    """
    # numpy
    error = 1e-6
    grad = np.array([[0.3, 0.4], [0.1, 0.2]], dtype=np.float32)
    np_lr = np.float32(0.001)
    np_l1 = np.float32(0.001)
    np_l2 = np.float32(0.001)
    np_global_step = np.int32(2)
    np_out = numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type=np.float32)
    # MindSpore
    ms_out = ms_forward_impl_vmap(grad, np_lr, np_l1, np_l2, np_global_step, data_type=np.float32)
    for i in range(3):
        np.testing.assert_allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)

    np_global_step = np.int64(2)
    np_out = numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type=np.float32)
    ms_out = ms_forward_impl_vmap(grad, np_lr, np_l1, np_l2, np_global_step, data_type=np.float32)
    for i in range(3):
        np.testing.assert_allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_apply_adagrad_da_float16_vmap2():
    """
    Feature: ApplyAdagradDA vmap test on GPU
    Description: test the rightness of basic ApplyAdagradDA vmap
    Expectation: success
    """
    # numpy
    error = 1e-6
    grad = np.array([[0.3, 0.4], [0.1, 0.2]], dtype=np.float16)
    np_lr = np.float16(0.001)
    np_l1 = np.float16(0.001)
    np_l2 = np.float16(0.001)
    np_global_step = np.int32(2)

    np_out = numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type=np.float16)
    #MindSpore
    ms_out = ms_forward_impl_vmap2(grad, np_lr, np_l1, np_l2, np_global_step, data_type=np.float16)
    for i in range(3):
        np.testing.assert_allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)

    np_global_step = np.int64(2)
    np_out = numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type=np.float16)
    ms_out = ms_forward_impl_vmap2(grad, np_lr, np_l1, np_l2, np_global_step, data_type=np.float16)
    for i in range(3):
        np.testing.assert_allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_apply_adagrad_da_float_vmap2():
    """
    Feature: ApplyAdagradDA vmap test on GPU
    Description: test the rightness of basic ApplyAdagradDA vmap
    Expectation: success
    """
    # numpy
    error = 1e-6
    grad = np.array([[0.3, 0.4], [0.1, 0.2]], dtype=np.float32)
    np_lr = np.float32(0.001)
    np_l1 = np.float32(0.001)
    np_l2 = np.float32(0.001)
    np_global_step = np.int32(2)
    np_out = numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type=np.float32)
    # MindSpore
    ms_out = ms_forward_impl_vmap2(grad, np_lr, np_l1, np_l2, np_global_step, data_type=np.float32)
    for i in range(3):
        np.testing.assert_allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)

    np_global_step = np.int64(2)
    np_out = numpy_impl(np_lr, np_l1, np_l2, np_global_step, data_type=np.float32)
    ms_out = ms_forward_impl_vmap(grad, np_lr, np_l1, np_l2, np_global_step, data_type=np.float32)
    for i in range(3):
        np.testing.assert_allclose(np_out[i], ms_out[i].asnumpy(), rtol=error, atol=error)
