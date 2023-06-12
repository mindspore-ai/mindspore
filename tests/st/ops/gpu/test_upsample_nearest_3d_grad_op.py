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
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops.operations import _grad_ops as G


class NetUpsampleNearest3DGrad(nn.Cell):

    def __init__(self):
        super(NetUpsampleNearest3DGrad, self).__init__()
        self.upsample_nearest_3d_grad = G.UpsampleNearest3DGrad()

    def construct(self, grad, input_size, output_size, scales):
        return self.upsample_nearest_3d_grad(grad, input_size, output_size,
                                             scales)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_upsample_nearest_3d_grad_output_size_fp32():
    """
    Feature: UpsampleNearest3DGrad
    Description: Test cases for UpsampleNearest3DGrad operator with output_size.
    Expectation: The result matches expected output.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_shape = (2, 2, 2, 2, 2)
    grad_shape = (2, 2, 4, 4, 5)
    grad = Tensor(
        np.arange(np.prod(grad_shape)).reshape(grad_shape).astype(np.float32))
    expect_x = Tensor(
        np.array([[[[[162, 128], [282, 208]], [[642, 448], [762, 528]]],
                   [[[1122, 768], [1242, 848]], [[1602, 1088], [1722, 1168]]]],
                  [[[[2082, 1408], [2202, 1488]], [[2562, 1728], [2682,
                                                                  1808]]],
                   [[[3042, 2048], [3162, 2128]],
                    [[3522, 2368], [3642, 2448]]]]]).astype(np.float32))
    error_x = np.ones(shape=expect_x.shape) * 1.0e-5
    upsample_trilinear_3d_grad = NetUpsampleNearest3DGrad()
    output = upsample_trilinear_3d_grad(grad, input_shape, [4, 4, 5], None)
    diff_x = output.asnumpy() - expect_x
    assert np.all(np.abs(diff_x) < error_x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_upsample_nearest_3d_grad_scales_fp16():
    """
    Feature: UpsampleNearest3DGrad
    Description: Test cases for UpsampleNearest3DGrad operator with output_size.
    Expectation: The result matches expected output.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_shape = (2, 2, 2, 2, 2)
    grad_shape = (2, 2, 4, 4, 5)
    grad = Tensor(
        np.arange(np.prod(grad_shape)).reshape(grad_shape).astype(np.float16))
    expect_x = Tensor(
        np.array([[[[[162, 128], [282, 208]], [[642, 448], [762, 528]]],
                   [[[1122, 768], [1242, 848]], [[1602, 1088], [1722, 1168]]]],
                  [[[[2082, 1408], [2202, 1488]], [[2562, 1728], [2682,
                                                                  1808]]],
                   [[[3042, 2048], [3162, 2128]],
                    [[3522, 2368], [3642, 2448]]]]]).astype(np.float16))
    error_x = np.ones(shape=expect_x.shape) * 1.0e-5
    upsample_trilinear_3d_grad = NetUpsampleNearest3DGrad()
    output = upsample_trilinear_3d_grad(grad, input_shape, None,
                                        [2.0, 2.0, 2.5])
    diff_x = output.asnumpy() - expect_x
    assert np.all(np.abs(diff_x) < error_x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_upsample_nearest_3d_error():
    """
    Feature: UpsampleNearest3D
    Description: Test cases for UpsampleNearest3D operator with errors.
    Expectation: Raise expected error type.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    with pytest.raises(ValueError):
        grad_tensor = Tensor(np.ones((2, 2, 2, 2), dtype=np.float32))
        net = NetUpsampleNearest3DGrad()
        net(grad_tensor, (2, 2, 2, 2, 2), [3, 4, 5], None)

    with pytest.raises(ValueError):
        grad_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = NetUpsampleNearest3DGrad()
        net(grad_tensor, (2, 2, 2, 2), [3, 4, 5], None)

    with pytest.raises(TypeError):
        grad_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.int32))
        net = NetUpsampleNearest3DGrad()
        net(grad_tensor, (2, 2, 2, 2, 2), [3, 4, 5], None)

    with pytest.raises(TypeError):
        grad_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = NetUpsampleNearest3DGrad()
        net(grad_tensor, (2, 2, 2, 2, 2), None, [1, 2, 3])

    with pytest.raises(ValueError):
        grad_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = NetUpsampleNearest3DGrad()
        net(grad_tensor, (2, 2, 2, 2, 2), [3, 4], None)

    with pytest.raises(ValueError):
        grad_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = NetUpsampleNearest3DGrad()
        net(grad_tensor, (2, 2, 2, 2, 2), None, [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError):
        grad_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = NetUpsampleNearest3DGrad()
        net(grad_tensor, (2, 2, 2, 2, 2), [3, 4, 5], [1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        grad_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = NetUpsampleNearest3DGrad()
        net(grad_tensor, (2, 2, 2, 2, 2), None, None)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vmap_upsample_nearest3d_grad():
    """
    Feature:  UpsampleNearest3DGrad GPU op vmap feature.
    Description: test the vmap feature of UpsampleNearest3DGrad.
    Expectation: success.
    """
    # 3 batches
    input_shape = (1, 1, 2, 2, 2)
    input_tensor = Tensor(
        np.arange(0, 7.2, 0.1).reshape((3, 1, 1, 2, 3, 4)).astype(np.float32))
    net = NetUpsampleNearest3DGrad()
    expect = np.array([[[[[[1.0, 1.8], [1.7, 2.1]], [[5.8, 6.6], [4.1,
                                                                  4.5]]]]],
                       [[[[[10.6, 11.4], [6.5, 6.9]],
                          [[15.4, 16.2], [8.9, 9.299999]]]]],
                       [[[[[20.2, 21.], [11.299999, 11.700001]],
                          [[25.0, 25.8], [13.700001,
                                          14.1]]]]]]).astype(np.float32)
    out_vmap = F.vmap(net,
                      in_axes=(0, None, None, None))(input_tensor, input_shape,
                                                     [2, 3, 4], None)
    error = np.ones(shape=expect.shape) * 1.0e-6
    assert np.all(abs(out_vmap.asnumpy() - expect) < error)
