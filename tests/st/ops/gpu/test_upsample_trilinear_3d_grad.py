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


class NetUpsampleTrilinear3DGrad(nn.Cell):
    def __init__(self, input_size, output_size=None, scales=None, align_corners=None):
        super(NetUpsampleTrilinear3DGrad, self).__init__()
        self.upsample_trilinear_3d_grad = G.UpsampleTrilinear3DGrad(input_size=input_size,
                                                                    output_size=output_size,
                                                                    scales=scales,
                                                                    align_corners=align_corners)

    def construct(self, grad):
        return self.upsample_trilinear_3d_grad(grad)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
def test_upsample_trilinear_3d_grad_main(data_type):
    """
    Feature: UpsampleTrilinear3DGrad
    Description: Test cases for UpsampleTrilinear3DGrad operator with output_size.
    Expectation: The result matches expected output.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    grad = Tensor(np.array([[[[[0.0000, 0.0010, 0.0039, 0.0049],
                               [0.0000, 0.0010, 0.0039, 0.0049],
                               [0.0039, 0.0049, 0.0078, 0.0088],
                               [0.0039, 0.0049, 0.0078, 0.0088]],
                              [[-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0039, -0.0029, 0.0000, 0.0010],
                               [-0.0039, -0.0029, 0.0000, 0.0010]],
                              [[-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0039, -0.0029, 0.0000, 0.0010],
                               [-0.0039, -0.0029, 0.0000, 0.0010]],
                              [[-0.0156, -0.0146, -0.0117, -0.0107],
                               [-0.0156, -0.0146, -0.0117, -0.0107],
                               [-0.0117, -0.0107, -0.0078, -0.0068],
                               [-0.0117, -0.0107, -0.0078, -0.0068]]],
                             [[[0.0000, 0.0010, 0.0039, 0.0049],
                               [0.0000, 0.0010, 0.0039, 0.0049],
                               [0.0039, 0.0049, 0.0078, 0.0088],
                               [0.0039, 0.0049, 0.0078, 0.0088]],
                              [[-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0039, -0.0029, 0.0000, 0.0010],
                               [-0.0039, -0.0029, 0.0000, 0.0010]],
                              [[-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0039, -0.0029, 0.0000, 0.0010],
                               [-0.0039, -0.0029, 0.0000, 0.0010]],
                              [[-0.0156, -0.0146, -0.0117, -0.0107],
                               [-0.0156, -0.0146, -0.0117, -0.0107],
                               [-0.0117, -0.0107, -0.0078, -0.0068],
                               [-0.0117, -0.0107, -0.0078, -0.0068]]]],
                            [[[[0.0000, 0.0010, 0.0039, 0.0049],
                               [0.0000, 0.0010, 0.0039, 0.0049],
                               [0.0039, 0.0049, 0.0078, 0.0088],
                               [0.0039, 0.0049, 0.0078, 0.0088]],
                              [[-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0039, -0.0029, 0.0000, 0.0010],
                               [-0.0039, -0.0029, 0.0000, 0.0010]],
                              [[-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0039, -0.0029, 0.0000, 0.0010],
                               [-0.0039, -0.0029, 0.0000, 0.0010]],
                              [[-0.0156, -0.0146, -0.0117, -0.0107],
                               [-0.0156, -0.0146, -0.0117, -0.0107],
                               [-0.0117, -0.0107, -0.0078, -0.0068],
                               [-0.0117, -0.0107, -0.0078, -0.0068]]],
                             [[[0.0000, 0.0010, 0.0039, 0.0049],
                               [0.0000, 0.0010, 0.0039, 0.0049],
                               [0.0039, 0.0049, 0.0078, 0.0088],
                               [0.0039, 0.0049, 0.0078, 0.0088]],
                              [[-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0039, -0.0029, 0.0000, 0.0010],
                               [-0.0039, -0.0029, 0.0000, 0.0010]],
                              [[-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0078, -0.0068, -0.0039, -0.0029],
                               [-0.0039, -0.0029, 0.0000, 0.0010],
                               [-0.0039, -0.0029, 0.0000, 0.0010]],
                              [[-0.0156, -0.0146, -0.0117, -0.0107],
                               [-0.0156, -0.0146, -0.0117, -0.0107],
                               [-0.0117, -0.0107, -0.0078, -0.0068],
                               [-0.0117, -0.0107, -0.0078, -0.0068]]]]]).astype(data_type))
    input_shape = (2, 2, 2, 2, 2)
    expect16 = Tensor(np.array([[[[[-0.0205, 0.0049],
                                   [0.0029, 0.0283]],
                                  [[-0.0830, -0.0576],
                                   [-0.0596, -0.0342]]],
                                 [[[-0.0205, 0.0049],
                                   [0.0029, 0.0283]],
                                  [[-0.0830, -0.0576],
                                   [-0.0596, -0.0342]]]],
                                [[[[-0.0205, 0.0049],
                                   [0.0029, 0.0283]],
                                  [[-0.0830, -0.0576],
                                   [-0.0596, -0.0342]]],
                                 [[[-0.0205, 0.0049],
                                   [0.0029, 0.0283]],
                                  [[-0.0830, -0.0576],
                                   [-0.0596, -0.0342]]]]]).astype(np.float16))
    expect32 = Tensor(np.array([[[[[-0.0205, 0.0049],
                                   [0.0029, 0.0283]],
                                  [[-0.0830, -0.0576],
                                   [-0.0596, -0.0342]]],
                                 [[[-0.0205, 0.0049],
                                   [0.0029, 0.0283]],
                                  [[-0.0830, -0.0576],
                                   [-0.0596, -0.0342]]]],
                                [[[[-0.0205, 0.0049],
                                   [0.0029, 0.0283]],
                                  [[-0.0830, -0.0576],
                                   [-0.0596, -0.0342]]],
                                 [[[-0.0205, 0.0049],
                                   [0.0029, 0.0283]],
                                  [[-0.0830, -0.0576],
                                   [-0.0596, -0.0342]]]]]).astype(np.float32))
    error = np.ones(shape=expect16.shape) * 1.0e-3
    upsample_trilinear_3d_grad = NetUpsampleTrilinear3DGrad(input_shape, output_size=[4, 4, 4], align_corners=False)
    output = upsample_trilinear_3d_grad(grad)
    if data_type == np.float32:
        diff = abs(output.asnumpy() - expect32)
    else:
        diff = abs(output.asnumpy() - expect16)
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
def test_upsample_trilinear_3d_grad_align_corners(data_type):
    """
    Feature: UpsampleTrilinear3DGrad
    Description: Test cases for UpsampleTrilinear3DGrad operator with output_size.
    Expectation: The result matches expected output.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    grad = Tensor(np.array([[[[[0.0000, 0.0016, 0.0033, 0.0049],
                               [0.0013, 0.0029, 0.0046, 0.0062],
                               [0.0026, 0.0042, 0.0059, 0.0075],
                               [0.0039, 0.0055, 0.0072, 0.0088]],
                              [[-0.0052, -0.0036, -0.0020, -0.0003],
                               [-0.0039, -0.0023, -0.0007, 0.0010],
                               [-0.0026, -0.0010, 0.0007, 0.0023],
                               [-0.0013, 0.0003, 0.0020, 0.0036]],
                              [[-0.0104, -0.0088, -0.0072, -0.0055],
                               [-0.0091, -0.0075, -0.0059, -0.0042],
                               [-0.0078, -0.0062, -0.0046, -0.0029],
                               [-0.0065, -0.0049, -0.0033, -0.0016]],
                              [[-0.0156, -0.0140, -0.0124, -0.0107],
                               [-0.0143, -0.0127, -0.0111, -0.0094],
                               [-0.0130, -0.0114, -0.0098, -0.0081],
                               [-0.0117, -0.0101, -0.0085, -0.0068]]],
                             [[[0.0000, 0.0016, 0.0033, 0.0049],
                               [0.0013, 0.0029, 0.0046, 0.0062],
                               [0.0026, 0.0042, 0.0059, 0.0075],
                               [0.0039, 0.0055, 0.0072, 0.0088]],
                              [[-0.0052, -0.0036, -0.0020, -0.0003],
                               [-0.0039, -0.0023, -0.0007, 0.0010],
                               [-0.0026, -0.0010, 0.0007, 0.0023],
                               [-0.0013, 0.0003, 0.0020, 0.0036]],
                              [[-0.0104, -0.0088, -0.0072, -0.0055],
                               [-0.0091, -0.0075, -0.0059, -0.0042],
                               [-0.0078, -0.0062, -0.0046, -0.0029],
                               [-0.0065, -0.0049, -0.0033, -0.0016]],
                              [[-0.0156, -0.0140, -0.0124, -0.0107],
                               [-0.0143, -0.0127, -0.0111, -0.0094],
                               [-0.0130, -0.0114, -0.0098, -0.0081],
                               [-0.0117, -0.0101, -0.0085, -0.0068]]]],
                            [[[[0.0000, 0.0016, 0.0033, 0.0049],
                               [0.0013, 0.0029, 0.0046, 0.0062],
                               [0.0026, 0.0042, 0.0059, 0.0075],
                               [0.0039, 0.0055, 0.0072, 0.0088]],
                              [[-0.0052, -0.0036, -0.0020, -0.0003],
                               [-0.0039, -0.0023, -0.0007, 0.0010],
                               [-0.0026, -0.0010, 0.0007, 0.0023],
                               [-0.0013, 0.0003, 0.0020, 0.0036]],
                              [[-0.0104, -0.0088, -0.0072, -0.0055],
                               [-0.0091, -0.0075, -0.0059, -0.0042],
                               [-0.0078, -0.0062, -0.0046, -0.0029],
                               [-0.0065, -0.0049, -0.0033, -0.0016]],
                              [[-0.0156, -0.0140, -0.0124, -0.0107],
                               [-0.0143, -0.0127, -0.0111, -0.0094],
                               [-0.0130, -0.0114, -0.0098, -0.0081],
                               [-0.0117, -0.0101, -0.0085, -0.0068]]],
                             [[[0.0000, 0.0016, 0.0033, 0.0049],
                               [0.0013, 0.0029, 0.0046, 0.0062],
                               [0.0026, 0.0042, 0.0059, 0.0075],
                               [0.0039, 0.0055, 0.0072, 0.0088]],
                              [[-0.0052, -0.0036, -0.0020, -0.0003],
                               [-0.0039, -0.0023, -0.0007, 0.0010],
                               [-0.0026, -0.0010, 0.0007, 0.0023],
                               [-0.0013, 0.0003, 0.0020, 0.0036]],
                              [[-0.0104, -0.0088, -0.0072, -0.0055],
                               [-0.0091, -0.0075, -0.0059, -0.0042],
                               [-0.0078, -0.0062, -0.0046, -0.0029],
                               [-0.0065, -0.0049, -0.0033, -0.0016]],
                              [[-0.0156, -0.0140, -0.0124, -0.0107],
                               [-0.0143, -0.0127, -0.0111, -0.0094],
                               [-0.0130, -0.0114, -0.0098, -0.0081],
                               [-0.0117, -0.0101, -0.0085, -0.0068]]]]]).astype(data_type))
    input_shape = (2, 2, 2, 2, 2)
    expect16 = Tensor(np.array([[[[[-0.0122, 0.0095],
                                   [0.0052, 0.0269]],
                                  [[-0.0816, -0.0599],
                                   [-0.0642, -0.0425]]],
                                 [[[-0.0122, 0.0095],
                                   [0.0052, 0.0269]],
                                  [[-0.0816, -0.0599],
                                   [-0.0642, -0.0425]]]],
                                [[[[-0.0122, 0.0095],
                                   [0.0052, 0.0269]],
                                  [[-0.0816, -0.0599],
                                   [-0.0642, -0.0425]]],
                                 [[[-0.0122, 0.0095],
                                   [0.0052, 0.0269]],
                                  [[-0.0816, -0.0599],
                                   [-0.0642, -0.0425]]]]]).astype(np.float16))
    expect32 = Tensor(np.array([[[[[-0.0122, 0.0095],
                                   [0.0052, 0.0269]],
                                  [[-0.0816, -0.0599],
                                   [-0.0642, -0.0425]]],
                                 [[[-0.0122, 0.0095],
                                   [0.0052, 0.0269]],
                                  [[-0.0816, -0.0599],
                                   [-0.0642, -0.0425]]]],
                                [[[[-0.0122, 0.0095],
                                   [0.0052, 0.0269]],
                                  [[-0.0816, -0.0599],
                                   [-0.0642, -0.0425]]],
                                 [[[-0.0122, 0.0095],
                                   [0.0052, 0.0269]],
                                  [[-0.0816, -0.0599],
                                   [-0.0642, -0.0425]]]]]).astype(np.float32))
    error = np.ones(shape=expect16.shape) * 1.0e-3
    upsample_trilinear_3d_grad = NetUpsampleTrilinear3DGrad(input_shape, scales=[2.0, 2.0, 2.0], align_corners=True)
    output = upsample_trilinear_3d_grad(grad)
    if data_type == np.float32:
        diff = abs(output.asnumpy() - expect32)
    else:
        diff = abs(output.asnumpy() - expect16)
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vmap_upsample_trilinear_3d_grad():
    """
    Feature:  UpsampleTrilinear3DGrad GPU op vmap feature.
    Description: test the vmap feature of UpsampleTrilinear3DGrad.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # 3 batches
    input_shape = (1, 1, 2, 2, 2)
    input_tensor = Tensor(np.arange(0, 5.4, 0.1).reshape(
        (3, 1, 1, 2, 3, 3)).astype(np.float32))
    net = NetUpsampleTrilinear3DGrad(input_shape, output_size=[2, 3, 3], align_corners=False)
    expect = np.array([[[[[[0.3, 0.6], [1.2, 1.5]],
                          [[2.325, 2.6250002], [3.2250001, 3.5250003]]]]],
                       [[[[[4.35, 4.65], [5.25, 5.5499997]],
                          [[6.375, 6.675], [7.275, 7.575]]]]],
                       [[[[[8.4, 8.700001], [9.299999, 9.599999]],
                          [[10.425001, 10.725], [11.325, 11.625001]]]]]]).astype(np.float32)
    out_vmap = F.vmap(net, in_axes=(0))(input_tensor)
    error = np.ones(shape=expect.shape) * 1.0e-6
    assert np.all(abs(out_vmap.asnumpy() - expect) < error)
