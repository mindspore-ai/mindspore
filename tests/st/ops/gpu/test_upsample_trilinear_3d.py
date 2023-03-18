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
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops.operations.nn_ops import UpsampleTrilinear3D


class UpsampleTrilinear3DNet(nn.Cell):
    def __init__(self, output_size=None, scales=None, align_corners=False):
        super(UpsampleTrilinear3DNet, self).__init__()
        self.upsample = UpsampleTrilinear3D(output_size, scales, align_corners)

    def construct(self, x):
        out = self.upsample(x)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
@pytest.mark.skip(reason="Have issues")
def test_upsample_trilinear_3d_dynamic_shape():
    """
    Feature: Test UpsampleTrilinear3D op in gpu.
    Description: Test the ops in dynamic shape.
    Expectation: Expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    output_size = [2, 4, 4]
    net = UpsampleTrilinear3DNet(output_size=output_size)
    x_dyn = Tensor(shape=[None, 1, 1, 2, 2], dtype=ms.float32)
    net.set_inputs(x_dyn)
    x = Tensor(np.arange(1, 5, dtype=np.float32).reshape((1, 1, 1, 2, 2)))
    output = net(x)
    expect_shape = (1, 1, 2, 4, 4)
    assert expect_shape == output.asnumpy().shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
@pytest.mark.skip(reason="Have issues")
def test_upsample_trilinear_3d_output_size(data_type):
    """
    Feature: UpsampleTrilinear3D
    Description: Test cases for UpsampleTrilinear3D operator with output_size.
    Expectation: The result matches expected output.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_tensor = Tensor(np.array([[[[[0.1, 0.2, 0.3],
                                       [0.4, 0.5, 0.6]],
                                      [[0.7, 0.8, 0.9],
                                       [1.0, 1.1, 1.2]]]]]).astype(data_type))
    fp32_expected = np.array([[[[[0.1, 0.14, 0.2, 0.26000002, 0.3],
                                 [0.175, 0.215, 0.275, 0.33500004, 0.375],
                                 [0.32500002, 0.36499998, 0.425, 0.48500004, 0.52500004],
                                 [0.4, 0.44, 0.5, 0.56000006, 0.6]],
                                [[0.4, 0.44, 0.5, 0.56, 0.6],
                                 [0.475, 0.515, 0.575, 0.63500005, 0.675],
                                 [0.625, 0.66499996, 0.725, 0.785, 0.82500005],
                                 [0.7, 0.74, 0.8, 0.8600001, 0.90000004]],
                                [[0.7, 0.74, 0.8, 0.86, 0.9],
                                 [0.775, 0.815, 0.875, 0.93500006, 0.975],
                                 [0.925, 0.965, 1.0250001, 1.085, 1.125],
                                 [1., 1.04, 1.1, 1.1600001, 1.2]]]]]).astype(np.float32)
    fp16_expected = np.array([[[[[0.1, 0.14, 0.2, 0.26, 0.3],
                                 [0.1749, 0.215, 0.275, 0.335, 0.375],
                                 [0.325, 0.365, 0.425, 0.485, 0.525],
                                 [0.4, 0.44, 0.5, 0.56, 0.6]],
                                [[0.4001, 0.44, 0.5, 0.56, 0.6],
                                 [0.475, 0.515, 0.5747, 0.635, 0.675],
                                 [0.625, 0.665, 0.7246, 0.785, 0.825],
                                 [0.7, 0.7397, 0.8, 0.86, 0.9004]],
                                [[0.7, 0.74, 0.8, 0.86, 0.9],
                                 [0.7754, 0.815, 0.875, 0.935, 0.975],
                                 [0.925, 0.965, 1.024, 1.085, 1.125],
                                 [1., 1.04, 1.1, 1.16, 1.2]]]]]).astype(np.float16)
    net = UpsampleTrilinear3DNet(output_size=[3, 4, 5])
    out = net(input_tensor)
    if data_type == np.float32:
        diff = abs(out.asnumpy() - fp32_expected)
    else:
        diff = abs(out.asnumpy() - fp16_expected)
    error = np.ones(shape=fp32_expected.shape) * 1.0e-5
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
@pytest.mark.skip(reason="Have issues")
def test_upsample_trilinear_3d_output_size_align_corners(data_type):
    """
    Feature: UpsampleTrilinear3D
    Description: Test cases for UpsampleTrilinear3D operator with output_size,
    with align corners parameter enabled
    Expectation: The result matches expected output.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_tensor = Tensor(np.array([[[[[0.1, 0.2, 0.3],
                                       [0.4, 0.5, 0.6]],
                                      [[0.7, 0.8, 0.9],
                                       [1.0, 1.1, 1.2]]]]]).astype(data_type))
    fp32_expected = np.array([[[[[0.1, 0.15, 0.2, 0.25, 0.3],
                                 [0.2, 0.25, 0.3, 0.35, 0.4],
                                 [0.3, 0.35000002, 0.4, 0.45, 0.50000006],
                                 [0.4, 0.45, 0.5, 0.55, 0.6]],
                                [[0.4, 0.45, 0.5, 0.55, 0.6],
                                 [0.49999997, 0.54999995, 0.6, 0.65000004, 0.7],
                                 [0.6, 0.65, 0.7, 0.75, 0.8000001],
                                 [0.7, 0.75, 0.8, 0.85, 0.90000004]],
                                [[0.7, 0.75, 0.8, 0.85, 0.9],
                                 [0.79999995, 0.84999996, 0.9, 0.95000005, 1.],
                                 [0.9, 0.95, 1., 1.0500001, 1.1],
                                 [1., 1.05, 1.1, 1.1500001, 1.2]]]]]).astype(np.float32)
    fp16_expected = np.array([[[[[0.1, 0.1499, 0.2, 0.25, 0.3],
                                 [0.2, 0.25, 0.3, 0.35, 0.4001],
                                 [0.2998, 0.3499, 0.4, 0.45, 0.5],
                                 [0.4, 0.45, 0.5, 0.55, 0.6]],
                                [[0.4001, 0.45, 0.5, 0.55, 0.6],
                                 [0.5, 0.55, 0.5996, 0.65, 0.7],
                                 [0.6, 0.65, 0.6997, 0.75, 0.8003],
                                 [0.7, 0.75, 0.8, 0.85, 0.9004]],
                                [[0.7, 0.75, 0.8, 0.8496, 0.9],
                                 [0.8003, 0.85, 0.9, 0.9497, 1.],
                                 [0.9, 0.9497, 0.9995, 1.05, 1.1],
                                 [1., 1.05, 1.1, 1.15, 1.2]]]]]).astype(np.float16)
    net = UpsampleTrilinear3DNet(output_size=(3, 4, 5), align_corners=True)
    out = net(input_tensor)
    if data_type == np.float32:
        diff = abs(out.asnumpy() - fp32_expected)
    else:
        diff = abs(out.asnumpy() - fp16_expected)
    error = np.ones(shape=fp32_expected.shape) * 1.0e-5
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
@pytest.mark.skip(reason="Have issues")
def test_upsample_trilinear_3d_scales(data_type):
    """
    Feature: UpsampleTrilinear3D
    Description: Test cases for UpsampleTrilinear3D operator with scales.
    Expectation: The result match expected output.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_tensor = Tensor(np.array([[[[[0.1, 0.2, 0.3],
                                       [0.4, 0.5, 0.6]],
                                      [[0.7, 0.8, 0.9],
                                       [1.0, 1.1, 1.2]]]]]).astype(data_type))
    fp16_expected = np.array([[[[[0.1, 0.11, 0.1499, 0.19, 0.23, 0.27, 0.3],
                                 [0.1749, 0.1849, 0.225, 0.265, 0.305, 0.345, 0.375],
                                 [0.325, 0.335, 0.375, 0.415, 0.455, 0.495, 0.525],
                                 [0.4, 0.41, 0.45, 0.49, 0.5303, 0.5703, 0.6]],
                                [[0.4001, 0.4102, 0.45, 0.49, 0.53, 0.57, 0.6],
                                 [0.475, 0.485, 0.525, 0.565, 0.605, 0.645, 0.675],
                                 [0.625, 0.635, 0.675, 0.715, 0.755, 0.795, 0.825],
                                 [0.7, 0.71, 0.75, 0.79, 0.83, 0.87, 0.9004]],
                                [[0.7, 0.71, 0.75, 0.79, 0.83, 0.87, 0.9],
                                 [0.7754, 0.785, 0.825, 0.8647, 0.905, 0.945, 0.975],
                                 [0.925, 0.935, 0.9746, 1.015, 1.055, 1.095, 1.125],
                                 [1., 1.01, 1.05, 1.09, 1.13, 1.17, 1.2]]]]]).astype(np.float16)
    fp32_expected = np.array([[[[[0.1, 0.11, 0.15, 0.19, 0.23000002, 0.27, 0.3],
                                 [0.175, 0.185, 0.225, 0.265, 0.305, 0.34500003, 0.375],
                                 [0.32500002, 0.335, 0.37499997, 0.41500002,
                                  0.45500004, 0.49500003, 0.52500004],
                                 [0.4, 0.41, 0.45, 0.49, 0.53000003, 0.57000005, 0.6]],
                                [[0.4, 0.41, 0.45, 0.49, 0.53000003, 0.57, 0.6],
                                 [0.475, 0.48499998, 0.525, 0.565, 0.605, 0.64500004, 0.675],
                                 [0.625, 0.635, 0.67499995, 0.71500003, 0.755, 0.795, 0.82500005],
                                 [0.7, 0.71, 0.75, 0.79, 0.83000004, 0.87000006, 0.90000004]],
                                [[0.7, 0.71, 0.75, 0.79, 0.83000004, 0.87, 0.9],
                                 [0.775, 0.78499997, 0.825, 0.865, 0.90500003, 0.94500005, 0.975],
                                 [0.925, 0.935, 0.97499996, 1.015, 1.055, 1.095, 1.125],
                                 [1., 1.01, 1.05, 1.09, 1.13,
                                  1.1700001, 1.2]]]]]).astype(np.float32)
    net = UpsampleTrilinear3DNet(scales=[1.5, 2.0, 2.5])
    out = net(input_tensor)
    if data_type == np.float32:
        diff = abs(out.asnumpy() - fp32_expected)
    else:
        diff = abs(out.asnumpy() - fp16_expected)
    error = np.ones(shape=fp32_expected.shape) * 1.0e-5
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
@pytest.mark.skip(reason="Have issues")
def test_upsample_trilinear_3d_scales_align_corners(data_type):
    """
    Feature: UpsampleTrilinear3D
    Description: Test cases for UpsampleTrilinear3D operator with scales,
    with align corners parameter enabled
    Expectation: The result match expected output.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_tensor = Tensor(np.array([[[[[0.1, 0.2, 0.3],
                                       [0.4, 0.5, 0.6]],
                                      [[0.7, 0.8, 0.9],
                                       [1.0, 1.1, 1.2]]]]]).astype(data_type))
    fp16_expected = np.array([[[[[0.1, 0.1333, 0.1666, 0.2, 0.2333, 0.2666, 0.3],
                                 [0.2, 0.2333, 0.2666, 0.3, 0.3333, 0.3667, 0.4001],
                                 [0.2998, 0.3333, 0.3667, 0.4, 0.4333, 0.4668, 0.5],
                                 [0.4, 0.4333, 0.4666, 0.5, 0.533, 0.567, 0.6]],
                                [[0.4001, 0.4333, 0.4666, 0.5, 0.533, 0.5664, 0.6],
                                 [0.5, 0.533, 0.5664, 0.5996, 0.6333, 0.6665, 0.7],
                                 [0.6, 0.6333, 0.6665, 0.6997, 0.7334, 0.7666, 0.8003],
                                 [0.7, 0.7334, 0.7666, 0.8, 0.833, 0.8667, 0.9004]],
                                [[0.7, 0.7334, 0.7666, 0.8, 0.833, 0.8667, 0.9],
                                 [0.8003, 0.8335, 0.8667, 0.9, 0.933, 0.967, 1.],
                                 [0.9, 0.933, 0.9663, 0.9995, 1.033, 1.066, 1.1],
                                 [1., 1.033, 1.066, 1.1, 1.133, 1.167, 1.2]]]]]).astype(np.float16)
    fp32_expected = np.array([[[[[0.1, 0.13333334, 0.16666667, 0.2, 0.23333335, 0.26666668, 0.3],
                                 [0.2, 0.23333333, 0.26666668, 0.3, 0.33333334, 0.36666667, 0.4],
                                 [0.3, 0.33333334, 0.36666667, 0.4, 0.43333337,
                                  0.4666667, 0.50000006],
                                 [0.4, 0.43333334, 0.46666667, 0.5, 0.53333336, 0.5666667, 0.6]],
                                [[0.4, 0.4333333, 0.46666667, 0.5, 0.53333336, 0.56666666, 0.6],
                                 [0.49999997, 0.5333333, 0.56666666, 0.6,
                                  0.6333333, 0.6666667, 0.7],
                                 [0.6, 0.6333333, 0.6666666, 0.7, 0.73333335, 0.7666667, 0.8000001],
                                 [0.7, 0.73333335, 0.76666665, 0.8, 0.8333334,
                                  0.86666673, 0.90000004]],
                                [[0.7, 0.7333333, 0.76666665, 0.8, 0.8333334, 0.8666667, 0.9],
                                 [0.79999995, 0.8333333, 0.8666666, 0.9, 0.93333334, 0.9666667, 1.],
                                 [0.9, 0.93333334, 0.9666666, 1., 1.0333333, 1.0666667, 1.1],
                                 [1., 1.0333333, 1.0666666, 1.1, 1.1333333,
                                  1.1666667, 1.2]]]]]).astype(np.float32)
    net = UpsampleTrilinear3DNet(scales=(1.5, 2.0, 2.5), align_corners=True)
    out = net(input_tensor)
    if data_type == np.float32:
        diff = abs(out.asnumpy() - fp32_expected)
    else:
        diff = abs(out.asnumpy() - fp16_expected)
    error = np.ones(shape=fp32_expected.shape) * 1.0e-5
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skip(reason="Have issues")
def test_upsample_trilinear_3d_error():
    """
    Feature: UpsampleTrilinear3D
    Description: Test cases for UpsampleTrilinear3D operator with errors.
    Expectation: Raise expected error type.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2), dtype=np.float32))
        net = UpsampleTrilinear3DNet(output_size=[3, 4, 5])
        net(input_tensor)

    with pytest.raises(TypeError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.int32))
        net = UpsampleTrilinear3DNet(output_size=[3, 4, 5])
        net(input_tensor)

    with pytest.raises(TypeError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = UpsampleTrilinear3DNet(scales=[1, 2, 3])
        net(input_tensor)

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = UpsampleTrilinear3DNet(output_size=[3, 4])
        net(input_tensor)

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = UpsampleTrilinear3DNet(scales=[1.0, 2.0, 3.0, 4.0])
        net(input_tensor)

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = UpsampleTrilinear3DNet(output_size=[3, 4, 5], scales=[1.0, 2.0, 3.0])
        net(input_tensor)

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = UpsampleTrilinear3DNet()
        net(input_tensor)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vmap_upsample_trilinear_3d():
    """
    Feature:  UpsampleTrilinear3D GPU op vmap feature.
    Description: test the vmap feature of UpsampleTrilinear3D.
    Expectation: success.
    """
    # 3 batches
    input_tensor = Tensor(np.arange(0, 4.8, 0.1).reshape([3, 1, 1, 2, 2, 4]).astype(np.float32))
    net = UpsampleTrilinear3DNet(output_size=[3, 2, 2], align_corners=True)
    expect = np.array([[[[[[0.0, 0.3], [0.4, 0.7]],
                          [[0.4, 0.70000005], [0.8, 1.1]],
                          [[0.8, 1.1], [1.2, 1.5]]]]],
                       [[[[[1.6, 1.9], [2.0, 2.3]],
                          [[2.0, 2.3], [2.4, 2.6999998]],
                          [[2.4, 2.7], [2.8, 3.1]]]]],
                       [[[[[3.2, 3.5], [3.6, 3.9]],
                          [[3.6, 3.9], [4.0, 4.3]],
                          [[4.0, 4.3], [4.4, 4.7]]]]]])
    out_vmap = F.vmap(net, in_axes=(0))(input_tensor)
    error = np.ones(shape=expect.shape) * 1.0e-6
    assert np.all(abs(out_vmap.asnumpy() - expect) < error)
