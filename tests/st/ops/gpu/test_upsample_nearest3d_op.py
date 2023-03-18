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
from mindspore.ops.operations.nn_ops import UpsampleNearest3D


class UpsampleNearest3DNet(nn.Cell):

    def __init__(self, output_size=None, scales=None):
        super(UpsampleNearest3DNet, self).__init__()
        self.upsample = UpsampleNearest3D(output_size, scales)

    def construct(self, x):
        out = self.upsample(x)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
@pytest.mark.skip(reason="Have issues")
def test_upsample_nearest_3d_dynamic_shape():
    """
    Feature: Test UpsampleNearest3D op in gpu.
    Description: Test the ops in dynamic shape.
    Expectation: Expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    output_size = [3, 4, 5]
    net = UpsampleNearest3DNet(output_size=output_size)
    x_dyn = Tensor(shape=[None, 1, 2, 2, 4], dtype=ms.float32)
    net.set_inputs(x_dyn)
    x = Tensor(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                  16]).reshape([1, 1, 2, 2, 4]), ms.float32)
    output = net(x)
    expect_shape = (1, 1, 3, 4, 5)
    assert expect_shape == output.asnumpy().shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
@pytest.mark.skip(reason="Have issues")
def test_upsample_nearest_3d_output_size_float(data_type):
    """
    Feature: UpsampleNearest3D
    Description: Test cases for UpsampleNearest3D operator with output_size.
    Expectation: The result match expected output.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_tensor = Tensor(
        np.array([[[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]]]).astype(data_type))
    expected = np.array([[[[[0.1000, 0.1000, 0.2000, 0.2000, 0.3000],
                            [0.1000, 0.1000, 0.2000, 0.2000, 0.3000],
                            [0.4000, 0.4000, 0.5000, 0.5000, 0.6000],
                            [0.4000, 0.4000, 0.5000, 0.5000, 0.6000]],
                           [[0.1000, 0.1000, 0.2000, 0.2000, 0.3000],
                            [0.1000, 0.1000, 0.2000, 0.2000, 0.3000],
                            [0.4000, 0.4000, 0.5000, 0.5000, 0.6000],
                            [0.4000, 0.4000, 0.5000, 0.5000, 0.6000]],
                           [[0.7000, 0.7000, 0.8000, 0.8000, 0.9000],
                            [0.7000, 0.7000, 0.8000, 0.8000, 0.9000],
                            [1.0000, 1.0000, 1.1000, 1.1000, 1.2000],
                            [1.0000, 1.0000, 1.1000, 1.1000,
                             1.2000]]]]]).astype(data_type)
    net = UpsampleNearest3DNet(output_size=[3, 4, 5])
    out = net(input_tensor)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-5
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
@pytest.mark.skip(reason="Have issues")
def test_upsample_nearest_3d_scales_float(data_type):
    """
    Feature: UpsampleNearest3D
    Description: Test cases for UpsampleNearest3D operator with scales.
    Expectation: The result match expected output.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_tensor = Tensor(
        np.array([[[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]]]).astype(data_type))
    expected = np.array(
        [[[[[0.1000, 0.1000, 0.1000, 0.2000, 0.2000, 0.3000, 0.3000],
            [0.1000, 0.1000, 0.1000, 0.2000, 0.2000, 0.3000, 0.3000],
            [0.4000, 0.4000, 0.4000, 0.5000, 0.5000, 0.6000, 0.6000],
            [0.4000, 0.4000, 0.4000, 0.5000, 0.5000, 0.6000, 0.6000]],
           [[0.1000, 0.1000, 0.1000, 0.2000, 0.2000, 0.3000, 0.3000],
            [0.1000, 0.1000, 0.1000, 0.2000, 0.2000, 0.3000, 0.3000],
            [0.4000, 0.4000, 0.4000, 0.5000, 0.5000, 0.6000, 0.6000],
            [0.4000, 0.4000, 0.4000, 0.5000, 0.5000, 0.6000, 0.6000]],
           [[0.7000, 0.7000, 0.7000, 0.8000, 0.8000, 0.9000, 0.9000],
            [0.7000, 0.7000, 0.7000, 0.8000, 0.8000, 0.9000, 0.9000],
            [1.0000, 1.0000, 1.0000, 1.1000, 1.1000, 1.2000, 1.2000],
            [1.0000, 1.0000, 1.0000, 1.1000, 1.1000, 1.2000,
             1.2000]]]]]).astype(data_type)
    net = UpsampleNearest3DNet(scales=[1.5, 2.0, 2.5])
    out = net(input_tensor)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-5
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skip(reason="Have issues")
def test_upsample_nearest_3d_error():
    """
    Feature: UpsampleNearest3D
    Description: Test cases for UpsampleNearest3D operator with errors.
    Expectation: Raise expected error type.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2), dtype=np.float32))
        net = UpsampleNearest3DNet(output_size=[3, 4, 5])
        net(input_tensor)

    with pytest.raises(TypeError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.int32))
        net = UpsampleNearest3DNet(output_size=[3, 4, 5])
        net(input_tensor)

    with pytest.raises(TypeError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = UpsampleNearest3DNet(scales=[1, 2, 3])
        net(input_tensor)

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = UpsampleNearest3DNet(output_size=[3, 4])
        net(input_tensor)

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = UpsampleNearest3DNet(scales=[1.0, 2.0, 3.0, 4.0])
        net(input_tensor)

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = UpsampleNearest3DNet(output_size=[3, 4, 5],
                                   scales=[1.0, 2.0, 3.0])
        net(input_tensor)

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net = UpsampleNearest3DNet()
        net(input_tensor)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vmap_upsample_nearest3d():
    """
    Feature:  UpsampleNearest3D GPU op vmap feature.
    Description: test the vmap feature of UpsampleNearest3D.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # 3 batches
    input_tensor = Tensor(
        np.arange(0, 4.8, 0.1).reshape([3, 1, 1, 2, 2, 4]).astype(np.float32))
    net = UpsampleNearest3DNet(output_size=[3, 2, 2])
    expect = np.array([[[[[[0.0, 0.2], [0.4, 0.6]], [[0.0, 0.2], [0.4, 0.6]],
                          [[0.8, 1.0], [1.2, 1.4]]]]],
                       [[[[[1.6, 1.8], [2.0, 2.2]], [[1.6, 1.8], [2.0, 2.2]],
                          [[2.4, 2.6], [2.8, 3.0]]]]],
                       [[[[[3.2, 3.4], [3.6, 3.8]], [[3.2, 3.4], [3.6, 3.8]],
                          [[4.0, 4.2], [4.4, 4.6]]]]]])
    out_vmap = F.vmap(net, in_axes=(0))(input_tensor)
    error = np.ones(shape=expect.shape) * 1.0e-6
    assert np.all(abs(out_vmap.asnumpy() - expect) < error)
