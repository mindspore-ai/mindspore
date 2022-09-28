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
"""
Test PSROIPoolingGrad.
"""
import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations import _inner_ops as inner


class NetPSROIPoolingGrad(nn.Cell):
    """Simple net for PSROIPoolingGrad."""

    def __init__(self, input_size, spatial_scale, group_size, output_dim, dynamic_shape=False):
        super(NetPSROIPoolingGrad, self).__init__()
        self.ps_roi_pooling_grad = G.PSROIPoolingGrad(
            input_size, spatial_scale,
            group_size, output_dim)
        self.to_dynamic = inner.GpuConvertToDynamicShape()
        self.dynamic_shape = dynamic_shape

    def construct(self, dy, rois):
        if self.dynamic_shape:
            dy = self.to_dynamic(dy)
            rois = self.to_dynamic(rois)
        return self.ps_roi_pooling_grad(dy, rois)


def ps_roi_pooling_grad_case(data_type, dynamic_shape=False):
    device_target = "GPU"
    size_scale = 10
    rois_np = np.array(
        [[[0.0000], [150.3563 / size_scale],
          [200.1320 / size_scale], [579.3563 / size_scale],
          [602.3452 / size_scale]],
         [[1.0000], [65.1263 / size_scale],
          [30.8564 / size_scale], [762.4214 / size_scale],
          [567.9854 / size_scale]]]).astype(data_type)
    batch_size = rois_np.shape[0]
    rois_number = rois_np.shape[2]
    rois_ms = ms.Tensor(rois_np)

    x_height = 5
    x_width = 4
    group_size = 2
    output_dim = 2

    y_size = (batch_size * rois_number, output_dim, group_size, group_size)
    dy_np = np.ones(y_size).astype(data_type)
    dy_ms = ms.Tensor(dy_np)

    input_size = (x_height, x_width)
    spatial_scale = 1.0 / 16

    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=device_target)
    ps_roi_pooling_grad = NetPSROIPoolingGrad(
        input_size, spatial_scale,
        group_size, output_dim,
        dynamic_shape=dynamic_shape)

    output = ps_roi_pooling_grad(dy_ms, rois_ms)
    output_ms = output.asnumpy()

    output_gt = np.array(
        [[[[0., 0., 0., 0.], [0., 0.25, 0.25, 0.],
           [0., 0.25, 0.25, 0.], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0.25, 0.25],
           [0., 0., 0.25, 0.25], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0.25, 0.25, 0.], [0., 0.25, 0.25, 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0.25, 0.25], [0., 0., 0.25, 0.25],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0.25, 0.25, 0.],
           [0., 0.25, 0.25, 0.], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0.25, 0.25],
           [0., 0., 0.25, 0.25], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0.25, 0.25, 0.], [0., 0.25, 0.25, 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0.25, 0.25], [0., 0., 0.25, 0.25],
           [0., 0., 0., 0.]]],

         [[[0.16666667, 0.16666667, 0.16666667, 0.],
           [0.16666667, 0.16666667, 0.16666667, 0.],
           [0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0.25, 0.25], [0., 0., 0.25, 0.25],
           [0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0.16666667, 0.16666667, 0.16666667, 0.],
           [0.16666667, 0.16666667, 0.16666667, 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0.25, 0.25], [0., 0., 0.25, 0.25],
           [0., 0., 0., 0.]],

          [[0.16666667, 0.16666667, 0.16666667, 0.],
           [0.16666667, 0.16666667, 0.16666667, 0.],
           [0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0.25, 0.25], [0., 0., 0.25, 0.25],
           [0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0.16666667, 0.16666667, 0.16666667, 0.],
           [0.16666667, 0.16666667, 0.16666667, 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0.25, 0.25], [0., 0., 0.25, 0.25],
           [0., 0., 0., 0.]]]], dtype=data_type)
    assert np.allclose(output_ms, output_gt, atol=1e-4, rtol=1e-4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the normal behavior of PSROIPooingGrad op.
    Expectation: success.
    """
    ps_roi_pooling_grad_case(np.float32)
