# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetROIAlignGrad(nn.Cell):
    def __init__(self, xdiff_shape, pooled_height, pooled_width, spatial_scale, sample_num):
        super(NetROIAlignGrad, self).__init__()
        self.roiAlignGrad = G.ROIAlignGrad(
            xdiff_shape,
            pooled_height,
            pooled_width,
            spatial_scale,
            sample_num)

    def construct(self, dy, rois):
        return self.roiAlignGrad(dy, rois)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_roi_align_grad_half():
    rois = Tensor(np.array([[0, -2.0, -2.0, 22.0, 22.0]], np.float16))

    dy = Tensor(np.array([[[
        [.1, .2, .3],
        [.1, .2, .3],
        [.1, .2, .3]
    ]]], np.float16))

    xdiff_shape = (1, 1, 6, 6)
    pooled_height, pooled_width, spatial_scale, sample_num = 3, 3, 0.25, 2

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    roi_align_grad = NetROIAlignGrad(
        xdiff_shape,
        pooled_height,
        pooled_width,
        spatial_scale,
        sample_num)
    output = roi_align_grad(dy, rois)
    print(output)
    expect = ([[[[0.0563, 0.0563, 0.0750, 0.0938, 0.1125, 0.0563],
                 [0.0375, 0.0375, 0.0500, 0.0625, 0.0750, 0.0375],
                 [0.0375, 0.0375, 0.0500, 0.0625, 0.0750, 0.0375],
                 [0.0375, 0.0375, 0.0500, 0.0625, 0.0750, 0.0375],
                 [0.0375, 0.0375, 0.0500, 0.0625, 0.0750, 0.0375],
                 [0.0188, 0.0188, 0.0250, 0.0312, 0.0375, 0.0188]]]])
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=4)
