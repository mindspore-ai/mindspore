# Copyright 2021 Huawei Technologies Co., Ltd
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

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


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
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_roi_align_grad():
    def roi_align_grad_case(data_type):
        rois = Tensor(np.array([[0, -2.0, -2.0, 21.0, 21.0]], data_type))

        dy = Tensor(np.array([[[
            [.1, .2, .3],
            [.1, .2, .3],
            [.1, .2, .3]
        ]]], data_type))

        xdiff_shape = (1, 1, 6, 6)
        pooled_height, pooled_width, spatial_scale, sample_num = 3, 3, 0.25, 2
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

        roi_align_grad = NetROIAlignGrad(
            xdiff_shape,
            pooled_height,
            pooled_width,
            spatial_scale,
            sample_num)
        output = roi_align_grad(dy, rois)
        #print(output)
        expect = ([[[[0.025, 0.025, 0.05, 0.05, 0.075, 0.075],
                     [0.025, 0.025, 0.05, 0.05, 0.075, 0.075],
                     [0.025, 0.025, 0.05, 0.05, 0.075, 0.075],
                     [0.025, 0.025, 0.05, 0.05, 0.075, 0.075],
                     [0.025, 0.025, 0.05, 0.05, 0.075, 0.075],
                     [0.025, 0.025, 0.05, 0.05, 0.075, 0.075]]]])
        np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=4)

    roi_align_grad_case(np.float32)
    roi_align_grad_case(np.float16)
