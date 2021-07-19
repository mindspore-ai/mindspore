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
from mindspore import Tensor
from mindspore.ops import operations as P


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_roi_align():
    def roi_align_case(data_type):
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
        x = Tensor(np.array([[
            [[1, 2, 3, 4, 5, 6],
             [7, 8, 9, 10, 11, 12],
             [13, 14, 15, 16, 17, 18],
             [19, 20, 21, 22, 23, 24],
             [25, 26, 27, 28, 29, 30],
             [31, 32, 33, 34, 35, 36]]
        ]], data_type))

        # test case 1
        rois = Tensor(np.array([[0, -2.0, -2.0, 21.0, 21.0]], data_type))
        pooled_height, pooled_width, spatial_scale, sample_num = 3, 3, 0.25, 2
        roi_align = P.ROIAlign(pooled_height, pooled_width,
                               spatial_scale, sample_num, 1)
        output = roi_align(x, rois)
        #print(output)
        expect = [[[[4.5, 6.5, 8.5],
                    [16.5, 18.5, 20.5],
                    [28.5, 30.5, 32.5]]]]
        assert (output.asnumpy() == expect).all()

        # test case 2
        rois = Tensor(np.array([[0, -2.0, -2.0, 22.0, 22.0]], data_type))
        pooled_height, pooled_width, spatial_scale, sample_num = 3, 3, 0.25, 2
        roi_align = P.ROIAlign(pooled_height, pooled_width,
                               spatial_scale, sample_num, 0)
        output = roi_align(x, rois)
        #print(output)
        expect = [[[[4.5, 6.5, 8.5],
                    [16.5, 18.5, 20.5],
                    [28.5, 30.5, 32.5]]]]
        assert (output.asnumpy() == expect).all()

        # test case 3
        pooled_height, pooled_width, spatial_scale, sample_num = 2, 2, 1.0, -1
        rois = Tensor(np.array([[0, -2.0, -2.0, 22.0, 22.0]], data_type))
        roi_align = P.ROIAlign(pooled_height, pooled_width,
                               spatial_scale, sample_num, 0)
        output = roi_align(x, rois)
        #print(output)
        expect = [[[[6.295, 0.],
                    [0., 0.]]]]
        np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=2)

    roi_align_case(np.float32)
    roi_align_case(np.float16)
