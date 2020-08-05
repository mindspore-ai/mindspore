# Copyright 2019 Huawei Technologies Co., Ltd
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
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_roi_align():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = Tensor(np.array([[
        [[1, 2, 3, 4, 5, 6],
         [7, 8, 9, 10, 11, 12],
         [13, 14, 15, 16, 17, 18],
         [19, 20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29, 30],
         [31, 32, 33, 34, 35, 36]]
    ]], np.float32))

    rois = Tensor(np.array([[0, -2.0, -2.0, 22.0, 22.0]], np.float32))

    # test case 1
    pooled_height, pooled_width, spatial_scale, sample_num = 3, 3, 0.25, 2
    roi_align = P.ROIAlign(pooled_height, pooled_width, spatial_scale, sample_num)
    output = roi_align(x, rois)
    print(output)
    expect = [[[[2.75, 4.5, 6.5],
                [13.25, 15., 17.],
                [25.25, 27., 29.]]]]
    assert (output.asnumpy() == expect).all()

    # test case 2
    pooled_height, pooled_width, spatial_scale, sample_num = 4, 4, 0.2, 3
    roi_align = P.ROIAlign(pooled_height, pooled_width, spatial_scale, sample_num)
    output = roi_align(x, rois)
    print(output)
    expect = [[[[1.2333, 2.1000, 3.3000, 4.5000],
                [6.4333, 7.3000, 8.5000, 9.7000],
                [13.6333, 14.5000, 15.7000, 16.9000],
                [20.8333, 21.7000, 22.9000, 24.1000]]]]
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=4)

    # test case 3
    pooled_height, pooled_width, spatial_scale, sample_num = 3, 3, 0.3, 3
    rois = Tensor(np.array([[0, -2.0, -2.0, 22.0, 22.0],
                            [0, 1.0, 0.0, 19.0, 18.0]],
                           np.float32))
    roi_align = P.ROIAlign(pooled_height, pooled_width, spatial_scale, sample_num)
    output = roi_align(x, rois)
    print(output)
    expect = [[[[3.3333, 5.5000, 7.6667],
                [16.3333, 18.5000, 20.6667],
                [29.3333, 31.5000, 33.6667]]],
              [[[4.5000, 6.3000, 8.1000],
                [14.9000, 16.7000, 18.5000],
                [25.7000, 27.5000, 29.3000]]]]
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=4)

    # test case 4
    pooled_height, pooled_width, spatial_scale, sample_num = 2, 2, 1.0, -1
    rois = Tensor(np.array([[0, -2.0, -2.0, 22.0, 22.0]], np.float32))
    roi_align = P.ROIAlign(pooled_height, pooled_width, spatial_scale, sample_num)
    output = roi_align(x, rois)
    print(output)
    expect = [[[[4.625, 0.],
                [0., 0.]]]]
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=4)
