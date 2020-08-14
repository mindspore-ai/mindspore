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
def test_roi_align_half():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = Tensor(np.array([[
        [[1, 2, 3, 4, 5, 6],
         [7, 8, 9, 10, 11, 12],
         [13, 14, 15, 16, 17, 18],
         [19, 20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29, 30],
         [31, 32, 33, 34, 35, 36]]
    ]], np.float16))

    rois = Tensor(np.array([[0, -2.0, -2.0, 22.0, 22.0]], np.float16))

    # test case 1
    pooled_height, pooled_width, spatial_scale, sample_num = 4, 4, 0.2, 3
    roi_align = P.ROIAlign(pooled_height, pooled_width, spatial_scale, sample_num)
    output = roi_align(x, rois)
    print(output)
    expect = [[[[1.2333, 2.1000, 3.3000, 4.5000],
                [6.4333, 7.3000, 8.5000, 9.7000],
                [13.6333, 14.5000, 15.7000, 16.9000],
                [20.8333, 21.7000, 22.9000, 24.1000]]]]
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=1)
