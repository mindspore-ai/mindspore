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
from mindspore import ops, nn, context, Tensor
from .test_grad_of_dynamic import TestDynamicGrad


class ROIAlignNet(nn.Cell):
    def __init__(self, pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode):
        super(ROIAlignNet, self).__init__()
        self.op = ops.ROIAlign(pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode)

    def construct(self, features, rois):
        return self.op(features, rois)


def dyn_grad_func(dtype=np.float16, is_dynamic_rank=False):
    features = Tensor(
        np.array(
            [
                [
                    [
                        [1, 2, 3, 4, 5, 6],
                        [7, 8, 9, 10, 11, 12],
                        [13, 14, 15, 16, 17, 18],
                        [19, 20, 21, 22, 23, 24],
                        [25, 26, 27, 28, 29, 30],
                        [31, 32, 33, 34, 35, 36],
                    ]
                ]
            ],
            dtype,
        )
    )
    rois = Tensor(np.array([[0, -2.0, -2.0, 21.0, 21.0]], dtype))
    pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode = 3, 3, 0.25, 2, 1
    inputs = [features, rois]
    test_dynamic = TestDynamicGrad(ROIAlignNet(pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode))
    test_dynamic.test_dynamic_grad_net(inputs, is_dynamic_rank=is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_roialign_dynamic_shape():
    """
    Feature: Test the bprop process of ROIAlign in PyNative mode with dynamic shape inputs
    Description: The inputs are dynamic shape and the bprop function invokes the ROIAlignGrad operator.
    Expectation: Assert the result is equal to that of static shape inputs
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    dyn_grad_func(dtype=np.float16)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_roialign_dynamic_rank():
    """
    Feature: Test the bprop process of ROIAlign in PyNative mode with dynamic rank inputs
    Description: The inputs are dynamic rank and the bprop function invokes the ROIAlignGrad operator.
    Expectation: Assert the result is equal to that of static shape inputs
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    dyn_grad_func(dtype=np.float32, is_dynamic_rank=True)
