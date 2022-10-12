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
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner


class NetROIAlign(nn.Cell):
    def __init__(self, pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode, is_dyn_rank=False):
        super(NetROIAlign, self).__init__()
        self.roi_align = P.ROIAlign(pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode)
        self.is_dyn_rank = is_dyn_rank
        self.convert_to_dynamic_rank = inner.ConvertToDynamic(is_dynamic_rank=is_dyn_rank).add_prim_attr(
            "primitive_target", "CPU"
        )

    def construct(self, features, rois):
        if self.is_dyn_rank:
            features = self.convert_to_dynamic_rank(features)
            rois = self.convert_to_dynamic_rank(rois)
        return self.roi_align(features, rois)


def roi_align_case(data_type=np.float16, is_dyn_shape=False, is_dyn_rank=False):
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
            data_type,
        )
    )

    dyn_features = Tensor(shape=(None, None, None, None), dtype=features.dtype)
    dyn_rois = Tensor(shape=(None, None), dtype=features.dtype)

    # test case 1
    rois = Tensor(np.array([[0, -2.0, -2.0, 21.0, 21.0]], data_type))
    pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode = 3, 3, 0.25, 2, 1
    roi_align = NetROIAlign(pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode, is_dyn_rank)
    if is_dyn_shape:
        roi_align.set_inputs(dyn_features, dyn_rois)
    output = roi_align(features, rois)
    expect = [[[[4.5, 6.5, 8.5], [16.5, 18.5, 20.5], [28.5, 30.5, 32.5]]]]
    assert (output.asnumpy() == expect).all()

    # test case 2
    rois = Tensor(np.array([[0, -2.0, -2.0, 22.0, 22.0]], data_type))
    pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode = 3, 3, 0.25, 2, 0
    roi_align = NetROIAlign(pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode, is_dyn_rank)
    if is_dyn_shape:
        roi_align.set_inputs(dyn_features, dyn_rois)
    output = roi_align(features, rois)
    expect = [[[[4.5, 6.5, 8.5], [16.5, 18.5, 20.5], [28.5, 30.5, 32.5]]]]
    assert (output.asnumpy() == expect).all()

    # test case 3
    rois = Tensor(np.array([[0, -2.0, -2.0, 22.0, 22.0]], data_type))
    pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode = 2, 2, 1.0, -1, 0
    roi_align = NetROIAlign(pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode, is_dyn_rank)
    if is_dyn_shape:
        roi_align.set_inputs(dyn_features, dyn_rois)
    output = roi_align(features, rois)
    expect = [[[[6.295, 0.0], [0.0, 0.0]]]]
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_roi_align_float16():
    """
    Feature: Test the operator ROIAlign
    Description:  Test in GRAPH and PYNATIVE mode using float16 inputs
    Expectation: Assert the result is equal to the expectation
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    roi_align_case(np.float16)
    context.set_context(mode=context.GRAPH_MODE)
    roi_align_case(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_roi_align_float32():
    """
    Feature: Test the operator ROIAlign
    Description:  Test in GRAPH and PYNATIVE mode using float32 inputs
    Expectation: Assert the result is equal to the expectation
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    roi_align_case(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    roi_align_case(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_roi_align_float16_dynamic_shape():
    """
    Feature: Test the operator ROIAlign with dynamic shape inputs
    Description:  Test in GRAPH and PYNATIVE mode using float16 dynamic shape inputs
    Expectation: Assert the result is equal to the expectation
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    roi_align_case(np.float16, True)
    context.set_context(mode=context.GRAPH_MODE)
    roi_align_case(np.float16, True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_roi_align_float32_dynamic_rank():
    """
    Feature: Test the operator ROIAlign with dynamic rank inputs
    Description:  Test in GRAPH and PYNATIVE mode using float32 dynamic rank inputs
    Expectation: Assert the result is equal to the expectation
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    roi_align_case(np.float32, False, True)
    context.set_context(mode=context.GRAPH_MODE)
    roi_align_case(np.float32, False, True)
