# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, ops
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations import _inner_ops as inner


class NetROIAlignGrad(nn.Cell):
    def __init__(self, pooled_height, pooled_width, spatial_scale, sample_num, is_dyn_rank=False):
        super(NetROIAlignGrad, self).__init__()
        self.shape = ops.Shape()
        self.dyn_shape = ops.TensorShape()
        self.roi_align_grad = G.ROIAlignGrad(pooled_height, pooled_width, spatial_scale, sample_num)
        self.is_dyn_rank = is_dyn_rank
        self.convert_to_dynamic_rank = inner.ConvertToDynamic(is_dynamic_rank=is_dyn_rank).add_prim_attr(
            "primitive_target", "CPU"
        )

    def construct(self, dy, rois, xdiff):
        if self.is_dyn_rank:
            dy = self.convert_to_dynamic_rank(dy)
            rois = self.convert_to_dynamic_rank(rois)
            xdiff = self.convert_to_dynamic_rank(xdiff)
        xdiff_shape = self.shape(xdiff)
        if -1 in xdiff_shape or -2 in xdiff_shape:
            xdiff_shape = self.dyn_shape(xdiff)
        return self.roi_align_grad(dy, rois, xdiff_shape)


def roi_align_grad_case(data_type=np.float16, is_dyn_shape=False, is_dyn_rank=False):
    rois = Tensor(np.array([[0, -2.0, -2.0, 21.0, 21.0]], data_type))
    dy = Tensor(np.array([[[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]]], data_type))

    xdiff_shape = (1, 1, 6, 6)
    xdiff = Tensor(np.random.rand(*xdiff_shape).astype(data_type))

    pooled_height, pooled_width, spatial_scale, sample_num = 3, 3, 0.25, 2

    roi_align_grad = NetROIAlignGrad(pooled_height, pooled_width, spatial_scale, sample_num, is_dyn_rank)

    if is_dyn_shape:
        dyn_dx_dy = Tensor(shape=(None, None, None, None), dtype=dy.dtype)
        dyn_rois = Tensor(shape=(None, None), dtype=dy.dtype)
        roi_align_grad.set_inputs(dyn_dx_dy, dyn_rois, dyn_dx_dy)

    output = roi_align_grad(dy, rois, xdiff)
    expect = [
        [
            [
                [0.025, 0.025, 0.05, 0.05, 0.075, 0.075],
                [0.025, 0.025, 0.05, 0.05, 0.075, 0.075],
                [0.025, 0.025, 0.05, 0.05, 0.075, 0.075],
                [0.025, 0.025, 0.05, 0.05, 0.075, 0.075],
                [0.025, 0.025, 0.05, 0.05, 0.075, 0.075],
                [0.025, 0.025, 0.05, 0.05, 0.075, 0.075],
            ]
        ]
    ]
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=4)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_roi_align_grad_float16():
    """
    Feature: Test the operator ROIAlignGrad
    Description:  Test in GRAPH and PYNATIVE mode using float16 inputs
    Expectation: Assert the result is equal to the expectation
    """
    context.set_context(mode=context.GRAPH_MODE)
    roi_align_grad_case(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE)
    roi_align_grad_case(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_roi_align_grad_float32():
    """
    Feature: Test the operator ROIAlignGrad
    Description:  Test in GRAPH and PYNATIVE mode using float32 inputs
    Expectation: Assert the result is equal to the expectation
    """
    context.set_context(mode=context.GRAPH_MODE)
    roi_align_grad_case(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE)
    roi_align_grad_case(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_roi_align_grad_float16_dynamic_shape():
    """
    Feature: Test the operator ROIAlignGrad with dynamic shape inputs
    Description:  Test in GRAPH and PYNATIVE mode using float16 dynamic shape inputs
    Expectation: Assert the result is equal to the expectation
    """
    context.set_context(mode=context.GRAPH_MODE)
    roi_align_grad_case(np.float16, True)
    context.set_context(mode=context.PYNATIVE_MODE)
    roi_align_grad_case(np.float16, True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_roi_align_grad_float32_dynamic_rank():
    """
    Feature: Test the operator ROIAlignGrad with dynamic rank inputs
    Description:  Test in GRAPH and PYNATIVE mode using float32 dynamic rank inputs
    Expectation: Assert the result is equal to the expectation
    """
    context.set_context(mode=context.GRAPH_MODE)
    roi_align_grad_case(np.float32, True)
    context.set_context(mode=context.PYNATIVE_MODE)
    roi_align_grad_case(np.float32, True)
