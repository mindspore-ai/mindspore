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
import mindspore.ops as ops
from mindspore import Tensor


class CheckValidDynNet(nn.Cell):
    def __init__(self, axis=0):
        super(CheckValidDynNet, self).__init__()
        self.unique = ops.Unique()
        self.gather = ops.Gather()
        self.check_valid = ops.CheckValid()
        self.axis = axis

    def construct(self, anchor, image_metas, indices_anchor, indices_image_metas):
        unique_indices_anchor, _ = self.unique(indices_anchor)
        anchor_ = self.gather(anchor, unique_indices_anchor, self.axis)
        unique_indices_image_metas, _ = self.unique(indices_image_metas)
        image_metas_ = self.gather(image_metas, unique_indices_image_metas, self.axis)
        return self.check_valid(anchor_, image_metas_)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16])
def test_dynamic_check_valid(dtype):
    """
    Feature: test CheckValid dynamic shape.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    anchor = np.array([[5, 0, 10, 70], [2, 2, 8, 10], [1, 2, 30, 200]]).astype(dtype)
    image_metas = np.array([76, 128, 1]).astype(dtype)
    anchor_box = Tensor(anchor)
    image_metas_box = Tensor(image_metas)
    indices_anchor_box = Tensor(np.array([i for i in range(anchor_box.shape[0])])).astype(np.int32)
    indices_image_metas_box = Tensor(np.array([i for i in range(image_metas_box.shape[0])])).astype(np.int32)
    expect = np.array([True, True, False], np.bool)

    context.set_context(mode=context.PYNATIVE_MODE)
    boundingbox_decode = CheckValidDynNet()
    output = boundingbox_decode(anchor_box, image_metas_box, indices_anchor_box, indices_image_metas_box)
    assert np.array_equal(output.asnumpy(), expect)

    context.set_context(mode=context.GRAPH_MODE)
    boundingbox_decode = CheckValidDynNet()
    output = boundingbox_decode(anchor_box, image_metas_box, indices_anchor_box, indices_image_metas_box)
    assert np.array_equal(output.asnumpy(), expect)
