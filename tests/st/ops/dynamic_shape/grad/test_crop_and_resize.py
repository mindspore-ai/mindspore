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
from mindspore import nn, context, Tensor
from mindspore.ops import operations as P
from .test_grad_of_dynamic import TestDynamicGrad

context.set_context(mode=context.PYNATIVE_MODE)


class NetCropAndResize(nn.Cell):
    def __init__(self, method_="bilinear", extrapolation_value_=0.0):
        super(NetCropAndResize, self).__init__()
        self.op = P.CropAndResize(
            method=method_, extrapolation_value=extrapolation_value_)

    def construct(self, image, boxes, box_index, channel):
        return self.op(image, boxes, box_index, channel)


def crop_and_resize_test(is_dyn_rank):
    batch_size = 2
    image_height = 512
    image_width = 256
    channels = 3
    crop_size = (5, 3)
    offset = 5000
    total_values = batch_size * image_height * image_width * channels
    input_data = np.arange(0 + offset, total_values + offset).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array(
        [[0.23, 0.5, 0.75, 0.0], [0, 0.1, 0.75, 1.75]]).astype(np.float32)
    input_box_index = np.array([1, 0]).astype(np.int32)
    input_data_tensor = Tensor(input_data.astype(np.float32))
    input_boxes_tensor = Tensor(input_boxes)
    input_box_index_tensor = Tensor(input_box_index)

    tester = TestDynamicGrad(NetCropAndResize("bilinear", 0.0))
    tester.test_dynamic_grad_net([input_data_tensor, input_boxes_tensor,
                                  input_box_index_tensor, crop_size],
                                 is_dyn_rank)



@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_crop_and_resize_dyn_shape():
    """
    Feature: CropAndResize Grad DynamicShape.
    Description: Test case of dynamic shape for CropAndResize grad operator.
    Expectation: success.
    """
    crop_and_resize_test(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_crop_and_resize_dyn_rank():
    """
    Feature: CropAndResize Grad DynamicShape.
    Description: Test case of dynamic rank for CropAndResize grad operator.
    Expectation: success.
    """
    crop_and_resize_test(True)
