# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor


class Net(nn.Cell):
    def __init__(self, method_="bilinear", extrapolation_value_=0.0):
        super(Net, self).__init__()
        self.op = ops.CropAndResize(method=method_,
                                    extrapolation_value=extrapolation_value_)

    def construct(self, image, boxes, box_index, channel):
        return self.op(image, boxes, box_index, channel)


class GradNet(nn.Cell):
    def __init__(self, network):
        super(GradNet, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network

    def construct(self, image, boxes, box_index, channel):
        gout = self.grad(self.network)(image, boxes, box_index, channel)
        return gout


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_crop_and_resize_grad(mode):
    """
    Feature: CropAndResize grad
    Description: CropAndResize Grad with list input
    Expectation: success
    """
    ms.set_context(mode=mode)
    batch_size = 2
    image_height = 50
    image_width = 30
    channels = 3
    crop_size = (5, 3)
    offset = 0
    total_values = batch_size * image_height * image_width * channels
    input_data = np.arange(0 + offset, total_values + offset).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array([[0.23, 0.5, 0.75, 0.0], [0, 0.1, 0.75,
                                                     1.75]]).astype(np.float32)
    input_box_index = np.array([1, 0]).astype(np.int32)
    input_data_tensor = Tensor(input_data.astype(np.float32))
    input_boxes_tensor = Tensor(input_boxes)
    input_box_index_tensor = Tensor(input_box_index)
    net = Net("bilinear", 0.0)
    output = GradNet(net)(input_data_tensor, input_boxes_tensor,
                          input_box_index_tensor, crop_size)
    print(output)
