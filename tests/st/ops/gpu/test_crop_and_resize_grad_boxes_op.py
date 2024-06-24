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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
from mindspore import context, Tensor
import mindspore.ops.operations.image_ops as P
from mindspore import nn


class NetCropAndResizeGradBoxes(nn.Cell):
    def __init__(self, method_="bilinear"):
        super(NetCropAndResizeGradBoxes, self).__init__()
        self.op = P.CropAndResizeGradBoxes(method_)

    def construct(self, grads, images, boxes, box_index):
        return self.op(grads, images, boxes, box_index)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("image_type", [np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64, np.float16,
                                        np.float32, np.float64])
def test_crop_and_resize_grad_boxes_float32(image_type):
    """
    Feature: Test CropAndResizeGradBoxes.
    Description: grads, boxes type is float32, output type is float32.
    Expectation: Check it by expected_output variable.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    others_type = np.float32
    batch_size = 2
    image_height = 32
    image_width = 18
    channels = 3
    crop_height = 8
    crop_width = 9
    num_boxes = 2
    total_values_1 = num_boxes * crop_height * crop_width * channels
    input_grads = 1e-5 * np.arange(0, total_values_1).reshape((num_boxes, crop_height, crop_width, channels))
    total_values_2 = batch_size * image_height * image_width * channels
    input_image_tmp = np.arange(0, 256)
    div = total_values_2 // 256
    mod = total_values_2 % 256
    input_image = np.append(np.repeat(input_image_tmp, div), input_image_tmp[:mod]).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array([[0.1, 0.5, 0.5, 0.0], [0.1, 0, 0.75, 1.75]])
    input_box_index = np.array([1, 0]).astype(np.int32)
    input_grads_tensor = Tensor(input_grads.astype(others_type))
    input_image_tensor = Tensor(input_image.astype(image_type))
    input_boxes_tensor = Tensor(input_boxes.astype(others_type))
    input_box_index_tensor = Tensor(input_box_index)
    net = NetCropAndResizeGradBoxes()
    output = net(input_grads_tensor, input_image_tensor,
                 input_boxes_tensor, input_box_index_tensor)
    output_ms = output.asnumpy()
    expected_output = np.array([[9.326791763305664, 0.4429844617843628, 20.578969955444336, 0.3551655411720276],
                                [21.320859909057617, 0.7584426403045654, 27.210113525390625,
                                 0.38604485988616943]]).astype(others_type)
    error = np.ones(shape=[num_boxes, 4]) * 1.0e-4
    diff = output_ms - expected_output
    assert np.all(abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("image_type", [np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64, np.float16,
                                        np.float32, np.float64])
def test_crop_and_resize_grad_boxes_float64(image_type):
    """
    Feature: Test CropAndResizeGradBoxes.
    Description: grads, boxes type is float64, output type is float64.
    Expectation: Check it by expected_output variable.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    others_type = np.float64
    batch_size = 2
    image_height = 34
    image_width = 34
    channels = 3
    crop_height = 7
    crop_width = 7
    num_boxes = 2
    total_values_1 = num_boxes * crop_height * crop_width * channels
    input_grads = 1e-5 * np.arange(0, total_values_1).reshape((num_boxes, crop_height, crop_width, channels))
    total_values_2 = batch_size * image_height * image_width * channels
    input_image_tmp = np.arange(0, 256)
    div = total_values_2 // 256
    mod = total_values_2 % 256
    input_image = np.append(np.repeat(input_image_tmp, div), input_image_tmp[:mod]).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array([[0.1, 0.5, 0.5, 0.7], [0.1, 0, 0.75, 0.85]])
    input_box_index = np.array([0, 1]).astype(np.int32)
    input_grads_tensor = Tensor(input_grads.astype(others_type))
    input_image_tensor = Tensor(input_image.astype(image_type))
    input_boxes_tensor = Tensor(input_boxes.astype(others_type))
    input_box_index_tensor = Tensor(input_box_index)
    net = NetCropAndResizeGradBoxes()
    output = net(input_grads_tensor, input_image_tensor,
                 input_boxes_tensor, input_box_index_tensor)
    output_ms = output.asnumpy()
    expected_output = np.array([[4.165656089782715, 0.12503701448440552, 9.360515594482422, 0.20364297926425934],
                                [18.26944351196289, 0.6215707063674927, 23.362707138061523,
                                 1.013537049293518]]).astype(others_type)
    error = np.ones(shape=[num_boxes, 4]) * 1.0e-5
    diff = output_ms - expected_output
    assert np.all(abs(diff) < error)
