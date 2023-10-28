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
from mindspore import context, Tensor
from mindspore.common import dtype as mstype
import mindspore.ops.operations.image_ops as P
from mindspore import nn


class NetCropAndResizeGradImage(nn.Cell):
    def __init__(self, grads_type, method_="bilinear"):
        super(NetCropAndResizeGradImage, self).__init__()
        self.op = P.CropAndResizeGradImage(grads_type, method_)

    def construct(self, grads, boxes, box_index, image_size):
        return self.op(grads, boxes, box_index, image_size)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("input_type", [np.float32, np.float64])
@pytest.mark.parametrize("output_type", [np.float16, np.float32, np.float64])
def test_crop_and_resize_grad_image_bilinear(input_type, output_type):
    """
    Feature: Test CropAndResizeGradImage.
    Description: Attributes method is bilinear.
    Expectation: Check it by expected_output variable.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    input_grads = 1e-4 * np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    input_boxes = np.array([[0.1, 0.2, 0.3, 0.4]])
    input_box_index = np.array([0]).astype(np.int32)
    input_image_size = np.array([1, 4, 4, 1]).astype(np.int32)
    input_grads_tensor = Tensor(input_grads.astype(input_type))
    input_boxes_tensor = Tensor(input_boxes.astype(input_type))
    input_box_index_tensor = Tensor(input_box_index, mstype.int32)
    input_image_size_tensor = Tensor(input_image_size, mstype.int32)
    if output_type == np.float16:
        grads_type = mstype.float16
    elif output_type == np.float32:
        grads_type = mstype.float32
    else:
        grads_type = mstype.float64
    loss = 1e-3
    net = NetCropAndResizeGradImage(grads_type, method_="bilinear")
    output = net(input_grads_tensor, input_boxes_tensor, input_box_index_tensor, input_image_size_tensor)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[0.00004], [0.000204], [0.000036], [0.]],
                                 [[0.00012], [0.000516], [0.000084], [0.]],
                                 [[0.], [0.], [0.], [0.]],
                                 [[0.], [0.], [0.], [0.]]]]).astype(output_type)
    error = np.ones(shape=[1, 4, 4, 1]) * loss
    diff = output_ms - expected_output
    assert np.all(abs(diff) < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("input_type", [np.float32, np.float64])
@pytest.mark.parametrize("output_type", [np.float16, np.float32, np.float64])
def test_crop_and_resize_grad_image_nearest(input_type, output_type):
    """
    Feature: Test CropAndResizeGradImage.
    Description: Attributes method is nearest.
    Expectation: Check it by expected_output variable.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_grads = 1e-6 * np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    input_boxes = np.array([[0.1, 0.2, 0.3, 0.4]])
    input_box_index = np.array([0]).astype(np.int32)
    input_image_size = np.array([1, 4, 4, 1]).astype(np.int32)
    input_grads_tensor = Tensor(input_grads.astype(input_type))
    input_boxes_tensor = Tensor(input_boxes.astype(input_type))
    input_box_index_tensor = Tensor(input_box_index, mstype.int32)
    input_image_size_tensor = Tensor(input_image_size, mstype.int32)
    if output_type == np.float16:
        grads_type = mstype.float16
    elif output_type == np.float32:
        grads_type = mstype.float32
    else:
        grads_type = mstype.float64
    loss = 1e-3
    net = NetCropAndResizeGradImage(grads_type, method_="nearest")
    output = net(input_grads_tensor, input_boxes_tensor, input_box_index_tensor, input_image_size_tensor)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[0], [0.000003], [0], [0]],
                                 [[0], [0.000007], [0], [0]],
                                 [[0], [0], [0], [0]],
                                 [[0], [0], [0], [0]]]]).astype(output_type)
    error = np.ones(shape=[1, 4, 4, 1]) * loss
    diff = output_ms - expected_output
    assert np.all(abs(diff) < error)
