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
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops.operations.image_ops import CombinedNonMaxSuppression


class NetCombinedNonMaxSuppression(nn.Cell):
    def __init__(self, pad_per_class, clip_boxes):
        super(NetCombinedNonMaxSuppression, self).__init__()
        self.combined_non_max_suppression = CombinedNonMaxSuppression(
            pad_per_class, clip_boxes)

    def construct(self, boxes, scores, max_output_size_per_class, max_total_size, iou_threshold, score_threshold):
        return self.combined_non_max_suppression(boxes, scores, max_output_size_per_class, max_total_size,
                                                 iou_threshold, score_threshold)


def cmp(output, expect):
    output_type = output.asnumpy().dtype
    expect_type = expect.asnumpy().dtype
    diff0 = output.asnumpy() - expect
    error0 = np.zeros(shape=expect.shape)
    assert np.all(diff0 == error0)
    assert output.shape == expect.shape
    assert output_type == expect_type


def test_combined_non_max_suppresion1():
    """
    Feature:   Combined non max suppression.
    Description: test case for Combined non max suppression.
    Expectation: The result are as expected.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    boxes = Tensor(np.array([[[[200, 100, 150, 100]], [[220, 120, 150, 100]], [[190, 110, 150, 100]],
                              [[210, 112, 150, 100]]]], dtype=np.float32))
    scores = Tensor(np.array([[[0.2000, 0.7000, 0.1000], [0.1000, 0.8000, 0.1000],
                               [0.3000, 0.6000, 0.1000], [0.0500, 0.9000, 0.0500]]], dtype=np.float32))
    max_output_size_per_class = Tensor(4, dtype=mstype.int32)
    max_total_size = Tensor(1, dtype=mstype.int32)
    iou_threshold = Tensor(0, dtype=mstype.float32)
    score_threshold = Tensor(0, dtype=mstype.float32)
    nmsed_boxes_expect = Tensor(
        np.array([[[1., 1., 1., 1.]]], dtype=np.float32))
    nmsed_scores_expect = Tensor(np.array([[0.9]], dtype=np.float32))
    nmsed_classes_expect = Tensor(np.array([[1.]], dtype=np.float32))
    valid_detections_expect = Tensor(np.array([1], dtype=np.int32))
    net = NetCombinedNonMaxSuppression(False, True)
    output = net(boxes, scores, max_output_size_per_class,
                 max_total_size, iou_threshold, score_threshold)
    output_nmsed_boxes = output[0]
    output_nmsed_scores = output[1]
    output_nmsed_classes = output[2]
    output_valid_detections = output[3]
    cmp(output_nmsed_boxes, nmsed_boxes_expect)
    cmp(output_nmsed_scores, nmsed_scores_expect)
    cmp(output_nmsed_classes, nmsed_classes_expect)
    cmp(output_valid_detections, valid_detections_expect)


def test_combined_non_max_suppresion2():
    """
    Feature:   Combined non max suppression.
    Description: test case for Combined non max suppression.
    Expectation: The result are as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    boxes = Tensor(np.array([[[[200, 100, 150, 100]], [[220, 120, 150, 100]], [[190, 110, 150, 100]],
                              [[210, 112, 150, 100]]]], dtype=np.float32))
    scores = Tensor(np.array([[[0.2000, 0.7000, 0.1000], [0.1000, 0.8000, 0.1000],
                               [0.3000, 0.6000, 0.1000], [0.0500, 0.9000, 0.0500]]], dtype=np.float32))
    max_output_size_per_class = Tensor(4, dtype=mstype.int32)
    max_total_size = Tensor(1, dtype=mstype.int32)
    iou_threshold = Tensor(0, dtype=mstype.float32)
    score_threshold = Tensor(0, dtype=mstype.float32)
    nmsed_boxes_expect = Tensor(
        np.array([[[1., 1., 1., 1.]]], dtype=np.float32))
    nmsed_scores_expect = Tensor(np.array([[0.9]], dtype=np.float32))
    nmsed_classes_expect = Tensor(np.array([[1.]], dtype=np.float32))
    valid_detections_expect = Tensor(np.array([1], dtype=np.int32))
    net = NetCombinedNonMaxSuppression(False, True)
    output = net(boxes, scores, max_output_size_per_class,
                 max_total_size, iou_threshold, score_threshold)
    output_nmsed_boxes = output[0]
    output_nmsed_scores = output[1]
    output_nmsed_classes = output[2]
    output_valid_detections = output[3]
    cmp(output_nmsed_boxes, nmsed_boxes_expect)
    cmp(output_nmsed_scores, nmsed_scores_expect)
    cmp(output_nmsed_classes, nmsed_classes_expect)
    cmp(output_valid_detections, valid_detections_expect)


def test_combined_non_max_suppresion3():
    """
    Feature:   Combined non max suppression.
    Description: test case for Combined non max suppression.
    Expectation: The result are as expected.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    boxes = Tensor(np.array([[[[200, 100, 150, 100]], [[220, 120, 150, 100]], [[190, 110, 150, 100]],
                              [[210, 112, 150, 100]]]], dtype=np.float32))
    scores = Tensor(np.array([[[0.2000, 0.7000, 0.1000], [0.1000, 0.8000, 0.1000],
                               [0.3000, 0.6000, 0.1000], [0.0500, 0.9000, 0.0500]]], dtype=np.float32))
    max_output_size_per_class = Tensor(4, dtype=mstype.int32)
    max_total_size = Tensor(1, dtype=mstype.int32)
    iou_threshold = Tensor(0.2, dtype=mstype.float32)
    score_threshold = Tensor(0.2, dtype=mstype.float32)
    nmsed_boxes_expect = Tensor(
        np.array([[[210., 112., 150., 100.]]], dtype=np.float32))
    nmsed_scores_expect = Tensor(np.array([[0.9]], dtype=np.float32))
    nmsed_classes_expect = Tensor(np.array([[1.]], dtype=np.float32))
    valid_detections_expect = Tensor(np.array([1], dtype=np.int32))
    net = NetCombinedNonMaxSuppression(True, False)
    output = net(boxes, scores, max_output_size_per_class,
                 max_total_size, iou_threshold, score_threshold)
    output_nmsed_boxes = output[0]
    output_nmsed_scores = output[1]
    output_nmsed_classes = output[2]
    output_valid_detections = output[3]
    cmp(output_nmsed_boxes, nmsed_boxes_expect)
    cmp(output_nmsed_scores, nmsed_scores_expect)
    cmp(output_nmsed_classes, nmsed_classes_expect)
    cmp(output_valid_detections, valid_detections_expect)


def test_combined_non_max_suppresion4():
    """
    Feature:   Combined non max suppression.
    Description: test case for Combined non max suppression.
    Expectation: The result are as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    boxes = Tensor(np.array([[[[200, 100, 150, 100]], [[220, 120, 150, 100]], [[190, 110, 150, 100]],
                              [[210, 112, 150, 100]]]], dtype=np.float32))
    scores = Tensor(np.array([[[0.2000, 0.7000, 0.1000], [0.1000, 0.8000, 0.1000],
                               [0.3000, 0.6000, 0.1000], [0.0500, 0.9000, 0.0500]]], dtype=np.float32))
    max_output_size_per_class = Tensor(4, dtype=mstype.int32)
    max_total_size = Tensor(1, dtype=mstype.int32)
    iou_threshold = Tensor(0, dtype=mstype.float32)
    score_threshold = Tensor(0, dtype=mstype.float32)
    nmsed_boxes_expect = Tensor(
        np.array([[[210., 112., 150., 100.]]], dtype=np.float32))
    nmsed_scores_expect = Tensor(np.array([[0.9]], dtype=np.float32))
    nmsed_classes_expect = Tensor(np.array([[1.]], dtype=np.float32))
    valid_detections_expect = Tensor(np.array([1], dtype=np.int32))
    net = NetCombinedNonMaxSuppression(True, False)
    output = net(boxes, scores, max_output_size_per_class,
                 max_total_size, iou_threshold, score_threshold)
    output_nmsed_boxes = output[0]
    output_nmsed_scores = output[1]
    output_nmsed_classes = output[2]
    output_valid_detections = output[3]
    cmp(output_nmsed_boxes, nmsed_boxes_expect)
    cmp(output_nmsed_scores, nmsed_scores_expect)
    cmp(output_nmsed_classes, nmsed_classes_expect)
    cmp(output_valid_detections, valid_detections_expect)
