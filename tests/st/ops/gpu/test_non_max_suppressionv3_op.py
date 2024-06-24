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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.nms = P.NonMaxSuppressionV3()

    def construct(self, boxes, scores, max_output_size, iou_threshold, score_threshold):
        return self.nms(boxes, scores, max_output_size, iou_threshold, score_threshold)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_boxes_float32_scores_float32():
    """
    Feature: test NonMaxSuppressionV3
    Description: test cases for NonMaxSuppressionV3
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    boxes = Tensor(np.array([[70, 70, 45, 75], [30, 33, 43, 29]]), mindspore.float32)
    scores = Tensor(np.array([0.6, 0.1]), mindspore.float32)
    max_output_size = Tensor(2, mindspore.int32)
    score_threshold = Tensor(0.05, mindspore.float16)
    iou_threshold = Tensor(0.7, mindspore.float16)
    expected_idx = np.array([0, 1])
    op = Net()
    sel_idx = op(boxes, scores, max_output_size, iou_threshold, score_threshold)
    assert np.array_equal(sel_idx.asnumpy(), expected_idx)
