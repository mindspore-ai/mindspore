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
from mindspore import Tensor
from mindspore.ops.operations.image_ops import NonMaxSuppressionWithOverlaps
import mindspore.common.dtype as ms


class NmsWithOverlaps(nn.Cell):
    def __init__(self):
        super().__init__()
        self.nms = NonMaxSuppressionWithOverlaps()

    def construct(self, overlaps, scores, max_output_size, overlap_threshold, score_threshold):
        return self.nms(overlaps, scores, max_output_size, overlap_threshold, score_threshold)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nms_with_overlaps():
    """
    Feature:  NonMaxSuppressionWithOverlaps 5 inputs and 1 output.
    Description: Compatible with Tensorflow's NonMaxSuppressionWithOverlaps.
    Expectation: The result matches numpy implementation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    overlaps = np.array([[1, 0.5, 0.8, 0.45], [0.5, 1, 0.5, 0.7], [0.8, 0.5, 1, 0.7],
                         [0.45, 0.7, 0.7, 1]])
    scores = np.array([4, 2, 3, 1]).astype(np.float32)
    max_output_size = 2
    overlap_threshold = 0.6
    score_threshold = 1.5
    expected_out = np.array([0, 1])

    net = NmsWithOverlaps()
    out = net(Tensor(overlaps, ms.float32), Tensor(scores, ms.float32), Tensor(max_output_size, ms.int32),
              Tensor(overlap_threshold, ms.float32), Tensor(score_threshold, ms.float32))
    np.testing.assert_almost_equal(out, expected_out)
