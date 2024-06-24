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

import pytest
import numpy as np

import mindspore.ops.operations.image_ops as P
from mindspore.common import dtype as mstype
from mindspore import Tensor, nn, context


class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()
        self.op = P.CombinedNonMaxSuppression()

    def construct(self, boxes, scores, max_output_size_per_class,
                  max_total_size, iou_threshold, score_threshold):
        return self.op(boxes, scores, max_output_size_per_class,
                       max_total_size, iou_threshold, score_threshold)


def dyn_case():
    net = Net()

    boxes_dyn = Tensor(shape=[None, None, None, 4], dtype=mstype.float32)
    scores_dyn = Tensor(shape=[None, None, None], dtype=mstype.float32)
    max_output_size_per_class = Tensor(4, mstype.int32)
    max_total_size = Tensor(1, mstype.int32)
    iou_threshold = Tensor(0, mstype.float32)
    score_threshold = Tensor(0, mstype.float32)

    net.set_inputs(boxes_dyn, scores_dyn, max_output_size_per_class,
                   max_total_size, iou_threshold, score_threshold)

    boxes = Tensor(
        np.array([[[[200, 100, 150, 100]], [[220, 120, 150, 100]],
                   [[190, 110, 150, 100]], [[210, 112, 150,
                                             100]]]])).astype('float32')
    scores = Tensor(
        np.array([[[0.2000, 0.7000, 0.1000], [0.1000, 0.8000, 0.1000],
                   [0.3000, 0.6000, 0.1000], [0.0500, 0.9000,
                                              0.0500]]])).astype('float32')

    out = net(boxes, scores, max_output_size_per_class, max_total_size,
              iou_threshold, score_threshold)
    expect_shapes = [(1, 1, 4), (1, 1), (1, 1), (1,)]
    for i in range(4):
        assert out[i].asnumpy().shape == expect_shapes[i]


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_combined_non_max_suppression_dyn():
    """
    Feature: test CombinedNonMaxSuppression in PyNative and Graph modes.
    Description: test dynamic shape case.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    dyn_case()
