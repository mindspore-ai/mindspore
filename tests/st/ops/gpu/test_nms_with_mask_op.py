# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore
from mindspore import Tensor
from mindspore.ops import operations as P


def runMSRun(op, bbox):
    inputs = Tensor(bbox, mindspore.float32)
    box, _, mask = op(inputs)
    box = box.asnumpy()
    mask = mask.asnumpy()
    sel_idx = np.where(mask)
    sel_rows = box[sel_idx][:, 0:4]
    sel_score = box[sel_idx][:, -1]
    return sel_rows, sel_score


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nms_with_mask_check_order():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    nms_op = P.NMSWithMask(0.5)
    for _ in range(10):
        count = 4000
        box = np.random.randint(1, 100, size=(count, 4))
        box[:, 2] = box[:, 0] + box[:, 2]
        box[:, 3] = box[:, 1] + box[:, 3]
        unsorted_scores = np.random.rand(count, 1)
        bbox = np.hstack((box, unsorted_scores))
        bbox = Tensor(bbox, dtype=mindspore.float32)
        prop, _, _ = nms_op(bbox)
        ms_sorted_scores = (prop.asnumpy()[:, -1])  # select just scores
        np_sorted_scores = (np.sort(unsorted_scores, axis=0)[::-1][:, 0])  # sort manually
        np.testing.assert_array_almost_equal(
            ms_sorted_scores, np_sorted_scores)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nms_with_mask_edge_case_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # CASE 1  - FULL OVERLAP BOXES - Every box is duplicated and has a different score
    nms_op1 = P.NMSWithMask(0.3)
    bbox1 = [[12, 4, 33, 17, 0.6], [20, 11, 38, 23, 0.1], [20, 10, 45, 26, 0.9], [15, 17, 35, 38, 0.5],
             [10, 20, 30, 40, 0.4], [35, 35, 89, 90, 0.8], [12, 4, 33, 17, 0.3], [20, 11, 38, 23, 0.2],
             [20, 10, 45, 26, 0.1], [15, 17, 35, 38, 0.8], [10, 20, 30, 40, 0.41], [35, 35, 89, 90, 0.82]]
    expected_bbox = np.array([[20., 10., 45., 26.],
                              [35., 35., 89., 90.],
                              [15., 17., 35., 38.],
                              [12., 4., 33., 17.]])
    expected_score = np.array([0.9, 0.82, 0.8, 0.6])

    sel_rows, sel_score = runMSRun(nms_op1, bbox1)
    np.testing.assert_almost_equal(sel_rows, expected_bbox)
    np.testing.assert_almost_equal(sel_score, expected_score)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nms_with_mask_edge_case_2():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # CASE 2 - 0 value boxes - with valid scores
    nms_op2 = P.NMSWithMask(0.5)
    bbox2 = [[0, 0, 0, 0, 0.6], [0, 0, 0, 0, 0.1]]
    expected_bbox = np.array([[0., 0., 0., 0.],
                              [0., 0., 0., 0.]])
    expected_score = np.array([0.6, 0.1])

    sel_rows, sel_score = runMSRun(nms_op2, bbox2)
    np.testing.assert_almost_equal(sel_rows, expected_bbox)
    np.testing.assert_almost_equal(sel_score, expected_score)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nms_with_mask_edge_case_3():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # CASE 3 - x2/x1 and y2/y1 sequence out of place
    nms_op3 = P.NMSWithMask(0.7)
    bbox3 = [[70, 70, 45, 75, 0.6], [30, 33, 43, 29, 0.1]]
    expected_bbox = np.array([[70., 70., 45., 75.],
                              [30., 33., 43., 29.]])
    expected_score = np.array([0.6, 0.1])

    sel_rows, sel_score = runMSRun(nms_op3, bbox3)
    np.testing.assert_almost_equal(sel_rows, expected_bbox)
    np.testing.assert_almost_equal(sel_score, expected_score)
