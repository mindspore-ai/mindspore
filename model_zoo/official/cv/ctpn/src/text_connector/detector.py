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
from src.config import config
from src.text_connector.utils import nms
from src.text_connector.connect_text_lines import connect_text_lines

def filter_proposal(proposals, scores):
    """
    Filter text proposals

    Args:
        proposals(numpy.array): Text proposals.
    Returns:
        proposals(numpy.array): Text proposals after filter.
    """
    inds = np.where(scores > config.text_proposals_min_scores)[0]
    keep_proposals = proposals[inds]
    keep_scores = scores[inds]
    sorted_inds = np.argsort(keep_scores.ravel())[::-1]
    keep_proposals, keep_scores = keep_proposals[sorted_inds], keep_scores[sorted_inds]
    nms_inds = nms(np.hstack((keep_proposals, keep_scores)), config.text_proposals_nms_thresh)
    keep_proposals, keep_scores = keep_proposals[nms_inds], keep_scores[nms_inds]
    return keep_proposals, keep_scores

def filter_boxes(boxes):
    """
    Filter text boxes

    Args:
        boxes(numpy.array): Text boxes.
    Returns:
        boxes(numpy.array): Text boxes after filter.
    """
    heights = np.zeros((len(boxes), 1), np.float)
    widths = np.zeros((len(boxes), 1), np.float)
    scores = np.zeros((len(boxes), 1), np.float)
    index = 0
    for box in boxes:
        widths[index] = abs(box[2] - box[0])
        heights[index] = abs(box[3] - box[1])
        scores[index] = abs(box[4])
        index += 1
    return np.where((widths / heights > config.min_ratio) & (scores > config.line_min_score) &\
        (widths > (config.text_proposals_width * config.min_num_proposals)))[0]

def detect(text_proposals, scores, size):
    """
    Detect text boxes

    Args:
        text_proposals(numpy.array): Predict text proposals.
        scores(numpy.array): Bbox predicts scores.
        size(numpy.array): Image size.
    Returns:
        boxes(numpy.array): Text boxes after connect.
    """
    keep_proposals, keep_scores = filter_proposal(text_proposals, scores)
    connect_boxes = connect_text_lines(keep_proposals, keep_scores, size)
    boxes = connect_boxes[filter_boxes(connect_boxes)]
    return boxes
