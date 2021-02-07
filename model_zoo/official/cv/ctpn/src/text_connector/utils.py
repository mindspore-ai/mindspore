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


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.

    Args:
        boxes(numpy.array):bounding box.
        im_shape(numpy.array): image shape.

    Return:
        boxes(numpy.array):boundding box after clip.
    """
    boxes[:, 0::2] = threshold(boxes[:, 0::2], 0, im_shape[1] - 1)
    boxes[:, 1::2] = threshold(boxes[:, 1::2], 0, im_shape[0] - 1)
    return boxes

def overlaps_v(text_proposals, index1, index2):
    """
    Calculate vertical overlap ratio.

    Args:
        text_proposals(numpy.array): Text proposlas.
        index1(int): First text proposal.
        index2(int): Second text proposal.

    Return:
        overlap(float32): vertical overlap.
    """
    h1 = text_proposals[index1][3] - text_proposals[index1][1] + 1
    h2 = text_proposals[index2][3] - text_proposals[index2][1] + 1
    y0 = max(text_proposals[index2][1], text_proposals[index1][1])
    y1 = min(text_proposals[index2][3], text_proposals[index1][3])
    return max(0, y1 - y0 + 1) / min(h1, h2)

def size_similarity(heights, index1, index2):
    """
    Calculate vertical size similarity ratio.

    Args:
        heights(numpy.array): Text proposlas heights.
        index1(int): First text proposal.
        index2(int): Second text proposal.

    Return:
        overlap(float32): vertical overlap.
    """
    h1 = heights[index1]
    h2 = heights[index2]
    return min(h1, h2) / max(h1, h2)

def fit_y(X, Y, x1, x2):
    if np.sum(X == X[0]) == len(X):
        return Y[0], Y[0]
    p = np.poly1d(np.polyfit(X, Y, 1))
    return p(x1), p(x2)

def nms(bboxs, thresh):
    """
    Args:
        text_proposals(numpy.array): tex proposals
        index1(int): text_proposal index
        tindex2(int): text proposal index
    """
    x1, y1, x2, y2, scores = np.split(bboxs, 5, axis=1)
    x1 = bboxs[:, 0]
    y1 = bboxs[:, 1]
    x2 = bboxs[:, 2]
    y2 = bboxs[:, 3]
    scores = bboxs[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    num_dets = bboxs.shape[0]
    suppressed = np.zeros(num_dets, dtype=np.int32)
    keep = []
    for _i in range(num_dets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        x1_i = x1[i]
        y1_i = y1[i]
        x2_i = x2[i]
        y2_i = y2[i]
        area_i = areas[i]
        for _j in range(_i + 1, num_dets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            x1_j = max(x1_i, x1[j])
            y1_j = max(y1_i, y1[j])
            x2_j = min(x2_i, x2[j])
            y2_j = min(y2_i, y2[j])
            w = max(0.0, x2_j - x1_j + 1)
            h = max(0.0, y2_j - y1_j + 1)
            inter = w*h
            overlap = inter / (area_i+areas[j]-inter)
            if overlap >= thresh:
                suppressed[j] = 1
    return keep
