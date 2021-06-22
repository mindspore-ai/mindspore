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
"""Utils."""
from itertools import product
from math import ceil
import numpy as np

def prior_box(image_size, min_sizes, steps, clip=False):
    """prior box"""
    feature_maps = [
        [ceil(image_size[0] / step), ceil(image_size[1] / step)]
        for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                if min_size == 32:
                    dense_cx = [x * steps[k] / image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
                    dense_cy = [y * steps[k] / image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
                elif min_size == 64:
                    dense_cx = [x * steps[k] / image_size[1] for x in [j+0, j+0.5]]
                    dense_cy = [y * steps[k] / image_size[0] for y in [i+0, i+0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
                else:
                    cx = (j + 0.5) * steps[k] / image_size[1]
                    cy = (i + 0.5) * steps[k] / image_size[0]
                    anchors += [cx, cy, s_kx, s_ky]

    output = np.asarray(anchors).reshape([-1, 4]).astype(np.float32)
    if clip:
        output = np.clip(output, 0, 1)
    return output

def center_point_2_box(boxes):
    return np.concatenate((boxes[:, 0:2] - boxes[:, 2:4] / 2,
                           boxes[:, 0:2] + boxes[:, 2:4] / 2), axis=1)

def compute_intersect(a, b):
    """compute_intersect"""
    A = a.shape[0]
    B = b.shape[0]
    max_xy = np.minimum(
        np.broadcast_to(np.expand_dims(a[:, 2:4], 1), [A, B, 2]),
        np.broadcast_to(np.expand_dims(b[:, 2:4], 0), [A, B, 2]))
    min_xy = np.maximum(
        np.broadcast_to(np.expand_dims(a[:, 0:2], 1), [A, B, 2]),
        np.broadcast_to(np.expand_dims(b[:, 0:2], 0), [A, B, 2]))
    inter = np.maximum((max_xy - min_xy), np.zeros_like(max_xy - min_xy))
    return inter[:, :, 0] * inter[:, :, 1]

def compute_overlaps(a, b):
    """compute_overlaps"""
    inter = compute_intersect(a, b)
    area_a = np.broadcast_to(
        np.expand_dims(
            (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1),
        np.shape(inter))
    area_b = np.broadcast_to(
        np.expand_dims(
            (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 0),
        np.shape(inter))
    union = area_a + area_b - inter
    return inter / union

def match(threshold, boxes, priors, var, labels):
    """match"""
    # compute IoU
    overlaps = compute_overlaps(boxes, center_point_2_box(priors))
    # bipartite matching
    best_prior_overlap = overlaps.max(1, keepdims=True)
    best_prior_idx = np.argsort(-overlaps, axis=1)[:, 0:1]
    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        loc = np.zeros((priors.shape[0], 4), dtype=np.float32)
        conf = np.zeros((priors.shape[0],), dtype=np.int32)
        return loc, conf
    # best ground truth for each prior
    best_truth_overlap = overlaps.max(0, keepdims=True)
    best_truth_idx = np.argsort(-overlaps, axis=0)[:1, :]
    best_truth_idx = best_truth_idx.squeeze(0)
    best_truth_overlap = best_truth_overlap.squeeze(0)
    best_prior_idx = best_prior_idx.squeeze(1)
    best_prior_idx_filter = best_prior_idx_filter.squeeze(1)
    best_truth_overlap[best_prior_idx_filter] = 2
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j
    matches = boxes[best_truth_idx]
    # encode boxes
    offset_cxcy = (matches[:, 0:2] + matches[:, 2:4]) / 2 - priors[:, 0:2]
    offset_cxcy /= (var[0] * priors[:, 2:4])
    wh = (matches[:, 2:4] - matches[:, 0:2]) / priors[:, 2:4]
    wh[wh == 0] = 1e-12
    wh = np.log(wh) / var[1]
    loc = np.concatenate([offset_cxcy, wh], axis=1)
    # set labels
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0

    return loc, np.array(conf, dtype=np.int32)

class bbox_encode():
    """bbox_encode"""
    def __init__(self, cfg):
        self.match_thresh = cfg['match_thresh']
        self.variances = cfg['variance']
        self.priors = prior_box(cfg['image_size'], cfg['min_sizes'], cfg['steps'], cfg['clip'])

    def __call__(self, image, targets):
        boxes = targets[:, :4]
        labels = targets[:, -1]
        priors = self.priors
        loc_t, conf_t = match(self.match_thresh, boxes, priors, self.variances, labels)

        return image, loc_t, conf_t

def decode_bbox(bbox, priors, var):
    """decode_bbox"""
    boxes = np.concatenate((
        priors[:, 0:2] + bbox[:, 0:2] * var[0] * priors[:, 2:4],
        priors[:, 2:4] * np.exp(bbox[:, 2:4] * var[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
