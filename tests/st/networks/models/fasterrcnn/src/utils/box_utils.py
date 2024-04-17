# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import ops


@ops.constexpr
def tensor(x, dtype=ms.float32):
    return ms.Tensor(x, dtype)


@ops.constexpr
def shape_prod(shape):
    size = 1
    for i in shape:
        size *= i
    return size


def delta2bbox(deltas, boxes, weights=(10.0, 10.0, 5.0, 5.0), max_shape=None):
    """Decode deltas to boxes.
    Note: return tensor shape [n,1,4]
    """
    clip_scale = 4

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0:1] / wx
    dy = deltas[:, 1:2] / wy
    dw = deltas[:, 2:3] / ww
    dh = deltas[:, 3:4] / wh
    # Prevent sending too large values into ops.exp()
    dw = ops.minimum(dw, clip_scale)
    dh = ops.minimum(dh, clip_scale)

    pred_ctr_x = dx * ops.expand_dims(widths, 1) + ops.expand_dims(ctr_x, 1)
    pred_ctr_y = dy * ops.expand_dims(heights, 1) + ops.expand_dims(ctr_y, 1)
    pred_w = ops.exp(dw) * ops.expand_dims(widths, 1)
    pred_h = ops.exp(dh) * ops.expand_dims(heights, 1)

    pred_boxes = []
    pred_boxes.append(pred_ctr_x - 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y - 0.5 * pred_h)
    pred_boxes.append(pred_ctr_x + 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y + 0.5 * pred_h)
    pred_boxes = ops.stack(pred_boxes, axis=-1)

    if max_shape is not None:
        h, w = max_shape
        x1 = ops.clip_by_value(pred_boxes[..., 0], 0, w - 1)
        y1 = ops.clip_by_value(pred_boxes[..., 1], 0, h - 1)
        x2 = ops.clip_by_value(pred_boxes[..., 2], 0, w - 1)
        y2 = ops.clip_by_value(pred_boxes[..., 3], 0, h - 1)
        pred_boxes = ops.stack((x1, y1, x2, y2), -1)

    return pred_boxes


def bbox2delta(src_boxes, tgt_boxes, weights=(10.0, 10.0, 5.0, 5.0)):
    """Encode bboxes to deltas."""
    src_w = src_boxes[:, 2] - src_boxes[:, 0]
    src_h = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_h

    tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_vaild = ops.logical_and(tgt_w > 0, tgt_h > 0)
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h
    tgt_w = ops.select(tgt_vaild, tgt_w, src_w)
    tgt_h = ops.select(tgt_vaild, tgt_h, src_h)
    tgt_ctr_x = ops.select(tgt_vaild, tgt_ctr_x, src_ctr_x)
    tgt_ctr_y = ops.select(tgt_vaild, tgt_ctr_y, src_ctr_y)

    wx, wy, ww, wh = weights
    dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
    dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
    dw = ww * ops.log(tgt_w / src_w)
    dh = wh * ops.log(tgt_h / src_h)

    deltas = ops.stack((dx, dy, dw, dh), axis=1)
    return deltas


def xywh2xyxy(box):
    """box shape is (N, 4), format is xywh"""
    x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    x1 = x - w // 2
    y1 = y - h // 2
    x2 = x1 + w
    y2 = y1 + h
    return ops.stack((x1, y1, x2, y2), -1)


def xyxy2xywh(box):
    """box shape is (N, 4), format is xyxy"""
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    w = x2 - x1
    h = y2 - y1
    x = x1 + w // 2
    y = y1 + h // 2
    return ops.stack((x, y, w, h), -1)
