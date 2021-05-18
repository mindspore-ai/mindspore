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
"""Coco metrics utils"""

import os
import json
import numpy as np
from src.model_utils.config import config
from .box_utils import ssd_bboxes_decode


def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return keep


def metrics(pred_data):
    """Calculate mAP of predicted bboxes."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    num_classes = config.num_classes

    coco_root = os.path.join(config.data_path, "coco_ori")
    data_type = config.val_data_type

    # Classes need to train or test.
    val_cls = config.coco_classes
    val_cls_dict = {}
    for i, cls in enumerate(val_cls):
        val_cls_dict[i] = cls

    anno_json = os.path.join(coco_root, config.instances_set.format(data_type))
    coco_gt = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco_gt.loadCats(coco_gt.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["name"]] = cat["id"]

    predictions = []
    img_ids = []

    for sample in pred_data:
        pred_boxes = sample['boxes']
        box_scores = sample['box_scores']
        img_id = sample['img_id']
        h, w = sample['image_shape']

        pred_boxes = ssd_bboxes_decode(pred_boxes)
        final_boxes = []
        final_label = []
        final_score = []
        img_ids.append(img_id)

        for c in range(1, num_classes):
            class_box_scores = box_scores[:, c]
            score_mask = class_box_scores > config.min_score
            class_box_scores = class_box_scores[score_mask]
            class_boxes = pred_boxes[score_mask] * [h, w, h, w]

            if score_mask.any():
                nms_index = apply_nms(
                    class_boxes, class_box_scores, config.nms_thershold, config.max_boxes)
                class_boxes = class_boxes[nms_index]
                class_box_scores = class_box_scores[nms_index]

                final_boxes += class_boxes.tolist()
                final_score += class_box_scores.tolist()
                final_label += [classs_dict[val_cls_dict[c]]] * \
                    len(class_box_scores)

        for loc, label, score in zip(final_boxes, final_label, final_score):
            res = {}
            res['image_id'] = img_id
            res['bbox'] = [loc[1], loc[0], loc[3] - loc[1], loc[2] - loc[0]]
            res['score'] = score
            res['category_id'] = label
            predictions.append(res)
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes('predictions.json')
    E = COCOeval(coco_gt, coco_dt, iouType='bbox')
    E.params.imgIds = img_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    return E.stats[0]
