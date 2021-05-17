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
"""metrics utils"""

import numpy as np
from model_utils.config import config


def calc_iou(bbox_pred, bbox_ground):
    """Calculate iou of predicted bbox and ground truth."""
    x1 = float(bbox_pred[0])
    y1 = float(bbox_pred[1])
    width1 = float(bbox_pred[2] - bbox_pred[0])
    height1 = float(bbox_pred[3] - bbox_pred[1])

    x2 = float(bbox_ground[0])
    y2 = float(bbox_ground[1])
    width2 = float(bbox_ground[2] - bbox_ground[0])
    height2 = float(bbox_ground[3] - bbox_ground[1])

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        iou = 0
    else:
        area = width * height
        area1 = width1 * height1
        area2 = width2 * height2
        iou = area * 1. / (area1 + area2 - area)

    return iou


def metrics(pred_data):
    """Calculate precision and recall of predicted bboxes."""
    num_classes = config.num_classes
    count_corrects = [1e-6 for _ in range(num_classes)]
    count_grounds = [1e-6 for _ in range(num_classes)]
    count_preds = [1e-6 for _ in range(num_classes)]
    ious = []
    for i, sample in enumerate(pred_data):
        gt_bboxes = sample['gt_bboxes']
        gt_labels = sample['gt_labels']

        print('gt_bboxes', gt_bboxes)
        print('gt_labels', gt_labels)

        boxes = sample['boxes']
        classes = sample['labels']
        print('boxes', boxes)
        print('labels', classes)

        # metric
        count_correct = [1e-6 for _ in range(num_classes)]
        count_ground = [1e-6 for _ in range(num_classes)]
        count_pred = [1e-6 for _ in range(num_classes)]

        for gt_label in gt_labels:
            count_ground[gt_label] += 1

        for box_index, box in enumerate(boxes):
            bbox_pred = [box[0], box[1], box[2], box[3]]
            count_pred[classes[box_index]] += 1

            for gt_index, gt_label in enumerate(gt_labels):
                class_ground = gt_label

                if classes[box_index] == class_ground:
                    iou = calc_iou(bbox_pred, gt_bboxes[gt_index])
                    ious.append(iou)
                    if iou >= 0.5:
                        count_correct[class_ground] += 1
                        break

        count_corrects = [count_corrects[i] + count_correct[i] for i in range(num_classes)]
        count_preds = [count_preds[i] + count_pred[i] for i in range(num_classes)]
        count_grounds = [count_grounds[i] + count_ground[i] for i in range(num_classes)]

    precision = np.array([count_corrects[ix] / count_preds[ix] for ix in range(num_classes)])
    recall = np.array([count_corrects[ix] / count_grounds[ix] for ix in range(num_classes)])
    return precision, recall * config.test_batch_size
