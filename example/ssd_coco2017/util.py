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
from config import ConfigSSD
from dataset import ssd_bboxes_decode


def calc_iou(bbox_pred, bbox_ground):
    """Calculate iou of predicted bbox and ground truth."""
    bbox_pred = np.expand_dims(bbox_pred, axis=0)

    pred_w = bbox_pred[:, 2] - bbox_pred[:, 0]
    pred_h = bbox_pred[:, 3] - bbox_pred[:, 1]
    pred_area = pred_w * pred_h

    gt_w = bbox_ground[:, 2] - bbox_ground[:, 0]
    gt_h = bbox_ground[:, 3] - bbox_ground[:, 1]
    gt_area = gt_w * gt_h

    iw = np.minimum(bbox_pred[:, 2], bbox_ground[:, 2]) - np.maximum(bbox_pred[:, 0], bbox_ground[:, 0])
    ih = np.minimum(bbox_pred[:, 3], bbox_ground[:, 3]) - np.maximum(bbox_pred[:, 1], bbox_ground[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    intersection_area = iw * ih

    union_area = pred_area + gt_area - intersection_area
    union_area = np.maximum(union_area, np.finfo(float).eps)

    iou = intersection_area *  1. / union_area
    return iou


def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]
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


def calc_ap(recall, precision):
    """Calculate AP."""
    correct_recall = np.concatenate(([0.], recall, [1.]))
    correct_precision = np.concatenate(([0.], precision, [0.]))

    for i in range(correct_recall.size - 1, 0, -1):
        correct_precision[i - 1] = np.maximum(correct_precision[i - 1], correct_precision[i])

    i = np.where(correct_recall[1:] != correct_recall[:-1])[0]

    ap = np.sum((correct_recall[i + 1] - correct_recall[i]) * correct_precision[i + 1])

    return ap

def metrics(pred_data):
    """Calculate mAP of predicted bboxes."""
    config = ConfigSSD()
    num_classes = config.NUM_CLASSES

    all_detections = [None for i in range(num_classes)]
    all_pred_scores = [None for i in range(num_classes)]
    all_annotations = [None for i in range(num_classes)]
    average_precisions = {}
    num = [0 for i in range(num_classes)]
    accurate_num = [0 for i in range(num_classes)]

    for sample in pred_data:
        pred_boxes = sample['boxes']
        boxes_scores = sample['box_scores']
        annotation = sample['annotation']
        image_shape = sample['image_shape']

        annotation = np.squeeze(annotation, axis=0)
        image_shape = np.squeeze(image_shape, axis=0)

        pred_labels = np.argmax(boxes_scores, axis=-1)
        index = np.nonzero(pred_labels)
        pred_boxes = ssd_bboxes_decode(pred_boxes, index, image_shape)

        pred_boxes = pred_boxes.clip(0, 1)
        boxes_scores = np.max(boxes_scores, axis=-1)
        boxes_scores = boxes_scores[index]
        pred_labels = pred_labels[index]

        top_k = 50

        for c in range(1, num_classes):
            if len(pred_labels) >= 1:
                class_box_scores = boxes_scores[pred_labels == c]
                class_boxes = pred_boxes[pred_labels == c]

                nms_index = apply_nms(class_boxes, class_box_scores, config.MATCH_THRESHOLD, top_k)

                class_boxes = class_boxes[nms_index]
                class_box_scores = class_box_scores[nms_index]

                cmask = class_box_scores > 0.5
                class_boxes = class_boxes[cmask]
                class_box_scores = class_box_scores[cmask]

                all_detections[c] = class_boxes
                all_pred_scores[c] = class_box_scores

        for c in range(1, num_classes):
            if len(annotation) >= 1:
                all_annotations[c] = annotation[annotation[:, 4] == c, :4]

        for c in range(1, num_classes):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            annotations = all_annotations[c]
            num_annotations += annotations.shape[0]
            detections = all_detections[c]
            pred_scores = all_pred_scores[c]

            for index, detection in enumerate(detections):
                scores = np.append(scores, pred_scores[index])
                if len(annotations) >= 1:
                    IoUs = calc_iou(detection, annotations)
                    assigned_anno = np.argmax(IoUs)
                    max_overlap = IoUs[assigned_anno]

                    if max_overlap >= 0.5:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

            if num_annotations == 0:
                if c not in average_precisions.keys():
                    average_precisions[c] = 0
                continue
            accurate_num[c] = 1
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            recall = true_positives * 1. / num_annotations
            precision = true_positives * 1. / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            average_precision = calc_ap(recall, precision)

            if c not in average_precisions.keys():
                average_precisions[c] = average_precision
            else:
                average_precisions[c] += average_precision

            num[c] += 1

    count = 0
    for key in average_precisions:
        if num[key] != 0:
            count += (average_precisions[key] / num[key])

    mAP = count * 1. / accurate_num.count(1)
    return mAP
