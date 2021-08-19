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
"""postprocess for 310 inference"""
import os
import json
import argparse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


parser = argparse.ArgumentParser("ssd quant postprocess")
parser.add_argument("--result_path", type=str, required=True, help="path to inference results.")
parser.add_argument("--image_shape", type=str, required=True, help="path to image shape directory.")
parser.add_argument("--image_id", type=str, required=True, help="path to image id directory.")
parser.add_argument("--ann_file", type=str, required=True, help="path to annotation file.")
parser.add_argument("--min_score", type=float, default=0.1, help="min box score threshold.")
parser.add_argument("--nms_threshold", type=float, default=0.6, help="threshold of nms process.")
parser.add_argument("--max_boxes", type=int, default=100, help="max number of detection boxes.")

args, _ = parser.parse_known_args()


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


def metrics(pred_data, anno_json):
    """Calculate mAP of predicted bboxes."""

    #Classes need to train or test.
    val_cls = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
               'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
    num_classes = len(val_cls)
    val_cls_dict = {}
    for i, cls in enumerate(val_cls):
        val_cls_dict[i] = cls
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

        final_boxes = []
        final_label = []
        final_score = []
        img_ids.append(img_id)

        for c in range(1, num_classes):
            class_box_scores = box_scores[:, c]
            score_mask = class_box_scores > args.min_score
            class_box_scores = class_box_scores[score_mask]
            class_boxes = pred_boxes[score_mask] * [h, w, h, w]

            if score_mask.any():
                nms_index = apply_nms(class_boxes, class_box_scores, args.nms_threshold, args.max_boxes)
                class_boxes = class_boxes[nms_index]
                class_box_scores = class_box_scores[nms_index]

                final_boxes += class_boxes.tolist()
                final_score += class_box_scores.tolist()
                final_label += [classs_dict[val_cls_dict[c]]] * len(class_box_scores)

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


def calculate_acc(result_path, image_id):
    """
    Calculate accuracy of VGG16 inference.

    Args:
        result_path (str): the directory or inference result.
        image_id (str): the path of image_id directory.
    """
    pred_data = []
    for file in os.listdir(image_id):
        id_num = int(np.fromfile(os.path.join(image_id, file), dtype=np.int32)[0])
        img_size = np.fromfile(os.path.join(args.image_shape, file), dtype=np.float32).reshape(1, -1)[0]
        img_ids_name = file.split(".")[0]
        image_shape = np.array([img_size[0], img_size[1]])
        result_path_0 = os.path.join(result_path, img_ids_name + "_output_0.bin")
        result_path_1 = os.path.join(result_path, img_ids_name + "_output_1.bin")
        boxes = np.fromfile(result_path_0, dtype=np.float32).reshape(1917, 4)
        box_scores = np.fromfile(result_path_1, dtype=np.float32).reshape(1917, 81)
        pred_data.append({"boxes": boxes, "box_scores": box_scores, "img_id": id_num, "image_shape": image_shape})
    mAP = metrics(pred_data, args.ann_file)
    print(f" mAP:{mAP}")


if __name__ == '__main__':
    calculate_acc(args.result_path, args.image_id)
