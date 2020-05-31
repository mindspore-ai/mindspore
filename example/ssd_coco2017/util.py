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

import os
import json
import math
import numpy as np
from mindspore import Tensor
from mindspore.common.initializer import initializer, TruncatedNormal
from config import ConfigSSD
from dataset import ssd_bboxes_decode


def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(int): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_end + \
                 (lr_max - lr_end) * \
                 (1. + math.cos(math.pi * (i - warmup_steps) / (total_steps - warmup_steps))) / 2.
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def init_net_param(network, initialize_mode='TruncatedNormal'):
    """Init the parameters in net."""
    params = network.trainable_params()
    for p in params:
        if isinstance(p.data, Tensor) and 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            if initialize_mode == 'TruncatedNormal':
                p.set_parameter_data(initializer(TruncatedNormal(0.03), p.data.shape(), p.data.dtype()))
            else:
                p.set_parameter_data(initialize_mode, p.data.shape(), p.data.dtype())


def load_backbone_params(network, param_dict):
    """Init the parameters from pre-train model, default is mobilenetv2."""
    for _, param in net.parameters_and_names():
        param_name = param.name.replace('network.backbone.', '')
        name_split = param_name.split('.')
        if 'features_1' in param_name:
            param_name = param_name.replace('features_1', 'features')
        if 'features_2' in param_name:
            param_name = '.'.join(['features', str(int(name_split[1]) + 14)] + name_split[2:])
        if param_name in param_dict:
            param.set_parameter_data(param_dict[param_name].data)


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
    config = ConfigSSD()
    num_classes = config.NUM_CLASSES

    coco_root = config.COCO_ROOT
    data_type = config.VAL_DATA_TYPE

    #Classes need to train or test.
    val_cls = config.COCO_CLASSES
    val_cls_dict = {}
    for i, cls in enumerate(val_cls):
        val_cls_dict[i] = cls

    anno_json = os.path.join(coco_root, config.INSTANCES_SET.format(data_type))
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
            score_mask = class_box_scores > config.MIN_SCORE
            class_box_scores = class_box_scores[score_mask]
            class_boxes = pred_boxes[score_mask] * [h, w, h, w]

            if score_mask.any():
                nms_index = apply_nms(class_boxes, class_box_scores, config.NMS_THRESHOLD, config.TOP_K)
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
