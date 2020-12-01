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
"""Face detection yolov3 data pre-process."""
import numpy as np

import mindspore.dataset.vision.py_transforms as P

from src.transforms import RandomCropLetterbox, RandomFlip, HSVShift, ResizeLetterbox
from src.config import config


class SingleScaleTrans:
    '''SingleScaleTrans'''
    def __init__(self, resize, max_anno_count=200):
        self.resize = (resize[0], resize[1])
        self.max_anno_count = max_anno_count

    def __call__(self, imgs, ann, image_names, image_size, batch_info):

        size = self.resize
        decode = P.Decode()
        resize_letter_box_op = ResizeLetterbox(input_dim=size)

        to_tensor = P.ToTensor()
        ret_imgs = []
        ret_anno = []

        for i, image in enumerate(imgs):
            img_pil = decode(image)
            input_data = img_pil, ann[i]
            input_data = resize_letter_box_op(*input_data)
            image_arr = to_tensor(input_data[0])
            ret_imgs.append(image_arr)
            ret_anno.append(input_data[1])

        for i, anno in enumerate(ret_anno):
            anno_count = anno.shape[0]
            if anno_count < self.max_anno_count:
                ret_anno[i] = np.concatenate(
                    (ret_anno[i], np.zeros((self.max_anno_count - anno_count, 6), dtype=float)), axis=0)
            else:
                ret_anno[i] = ret_anno[i][:self.max_anno_count]

        return np.array(ret_imgs), np.array(ret_anno), image_names, image_size


def check_gt_negative_or_empty(gt):
    new_gt = []
    for anno in gt:
        for data in anno:
            if data not in (0, -1):
                new_gt.append(anno)
                break
    if not new_gt:
        return True, new_gt
    return False, new_gt


def bbox_ious_numpy(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Args:
        boxes1 (np.array): List of bounding boxes
        boxes2 (np.array): List of bounding boxes

    Note:
        List format: [[xc, yc, w, h],...]
    """
    b1x1, b1y1 = np.split((boxes1[:, :2] - (boxes1[:, 2:4] / 2)), 2, axis=1)
    b1x2, b1y2 = np.split((boxes1[:, :2] + (boxes1[:, 2:4] / 2)), 2, axis=1)
    b2x1, b2y1 = np.split((boxes2[:, :2] - (boxes2[:, 2:4] / 2)), 2, axis=1)
    b2x2, b2y2 = np.split((boxes2[:, :2] + (boxes2[:, 2:4] / 2)), 2, axis=1)

    dx = np.minimum(b1x2, b2x2.transpose()) - np.maximum(b1x1, b2x1.transpose())
    dx = np.maximum(dx, 0)
    dy = np.minimum(b1y2, b2y2.transpose()) - np.maximum(b1y1, b2y1.transpose())
    dy = np.maximum(dy, 0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.transpose()) - intersections

    return intersections / unions


def build_targets_brambox(img, anno, reduction, img_shape_para, anchors_mask, anchors):
    """
    Compare prediction boxes and ground truths, convert ground truths to network output tensors
    """
    ground_truth = anno
    img_shape = img.shape
    n_h = int(img_shape[1] / img_shape_para)  # height
    n_w = int(img_shape[2] / img_shape_para)  # width
    anchors_ori = np.array(anchors) / reduction
    num_anchor = len(anchors_mask)
    conf_pos_mask = np.zeros((num_anchor, n_h * n_w), dtype=np.float32)  # pos mask
    conf_neg_mask = np.ones((num_anchor, n_h * n_w), dtype=np.float32)  # neg mask

    # coordination mask and classification mask
    coord_mask = np.zeros((num_anchor, 1, n_h * n_w), dtype=np.float32)  # coord mask
    cls_mask = np.zeros((num_anchor, n_h * n_w), dtype=np.int)

    # for target coordination confidence classification
    t_coord = np.zeros((num_anchor, 4, n_h * n_w), dtype=np.float32)
    t_conf = np.zeros((num_anchor, n_h * n_w), dtype=np.float32)
    t_cls = np.zeros((num_anchor, n_h * n_w), dtype=np.float32)

    gt_list = None
    is_empty_or_negative, filtered_ground_truth = check_gt_negative_or_empty(ground_truth)
    if is_empty_or_negative:
        gt_np = np.zeros((len(ground_truth), 4), dtype=np.float32)
        gt_temp = gt_np[:]
        gt_list = gt_temp
        # continue
        return coord_mask, conf_pos_mask, conf_neg_mask, cls_mask, t_coord, t_conf, t_cls, gt_list
    # Build up tensors
    anchors = np.concatenate([np.zeros_like(anchors_ori), anchors_ori], axis=1)
    gt = np.zeros((len(filtered_ground_truth), 4), dtype=np.float32)
    gt_np = np.zeros((len(ground_truth), 4), dtype=np.float32)
    for i, annotation in enumerate(filtered_ground_truth):
        # gt x y x h->x_c y_c w h
        # reduction for remap the gt to the feature
        gt[i, 0] = (annotation[1] + annotation[3] / 2) / reduction
        gt[i, 1] = (annotation[2] + annotation[4] / 2) / reduction
        gt[i, 2] = annotation[3] / reduction
        gt[i, 3] = annotation[4] / reduction

        gt_np[i, 0] = annotation[1] / reduction
        gt_np[i, 1] = annotation[2] / reduction
        gt_np[i, 2] = (annotation[1] + annotation[3]) / reduction
        gt_np[i, 3] = (annotation[2] + annotation[4]) / reduction
    gt_temp = gt_np[:]
    gt_list = gt_temp

    # Find best anchor for each gt

    gt_wh = np.copy(gt)
    gt_wh[:, :2] = 0
    iou_gt_anchors = bbox_ious_numpy(gt_wh, anchors)
    best_anchors = np.argmax(iou_gt_anchors, axis=1)
    # Set masks and target values for each gt

    for i, annotation in enumerate(filtered_ground_truth):
        annotation_ignore = annotation[5]
        annotation_width = annotation[3]
        annotation_height = annotation[4]
        annotation_class_id = annotation[0]

        gi = min(n_w - 1, max(0, int(gt[i, 0])))
        gj = min(n_h - 1, max(0, int(gt[i, 1])))
        cur_n = best_anchors[i]  # best anchors for current ground truth

        if cur_n in anchors_mask:
            best_n = np.where(np.array(anchors_mask) == cur_n)[0][0]
        else:
            continue

        if annotation_ignore:
            # current annotation is ignore for difficult
            conf_pos_mask[best_n][gj * n_w + gi] = 0
            conf_neg_mask[best_n][gj * n_w + gi] = 0
        else:
            coord_mask[best_n][0][gj * n_w + gi] = 2 - annotation_width * annotation_height / \
                                                   (n_w * n_h * reduction * reduction)
            cls_mask[best_n][gj * n_w + gi] = 1
            conf_pos_mask[best_n][gj * n_w + gi] = 1
            conf_neg_mask[best_n][gj * n_w + gi] = 0
            t_coord[best_n][0][gj * n_w + gi] = gt[i, 0] - gi
            t_coord[best_n][1][gj * n_w + gi] = gt[i, 1] - gj
            t_coord[best_n][2][gj * n_w + gi] = np.log(gt[i, 2] / anchors[cur_n, 2])
            t_coord[best_n][3][gj * n_w + gi] = np.log(gt[i, 3] / anchors[cur_n, 3])
            t_conf[best_n][gj * n_w + gi] = 1
            t_cls[best_n][gj * n_w + gi] = annotation_class_id

    return coord_mask, conf_pos_mask, conf_neg_mask, cls_mask, t_coord, t_conf, t_cls, gt_list


def preprocess_fn(image, annotation):
    '''preprocess_fn'''
    jitter = config.jitter
    flip = config.flip
    hue = config.hue
    sat = config.sat
    val = config.val
    size = config.input_shape
    max_anno_count = 200
    reduction_0 = 64.0
    reduction_1 = 32.0
    reduction_2 = 16.0
    anchors = config.anchors
    anchors_mask = config.anchors_mask

    decode = P.Decode()
    random_crop_letter_box_op = RandomCropLetterbox(jitter=jitter, input_dim=size)
    random_flip_op = RandomFlip(flip)
    hsv_shift_op = HSVShift(hue, sat, val)
    to_tensor = P.ToTensor()

    img_pil = decode(image)
    input_data = img_pil, annotation
    input_data = random_crop_letter_box_op(*input_data)
    input_data = random_flip_op(*input_data)
    input_data = hsv_shift_op(*input_data)
    image_arr = to_tensor(input_data[0])
    ret_img = image_arr
    ret_anno = input_data[1]

    anno_count = ret_anno.shape[0]

    if anno_count < max_anno_count:
        ret_anno = np.concatenate((ret_anno, np.zeros((max_anno_count - anno_count, 6), dtype=float)), axis=0)
    else:
        ret_anno = ret_anno[:max_anno_count]

    ret_img = np.array(ret_img)
    ret_anno = np.array(ret_anno)

    coord_mask_0, conf_pos_mask_0, conf_neg_mask_0, cls_mask_0, t_coord_0, t_conf_0, t_cls_0, gt_list_0 = \
        build_targets_brambox(ret_img, ret_anno, reduction_0, int(reduction_0), anchors_mask[0], anchors)
    coord_mask_1, conf_pos_mask_1, conf_neg_mask_1, cls_mask_1, t_coord_1, t_conf_1, t_cls_1, gt_list_1 = \
        build_targets_brambox(ret_img, ret_anno, reduction_1, int(reduction_1), anchors_mask[1], anchors)
    coord_mask_2, conf_pos_mask_2, conf_neg_mask_2, cls_mask_2, t_coord_2, t_conf_2, t_cls_2, gt_list_2 = \
        build_targets_brambox(ret_img, ret_anno, reduction_2, int(reduction_2), anchors_mask[2], anchors)

    return ret_img, ret_anno, coord_mask_0, conf_pos_mask_0, conf_neg_mask_0, cls_mask_0, t_coord_0, t_conf_0,\
        t_cls_0, gt_list_0, coord_mask_1, conf_pos_mask_1, conf_neg_mask_1, cls_mask_1, t_coord_1, t_conf_1,\
        t_cls_1, gt_list_1, coord_mask_2, conf_pos_mask_2, conf_neg_mask_2, cls_mask_2, t_coord_2, t_conf_2, \
        t_cls_2, gt_list_2


compose_map_func = (preprocess_fn)
