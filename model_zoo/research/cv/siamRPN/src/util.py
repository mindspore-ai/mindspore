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
""" function for data preprocessing"""

import numpy as np
import cv2

def round_up(value):
    """Replace the built-in round function to achieve accurate rounding with 2 decimal places

    :param value:object
    :return:object
    """

    return round(value + 1e-6 + 1000) - 1000


def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    """change img size

    :param img:rgb
    :param cx: center x
    :param cy: center y
    :param model_sz: changed size
    :param original_sz: origin size
    :param img_mean: mean of img
    :return: changed img ,scale for origin to changed
    """
    im_h, im_w, _ = img.shape

    xmin = cx - (original_sz - 1) / 2
    xmax = xmin + original_sz - 1
    ymin = cy - (original_sz - 1) / 2
    ymax = ymin + original_sz - 1

    left = int(round_up(max(0., -xmin)))
    top = int(round_up(max(0., -ymin)))
    right = int(round_up(max(0., xmax - im_w + 1)))
    bottom = int(round_up(max(0., ymax - im_h + 1)))

    xmin = int(round_up(xmin + left))
    xmax = int(round_up(xmax + left))
    ymin = int(round_up(ymin + top))
    ymax = int(round_up(ymax + top))
    r, c, k = img.shape
    if any([top, bottom, left, right]):
        te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)  # 0 is better than 1 initialization
        te_im[top:top + r, left:left + c, :] = img
        if top:
            te_im[0:top, left:left + c, :] = img_mean
        if bottom:
            te_im[r + top:, left:left + c, :] = img_mean
        if left:
            te_im[:, 0:left, :] = img_mean
        if right:
            te_im[:, c + left:, :] = img_mean
        im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    else:
        im_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
    else:
        im_patch = im_patch_original
    scale = model_sz / im_patch_original.shape[0]
    return im_patch, scale


def get_exemplar_image(img, bbox, size_z, context_amount, img_mean=None):
    """ preprocessing exemplar

    :param img: exemplar img
    :param bbox: init bbox
    :param size_z: changed size
    :param context_amount: context amount
    :param img_mean: mean of img
    :return: img ,scale
    """
    cx, cy, w, h = bbox
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z
    exemplar_img, _ = crop_and_pad(img, cx, cy, size_z, s_z, img_mean)
    return exemplar_img, scale_z, s_z


def get_instance_image(img, bbox, size_z, size_x, context_amount, img_mean=None):
    """ preprocessing instance

    :param img: instance img
    :param bbox: init bbox
    :param size_z: changed size
    :param context_amount: context amount
    :param img_mean: mean of img
    :return: img ,scale
    """
    cx, cy, w, h = bbox  # float type
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)  # the width of the crop box

    s_x = s_z * size_x / size_z
    instance_img, scale_x = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    w_x = w * scale_x
    h_x = h * scale_x
    return instance_img, w_x, h_x, scale_x


def box_transform(anchors, gt_box):
    """transform box

    :param anchors: object
    :param gt_box: object
    :return: object
    """
    anchor_xctr = anchors[:, :1]
    anchor_yctr = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    gt_cx, gt_cy, gt_w, gt_h = gt_box

    target_x = (gt_cx - anchor_xctr) / anchor_w
    target_y = (gt_cy - anchor_yctr) / anchor_h
    target_w = np.log(gt_w / anchor_w)
    target_h = np.log(gt_h / anchor_h)
    regression_target = np.hstack((target_x, target_y, target_w, target_h))
    return regression_target


def box_transform_inv(anchors, offset):
    """invert transform box

    :param anchors: object
    :param offset: object
    :return: object
    """
    anchor_xctr = anchors[:, :1]
    anchor_yctr = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    offset_x, offset_y, offset_w, offset_h = offset[:, :1], offset[:, 1:2], offset[:, 2:3], offset[:, 3:]

    box_cx = anchor_w * offset_x + anchor_xctr
    box_cy = anchor_h * offset_y + anchor_yctr
    box_w = anchor_w * np.exp(offset_w)
    box_h = anchor_h * np.exp(offset_h)
    box = np.hstack([box_cx, box_cy, box_w, box_h])
    return box


def compute_iou(anchors, box):
    """compute iou

    :param anchors: object
    :param box:object
    :return: iou value
    """
    if np.array(anchors).ndim == 1:
        anchors = np.array(anchors)[None, :]
    else:
        anchors = np.array(anchors)
    if np.array(box).ndim == 1:
        box = np.array(box)[None, :]
    else:
        box = np.array(box)
    gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

    anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5
    anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5
    anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5
    anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5

    gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
    gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
    gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
    gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

    xx1 = np.max([anchor_x1, gt_x1], axis=0)
    xx2 = np.min([anchor_x2, gt_x2], axis=0)
    yy1 = np.max([anchor_y1, gt_y1], axis=0)
    yy2 = np.min([anchor_y2, gt_y2], axis=0)

    inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                           axis=0)
    area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
    return iou
