"""
MIT License

Copyright (c) 2019 Xingyi Zhou
All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import cv2

from dependency.centernet.src.lib.utils.image import color_aug
from dependency.centernet.src.lib.utils.image import get_affine_transform, affine_transform
from dependency.centernet.src.lib.utils.image import gaussian_radius, draw_umich_gaussian
from dependency.extd.utils.augmentations import anchor_crop_image_sampling

def get_border(border, size):
    """
    Get border
    """
    i = 1
    while size - border // i <= border // i:  # size > 2 * (border // i)
        i *= 2
    return border // i

def coco_box_to_bbox(box):
    """
    (x1, y1, w, h) -> (x1, y1, x2, y2)
    """
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
    return bbox

def preprocess_train(image, target, config):
    """
    Preprocess training data
    """
    data_rng = np.random.RandomState(123)
    eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
    eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape((1, 1, 3))
    num_objs = len(target)

    anns = []
    for each in target:
        ann = {}
        ann['bbox'] = each[0:4]
        ann['keypoints'] = each[4:]
        anns.append(ann)

    cv2.setNumThreads(0)
    img, anns = anchor_crop_image_sampling(image, anns)

    _, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0
    rot = 0
    flipped = False
    if config.rand_crop:
        #s = s * np.random.choice(np.arange(0.8, 1.3, 0.05)) # for 768*768 or 800* 800
        s = s * np.random.choice(np.arange(0.6, 1.0, 0.05)) # for 512 * 512
        border = s * np.random.choice([0.1, 0.2, 0.25])
        w_border = get_border(border, img.shape[1])  # w > 2 * w_border
        h_border = get_border(border, img.shape[0])  # h > 2 * h_border
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
    else:
        sf = config.scale
        cf = config.shift
        c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
    if np.random.random() < config.rotate:
        rf = config.rotate
        rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

    if np.random.random() < config.flip:  # opt.flip = 0.5
        flipped = True
        img = img[:, ::-1, :]
        c[0] = width - c[0] - 1

    trans_input = get_affine_transform(c, s, rot, [config.input_res, config.input_res])
    inp = cv2.warpAffine(img, trans_input, (config.input_res, config.input_res), flags=cv2.INTER_LINEAR)

    inp = (inp.astype(np.float32) / 255.)
    if config.color_aug:
        color_aug(data_rng, inp, eig_val, eig_vec)

    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    output_res = config.output_res
    num_joints = config.num_joints
    max_objs = config.max_objs
    trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
    trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

    # map
    hm = np.zeros((config.num_classes, output_res, output_res), dtype=np.float32)
    hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)

    wh = np.zeros((output_res, output_res, 2), dtype=np.float32)
    reg = np.zeros((output_res, output_res, 2), dtype=np.float32)
    ind = np.zeros((output_res, output_res), dtype=np.float32) # as float32, need no data_type change later

    reg_mask = np.zeros((max_objs), dtype=np.uint8)
    wight_mask = np.zeros((output_res, output_res, 2), dtype=np.float32)

    kps = np.zeros((output_res, output_res, num_joints * 2), dtype=np.float32)
    kps_mask = np.zeros((output_res, output_res, num_joints * 2), dtype=np.float32)
    #
    hp_offset = np.zeros((max_objs * num_joints, 2), dtype=np.float32)
    hp_ind = np.zeros((max_objs * num_joints), dtype=np.int64)
    hp_mask = np.zeros((max_objs * num_joints), dtype=np.int64)

    draw_gaussian = draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
        ann = anns[k]
        bbox = coco_box_to_bbox(ann['bbox'])  # [x,y,w,h]--[x1,y1,x2,y2]
        cls_id = 0 #int(ann['category_id']) - 1
        pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)  # (x,y,0/1)
        if flipped:
            bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            pts[:, 0] = width - pts[:, 0] - 1
            for e in config.flip_idx: # flip_idx = [[0, 1], [3, 4]]
                pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

        bbox[:2] = affine_transform(bbox[:2], trans_output)  # [0, 1] -- (x1, y1)
        bbox[2:] = affine_transform(bbox[2:], trans_output)  # [2, 3] -- (x2, y2)
        bbox = np.clip(bbox, 0, output_res - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if (h > 0 and w > 0) or (rot != 0):
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)

            ind[ct_int[1], ct_int[0]] = 1.0
            wh[ct_int[1], ct_int[0], :] = np.log(1. * w / 4), np.log(1. * h / 4)
            reg[ct_int[1], ct_int[0], :] = ct[0] - ct_int[0], ct[1] - ct_int[1]

            reg_mask[k] = 1.0
            wight_mask[ct_int[1], ct_int[0], 0] = 1
            wight_mask[ct_int[1], ct_int[0], 1] = 1

            # if w*h <= 20: # can get what we want sometime, but unstable
            #     wight_mask[k] = 15
            if w*h <= 40:
                wight_mask[ct_int[1], ct_int[0], 0] = 5
                wight_mask[ct_int[1], ct_int[0], 1] = 5
            if w*h <= 20:
                wight_mask[ct_int[1], ct_int[0], 0] = 10
                wight_mask[ct_int[1], ct_int[0], 1] = 10
            if w*h <= 10:
                wight_mask[ct_int[1], ct_int[0], 0] = 15
                wight_mask[ct_int[1], ct_int[0], 1] = 15
            if w*h <= 4:
                wight_mask[ct_int[1], ct_int[0], 0] = 0.1
                wight_mask[ct_int[1], ct_int[0], 1] = 0.1

            num_kpts = pts[:, 2].sum()
            if num_kpts == 0:
                hm[cls_id, ct_int[1], ct_int[0]] = 0.9999

            hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            hp_radius = max(0, int(hp_radius))
            for j in range(num_joints):
                if pts[j, 2] > 0:
                    pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                    if pts[j, 0] >= 0 and pts[j, 0] < output_res and pts[j, 1] >= 0 and pts[j, 1] < output_res:
                        kps[ct_int[1], ct_int[0], j * 2 : j * 2 + 2] = pts[j, :2] - ct_int
                        kps[ct_int[1], ct_int[0], j * 2 : j * 2 + 1] = kps[ct_int[1], ct_int[0], j * 2 : j * 2 + 1] / w
                        kps[ct_int[1], ct_int[0], j * 2 + 1: j * 2 + 2] = kps[ct_int[1], ct_int[0],
                                                                              j * 2 + 1 : j * 2 + 2] / h
                        kps_mask[ct_int[1], ct_int[0], j * 2 : j * 2 + 2] = 1.0

                        pt_int = pts[j, :2].astype(np.int32)
                        hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                        hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                        hp_mask[k * num_joints + j] = 1

                        draw_gaussian(hm_hp[j], pt_int, hp_radius)
                        kps_mask[ct_int[1], ct_int[0], j * 2 : j * 2 + 2] = \
                            0.0 if ann['bbox'][2] * ann['bbox'][3] <= 8.0 else 1.0
            draw_gaussian(hm[cls_id], ct_int, radius)
            gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                           ct[0] + w / 2, ct[1] + h / 2, 1] +
                          pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])

    return inp, hm, reg_mask, ind, wh, wight_mask, reg, kps_mask, kps
