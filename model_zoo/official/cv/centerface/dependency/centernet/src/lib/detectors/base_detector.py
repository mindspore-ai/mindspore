###modified based on centernet###
#MIT License
#Copyright (c) 2019 Xingyi Zhou
#All rights reserved.
"""Basic definition of detector"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

from mindspore import Tensor

from dependency.centernet.src.lib.external.nms import soft_nms
from dependency.centernet.src.lib.utils.image import get_affine_transform, affine_transform

def transform_preds(coords, center, scale, output_size):
    """
    Transform target coords
    """
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def multi_pose_post_process(dets, c, s, h, w):
    """
    Multi pose post process
    dets_result: 4 + score:1 + kpoints:10 + class:1 = 16
    dets: batch x max_dets x 40
    return list of 39 in image coord
    """
    ret = []
    for i in range(dets.shape[0]):
        bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
        pts = transform_preds(dets[i, :, 5:15].reshape(-1, 2), c[i], s[i], (w, h))
        top_preds = np.concatenate([bbox.reshape(-1, 4), dets[i, :, 4:5], pts.reshape(-1, 10)],
                                   axis=1).astype(np.float32).tolist()
        ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
    return ret

class CenterFaceDetector():
    """
    Centerface detector
    """
    def __init__(self, opt, model):
        self.flip_idx = opt.flip_idx

        print('Creating model...')
        self.model = model

        self.mean = np.array(opt.mean, dtype=np.float32).reshape((1, 1, 3))
        self.std = np.array(opt.std, dtype=np.float32).reshape((1, 1, 3))
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = False

    def pre_process(self, image, scale, meta=None):
        """
        Preprocess method
        """
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_res: # True
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = int(np.ceil(new_height / 32) * 32)
            inp_width = int(np.ceil(new_width / 32) * 32)
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)

        meta = {'c': c, 's': s, 'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta

    def process(self, images):
        """
        Process method
        """
        images = Tensor(images)
        # test with mindspore model
        output_hm, output_wh, output_off, output_kps, topk_inds = self.model(images)
        # Tensor to numpy
        output_hm = output_hm.asnumpy().astype(np.float32)
        output_wh = output_wh.asnumpy().astype(np.float32)
        output_off = output_off.asnumpy().astype(np.float32)
        output_kps = output_kps.asnumpy().astype(np.float32)
        topk_inds = topk_inds.asnumpy().astype(np.long)

        reg = output_off if self.opt.reg_offset else None

        dets = self.centerface_decode(output_hm, output_wh, output_kps, reg=reg, opt_k=self.opt.K, topk_inds=topk_inds)

        return dets

    def post_process(self, dets, meta, scale=1):
        """
        Post process process
        """
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = multi_pose_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'])
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 15)
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        """
        Merge detection outputs
        """
        results = {}
        results[1] = np.concatenate([detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            soft_nms(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def run(self, image_or_path_or_tensor, meta=None):
        """
        Run method
        """
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif isinstance(image_or_path_or_tensor, str):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        detections = []
        for scale in self.scales: # [1]
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta) # --1: pre_process
            else:
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}

            dets = self.process(images) # --2: process

            dets = self.post_process(dets, meta, scale)     # box:4+score:1+kpoints:10+class:1=16     ## --3: post_process

            detections.append(dets)

        results = self.merge_outputs(detections) # --4: merge_outputs
        return {'results': results}

    def centerface_decode(self, heat, wh, kps, reg=None, opt_k=100, topk_inds=None):
        """
        Decode detection bbox
        """
        batch, _, _, width = wh.shape

        num_joints = kps.shape[1] // 2

        scores = heat
        inds = topk_inds
        ys_int = (topk_inds / width).astype(np.int32).astype(np.float32)
        xs_int = (topk_inds % width).astype(np.int32).astype(np.float32)

        reg = reg.reshape(batch, 2, -1)
        reg_tmp = np.zeros((batch, 2, opt_k), dtype=np.float32)
        for i in range(batch):
            reg_tmp[i, 0, :] = reg[i, 0, inds[i]]
            reg_tmp[i, 1, :] = reg[i, 1, inds[i]]
        reg = reg_tmp.transpose(0, 2, 1)

        if reg is not None:
            xs = xs_int.reshape(batch, opt_k, 1) + reg[:, :, 0:1]
            ys = ys_int.reshape(batch, opt_k, 1) + reg[:, :, 1:2]
        else:
            xs = xs_int.reshape(batch, opt_k, 1) + 0.5
            ys = ys_int.reshape(batch, opt_k, 1) + 0.5

        wh = wh.reshape(batch, 2, -1)
        wh_tmp = np.zeros((batch, 2, opt_k), dtype=np.float32)
        for i in range(batch):
            wh_tmp[i, 0, :] = wh[i, 0, inds[i]]
            wh_tmp[i, 1, :] = wh[i, 1, inds[i]]

        wh = wh_tmp.transpose(0, 2, 1)
        wh = np.exp(wh) * 4.
        scores = scores.reshape(batch, opt_k, 1)
        bboxes = np.concatenate([xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2,
                                 ys + wh[..., 1:2] / 2], axis=2)

        clses = np.zeros((batch, opt_k, 1), dtype=np.float32)
        kps = np.zeros((batch, opt_k, num_joints * 2), dtype=np.float32)
        detections = np.concatenate([bboxes, scores, kps, clses], axis=2)    # box:4+score:1+kpoints:10+class:1=16
        return detections
