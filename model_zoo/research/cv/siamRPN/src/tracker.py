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
""" eval tracker"""
import numpy as np
import mindspore as ms
from mindspore import Tensor
from src.config import config
from src.util import get_exemplar_image, get_instance_image, box_transform_inv
from src.generate_anchors import generate_anchors


class SiamRPNTracker:
    """ Tracker for SiamRPN"""
    def __init__(self, model):
        self.model = model
        valid_scope = 2 * config.valid_scope + 1
        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                        config.anchor_ratios,
                                        valid_scope)
        self.window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                              [config.anchor_num, 1, 1]).flatten()

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """ initialize siamfc tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        self.pos = np.array(
            [bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2])  # center x, center y, zero based
        # same to original code
        self.target_sz = np.array([bbox[2], bbox[3]])  # width, height
        self.bbox = np.array([bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]])
        # same to original code
        self.origin_target_sz = np.array([bbox[2], bbox[3]])
        # get exemplar img
        self.img_mean = np.mean(frame, axis=(0, 1))

        exemplar_img, _, _ = get_exemplar_image(frame, self.bbox,
                                                config.exemplar_size, config.context_amount, self.img_mean)
        exemplar_img = Tensor(exemplar_img, ms.float32)
        self.model.is_train = False
        self.model.is_trackinit = True
        self.model.is_track = False
        self.ckernal, self.rkernal = self.model(template=exemplar_img, detection=exemplar_img,
                                                ckernal=exemplar_img, rkernal=exemplar_img)




    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        instance_img_np, _, _, scale_x = get_instance_image(frame, self.bbox, config.exemplar_size,
                                                            config.instance_size,
                                                            config.context_amount, self.img_mean)

        self.model.is_train = False
        self.model.is_trackinit = False
        self.model.is_track = True
        instance_img_np = Tensor(instance_img_np, ms.float32)
        pred_score, pred_regression = self.model(template=instance_img_np, detection=instance_img_np,
                                                 ckernal=self.ckernal, rkernal=self.rkernal)
        delta = pred_regression[0].asnumpy()
        box_pred = box_transform_inv(self.anchors, delta)
        pred_score = pred_score.asnumpy()

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        s_c = change(sz(box_pred[:, 2], box_pred[:, 3]) / (sz_wh(self.target_sz * scale_x)))  # scale penalty
        r_c = change((self.target_sz[0] / self.target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k)
        pscore = penalty * pred_score
        pscore = pscore * (1 - config.window_influence) + self.window * config.window_influence
        best_pscore_id = np.argmax(pscore)

        target = box_pred[best_pscore_id, :] / scale_x

        lr = penalty[best_pscore_id] * pred_score[best_pscore_id] * config.lr_box

        res_x = np.clip(target[0] + self.pos[0], 0, frame.shape[1])
        res_y = np.clip(target[1] + self.pos[1], 0, frame.shape[0])

        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, config.min_scale * self.origin_target_sz[0],
                        config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, config.min_scale * self.origin_target_sz[1],
                        config.max_scale * self.origin_target_sz[1])

        self.pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])
        bbox = np.array([res_x, res_y, res_w, res_h])
        self.bbox = (
            np.clip(bbox[0], 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64))
        return self.bbox, pred_score[best_pscore_id]
