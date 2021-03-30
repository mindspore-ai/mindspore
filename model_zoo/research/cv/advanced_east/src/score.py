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
"""
#################calculate precision ########################
"""
import numpy as np
from shapely.geometry import Polygon

from src.config import config as cfg
from src.nms import nms

class Averager():
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


class eval_pre_rec_f1():
    '''eval batch'''

    def __init__(self):
        self.pixel_threshold = float(cfg.pixel_threshold)
        self.reset()

    def reset(self):
        self.img_num = 0
        self.pre = 0
        self.rec = 0
        self.f1_score = 0

    def val(self):
        mpre = self.pre / self.img_num * 100
        mrec = self.rec / self.img_num * 100
        mf1_score = self.f1_score / self.img_num * 100
        return mpre, mrec, mf1_score

    def sigmoid(self, x):
        """`y = 1 / (1 + exp(-x))`"""
        return 1 / (1 + np.exp(-x))

    def get_iou(self, g, p):
        """`get_iou`"""
        g = Polygon(g)
        p = Polygon(p)
        if not g.is_valid or not p.is_valid:
            return 0
        inter = Polygon(g).intersection(Polygon(p)).area
        union = g.area + p.area - inter
        if union == 0:
            return 0
        return inter / union

    def eval_one(self, quad_scores, quad_after_nms, gt_xy, quiet=cfg.quiet):
        """`eval_one`"""
        num_gts = len(gt_xy)
        quad_scores_no_zero = []
        quad_after_nms_no_zero = []
        for score, geo in zip(quad_scores, quad_after_nms):
            if np.amin(score) > 0:
                quad_scores_no_zero.append(sum(score))
                quad_after_nms_no_zero.append(geo)
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
                continue
        num_quads = len(quad_after_nms_no_zero)
        if num_quads == 0:
            return 0, 0, 0
        quad_flag = np.zeros(num_quads)
        gt_flag = np.zeros(num_gts)
        quad_scores_no_zero = np.array(quad_scores_no_zero)
        scores_idx = np.argsort(quad_scores_no_zero)[::-1]
        for i in range(num_quads):
            idx = scores_idx[i]
            geo = quad_after_nms_no_zero[idx]
            for j in range(num_gts):
                if gt_flag[j] == 0:
                    gt_geo = gt_xy[j]
                    iou = self.get_iou(geo, gt_geo)
                    if iou >= cfg.iou_threshold:
                        gt_flag[j] = 1
                        quad_flag[i] = 1
        tp = np.sum(quad_flag)
        fp = num_quads - tp
        fn = num_gts - tp
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        if pre + rec == 0:
            f1_score = 0
        else:
            f1_score = 2 * pre * rec / (pre + rec)
        return pre, rec, f1_score

    def add(self, out, gt_xy_list):
        """`add`"""
        self.img_num += len(gt_xy_list)
        ys = out.asnumpy()  # (N, 7, 64, 64)
        if ys.shape[1] == 7:
            ys = ys.transpose((0, 2, 3, 1))  # NCHW->NHWC
        for y, gt_xy in zip(ys, gt_xy_list):
            y[:, :, :3] = self.sigmoid(y[:, :, :3])
            cond = np.greater_equal(y[:, :, 0], self.pixel_threshold)
            activation_pixels = np.where(cond)
            quad_scores, quad_after_nms = nms(y, activation_pixels)
            lens = len(quad_after_nms)
            if lens == 0 or (sum(sum(quad_scores)) == 0):
                if not cfg.quiet:
                    print('NMS')
                continue
            else:
                pre, rec, f1_score = self.eval_one(quad_scores, quad_after_nms, gt_xy)
                self.pre += pre
                self.rec += rec
                self.f1_score += f1_score
