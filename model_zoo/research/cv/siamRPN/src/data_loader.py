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
""" data loader class """
import pickle
import glob
import os
import cv2
import numpy as np
from src.generate_anchors import generate_anchors
from src.util import box_transform, compute_iou, crop_and_pad
from src.config import config

class TrainDataLoader:
    """ dataloader """
    def __init__(self, data_dir):
        self.ret = {}
        self.data_dir = data_dir
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        config.pairs_per_video_per_epoch = 2
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.video_names = [x[0] for x in self.meta_data]
        self.meta_data = {x[0]: x[1] for x in self.meta_data}
        self.training = True
        for key in self.meta_data.keys():
            trajs = self.meta_data[key]
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]
        #dataset config
        self.num = len(self.video_names) if config.pairs_per_video_per_epoch is None or not self.training \
            else config.pairs_per_video_per_epoch * len(self.video_names)
        self.valid_scope = int((config.instance_size - config.exemplar_size) / 8 / 2)*2+1
        self.anchors = generate_anchors(total_stride=config.total_stride, base_size=config.anchor_base_size,
                                        scales=config.anchor_scales,  \
                                        ratios=config.anchor_ratios, score_size=self.valid_scope)

    def imread(self, image_name):
        img = cv2.imread(image_name)
        return img

    def RandomStretch(self, sample, gt_w, gt_h):
        scale_h = 1.0 + np.random.uniform(-config.max_stretch, config.max_stretch)
        scale_w = 1.0 + np.random.uniform(-config.max_stretch, config.max_stretch)
        h, w = sample.shape[:2]
        shape = int(w * scale_w), int(h * scale_h)
        scale_w = int(w * scale_w) / w
        scale_h = int(h * scale_h) / h
        gt_w = gt_w * scale_w
        gt_h = gt_h * scale_h
        return cv2.resize(sample, shape, cv2.INTER_LINEAR), gt_w, gt_h

    def compute_target(self, anchors, box, pos_threshold=0.6, neg_threshold=0.3, pos_num=16, num_neg=48):
        """ compute iou to label """
        total_num = pos_num + num_neg
        regression_target = box_transform(anchors, box)
        iou = compute_iou(anchors, box).flatten()
        pos_cand = np.where(iou > pos_threshold)[0]
        if len(pos_cand) > pos_num:
            pos_index = np.random.choice(pos_cand, pos_num, replace=False)

        else:
            pos_index = pos_cand
        pos_num = len(pos_index)
        neg_cand = np.where(iou < neg_threshold)[0]
        neg_num = total_num - pos_num
        neg_index = np.random.choice(neg_cand, neg_num, replace=False)
        label = np.ones_like(iou) * -100
        label[pos_index] = 1
        label[neg_index] = 0
        pos_neg_diff = np.hstack((label.reshape(-1, 1), regression_target))
        return pos_neg_diff

    def __getitem__(self, idx):
        all_idx = np.arange(self.num)
        np.random.shuffle(all_idx)
        all_idx = np.insert(all_idx, 0, idx, 0)
        for vedio_idx in all_idx:
            vedio_idx = vedio_idx % len(self.video_names)
            video = self.video_names[vedio_idx]
            trajs = self.meta_data[video]
            # sample one trajs
            if not trajs.keys():
                continue

            trkid = np.random.choice(list(trajs.keys()))
            traj = trajs[trkid]
            assert len(traj) > 1, "video_name: {}".format(video)
            # sample exemplar
            exemplar_idx = np.random.choice(list(range(len(traj))))
            if 'ILSVRC2015' in video:
                exemplar_name = \
                    glob.glob(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid)))[
                        0]
            else:
                exemplar_name = \
                    glob.glob(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{}.x*.jpg".format(trkid)))[0]
            exemplar_gt_w, exemplar_gt_h, exemplar_w_image, exemplar_h_image = \
                float(exemplar_name.split('_')[-4]), float(exemplar_name.split('_')[-3]), \
                float(exemplar_name.split('_')[-2]), float(exemplar_name.split('_')[-1][:-4])
            exemplar_ratio = min(exemplar_gt_w / exemplar_gt_h, exemplar_gt_h / exemplar_gt_w)
            exemplar_scale = exemplar_gt_w * exemplar_gt_h / (exemplar_w_image * exemplar_h_image)
            if not config.scale_range[0] <= exemplar_scale < config.scale_range[1]:
                continue
            if not config.ratio_range[0] <= exemplar_ratio < config.ratio_range[1]:
                continue

            exemplar_img = self.imread(exemplar_name)
            # sample instance
            if 'ILSVRC2015' in exemplar_name:
                frame_range = config.frame_range_vid
            else:
                frame_range = config.frame_range_ytb
            low_idx = max(0, exemplar_idx - frame_range)
            up_idx = min(len(traj), exemplar_idx + frame_range + 1)
            weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type)
            instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx + 1:up_idx], p=weights)

            if 'ILSVRC2015' in video:
                instance_name = \
                    glob.glob(os.path.join(self.data_dir, video, instance + ".{:02d}.x*.jpg".format(trkid)))[0]
            else:
                instance_name = glob.glob(os.path.join(self.data_dir, video, instance + ".{}.x*.jpg".format(trkid)))[0]

            instance_gt_w, instance_gt_h, instance_w_image, instance_h_image = \
                float(instance_name.split('_')[-4]), float(instance_name.split('_')[-3]), \
                float(instance_name.split('_')[-2]), float(instance_name.split('_')[-1][:-4])
            instance_ratio = min(instance_gt_w / instance_gt_h, instance_gt_h / instance_gt_w)
            instance_scale = instance_gt_w * instance_gt_h / (instance_w_image * instance_h_image)
            if not config.scale_range[0] <= instance_scale < config.scale_range[1]:
                continue
            if not config.ratio_range[0] <= instance_ratio < config.ratio_range[1]:
                continue

            instance_img = self.imread(instance_name)

            if np.random.rand(1) < config.gray_ratio:
                exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
                exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)
                instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
                instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)
            if config.exem_stretch:
                exemplar_img, exemplar_gt_w, exemplar_gt_h = self.RandomStretch(exemplar_img, exemplar_gt_w,
                                                                                exemplar_gt_h)
            exemplar_img, _ = crop_and_pad(exemplar_img, (exemplar_img.shape[1] - 1) / 2,
                                           (exemplar_img.shape[0] - 1) / 2, config.exemplar_size,
                                           config.exemplar_size)

            instance_img, gt_w, gt_h = self.RandomStretch(instance_img, instance_gt_w, instance_gt_h)
            im_h, im_w, _ = instance_img.shape
            cy_o = (im_h - 1) / 2
            cx_o = (im_w - 1) / 2
            cy = cy_o + np.random.randint(- config.max_translate, config.max_translate + 1)
            cx = cx_o + np.random.randint(- config.max_translate, config.max_translate + 1)
            gt_cx = cx_o - cx
            gt_cy = cy_o - cy

            instance_img_1, _ = crop_and_pad(instance_img, cx, cy, config.instance_size, config.instance_size)


            pos_neg_diff = self.compute_target(self.anchors, np.array(list(map(round, [gt_cx, gt_cy, gt_w, gt_h]))),
                                               pos_threshold=config.pos_threshold, neg_threshold=config.neg_threshold,
                                               pos_num=config.pos_num, num_neg=config.neg_num)
            self.ret['template_cropped_resized'] = exemplar_img
            self.ret['detection_cropped_resized'] = instance_img_1
            self.ret['pos_neg_diff'] = pos_neg_diff
            self._tranform()
            return (self.ret['template_tensor'], self.ret['detection_tensor'], self.ret['pos_neg_diff_tensor'])

    def _tranform(self):
        """PIL to Tensor"""
        template_pil = self.ret['template_cropped_resized'].copy()
        detection_pil = self.ret['detection_cropped_resized'].copy()
        pos_neg_diff = self.ret['pos_neg_diff'].copy()

        template_tensor = (np.transpose(np.array(template_pil), (2, 0, 1))).astype(np.float32)
        detection_tensor = (np.transpose(np.array(detection_pil), (2, 0, 1))).astype(np.float32)
        self.ret['template_tensor'] = template_tensor
        self.ret['detection_tensor'] = detection_tensor

        self.ret['pos_neg_diff_tensor'] = pos_neg_diff

    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'):
        """ sample weights"""
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def __len__(self):
        return self.num
