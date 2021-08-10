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
'''
dataset processing
'''
from __future__ import division

import json
import os
from copy import deepcopy
import random

import numpy as np
import cv2

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
from src.utils.transforms import fliplr_joints, get_affine_transform, affine_transform
from src.config import config

ds.config.set_seed(config.GENERAL.DATASET_SEED) # Set Random Seed
flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
              [9, 10], [11, 12], [13, 14], [15, 16]]

class CocoDatasetGenerator:
    '''
    About the specific operations of coco2017 data set processing
    '''
    def __init__(self, cfg, is_train=False):
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE, dtype=np.int32)
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.heatmap_size = np.array(cfg.NETWORK.HEATMAP_SIZE, dtype=np.int32)
        self.sigma = cfg.NETWORK.SIGMA
        self.target_type = cfg.NETWORK.TARGET_TYPE
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.db = []
        self.is_train = is_train
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.num_joints = 17

    def load_gt_dataset(self, image_path, ann_file):
        '''
        load_gt_dataset
        '''
        self.db = []

        with open(ann_file, "rb") as f:
            lines = f.readlines()
        json_dict = json.loads(lines[0].decode("utf-8"))

        objs = {}
        cnt = 0
        for item in json_dict['annotations']:
            # exclude iscrowd and no-keypoint record
            if item['iscrowd'] != 0 or item['num_keypoints'] == 0:
                continue

            # assert the record is valid
            assert item['iscrowd'] == 0, 'is crowd'
            assert item['category_id'] == 1, 'is not people'
            assert item['area'] > 0, 'area le 0'
            assert item['num_keypoints'] > 0, 'has no keypoint'
            assert max(item['keypoints']) > 0

            image_id = item['image_id']
            obj = [{'num_keypoints': item['num_keypoints'], 'keypoints': item['keypoints'], 'bbox': item['bbox']}]
            objs[image_id] = obj if image_id not in objs else objs[image_id] + obj

            cnt += 1

        print('loaded %d records from coco dataset.' % cnt)

        for item in json_dict['images']:
            image_id = item['id']
            width = item['width']
            height = item['height']
            if image_id not in objs:
                continue
            valid_objs = []
            for obj in objs[image_id]:
                x, y, w, h = obj['bbox']
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width - 1, x1 + max(0, w - 1))
                y2 = min(height - 1, y1 + max(0, h - 1))
                if x2 >= x1 and y2 >= y1:
                    tmp_obj = deepcopy(obj)
                    tmp_obj['bbox'] = np.array((x1, y1, x2, y2)) - np.array((0, 0, x1, y1))
                    valid_objs.append(tmp_obj)
                else:
                    assert False, 'invalid bbox!'
            objs[image_id] = valid_objs

            for obj in objs[image_id]:
                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
                for ipt in range(self.num_joints):
                    joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    joints_3d[ipt, 2] = 0
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis > 1:
                        t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

                scale, center = self._bbox2sc(obj['bbox'])

                self.db.append({
                    'id': int(item['id']),
                    'image': os.path.join(image_path, item['file_name']),
                    'center': center,
                    'scale': scale,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                })

    def load_detect_dataset(self, image_path, ann_file, bbox_file):
        '''
        load_detect_dataset
        '''
        self.db = []
        all_boxes = None
        with open(bbox_file, 'r') as f:
            all_boxes = json.load(f)

        assert all_boxes, 'Loading %s fail!' % bbox_file
        print('Total boxes: {}'.format(len(all_boxes)))

        with open(ann_file, "rb") as f:
            lines = f.readlines()
        json_dict = json.loads(lines[0].decode("utf-8"))
        index_to_filename = {}
        for item in json_dict['images']:
            index_to_filename[item['id']] = item['file_name']
        for det_res in all_boxes:
            if det_res['category_id'] != 1:
                continue
            image = os.path.join(image_path,
                                 index_to_filename[det_res['image_id']])

            bbox = det_res['bbox']
            score = det_res['score']
            if score < self.image_thre:
                continue

            scale, center = self._bbox2sc(bbox)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones((self.num_joints, 3), dtype=np.float)

            self.db.append({
                'id': int(det_res['image_id']),
                'image': image,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

    def _bbox2sc(self, bbox):
        """
        reform xywh to meet the need of aspect ratio
        """
        x, y, w, h = bbox[:4]
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / 200, h * 1.0 / 200], dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return scale, center

    def __getitem__(self, idx):
        db_rec = deepcopy(self.db[idx])

        image_file = db_rec['image']

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            print('[ERROR] fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        image = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_heatmap(joints, joints_vis)

        return image, target, target_weight, s, c, score, db_rec['id']

    def generate_heatmap(self, joints, joints_vis):
        '''
        generate_heatmap
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def __len__(self):
        return len(self.db)

def CreateDatasetCoco(rank=0,
                      group_size=1,
                      train_mode=True,
                      num_parallel_workers=8,
                      transform=None,
                      shuffle=None):
    '''
    CreateDatasetCoco
    '''
    per_batch_size = config.TRAIN.BATCH_SIZE if train_mode else config.TEST.BATCH_SIZE

    image_path = ''
    ann_file = ''
    bbox_file = ''
    if config.MODELARTS.IS_MODEL_ARTS:
        image_path = config.MODELARTS.CACHE_INPUT
        ann_file = config.MODELARTS.CACHE_INPUT
        bbox_file = config.MODELARTS.CACHE_INPUT
    else:
        image_path = config.DATASET.ROOT
        ann_file = config.DATASET.ROOT
        bbox_file = config.DATASET.ROOT

    if train_mode:
        image_path = image_path + config.DATASET.TRAIN_SET
        ann_file = ann_file + config.DATASET.TRAIN_JSON
    else:
        image_path = image_path + config.DATASET.TEST_SET
        ann_file = ann_file + config.DATASET.TEST_JSON
    bbox_file = bbox_file + config.TEST.COCO_BBOX_FILE

    print('loading dataset from {}'.format(image_path))

    shuffle = shuffle if shuffle is not None else train_mode
    dataset_generator = CocoDatasetGenerator(config, is_train=train_mode)

    if not train_mode and config.TEST.USE_GT_BBOX:
        print('loading bbox file from {}'.format(bbox_file))
        dataset_generator.load_detect_dataset(image_path, ann_file, bbox_file)
    else:
        dataset_generator.load_gt_dataset(image_path, ann_file)

    coco_dataset = ds.GeneratorDataset(dataset_generator,
                                       column_names=["image", "target", "weight", "scale", "center", "score", "id"],
                                       num_parallel_workers=num_parallel_workers,
                                       num_shards=group_size,
                                       shard_id=rank,
                                       shuffle=shuffle)
    if transform is None:
        transform_img = [
            C.Rescale(1.0 / 255.0, 0.0),
            C.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            C.HWC2CHW()
        ]
    else:
        transform_img = transform
    coco_dataset = coco_dataset.map(input_columns="image",
                                    num_parallel_workers=num_parallel_workers,
                                    operations=transform_img)
    coco_dataset = coco_dataset.batch(per_batch_size, drop_remainder=train_mode)

    return coco_dataset
