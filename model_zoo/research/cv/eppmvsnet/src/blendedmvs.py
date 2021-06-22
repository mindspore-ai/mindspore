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
"""blendedmvs dataset"""

import os
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image

import mindspore.dataset.vision.py_transforms as py_vision

from src.utils import read_pfm


class Compose:
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class BlendedMVSDataset:
    """blendedmvs dataset"""

    def __init__(self, root_dir, split, n_views=3, levels=3, depth_interval=128.0, img_wh=(768, 576),
                 crop_wh=(640, 512), scale=False, scan=None, training_tag=False):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.root_dir = root_dir
        self.split = split
        self.scale = scale
        self.training_tag = training_tag
        assert self.split in ['train', 'val', 'all'], \
            'split must be either "train", "val" or "all"!'
        self.img_wh = img_wh
        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, \
                'img_wh must both be multiples of 32!'
        self.single_scan = scan
        self.crop_wh = crop_wh
        if crop_wh is not None:
            assert crop_wh[0] % 32 == 0 and crop_wh[1] % 32 == 0, \
                'img_wh must both be multiples of 32!'
        self.n_views = n_views
        self.levels = levels  # FPN levels
        self.n_depths = depth_interval

        self.build_metas()
        self.cal_crop_factors()
        self.build_proj_mats()
        self.define_transforms()

    def cal_crop_factors(self):
        """"calculate crop factors"""
        self.start_w = (self.img_wh[0] - self.crop_wh[0]) // 2
        self.start_h = (self.img_wh[1] - self.crop_wh[1]) // 2
        self.finish_w = self.start_w + self.crop_wh[0]
        self.finish_h = self.start_h + self.crop_wh[1]

    def build_metas(self):
        """"build meta information"""
        self.metas = []
        self.ref_views_per_scan = defaultdict(list)
        if self.split == 'train':
            list_txt = os.path.join(self.root_dir, 'training_list.txt')
        elif self.split == 'val':
            list_txt = os.path.join(self.root_dir, 'validation_list.txt')
        else:
            list_txt = os.path.join(self.root_dir, 'all_list.txt')

        if self.single_scan is not None:
            self.scans = self.single_scan if isinstance(self.single_scan, list) else [self.single_scan]
        else:
            with open(list_txt) as f:
                self.scans = [line.rstrip() for line in f.readlines()]

        for scan in self.scans:
            with open(os.path.join(self.root_dir, scan, "cams/pair.txt")) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    self.ref_views_per_scan[scan] += [ref_view]
                    line = f.readline().rstrip().split()
                    n_views_valid = int(line[0])  # valid views
                    if n_views_valid < self.n_views:  # skip no enough valid views
                        continue
                    src_views = [int(x) for x in line[1::2]]
                    self.metas += [(scan, -1, ref_view, src_views)]

    def build_proj_mats(self):
        """"build projection matrix"""
        self.proj_mats = {}  # proj mats for each scan
        if self.root_dir.endswith('dataset_low_res') \
                or self.root_dir.endswith('dataset_low_res/'):
            img_w, img_h = 768, 576
        else:
            img_w, img_h = 2048, 1536
        for scan in self.scans:
            self.proj_mats[scan] = {}
            for vid in self.ref_views_per_scan[scan]:
                proj_mat_filename = os.path.join(self.root_dir, scan,
                                                 f'cams/{vid:08d}_cam.txt')
                intrinsics, extrinsics, depth_min, depth_max = \
                    self.read_cam_file(scan, proj_mat_filename)
                intrinsics[0] *= self.img_wh[0] / img_w / 8
                intrinsics[1] *= self.img_wh[1] / img_h / 8
                # center crop
                if self.training_tag:
                    intrinsics[0, 2] = intrinsics[0, 2] - self.start_w / 8
                    intrinsics[1, 2] = intrinsics[1, 2] - self.start_h / 8

                # multiply intrinsics and extrinsics to get projection matrix
                proj_mat_ls = []
                for _ in reversed(range(self.levels)):
                    proj_mat_l = np.eye(4)
                    proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                    intrinsics[:2] *= 2  # 1/8->1/4->1/2
                    proj_mat_ls += [proj_mat_l]
                proj_mat_ls = np.stack(proj_mat_ls[::-1]).astype(dtype=np.float32)
                self.proj_mats[scan][vid] = (proj_mat_ls, depth_min, depth_max)

    def read_cam_file(self, scan, filename):
        """"read camera file"""
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[3])
        return intrinsics, extrinsics, depth_min, depth_max

    def read_depth_and_mask(self, scan, filename, depth_min):
        """"read depth and mask"""
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        h, w = depth.shape
        if (h, w) != self.img_wh:
            depth_0 = cv2.resize(depth, self.img_wh,
                                 interpolation=cv2.INTER_LINEAR)
            if self.training_tag:
                depth_0 = depth_0[self.start_h:self.finish_h, self.start_w:self.finish_w]
        depth_0 = cv2.resize(depth_0, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_LINEAR)
        depth_1 = cv2.resize(depth_0, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_LINEAR)
        depth_2 = cv2.resize(depth_1, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_LINEAR)

        depths = {"level_0": depth_0,
                  "level_1": depth_1,
                  "level_2": depth_2}

        masks = {"level_0": depth_0 > depth_min,
                 "level_1": depth_1 > depth_min,
                 "level_2": depth_2 > depth_min}
        depth_max = depth_0.max()
        return depths, masks, depth_max

    def define_transforms(self):
        if self.training_tag and self.split == 'train':  # you can add augmentation here
            self.transform = Compose([
                py_vision.ToTensor(),
                py_vision.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = Compose([
                py_vision.ToTensor(),
                py_vision.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        sample = {}
        scan, _, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views - 1]

        imgs = []
        proj_mats = []  # record proj mats between views
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.root_dir, f'{scan}/blended_images/{vid:08d}.jpg')
            depth_filename = os.path.join(self.root_dir, f'{scan}/rendered_depth_maps/{vid:08d}.pfm')

            img = Image.open(img_filename)
            w, h = img.size
            if (h, w) != self.img_wh:
                img = img.resize(self.img_wh, Image.BILINEAR)
            if self.training_tag:
                img = img.crop((self.start_w, self.start_h, self.finish_w, self.finish_h))
            img = self.transform(img)
            imgs += [img]

            proj_mat_ls, depth_min, depth_max = deepcopy(self.proj_mats[scan][vid])

            if i == 0:  # reference view
                if self.split == 'train':
                    depths, masks, depth_max = self.read_depth_and_mask(scan, depth_filename, depth_min)
                elif self.split == 'val':
                    if self.training_tag:
                        depths, masks, depth_max = self.read_depth_and_mask(scan, depth_filename, depth_min)
                    else:
                        depths, masks, _ = self.read_depth_and_mask(scan, depth_filename, depth_min)
                else:
                    raise ValueError
                fix_depth_interval = (depth_max - depth_min) / self.n_depths
                depth_interval = fix_depth_interval
                sample['init_depth_min'] = [depth_min]
                sample['depth_interval'] = [depth_interval]
                sample['fix_depth_interval'] = [fix_depth_interval]
                ref_proj_inv = np.asarray(proj_mat_ls)
                for j in range(proj_mat_ls.shape[0]):
                    ref_proj_inv[j] = np.mat(proj_mat_ls[j]).I
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]

        imgs = np.stack(imgs)
        proj_mats = np.stack(proj_mats)[:, :, :3]  # (V-1, self.levels, 3, 4) from fine to coarse
        depth_0 = depths['level_0']
        mask_0 = masks["level_0"]

        sample['imgs'] = imgs
        sample['proj_mats'] = proj_mats
        sample['depths'] = depths
        sample['masks'] = masks
        sample['scan_vid'] = (scan, ref_view)

        return imgs, proj_mats, np.array(sample['init_depth_min'], dtype=np.float32), \
               np.array(sample['depth_interval'], dtype=np.float32), np.fromstring(scan, dtype=np.uint8), \
               np.array(ref_view), depth_0, mask_0, np.array(sample['fix_depth_interval'], dtype=np.float32)
