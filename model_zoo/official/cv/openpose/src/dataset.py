# Copyright 2020 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import math
import random
import numpy as np
import cv2
from pycocotools.coco import COCO as ReadJson

import mindspore.dataset as de

from src.config import JointType, params

cv2.setNumThreads(0)

class txtdataset():
    def __init__(self, train, imgpath, maskpath, insize, mode='train', n_samples=None):
        self.train = train
        self.mode = mode
        self.imgpath = imgpath
        self.maskpath = maskpath
        self.insize = insize
        self.maxtime = 0
        self.catIds = train.getCatIds(catNms=['person'])
        self.imgIds = sorted(train.getImgIds(catIds=self.catIds))
        if self.mode == 'train':
            self.clean_imgIds()
        if self.mode in ['val', 'eval'] and n_samples is not None:
            self.imgIds = random.sample(self.imgIds, n_samples)
        print('{} images: {}'.format(mode, len(self)))


    def __len__(self):
        return len(self.imgIds)

    def clean_imgIds(self):
        print("cleaning imgids")

        for img_id in self.imgIds.copy():
            annotations = None
            anno_ids = self.train.getAnnIds(imgIds=[img_id], iscrowd=None)

            # annotation for that image
            if anno_ids:
                annotations_for_img = self.train.loadAnns(anno_ids)

                person_cnt = 0
                valid_annotations_for_img = []
                for annotation in annotations_for_img:
                    # if too few keypoints or too small
                    if annotation['num_keypoints'] >= params['min_keypoints'] and \
                            annotation['area'] > params['min_area']:
                        person_cnt += 1
                        valid_annotations_for_img.append(annotation)

                # if person annotation
                if person_cnt > 0:
                    annotations = valid_annotations_for_img
            if annotations is None:
                self.imgIds.remove(img_id)

    def overlay_paf(self, img, paf):
        hue = ((np.arctan2(paf[1], paf[0]) / np.pi) / -2 + 0.5)
        saturation = np.sqrt(paf[0] ** 2 + paf[1] ** 2)
        saturation[saturation > 1.0] = 1.0
        value = saturation.copy()
        hsv_paf = np.vstack((hue[np.newaxis], saturation[np.newaxis], value[np.newaxis])).transpose(1, 2, 0)
        rgb_paf = cv2.cvtColor((hsv_paf * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        img = cv2.addWeighted(img, 0.6, rgb_paf, 0.4, 0)
        return img

    def overlay_pafs(self, img, pafs):
        mix_paf = np.zeros((2,) + img.shape[:-1])
        paf_flags = np.zeros(mix_paf.shape) # for constant paf

        for paf in pafs.reshape((int(pafs.shape[0]/2), 2,) + pafs.shape[1:]):
            paf_flags = paf != 0
            paf_flags += np.broadcast_to(paf_flags[0] | paf_flags[1], paf.shape)
            mix_paf += paf

        mix_paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
        img = self.overlay_paf(img, mix_paf)
        return img

    def overlay_heatmap(self, img, heatmap):
        rgb_heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, 0.6, rgb_heatmap, 0.4, 0)
        return img

    def overlay_ignore_mask(self, img, ignore_mask):
        img = img * np.repeat((ignore_mask == 0).astype(np.uint8)[:, :, None], 3, axis=2)
        return img

    # -------------------- augment code --------------------------------
    def get_pose_bboxes(self, poses):
        pose_bboxes = []
        for pose in poses:
            x1 = pose[pose[:, 2] > 0][:, 0].min()
            y1 = pose[pose[:, 2] > 0][:, 1].min()
            x2 = pose[pose[:, 2] > 0][:, 0].max()
            y2 = pose[pose[:, 2] > 0][:, 1].max()
            pose_bboxes.append([x1, y1, x2, y2])
        pose_bboxes = np.array(pose_bboxes)
        return pose_bboxes

    def resize_data(self, img, ignore_mask, poses, shape):
        """resize img, mask and annotations"""
        img_h, img_w, _ = img.shape

        resized_img = cv2.resize(img, shape)
        ignore_mask = cv2.resize(ignore_mask.astype(np.uint8), shape).astype('bool')
        poses[:, :, :2] = (poses[:, :, :2] * np.array(shape) / np.array((img_w, img_h)))
        return resized_img, ignore_mask, poses

    def random_resize_img(self, img, ignore_mask, poses):
        h, w, _ = img.shape
        joint_bboxes = self.get_pose_bboxes(poses)
        bbox_sizes = ((joint_bboxes[:, 2:] - joint_bboxes[:, :2] + 1) ** 2).sum(axis=1) ** 0.5

        min_scale = params['min_box_size'] / bbox_sizes.min()
        max_scale = params['max_box_size'] / bbox_sizes.max()

        min_scale = min(max(min_scale, params['min_scale']), 1)
        max_scale = min(max(max_scale, 1), params['max_scale'])

        scale = float((max_scale - min_scale) * random.random() + min_scale)
        shape = (round(w * scale), round(h * scale))

        resized_img, resized_mask, resized_poses = self.resize_data(img, ignore_mask, poses, shape)
        return resized_img, resized_mask, resized_poses

    def random_rotate_img(self, img, mask, poses):
        h, w, _ = img.shape
        degree = np.random.randn() / 3 * params['max_rotate_degree']
        rad = degree * math.pi / 180
        center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(center, degree, 1)
        bbox = (w * abs(math.cos(rad)) + h * abs(math.sin(rad)), w * abs(math.sin(rad)) + h * abs(math.cos(rad)))
        R[0, 2] += bbox[0] / 2 - center[0]
        R[1, 2] += bbox[1] / 2 - center[1]
        rotate_img = cv2.warpAffine(img, R, (int(bbox[0]+0.5), int(bbox[1]+0.5)), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=[127.5, 127.5, 127.5])
        rotate_mask = cv2.warpAffine(mask.astype('uint8')*255, R, (int(bbox[0]+0.5), int(bbox[1]+0.5))) > 0

        tmp_poses = np.ones_like(poses)
        tmp_poses[:, :, :2] = poses[:, :, :2].copy()
        tmp_rotate_poses = np.dot(tmp_poses, R.T)  # apply rotation matrix to the poses
        rotate_poses = poses.copy()  # to keep visibility flag
        rotate_poses[:, :, :2] = tmp_rotate_poses
        return rotate_img, rotate_mask, rotate_poses

    def random_crop_img(self, img, ignore_mask, poses):
        h, w, _ = img.shape
        insize = self.insize
        joint_bboxes = self.get_pose_bboxes(poses)
        bbox = random.choice(joint_bboxes)  # select a bbox randomly
        bbox_center = bbox[:2] + (bbox[2:] - bbox[:2]) / 2

        r_xy = np.random.rand(2)
        perturb = ((r_xy - 0.5) * 2 * params['center_perterb_max'])
        center = (bbox_center + perturb + 0.5).astype('i')

        crop_img = np.zeros((insize, insize, 3), 'uint8') + 127.5
        crop_mask = np.zeros((insize, insize), 'bool')

        offset = (center - (insize - 1) / 2 + 0.5).astype('i')
        offset_ = (center + (insize - 1) / 2 - (w - 1, h - 1) + 0.5).astype('i')

        x1, y1 = (center - (insize-1)/2 + 0.5).astype('i')
        x2, y2 = (center + (insize-1)/2 + 0.5).astype('i')

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w-1)
        y2 = min(y2, h-1)

        x_from = -offset[0] if offset[0] < 0 else 0
        y_from = -offset[1] if offset[1] < 0 else 0
        x_to = insize - offset_[0] - 1 if offset_[0] >= 0 else insize - 1
        y_to = insize - offset_[1] - 1 if offset_[1] >= 0 else insize - 1

        crop_img[y_from:y_to+1, x_from:x_to+1] = img[y1:y2+1, x1:x2+1].copy()
        crop_mask[y_from:y_to+1, x_from:x_to+1] = ignore_mask[y1:y2+1, x1:x2+1].copy()

        poses[:, :, :2] -= offset
        return crop_img.astype('uint8'), crop_mask, poses

    def distort_color(self, img):
        img_max = np.broadcast_to(np.array(255, dtype=np.uint8), img.shape[:-1])
        img_min = np.zeros(img.shape[:-1], dtype=np.uint8)

        hsv_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv_img[:, :, 0] = np.maximum(np.minimum(hsv_img[:, :, 0] - 10 + np.random.randint(20 + 1), img_max), img_min) # hue
        hsv_img[:, :, 1] = np.maximum(np.minimum(hsv_img[:, :, 1] - 40 + np.random.randint(80 + 1), img_max), img_min) # saturation
        hsv_img[:, :, 2] = np.maximum(np.minimum(hsv_img[:, :, 2] - 30 + np.random.randint(60 + 1), img_max), img_min) # value
        hsv_img = hsv_img.astype(np.uint8)

        distorted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return distorted_img

    def flip_img(self, img, mask, poses):
        flipped_img = cv2.flip(img, 1)
        flipped_mask = cv2.flip(mask.astype(np.uint8), 1).astype('bool')
        poses[:, :, 0] = img.shape[1] - 1 - poses[:, :, 0]

        def swap_joints(poses, joint_type_, joint_type_2):
            tmp = poses[:, joint_type_].copy()
            poses[:, joint_type_] = poses[:, joint_type_2]
            poses[:, joint_type_2] = tmp

        swap_joints(poses, JointType.LeftEye, JointType.RightEye)
        swap_joints(poses, JointType.LeftEar, JointType.RightEar)
        swap_joints(poses, JointType.LeftShoulder, JointType.RightShoulder)
        swap_joints(poses, JointType.LeftElbow, JointType.RightElbow)
        swap_joints(poses, JointType.LeftHand, JointType.RightHand)
        swap_joints(poses, JointType.LeftWaist, JointType.RightWaist)
        swap_joints(poses, JointType.LeftKnee, JointType.RightKnee)
        swap_joints(poses, JointType.LeftFoot, JointType.RightFoot)
        return flipped_img, flipped_mask, poses

    def augment_data(self, img, ignore_mask, poses):
        aug_img = img.copy()
        aug_img, ignore_mask, poses = self.random_resize_img(aug_img, ignore_mask, poses)
        aug_img, ignore_mask, poses = self.random_rotate_img(aug_img, ignore_mask, poses)
        aug_img, ignore_mask, poses = self.random_crop_img(aug_img, ignore_mask, poses)
        if np.random.randint(2):
            aug_img = self.distort_color(aug_img)
        if np.random.randint(2):
            aug_img, ignore_mask, poses = self.flip_img(aug_img, ignore_mask, poses)

        return aug_img, ignore_mask, poses
    # ------------------------------- end -----------------------------------


    # ------------------------------ Heatmap ------------------------------------
    # return shape: (height, width)
    def generate_gaussian_heatmap(self, shape, joint, sigma):
        x, y = joint
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        return gaussian_heatmap

    def generate_heatmaps(self, img, poses, heatmap_sigma):
        heatmaps = np.zeros((0,) + img.shape[:-1])
        sum_heatmap = np.zeros(img.shape[:-1])
        for joint_index in range(len(JointType)):
            heatmap = np.zeros(img.shape[:-1])
            for pose in poses:
                if pose[joint_index, 2] > 0:
                    jointmap = self.generate_gaussian_heatmap(img.shape[:-1], pose[joint_index][:2], heatmap_sigma)
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                    sum_heatmap[jointmap > sum_heatmap] = jointmap[jointmap > sum_heatmap]
            heatmaps = np.vstack((heatmaps, heatmap.reshape((1,) + heatmap.shape)))
        bg_heatmap = 1 - sum_heatmap  # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
        return heatmaps.astype('f')

    def generate_gaussian_heatmap_fast(self, shape, joint, sigma):
        x, y = joint
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_x = grid_x + 0.4375
        grid_y = grid_y + 0.4375
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        return gaussian_heatmap

    def generate_heatmaps_fast(self, img, poses, heatmap_sigma):
        resize_shape = (img.shape[0] // 8, img.shape[1] // 8)
        heatmaps = np.zeros((0,) + resize_shape)
        sum_heatmap = np.zeros(resize_shape)
        for joint_index in range(len(JointType)):
            heatmap = np.zeros(resize_shape)
            for pose in poses:
                if pose[joint_index, 2] > 0:
                    jointmap = self.generate_gaussian_heatmap_fast(resize_shape, pose[joint_index][:2]/8,
                                                                   heatmap_sigma/8)
                    index_1 = jointmap > heatmap
                    heatmap[index_1] = jointmap[index_1]
                    index_2 = jointmap > sum_heatmap
                    sum_heatmap[index_2] = jointmap[index_2]
            heatmaps = np.vstack((heatmaps, heatmap.reshape((1,) + heatmap.shape)))

        bg_heatmap = 1 - sum_heatmap  # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
        return heatmaps.astype('f')
    # ------------------------------ end ------------------------------------

    # ------------------------------ PAF ------------------------------------
    # return shape: (2, height, width)
    def generate_constant_paf(self, shape, joint_from, joint_to, paf_width):
        if np.array_equal(joint_from, joint_to): # same joint
            return np.zeros((2,) + shape[:-1])

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance
        rad = np.pi / 2
        # [[0, 1], [-1, 0]]
        rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        # [[u_y], [-u_x]]
        vertical_unit_vector = np.dot(rot_matrix, unit_vector)
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
        horizontal_paf_flag = (horizontal_inner_product >= 0) & (horizontal_inner_product <= joint_distance)
        vertical_inner_product = vertical_unit_vector[0] * (grid_x - joint_from[0]) + vertical_unit_vector[1] *\
                                 (grid_y - joint_from[1])
        vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width  # paf_width : 8
        paf_flag = horizontal_paf_flag & vertical_paf_flag
        constant_paf = np.stack((paf_flag, paf_flag)) *\
                       np.broadcast_to(unit_vector, shape[:-1] + (2,)).transpose(2, 0, 1)

        return constant_paf

    def generate_pafs(self, img, poses, paf_sigma):
        pafs = np.zeros((0,) + img.shape[:-1])

        for limb in params['limbs_point']:
            paf = np.zeros((2,) + img.shape[:-1])
            paf_flags = np.zeros(paf.shape) # for constant paf

            for pose in poses:
                joint_from, joint_to = pose[limb]
                if joint_from[2] > 0 and joint_to[2] > 0:
                    limb_paf = self.generate_constant_paf(img.shape, joint_from[:2], joint_to[:2], paf_sigma) # [2, 368, 368]
                    limb_paf_flags = limb_paf != 0
                    paf_flags += np.broadcast_to(limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)

                    paf += limb_paf

            paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
            pafs = np.vstack((pafs, paf))
        return pafs.astype('f')

    def generate_constant_paf_fast(self, shape, joint_from, joint_to, paf_width):
        if np.array_equal(joint_from, joint_to): # same joint
            return np.zeros((2,) + shape[:-1])

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance
        rad = np.pi / 2
        # [[0, 1], [-1, 0]]
        rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        # [[u_y], [-u_x]]
        vertical_unit_vector = np.dot(rot_matrix, unit_vector)
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_x = grid_x + 0.4375
        grid_y = grid_y + 0.4375
        horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
        horizontal_paf_flag = (horizontal_inner_product >= 0) & (horizontal_inner_product <= joint_distance)
        vertical_inner_product = vertical_unit_vector[0] * (grid_x - joint_from[0]) + vertical_unit_vector[1] *\
                                 (grid_y - joint_from[1])
        vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width  # paf_width : 8/8 = 1
        paf_flag = horizontal_paf_flag & vertical_paf_flag
        constant_paf = np.stack((paf_flag, paf_flag)) *\
                       np.broadcast_to(unit_vector, shape[:-1] + (2,)).transpose(2, 0, 1)

        return constant_paf

    def generate_pafs_fast(self, img, poses, paf_sigma):
        resize_shape = (img.shape[0]//8, img.shape[1]//8, 3)
        pafs = np.zeros((0,) + resize_shape[:-1])

        for limb in params['limbs_point']:
            paf = np.zeros((2,) + resize_shape[:-1])
            paf_flags = np.zeros(paf.shape) # for constant paf

            for pose in poses:
                joint_from, joint_to = pose[limb]
                if joint_from[2] > 0 and joint_to[2] > 0:
                    limb_paf = self.generate_constant_paf_fast(resize_shape, joint_from[:2]/8, joint_to[:2]/8, paf_sigma/8) # [2, 368, 368]
                    limb_paf_flags = limb_paf != 0
                    paf_flags += np.broadcast_to(limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)

                    paf += limb_paf

            index_1 = paf_flags > 0
            paf[index_1] /= paf_flags[index_1]
            pafs = np.vstack((pafs, paf))
        return pafs.astype('f')
    # ------------------------------ end ------------------------------------

    def get_img_annotation(self, ind=None, img_id=None):
        annotations = None

        if ind is not None:
            img_id = self.imgIds[ind]
        anno_ids = self.train.getAnnIds(imgIds=[img_id], iscrowd=None)

        # annotation for that image
        if anno_ids:
            annotations_for_img = self.train.loadAnns(anno_ids)

            person_cnt = 0
            valid_annotations_for_img = []
            for annotation in annotations_for_img:
                # if too few keypoints or too small
                if annotation['num_keypoints'] >= params['min_keypoints'] and annotation['area'] > params['min_area']:
                    person_cnt += 1
                    valid_annotations_for_img.append(annotation)

            # if person annotation
            if person_cnt > 0:
                annotations = valid_annotations_for_img

        img_path = os.path.join(self.imgpath, self.train.loadImgs([img_id])[0]['file_name'])
        mask_path = os.path.join(self.maskpath, '{:012d}.png'.format(img_id))
        img = cv2.imread(img_path)
        ignore_mask = cv2.imread(mask_path, 0)
        if ignore_mask is None:
            ignore_mask = np.zeros(img.shape[:2], np.float32)
        else:
            ignore_mask[ignore_mask == 255] = 1

        if self.mode == 'eval':
            return img, img_id, annotations_for_img, ignore_mask

        return img, img_id, annotations, ignore_mask.astype('f')

    def parse_annotation(self, annotations):
        poses = np.zeros((0, len(JointType), 3), dtype=np.int32)

        for ann in annotations:
            ann_pose = np.array(ann['keypoints']).reshape(-1, 3)
            pose = np.zeros((1, len(JointType), 3), dtype=np.int32)

            # convert poses position
            for i, joint_index in enumerate(params['joint_indices']):
                pose[0][joint_index] = ann_pose[i]

            # compute neck position
            if pose[0][JointType.LeftShoulder][2] > 0 and pose[0][JointType.RightShoulder][2] > 0:
                pose[0][JointType.Neck][0] = int((pose[0][JointType.LeftShoulder][0] +
                                                  pose[0][JointType.RightShoulder][0]) / 2)
                pose[0][JointType.Neck][1] = int((pose[0][JointType.LeftShoulder][1] +
                                                  pose[0][JointType.RightShoulder][1]) / 2)
                pose[0][JointType.Neck][2] = 2

            poses = np.vstack((poses, pose))
        return poses

    def resize_output(self, input_np, map_h=46, map_w=46):
        if len(input_np.shape) == 3:
            output = np.zeros((input_np.shape[0], map_h, map_w))
            for i in range(input_np.shape[0]):
                output[i] = cv2.resize(input_np[i], (map_w, map_h))
            return output.astype('f')

        input_np = input_np.astype('f')
        output = cv2.resize(input_np, (map_h, map_w))
        return output

    def generate_labels(self, img, poses, ignore_mask):
        img, ignore_mask, poses = self.augment_data(img, ignore_mask, poses)
        resized_img, ignore_mask, resized_poses = self.resize_data(img, ignore_mask, poses,
                                                                   shape=(self.insize, self.insize))

        resized_heatmaps = self.generate_heatmaps_fast(resized_img, resized_poses, params['heatmap_sigma'])

        resized_pafs = self.generate_pafs_fast(resized_img, resized_poses, params['paf_sigma'])

        ignore_mask = cv2.morphologyEx(ignore_mask.astype('uint8'), cv2.MORPH_DILATE, np.ones((16, 16))).astype('bool')
        resized_ignore_mask = self.resize_output(ignore_mask)


        return resized_img, resized_pafs, resized_heatmaps, resized_ignore_mask

    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)
        return x_data

    def __getitem__(self, i):
        img, img_id, annotations, ignore_mask = self.get_img_annotation(ind=i)

        if self.mode in ['eval', 'val']:
            # don't need to make heatmaps/pafs
            return img, np.array([img_id])

        # if no annotations are available
        while annotations is None:
            print("none annotations", img_id)
            img_id = self.imgIds[np.random.randint(len(self))]
            img, img_id, annotations, ignore_mask = self.get_img_annotation(img_id=img_id)

        poses = self.parse_annotation(annotations)

        # TEST
        # return img, poses, ignore_mask

        resized_img, pafs, heatmaps, ignore_mask = self.generate_labels(img, poses, ignore_mask)
        resized_img = self.preprocess(resized_img)
        ignore_mask = 1. - ignore_mask

        # # TEST
        # print("Shape: ", resized_img.dtype, " ", pafs.dtype, " ", heatmaps.dtype, " ", ignore_mask.dtype)

        return resized_img, pafs, heatmaps, ignore_mask


class DistributedSampler():
    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_len = len(self.dataset)
        self.num_samplers = int(math.ceil(self.dataset_len * 1.0 / self.group_size))
        self.total_size = self.num_samplers * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_len).tolist()
        else:
            indices = list(range(len(self.dataset_len)))
        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.num_samplers

def valdata(jsonpath, imgpath, rank, group_size, mode='val', maskpath=''):
    #cv2.setNumThreads(0)
    val = ReadJson(jsonpath)
    dataset = txtdataset(val, imgpath, maskpath, params['insize'], mode=mode)
    sampler = DistributedSampler(dataset, rank, group_size)
    ds = de.GeneratorDataset(dataset, ['img', 'img_id'], num_parallel_workers=8, sampler=sampler)
    ds = ds.repeat(1)
    return ds


def create_dataset(jsonpath, imgpath, maskpath, batch_size, rank, group_size, mode='train', repeat_num=1, shuffle=True,
                   multiprocessing=True, num_worker=20):

    train = ReadJson(jsonpath)
    dataset = txtdataset(train, imgpath, maskpath, params['insize'], mode=mode)
    if group_size == 1:
        de_dataset = de.GeneratorDataset(dataset, ["image", "pafs", "heatmaps", "ignore_mask"],
                                         shuffle=shuffle,
                                         num_parallel_workers=num_worker,
                                         python_multiprocessing=multiprocessing)
    else:
        de_dataset = de.GeneratorDataset(dataset, ["image", "pafs", "heatmaps", "ignore_mask"],
                                         shuffle=shuffle,
                                         num_parallel_workers=num_worker,
                                         python_multiprocessing=multiprocessing,
                                         num_shards=group_size,
                                         shard_id=rank)

    de_dataset = de_dataset.batch(batch_size=batch_size, drop_remainder=True)
    de_dataset = de_dataset.repeat(repeat_num)

    return de_dataset
