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


import math
import os
import random
import cv2
from PIL import Image
import numpy as np
import Polygon as plg
import pyclipper

import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_transforms

from src.config import config

__all__ = ['train_dataset_creator', 'test_dataset_creator']


def get_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_imgs_names(root_dir):
    img_paths = [i for i in os.listdir(root_dir)
                 if os.path.splitext(i)[-1].lower() in ['.jpg', '.jpeg', '.png']]
    return img_paths


def get_bboxes(img, gt_path):
    h, w = img.shape[0:2]
    with open(gt_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    bboxes = []
    tags = []
    for line in lines:
        line = line.replace('\xef\xbb\xbf', '')
        line = line.replace('\ufeff', '')
        line = line.replace('\n', '')
        gt = line.split(",", 8)
        tag = gt[-1][0] != '#'
        box = [int(gt[i]) for i in range(8)]
        box = np.asarray(box) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(box)
        tags.append(tag)
    return np.array(bboxes), tags


def random_scale(img, min_size):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale1 = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale1, fy=scale1)

    h, w = img.shape[0:2]
    random_scale1 = np.array([0.5, 1.0, 2.0, 3.0])
    scale2 = np.random.choice(random_scale1)
    if min(h, w) * scale2 <= min_size:
        scale3 = (min_size + 10) * 1.0 / min(h, w)
        img = cv2.resize(img, dsize=None, fx=scale3, fy=scale3)
    else:
        img = cv2.resize(img, dsize=None, fx=scale2, fy=scale2)
    return img


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i, _ in enumerate(imgs):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i, _ in enumerate(imgs):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs


def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis=1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)

        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    for idx, _ in enumerate(imgs):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs


def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale_long = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale_long, fy=scale_long)
    return img


def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

        shrinked_bbox = pco.Execute(-offset)
        if not shrinked_bbox:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bbox = np.array(shrinked_bbox)[0]
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bboxes.append(shrinked_bbox)

    return np.array(shrinked_bboxes)


class TrainDataset:
    def __init__(self):
        cv2.setNumThreads(2)

        self.is_transform = True
        self.img_size = config.TRAIN_LONG_SIZE
        self.kernel_num = config.KERNEL_NUM
        self.min_scale = config.TRAIN_MIN_SCALE

        root_dir = os.path.join(os.path.join(os.path.dirname(__file__), '..'), config.TRAIN_ROOT_DIR)
        ic15_train_data_dir = root_dir + 'ch4_training_images/'
        ic15_train_gt_dir = root_dir + 'ch4_training_localization_transcription_gt/'

        self.img_size = self.img_size if \
            (self.img_size is None or isinstance(self.img_size, tuple)) \
            else (self.img_size, self.img_size)

        data_dirs = [ic15_train_data_dir]
        gt_dirs = [ic15_train_gt_dir]

        self.all_img_paths = []
        self.all_gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = [i for i in os.listdir(data_dir)
                         if os.path.splitext(i)[-1].lower()
                         in ['.jpg', '.jpeg', '.png']]

            img_paths = []
            gt_paths = []
            for _, img_name in enumerate(img_names):
                img_path = os.path.join(data_dir, img_name)
                gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
                gt_path = os.path.join(gt_dir, gt_name)
                img_paths.append(img_path)
                gt_paths.append(gt_path)

            self.all_img_paths.extend(img_paths)
            self.all_gt_paths.extend(gt_paths)

    def __getitem__(self, index):
        img_path = self.all_img_paths[index]
        gt_path = self.all_gt_paths[index]

        img = get_img(img_path)
        bboxes, tags = get_bboxes(img, gt_path)

        # multi-scale training
        if self.is_transform:
            img = random_scale(img, min_size=self.img_size[0])

        # get gt_text and training_mask
        img_h, img_w = img.shape[0: 2]
        gt_text = np.zeros((img_h, img_w), dtype=np.float32)
        training_mask = np.ones((img_h, img_w), dtype=np.float32)
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img_w, img_h] * 4), (bboxes.shape[0], -1, 2)).astype('int32')
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_text, [bboxes[i]], 0, i + 1, -1)
                if not tags[i]:
                    cv2.drawContours(training_mask, [bboxes[i]], 0, 0, -1)

        # get gt_kernels
        gt_kernels = []
        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            gt_kernel = np.zeros(img.shape[0:2], dtype=np.float32)
            kernel_bboxes = shrink(bboxes, rate)
            for j in range(kernel_bboxes.shape[0]):
                cv2.drawContours(gt_kernel, [kernel_bboxes[j]], 0, 1, -1)
            gt_kernels.append(gt_kernel)

        # data augmentation
        if self.is_transform:
            imgs = [img, gt_text, training_mask]
            imgs.extend(gt_kernels)
            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop(imgs, self.img_size)
            img, gt_text, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]

        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = py_transforms.RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = py_transforms.ToTensor()(img)
        img = py_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = gt_text.astype(np.float32)
        gt_kernels = gt_kernels.astype(np.float32)
        training_mask = training_mask.astype(np.float32)

        return img, gt_text, gt_kernels, training_mask

    def __len__(self):
        return len(self.all_img_paths)


def IC15_TEST_Generator():
    ic15_test_data_dir = config.TEST_ROOT_DIR + 'ch4_test_images/'
    img_size = config.INFER_LONG_SIZE

    img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)

    data_dirs = [ic15_test_data_dir]
    all_img_paths = []

    for data_dir in data_dirs:
        img_names = [i for i in os.listdir(data_dir) if os.path.splitext(i)[-1].lower() in ['.jpg', '.jpeg', '.png']]

        img_paths = []
        for _, img_name in enumerate(img_names):
            img_path = data_dir + img_name
            img_paths.append(img_path)

        all_img_paths.extend(img_paths)

    dataset_length = len(all_img_paths)

    for index in range(dataset_length):
        img_path = all_img_paths[index]
        img_name = np.array(os.path.split(img_path)[-1])
        img = get_img(img_path)

        long_size = max(img.shape[:2])
        img_resized = np.zeros((long_size, long_size, 3), np.uint8)
        img_resized[:img.shape[0], :img.shape[1], :] = img
        img_resized = cv2.resize(img_resized, dsize=img_size)

        img_resized = Image.fromarray(img_resized)
        img_resized = img_resized.convert('RGB')
        img_resized = py_transforms.ToTensor()(img_resized)
        img_resized = py_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_resized)

        yield img, img_resized, img_name


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


def train_dataset_creator(rank, group_size, shuffle=True):
    cv2.setNumThreads(0)
    dataset = TrainDataset()
    sampler = DistributedSampler(dataset, rank, group_size, shuffle)
    data_set = ds.GeneratorDataset(dataset, ['img', 'gt_text', 'gt_kernels', 'training_mask'], num_parallel_workers=8,
                                   sampler=sampler)
    data_set = data_set.repeat(1)
    data_set = data_set.batch(config.TRAIN_BATCH_SIZE, drop_remainder=config.TRAIN_DROP_REMAINDER)
    return data_set


def test_dataset_creator():
    data_set = ds.GeneratorDataset(IC15_TEST_Generator, ['img', 'img_resized', 'img_name'])
    data_set = data_set.shuffle(config.TEST_BUFFER_SIZE)
    data_set = data_set.batch(1, drop_remainder=config.TEST_DROP_REMAINDER)
    return data_set
