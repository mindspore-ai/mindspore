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

"""YOLOv3 dataset"""
from __future__ import division

import abc
import io
import os
import math
import json
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import mindspore.dataset as de
import mindspore.dataset.transforms.vision.py_transforms as P
from config import ConfigYOLOV3ResNet18

iter_cnt = 0
_NUM_BOXES = 50

def preprocess_fn(image, box, is_training):
    """Preprocess function for dataset."""
    config_anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 163, 326]
    anchors = np.array([float(x) for x in config_anchors]).reshape(-1, 2)
    do_hsv = False
    max_boxes = 20
    num_classes = ConfigYOLOV3ResNet18.num_classes

    def _rand(a=0., b=1.):
        return np.random.rand() * (b - a) + a

    def _preprocess_true_boxes(true_boxes, anchors, in_shape=None):
        """Get true boxes."""
        num_layers = anchors.shape[0] // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        true_boxes = np.array(true_boxes, dtype='float32')
        # input_shape = np.array([in_shape, in_shape], dtype='int32')
        input_shape = np.array(in_shape, dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2.
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]
        y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]),
                            5 + num_classes), dtype='float32') for l in range(num_layers)]

        anchors = np.expand_dims(anchors, 0)
        anchors_max = anchors / 2.
        anchors_min = -anchors_max

        valid_mask = boxes_wh[..., 0] >= 1

        wh = boxes_wh[valid_mask]


        if len(wh) >= 1:
            wh = np.expand_dims(wh, -2)
            boxes_max = wh / 2.
            boxes_min = -boxes_max

            intersect_min = np.maximum(boxes_min, anchors_min)
            intersect_max = np.minimum(boxes_max, anchors_max)
            intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            best_anchor = np.argmax(iou, axis=-1)
            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)

                        c = true_boxes[t, 4].astype('int32')
                        y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                        y_true[l][j, i, k, 4] = 1.
                        y_true[l][j, i, k, 5 + c] = 1.

        pad_gt_box0 = np.zeros(shape=[50, 4], dtype=np.float32)
        pad_gt_box1 = np.zeros(shape=[50, 4], dtype=np.float32)
        pad_gt_box2 = np.zeros(shape=[50, 4], dtype=np.float32)

        mask0 = np.reshape(y_true[0][..., 4:5], [-1])
        gt_box0 = np.reshape(y_true[0][..., 0:4], [-1, 4])
        gt_box0 = gt_box0[mask0 == 1]
        pad_gt_box0[:gt_box0.shape[0]] = gt_box0

        mask1 = np.reshape(y_true[1][..., 4:5], [-1])
        gt_box1 = np.reshape(y_true[1][..., 0:4], [-1, 4])
        gt_box1 = gt_box1[mask1 == 1]
        pad_gt_box1[:gt_box1.shape[0]] = gt_box1

        mask2 = np.reshape(y_true[2][..., 4:5], [-1])
        gt_box2 = np.reshape(y_true[2][..., 0:4], [-1, 4])
        gt_box2 = gt_box2[mask2 == 1]
        pad_gt_box2[:gt_box2.shape[0]] = gt_box2

        return y_true[0], y_true[1], y_true[2], pad_gt_box0, pad_gt_box1, pad_gt_box2

    def _data_aug(image, box, is_training, jitter=0.3, hue=0.1, sat=1.5, val=1.5, image_size=(352, 640)):
        """Data augmentation function."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        iw, ih = image.size
        ori_image_shape = np.array([ih, iw], np.int32)
        h, w = image_size

        if not is_training:
            image = image.resize((w, h), Image.BICUBIC)
            image_data = np.array(image) / 255.
            if len(image_data.shape) == 2:
                image_data = np.expand_dims(image_data, axis=-1)
                image_data = np.concatenate([image_data, image_data, image_data], axis=-1)
            image_data = image_data.astype(np.float32)

            # correct boxes
            box_data = np.zeros((max_boxes, 5))
            if len(box) >= 1:
                np.random.shuffle(box)
                if len(box) > max_boxes:
                    box = box[:max_boxes]
                # xmin ymin xmax ymax
                box[:, [0, 2]] = box[:, [0, 2]] * float(w) / float(iw)
                box[:, [1, 3]] = box[:, [1, 3]] * float(h) / float(ih)
                box_data[:len(box)] = box
            else:
                image_data, box_data = None, None

            # preprocess bounding boxes
            bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 = \
                _preprocess_true_boxes(box_data, anchors, image_size)

            return image_data, bbox_true_1, bbox_true_2, bbox_true_3, \
                   ori_image_shape, gt_box1, gt_box2, gt_box3

        flip = _rand() < .5
        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        while True:
            # Prevent the situation that all boxes are eliminated
            new_ar = float(w) / float(h) * _rand(1 - jitter, 1 + jitter) / \
                     _rand(1 - jitter, 1 + jitter)
            scale = _rand(0.25, 2)

            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)

            dx = int(_rand(0, w - nw))
            dy = int(_rand(0, h - nh))

            if len(box) >= 1:
                t_box = box.copy()
                np.random.shuffle(t_box)
                t_box[:, [0, 2]] = t_box[:, [0, 2]] * float(nw) / float(iw) + dx
                t_box[:, [1, 3]] = t_box[:, [1, 3]] * float(nh) / float(ih) + dy
                if flip:
                    t_box[:, [0, 2]] = w - t_box[:, [2, 0]]
                t_box[:, 0:2][t_box[:, 0:2] < 0] = 0
                t_box[:, 2][t_box[:, 2] > w] = w
                t_box[:, 3][t_box[:, 3] > h] = h
                box_w = t_box[:, 2] - t_box[:, 0]
                box_h = t_box[:, 3] - t_box[:, 1]
                t_box = t_box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            if len(t_box) >= 1:
                box = t_box
                break

        box_data[:len(box)] = box
        # resize image
        image = image.resize((nw, nh), Image.BICUBIC)
        # place image
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # convert image to gray or not
        gray = _rand() < .25
        if gray:
            image = image.convert('L').convert('RGB')

        # when the channels of image is 1
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        # distort image
        hue = _rand(-hue, hue)
        sat = _rand(1, sat) if _rand() < .5 else 1 / _rand(1, sat)
        val = _rand(1, val) if _rand() < .5 else 1 / _rand(1, val)
        image_data = image / 255.
        if do_hsv:
            x = rgb_to_hsv(image_data)
            x[..., 0] += hue
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x > 1] = 1
            x[x < 0] = 0
            image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
        image_data = image_data.astype(np.float32)

        # preprocess bounding boxes
        bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 = \
            _preprocess_true_boxes(box_data, anchors, image_size)

        return image_data, bbox_true_1, bbox_true_2, bbox_true_3, \
               ori_image_shape, gt_box1, gt_box2, gt_box3

    images, bbox_1, bbox_2, bbox_3, _, gt_box1, gt_box2, gt_box3 = _data_aug(image, box, is_training)
    return images, bbox_1, bbox_2, bbox_3, gt_box1, gt_box2, gt_box3


def anno_parser(annos_str):
    """Annotation parser."""
    annos = []
    for anno_str in annos_str:
        anno = list(map(int, anno_str.strip().split(',')))
        annos.append(anno)
    return annos


def expand_path(path):
    """Get file list from path."""
    files = []
    if os.path.isdir(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                files.append(file)
    else:
        raise RuntimeError("Path given is not valid.")
    return files


def read_image(img_path):
    """Read image with PIL."""
    with open(img_path, "rb") as f:
        img = f.read()
    data = io.BytesIO(img)
    img = Image.open(data)
    return np.array(img)


class BaseDataset():
    """BaseDataset for GeneratorDataset iterator."""
    def __init__(self, image_dir, anno_path):
        self.image_dir = image_dir
        self.anno_path = anno_path
        self.cur_index = 0
        self.samples = []
        self.image_anno_dict = {}
        self._load_samples()

    def __getitem__(self, item):
        sample = self.samples[item]
        return self._next_data(sample, self.image_dir, self.image_anno_dict)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _next_data(sample, image_dir, image_anno_dict):
        """Get next data."""
        image = read_image(os.path.join(image_dir, sample))
        annos = image_anno_dict[sample]
        return [np.array(image), np.array(annos)]

    @abc.abstractmethod
    def _load_samples(self):
        """Base load samples."""


class YoloDataset(BaseDataset):
    """YoloDataset for GeneratorDataset iterator."""
    def _load_samples(self):
        """Load samples."""
        image_files_raw = expand_path(self.image_dir)
        self.samples = self._filter_valid_data(self.anno_path, image_files_raw)
        self.dataset_size = len(self.samples)
        if self.dataset_size == 0:
            raise RuntimeError("Valid dataset is none!")

    def _filter_valid_data(self, anno_path, image_files_raw):
        """Filter valid data."""
        image_files = []
        anno_dict = {}
        print("Start filter valid data.")
        with open(anno_path, "rb") as f:
            lines = f.readlines()
            for line in lines:
                line_str = line.decode("utf-8")
                line_split = str(line_str).split(' ')
                anno_dict[line_split[0].split("/")[-1]] = line_split[1:]
        anno_set = set(anno_dict.keys())
        image_set = set(image_files_raw)
        for image_file in (anno_set & image_set):
            image_files.append(image_file)
            self.image_anno_dict[image_file] = anno_parser(anno_dict[image_file])
        image_files.sort()
        print("Filter valid data done!")
        return image_files


class DistributedSampler():
    """DistributedSampler for YOLOv3"""
    def __init__(self, dataset_size, batch_size, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank % num_replicas
        self.epoch = 0
        self.num_samples = max(batch_size, int(math.ceil(dataset_size * 1.0 / self.num_replicas)))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            indices = np.random.RandomState(seed=self.epoch).permutation(self.dataset_size)
            indices = indices.tolist()
        else:
            indices = list(range(self.dataset_size))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def create_yolo_dataset(image_dir, anno_path, batch_size=32, repeat_num=10, device_num=1, rank=0,
                        is_training=True, num_parallel_workers=8):
    """Creatr YOLOv3 dataset with GeneratorDataset."""
    yolo_dataset = YoloDataset(image_dir=image_dir, anno_path=anno_path)
    distributed_sampler = DistributedSampler(yolo_dataset.dataset_size, batch_size, device_num, rank)
    ds = de.GeneratorDataset(yolo_dataset, column_names=["image", "annotation"], sampler=distributed_sampler)
    ds.set_dataset_size(len(distributed_sampler))
    compose_map_func = (lambda image, annotation: preprocess_fn(image, annotation, is_training))
    hwc_to_chw = P.HWC2CHW()
    ds = ds.map(input_columns=["image", "annotation"],
                output_columns=["image", "bbox_1", "bbox_2", "bbox_3", "gt_box1", "gt_box2", "gt_box3"],
                columns_order=["image", "bbox_1", "bbox_2", "bbox_3", "gt_box1", "gt_box2", "gt_box3"],
                operations=compose_map_func, num_parallel_workers=num_parallel_workers)
    ds = ds.map(input_columns=["image"], operations=hwc_to_chw, num_parallel_workers=num_parallel_workers)
    ds = ds.shuffle(buffer_size=256)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_num)
    return ds
