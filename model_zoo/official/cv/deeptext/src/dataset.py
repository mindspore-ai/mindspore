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

"""Deeptext dataset"""
from __future__ import division

import os
import numpy as np
from numpy import random

import mmcv
import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as CC
import mindspore.common.dtype as mstype
from mindspore.mindrecord import FileWriter
from src.config import config


def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


class PhotoMetricDistortion:
    """Photo Metric Distortion"""

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        img = img.astype('float32')

        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


class Expand:
    """expand image"""

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels


def resize_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """resize operation for image"""
    img_data = img
    img_data, w_scale, h_scale = mmcv.imresize(
        img_data, (config.img_width, config.img_height), return_scale=True)
    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def resize_column_test(img, img_shape, gt_bboxes, gt_label, gt_num):
    """resize operation for image of eval"""
    img_data = img
    img_data, w_scale, h_scale = mmcv.imresize(
        img_data, (config.img_width, config.img_height), return_scale=True)
    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = (config.img_height, config.img_width)
    img_shape = np.append(img_shape, (h_scale, w_scale))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def impad_to_multiple_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """impad operation for image"""
    img_data = mmcv.impad(img, (config.img_height, config.img_width))
    img_data = img_data.astype(np.float32)
    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def imnormalize_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """imnormalize operation for image"""
    img_data = mmcv.imnormalize(img, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375], True)
    img_data = img_data.astype(np.float32)
    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def flip_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """flip operation for image"""
    img_data = img
    img_data = mmcv.imflip(img_data)
    flipped = gt_bboxes.copy()
    _, w, _ = img_data.shape

    flipped[..., 0::4] = w - gt_bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - gt_bboxes[..., 0::4] - 1

    return (img_data, img_shape, flipped, gt_label, gt_num)


def flipped_generation(img, img_shape, gt_bboxes, gt_label, gt_num):
    """flipped generation"""
    img_data = img
    flipped = gt_bboxes.copy()
    _, w, _ = img_data.shape

    flipped[..., 0::4] = w - gt_bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - gt_bboxes[..., 0::4] - 1

    return (img_data, img_shape, flipped, gt_label, gt_num)


def image_bgr_rgb(img, img_shape, gt_bboxes, gt_label, gt_num):
    img_data = img[:, :, ::-1]
    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def transpose_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """transpose operation for image"""
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np.float32)
    img_shape = img_shape.astype(np.float32)
    gt_bboxes = gt_bboxes.astype(np.float32)
    gt_label = gt_label.astype(np.int32)
    gt_num = gt_num.astype(np.bool)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def photo_crop_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """photo crop operation for image"""
    random_photo = PhotoMetricDistortion()
    img_data, gt_bboxes, gt_label = random_photo(img, gt_bboxes, gt_label)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def expand_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """expand operation for image"""
    expand = Expand()
    img, gt_bboxes, gt_label = expand(img, gt_bboxes, gt_label)

    return (img, img_shape, gt_bboxes, gt_label, gt_num)


def preprocess_fn(image, box, is_training):
    """Preprocess function for dataset."""

    def _infer_data(image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert):
        image_shape = image_shape[:2]
        input_data = image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert
        input_data = resize_column_test(*input_data)
        input_data = image_bgr_rgb(*input_data)
        output_data = input_data
        return output_data

    def _data_aug(image, box, is_training):
        """Data augmentation function."""
        image_bgr = image.copy()
        image_bgr[:, :, 0] = image[:, :, 2]
        image_bgr[:, :, 1] = image[:, :, 1]
        image_bgr[:, :, 2] = image[:, :, 0]
        image_shape = image_bgr.shape[:2]
        gt_box = box[:, :4]
        gt_label = box[:, 4]
        gt_iscrowd = box[:, 5]

        pad_max_number = 128
        if box.shape[0] < 128:
            gt_box_new = np.pad(gt_box, ((0, pad_max_number - box.shape[0]), (0, 0)), mode="constant",
                                constant_values=0)
            gt_label_new = np.pad(gt_label, ((0, pad_max_number - box.shape[0])), mode="constant", constant_values=-1)
            gt_iscrowd_new = np.pad(gt_iscrowd, ((0, pad_max_number - box.shape[0])), mode="constant",
                                    constant_values=1)
        else:
            gt_box_new = gt_box[0:pad_max_number]
            gt_label_new = gt_label[0:pad_max_number]
            gt_iscrowd_new = gt_iscrowd[0:pad_max_number]

        gt_iscrowd_new_revert = (~(gt_iscrowd_new.astype(np.bool))).astype(np.int32)

        if not is_training:
            return _infer_data(image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert)

        input_data = image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert

        expand = (np.random.rand() < config.expand_ratio)
        if expand:
            input_data = expand_column(*input_data)

        input_data = photo_crop_column(*input_data)
        input_data = resize_column(*input_data)
        input_data = image_bgr_rgb(*input_data)

        output_data = input_data
        return output_data

    return _data_aug(image, box, is_training)


def get_imgs_and_annos(img_dir, txt_dir, image_files, image_anno_dict):
    img_basenames = []
    for file in os.listdir(img_dir):
        # Filter git file.
        if 'gif' not in file:
            img_basenames.append(os.path.basename(file))

    img_names = []
    for item in img_basenames:
        temp1, _ = os.path.splitext(item)
        img_names.append((temp1, item))
    for img, img_basename in img_names:
        image_path = img_dir + '/' + img_basename
        annos = []
        # Parse annotation of dataset in paper.
        if len(img) == 6 and '_' not in img_basename:
            gt = open(txt_dir + '/' + img + '.txt').read().splitlines()
            if img.isdigit() and int(img) > 1200:
                continue
            for img_each_label in gt:
                spt = img_each_label.replace(',', '').split(' ')
                if ' ' not in img_each_label:
                    spt = img_each_label.split(',')
                annos.append(
                    [spt[0], spt[1], str(int(spt[0]) + int(spt[2])), str(int(spt[1]) + int(spt[3]))] + [1] + [
                        int(0)])
        else:
            anno_file = txt_dir + '/gt_img_' + img.split('_')[-1] + '.txt'
            if not os.path.exists(anno_file):
                anno_file = txt_dir + '/gt_' + img.split('_')[-1] + '.txt'
            if not os.path.exists(anno_file):
                anno_file = txt_dir + '/img_' + img.split('_')[-1] + '.txt'
            gt = open(anno_file).read().splitlines()
            for img_each_label in gt:
                spt = img_each_label.replace(',', '').split(' ')
                if ' ' not in img_each_label:
                    spt = img_each_label.split(',')
                annos.append([spt[0], spt[1], spt[2], spt[3]] + [1] + [int(0)])

        image_files.append(image_path)
        if annos:
            image_anno_dict[image_path] = np.array(annos)
        else:
            image_anno_dict[image_path] = np.array([0, 0, 0, 0, 0, 1])


def create_label(is_training):
    """Create image label."""
    image_files = []
    image_anno_dict = {}

    if is_training:
        img_dirs = config.train_images.split(',')
        txt_dirs = config.train_txts.split(',')
    else:
        img_dirs = config.test_images.split(',')
        txt_dirs = config.test_txts.split(',')

    for img_dir, txt_dir in zip(img_dirs, txt_dirs):
        get_imgs_and_annos(img_dir, txt_dir, image_files, image_anno_dict)

    if is_training and config.use_coco:
        coco_root = config.coco_root
        data_type = config.coco_train_data_type
        from src.coco_text import COCO_Text
        anno_json = config.cocotext_json
        ct = COCO_Text(anno_json)
        image_ids = ct.getImgIds(imgIds=ct.train,
                                 catIds=[('legibility', 'legible')])
        for img_id in image_ids:
            image_info = ct.loadImgs(img_id)[0]
            file_name = image_info['file_name'][15:]
            anno_ids = ct.getAnnIds(imgIds=img_id)
            anno = ct.loadAnns(anno_ids)
            image_path = os.path.join(coco_root, data_type, file_name)
            annos = []
            for label in anno:
                bbox = label["bbox"]
                x1, x2 = bbox[0], bbox[0] + bbox[2]
                y1, y2 = bbox[1], bbox[1] + bbox[3]
                annos.append([x1, y1, x2, y2] + [1] + [int(0)])

            image_files.append(image_path)
            if annos:
                image_anno_dict[image_path] = np.array(annos)
            else:
                image_anno_dict[image_path] = np.array([0, 0, 0, 0, 0, 1])

    return image_files, image_anno_dict


def anno_parser(annos_str):
    """Parse annotation from string to list."""
    annos = []
    for anno_str in annos_str:
        anno = list(map(int, anno_str.strip().split(',')))
        annos.append(anno)
    return annos


def data_to_mindrecord_byte_image(is_training=True, prefix="deeptext.mindrecord", file_num=8):
    """Create MindRecord file."""
    mindrecord_dir = config.mindrecord_dir
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)
    image_files, image_anno_dict = create_label(is_training)

    deeptext_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 6]},
    }
    writer.add_schema(deeptext_json, "deeptext_json")

    for image_name in image_files:
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        row = {"image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()


def create_deeptext_dataset(mindrecord_file, batch_size=2, repeat_num=12, device_num=1, rank_id=0,
                            is_training=True, num_parallel_workers=4):
    """Creatr deeptext dataset with MindDataset."""
    ds = de.MindDataset(mindrecord_file, columns_list=["image", "annotation"], num_shards=device_num, shard_id=rank_id,
                        num_parallel_workers=1, shuffle=is_training)
    decode = C.Decode()
    ds = ds.map(operations=decode, input_columns=["image"], num_parallel_workers=1)
    compose_map_func = (lambda image, annotation: preprocess_fn(image, annotation, is_training))

    hwc_to_chw = C.HWC2CHW()
    normalize_op = C.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
    horizontally_op = C.RandomHorizontalFlip(1)
    type_cast0 = CC.TypeCast(mstype.float32)
    type_cast1 = CC.TypeCast(mstype.float32)
    type_cast2 = CC.TypeCast(mstype.int32)
    type_cast3 = CC.TypeCast(mstype.bool_)

    if is_training:
        ds = ds.map(operations=compose_map_func, input_columns=["image", "annotation"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num"],
                    column_order=["image", "image_shape", "box", "label", "valid_num"],
                    num_parallel_workers=num_parallel_workers)

        flip = (np.random.rand() < config.flip_ratio)
        if flip:
            ds = ds.map(operations=[normalize_op, type_cast0, horizontally_op], input_columns=["image"],
                        num_parallel_workers=12)
            ds = ds.map(operations=flipped_generation,
                        input_columns=["image", "image_shape", "box", "label", "valid_num"],
                        num_parallel_workers=num_parallel_workers)
        else:
            ds = ds.map(operations=[normalize_op, type_cast0], input_columns=["image"],
                        num_parallel_workers=12)
        ds = ds.map(operations=[hwc_to_chw, type_cast1], input_columns=["image"],
                    num_parallel_workers=12)

    else:
        ds = ds.map(operations=compose_map_func,
                    input_columns=["image", "annotation"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num"],
                    column_order=["image", "image_shape", "box", "label", "valid_num"],
                    num_parallel_workers=num_parallel_workers)

        ds = ds.map(operations=[normalize_op, hwc_to_chw, type_cast1], input_columns=["image"],
                    num_parallel_workers=24)

    # transpose_column from python to c
    ds = ds.map(operations=[type_cast1], input_columns=["image_shape"])
    ds = ds.map(operations=[type_cast1], input_columns=["box"])
    ds = ds.map(operations=[type_cast2], input_columns=["label"])
    ds = ds.map(operations=[type_cast3], input_columns=["valid_num"])
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_num)

    return ds
