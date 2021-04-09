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

"""MaskRcnn dataset"""
from __future__ import division

import os
import numpy as np
from numpy import random
import cv2
import mmcv

import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C
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

    def __call__(self, img, boxes, labels, mask):
        if random.randint(2):
            return img, boxes, labels, mask

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)

        mask_count, mask_h, mask_w = mask.shape
        expand_mask = np.zeros((mask_count, int(mask_h * ratio), int(mask_w * ratio))).astype(mask.dtype)
        expand_mask[:, top:top + h, left:left + w] = mask
        mask = expand_mask

        return img, boxes, labels, mask

def rescale_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """rescale operation for image"""
    img_data, scale_factor = mmcv.imrescale(img, (config.img_width, config.img_height), return_scale=True)
    if img_data.shape[0] > config.img_height:
        img_data, scale_factor2 = mmcv.imrescale(img_data, (config.img_height, config.img_height), return_scale=True)
        scale_factor = scale_factor*scale_factor2

    gt_bboxes = gt_bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_data.shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_data.shape[0] - 1)

    gt_mask_data = np.array([
        mmcv.imrescale(mask, scale_factor, interpolation='nearest')
        for mask in gt_mask
    ])

    pad_h = config.img_height - img_data.shape[0]
    pad_w = config.img_width - img_data.shape[1]
    assert ((pad_h >= 0) and (pad_w >= 0))

    pad_img_data = np.zeros((config.img_height, config.img_width, 3)).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

    mask_count, mask_h, mask_w = gt_mask_data.shape
    pad_mask = np.zeros((mask_count, config.img_height, config.img_width)).astype(gt_mask_data.dtype)
    pad_mask[:, 0:mask_h, 0:mask_w] = gt_mask_data

    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return  (pad_img_data, img_shape, gt_bboxes, gt_label, gt_num, pad_mask)

def rescale_column_test(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """rescale operation for image of eval"""
    img_data, scale_factor = mmcv.imrescale(img, (config.img_width, config.img_height), return_scale=True)
    if img_data.shape[0] > config.img_height:
        img_data, scale_factor2 = mmcv.imrescale(img_data, (config.img_height, config.img_height), return_scale=True)
        scale_factor = scale_factor*scale_factor2

    pad_h = config.img_height - img_data.shape[0]
    pad_w = config.img_width - img_data.shape[1]
    assert ((pad_h >= 0) and (pad_w >= 0))

    pad_img_data = np.zeros((config.img_height, config.img_width, 3)).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

    img_shape = np.append(img_shape, (scale_factor, scale_factor))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return  (pad_img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask)

def resize_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """resize operation for image"""
    img_data = img
    img_data, w_scale, h_scale = mmcv.imresize(
        img_data, (config.img_width, config.img_height), return_scale=True)
    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1) # x1, x2   [0, W-1]
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1) # y1, y2   [0, H-1]

    gt_mask_data = np.array([
        mmcv.imresize(mask, (config.img_width, config.img_height), interpolation='nearest')
        for mask in gt_mask
    ])
    return  (img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask_data)

def resize_column_test(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """resize operation for image of eval"""
    img_data = img
    img_data, w_scale, h_scale = mmcv.imresize(
        img_data, (config.img_width, config.img_height), return_scale=True)
    img_shape = np.append(img_shape, (h_scale, w_scale))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return  (img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask)

def impad_to_multiple_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """impad operation for image"""
    img_data = mmcv.impad(img, (config.img_height, config.img_width))
    img_data = img_data.astype(np.float32)
    return (img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask)

def imnormalize_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """imnormalize operation for image"""
    img_data = mmcv.imnormalize(img, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375], True)
    img_data = img_data.astype(np.float32)
    return (img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask)

def flip_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """flip operation for image"""
    img_data = img
    img_data = mmcv.imflip(img_data)
    flipped = gt_bboxes.copy()
    _, w, _ = img_data.shape

    flipped[..., 0::4] = w - gt_bboxes[..., 2::4] - 1  # x1 = W-x2-1
    flipped[..., 2::4] = w - gt_bboxes[..., 0::4] - 1  # x2 = W-x1-1

    gt_mask_data = np.array([mask[:, ::-1] for mask in gt_mask])

    return  (img_data, img_shape, flipped, gt_label, gt_num, gt_mask_data)

def transpose_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """transpose operation for image"""
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np.float16)
    img_shape = img_shape.astype(np.float16)
    gt_bboxes = gt_bboxes.astype(np.float16)
    gt_label = gt_label.astype(np.int32)
    gt_num = gt_num.astype(np.bool)
    gt_mask_data = gt_mask.astype(np.bool)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask_data)

def photo_crop_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """photo crop operation for image"""
    random_photo = PhotoMetricDistortion()
    img_data, gt_bboxes, gt_label = random_photo(img, gt_bboxes, gt_label)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num, gt_mask)

def expand_column(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask):
    """expand operation for image"""
    expand = Expand()
    img, gt_bboxes, gt_label, gt_mask = expand(img, gt_bboxes, gt_label, gt_mask)

    return (img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask)

def pad_to_max(img, img_shape, gt_bboxes, gt_label, gt_num, gt_mask, instance_count):
    pad_max_number = config.max_instance_count
    gt_box_new = np.pad(gt_bboxes, ((0, pad_max_number - instance_count), (0, 0)), mode="constant", constant_values=0)
    gt_label_new = np.pad(gt_label, ((0, pad_max_number - instance_count)), mode="constant", constant_values=-1)
    gt_iscrowd_new = np.pad(gt_num, ((0, pad_max_number - instance_count)), mode="constant", constant_values=1)
    gt_iscrowd_new_revert = ~(gt_iscrowd_new.astype(np.bool))

    return img, img_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert, gt_mask

def preprocess_fn(image, box, mask, mask_shape, is_training):
    """Preprocess function for dataset."""
    def _infer_data(image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert,
                    gt_mask_new, instance_count):
        image_shape = image_shape[:2]
        input_data = image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert, gt_mask_new

        if config.keep_ratio:
            input_data = rescale_column_test(*input_data)
        else:
            input_data = resize_column_test(*input_data)
        input_data = imnormalize_column(*input_data)

        input_data = pad_to_max(*input_data, instance_count)
        output_data = transpose_column(*input_data)
        return output_data

    def _data_aug(image, box, mask, mask_shape, is_training):
        """Data augmentation function."""
        image_bgr = image.copy()
        image_bgr[:, :, 0] = image[:, :, 2]
        image_bgr[:, :, 1] = image[:, :, 1]
        image_bgr[:, :, 2] = image[:, :, 0]
        image_shape = image_bgr.shape[:2]
        instance_count = box.shape[0]
        gt_box = box[:, :4]
        gt_label = box[:, 4]
        gt_iscrowd = box[:, 5]
        gt_mask = mask.copy()
        n, h, w = mask_shape
        gt_mask = gt_mask.reshape(n, h, w)
        assert n == box.shape[0]

        if not is_training:
            return _infer_data(image_bgr, image_shape, gt_box, gt_label, gt_iscrowd, gt_mask, instance_count)

        flip = (np.random.rand() < config.flip_ratio)
        expand = (np.random.rand() < config.expand_ratio)

        input_data = image_bgr, image_shape, gt_box, gt_label, gt_iscrowd, gt_mask

        if expand:
            input_data = expand_column(*input_data)
        if config.keep_ratio:
            input_data = rescale_column(*input_data)
        else:
            input_data = resize_column(*input_data)

        input_data = imnormalize_column(*input_data)
        if flip:
            input_data = flip_column(*input_data)

        input_data = pad_to_max(*input_data, instance_count)
        output_data = transpose_column(*input_data)
        return output_data

    return _data_aug(image, box, mask, mask_shape, is_training)

def annToMask(ann, height, width):
    """Convert annotation to RLE and then to binary mask."""
    from pycocotools import mask as maskHelper
    segm = ann['segmentation']
    if isinstance(segm, list):
        rles = maskHelper.frPyObjects(segm, height, width)
        rle = maskHelper.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = maskHelper.frPyObjects(segm, height, width)
    else:
        rle = ann['segmentation']
    m = maskHelper.decode(rle)
    return m

def create_coco_label(is_training):
    """Get image path and annotation from COCO."""
    from pycocotools.coco import COCO

    coco_root = config.coco_root
    data_type = config.val_data_type
    if is_training:
        data_type = config.train_data_type

    #Classes need to train or test.
    train_cls = config.coco_classes
    train_cls_dict = {}
    for i, cls in enumerate(train_cls):
        train_cls_dict[cls] = i

    anno_json = os.path.join(coco_root, config.instance_set.format(data_type))

    coco = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["id"]] = cat["name"]

    image_ids = coco.getImgIds()
    image_files = []
    image_anno_dict = {}
    masks = {}
    masks_shape = {}
    images_num = len(image_ids)
    for ind, img_id in enumerate(image_ids):
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        image_path = os.path.join(coco_root, data_type, file_name)
        if not os.path.isfile(image_path):
            print("{}/{}: {} is in annotations but not exist".format(ind + 1, images_num, image_path))
            continue
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        instance_masks = []
        image_height = coco.imgs[img_id]["height"]
        image_width = coco.imgs[img_id]["width"]
        if (ind + 1) % 10 == 0:
            print("{}/{}: parsing annotation for image={}".format(ind + 1, images_num, file_name))
        if not is_training:
            image_files.append(image_path)
            image_anno_dict[image_path] = np.array([0, 0, 0, 0, 0, 1])
            masks[image_path] = np.zeros([1, 1, 1], dtype=np.bool).tobytes()
            masks_shape[image_path] = np.array([1, 1, 1], dtype=np.int32)
        else:
            for label in anno:
                bbox = label["bbox"]
                class_name = classs_dict[label["category_id"]]
                if class_name in train_cls:
                    # get coco mask
                    m = annToMask(label, image_height, image_width)
                    if m.max() < 1:
                        print("all black mask!!!!")
                        continue
                    # Resize mask for the crowd
                    if label['iscrowd'] and (m.shape[0] != image_height or m.shape[1] != image_width):
                        m = np.ones([image_height, image_width], dtype=np.bool)
                    instance_masks.append(m)

                    # get coco bbox
                    x1, x2 = bbox[0], bbox[0] + bbox[2]
                    y1, y2 = bbox[1], bbox[1] + bbox[3]
                    annos.append([x1, y1, x2, y2] + [train_cls_dict[class_name]] + [int(label["iscrowd"])])
                else:
                    print("not in classes: ", class_name)

            image_files.append(image_path)
            if annos:
                image_anno_dict[image_path] = np.array(annos)
                instance_masks = np.stack(instance_masks, axis=0).astype(np.bool)
                masks[image_path] = np.array(instance_masks).tobytes()
                masks_shape[image_path] = np.array(instance_masks.shape, dtype=np.int32)
            else:
                print("no annotations for image ", file_name)
                image_anno_dict[image_path] = np.array([0, 0, 0, 0, 0, 1])
                masks[image_path] = np.zeros([1, image_height, image_width], dtype=np.bool).tobytes()
                masks_shape[image_path] = np.array([1, image_height, image_width], dtype=np.int32)

    return image_files, image_anno_dict, masks, masks_shape

def data_to_mindrecord_byte_image(dataset="coco", is_training=True, prefix="maskrcnn.mindrecord", file_num=8):
    """Create MindRecord file."""
    mindrecord_dir = config.mindrecord_dir
    mindrecord_path = os.path.join(mindrecord_dir, prefix)

    writer = FileWriter(mindrecord_path, file_num)
    if dataset == "coco":
        image_files, image_anno_dict, masks, masks_shape = create_coco_label(is_training)
    else:
        print("Error unsupported other dataset")
        return

    maskrcnn_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 6]},
        "mask": {"type": "bytes"},
        "mask_shape": {"type": "int32", "shape": [-1]},
    }
    writer.add_schema(maskrcnn_json, "maskrcnn_json")

    image_files_num = len(image_files)
    for ind, image_name in enumerate(image_files):
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        mask = masks[image_name]
        mask_shape = masks_shape[image_name]
        row = {"image": img, "annotation": annos, "mask": mask, "mask_shape": mask_shape}
        if (ind + 1) % 10 == 0:
            print("writing {}/{} into mindrecord".format(ind + 1, image_files_num))
        writer.write_raw_data([row])
    writer.commit()

def create_maskrcnn_dataset(mindrecord_file, batch_size=2, device_num=1, rank_id=0,
                            is_training=True, num_parallel_workers=8):
    """Create MaskRcnn dataset with MindDataset."""
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(8)
    ds = de.MindDataset(mindrecord_file, columns_list=["image", "annotation", "mask", "mask_shape"],
                        num_shards=device_num, shard_id=rank_id,
                        num_parallel_workers=4, shuffle=is_training)

    decode = C.Decode()
    ds = ds.map(operations=decode, input_columns=["image"])
    compose_map_func = (lambda image, annotation, mask, mask_shape:
                        preprocess_fn(image, annotation, mask, mask_shape, is_training))

    if is_training:
        ds = ds.map(operations=compose_map_func,
                    input_columns=["image", "annotation", "mask", "mask_shape"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    column_order=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    python_multiprocessing=False,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True, pad_info={"mask": ([config.max_instance_count, None, None], 0)})

    else:
        ds = ds.map(operations=compose_map_func,
                    input_columns=["image", "annotation", "mask", "mask_shape"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    column_order=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)

    return ds
