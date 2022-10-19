# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

import os
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
import mindspore.dataset as de
from mindspore.mindrecord import FileWriter
import mindspore.dataset.vision as C
from src.config import ConfigYOLOV3ResNet18

_NUM_BOXES = 50
np.random.seed(1)
de.config.set_seed(1)


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

    def _infer_data(img_data, input_shape, box):
        w, h = img_data.size
        input_h, input_w = input_shape
        scale = min(float(input_w) / float(w), float(input_h) / float(h))
        nw = int(w * scale)
        nh = int(h * scale)
        img_data = img_data.resize((nw, nh), Image.BICUBIC)

        new_image = np.zeros((input_h, input_w, 3), np.float32)
        new_image.fill(128)
        img_data = np.array(img_data)
        if len(img_data.shape) == 2:
            img_data = np.expand_dims(img_data, axis=-1)
            img_data = np.concatenate([img_data, img_data, img_data], axis=-1)

        dh = int((input_h - nh) / 2)
        dw = int((input_w - nw) / 2)
        new_image[dh:(nh + dh), dw:(nw + dw), :] = img_data
        new_image /= 255.
        new_image = np.transpose(new_image, (2, 0, 1))
        new_image = np.expand_dims(new_image, 0)
        return new_image, np.array([h, w], np.float32), box

    def _data_aug(image, box, is_training, jitter=0.3, hue=0.1, sat=1.5, val=1.5, image_size=(352, 640)):
        """Data augmentation function."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        iw, ih = image.size
        ori_image_shape = np.array([ih, iw], np.int32)
        h, w = image_size

        if not is_training:
            return _infer_data(image, image_size, box)

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

    if is_training:
        images, bbox_1, bbox_2, bbox_3, _, gt_box1, gt_box2, gt_box3 = _data_aug(image, box, is_training)
        return images, bbox_1, bbox_2, bbox_3, gt_box1, gt_box2, gt_box3

    images, shape, anno = _data_aug(image, box, is_training)
    return images, shape, anno


def anno_parser(annos_str):
    """Parse annotation from string to list."""
    annos = []
    for anno_str in annos_str:
        anno = list(map(int, anno_str.strip().split(',')))
        annos.append(anno)
    return annos


def filter_valid_data(image_dir, anno_path):
    """Filter valid image file, which both in image_dir and anno_path."""
    image_files = []
    image_anno_dict = {}
    if not os.path.isdir(image_dir):
        raise RuntimeError("Path given is not valid.")
    if not os.path.isfile(anno_path):
        raise RuntimeError("Annotation file is not valid.")

    with open(anno_path, "rb") as f:
        lines = f.readlines()
    for line in lines:
        line_str = line.decode("utf-8").strip()
        line_split = str(line_str).split(' ')
        file_name = line_split[0]
        if os.path.isfile(os.path.join(image_dir, file_name)):
            image_anno_dict[file_name] = anno_parser(line_split[1:])
            image_files.append(file_name)
    return image_files, image_anno_dict


def data_to_mindrecord_byte_image(image_dir, anno_path, mindrecord_dir, prefix="yolo.mindrecord", file_num=8):
    """Create MindRecord file by image_dir and anno_path."""
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)
    image_files, image_anno_dict = filter_valid_data(image_dir, anno_path)

    yolo_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int64", "shape": [-1, 5]},
    }
    writer.add_schema(yolo_json, "yolo_json")

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        with open(image_path, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name])
        row = {"image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()


def create_yolo_dataset(mindrecord_dir, batch_size=32, repeat_num=10, device_num=1, rank=0,
                        is_training=True, num_parallel_workers=8):
    """Create YOLOv3 dataset with MindDataset."""
    ds = de.MindDataset(mindrecord_dir, columns_list=["image", "annotation"], num_shards=device_num, shard_id=rank,
                        num_parallel_workers=num_parallel_workers, shuffle=False)
    decode = C.Decode()
    ds = ds.map(operations=decode, input_columns=["image"])
    compose_map_func = (lambda image, annotation: preprocess_fn(image, annotation, is_training))

    if is_training:
        hwc_to_chw = C.HWC2CHW()
        ds = ds.map(operations=compose_map_func, input_columns=["image", "annotation"],
                    output_columns=["image", "bbox_1", "bbox_2", "bbox_3", "gt_box1", "gt_box2", "gt_box3"],
                    num_parallel_workers=num_parallel_workers)
        ds = ds.project(["image", "bbox_1", "bbox_2", "bbox_3", "gt_box1", "gt_box2", "gt_box3"])
        ds = ds.map(operations=hwc_to_chw, input_columns=["image"], num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.repeat(repeat_num)
    else:
        ds = ds.map(operations=compose_map_func, input_columns=["image", "annotation"],
                    output_columns=["image", "image_shape", "annotation"],
                    num_parallel_workers=num_parallel_workers)
    return ds
