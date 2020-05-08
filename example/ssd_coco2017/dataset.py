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

"""SSD dataset"""
from __future__ import division

import os
import math
import itertools as it
import numpy as np
import cv2

import mindspore.dataset as de
import mindspore.dataset.transforms.vision.c_transforms as C
from mindspore.mindrecord import FileWriter
from config import ConfigSSD

config = ConfigSSD()

class GeneratDefaultBoxes():
    """
    Generate Default boxes for SSD, follows the order of (W, H, archor_sizes).
    `self.default_boxes` has a shape of [archor_sizes, H, W, 4], the last dimension is [x, y, w, h].
    `self.default_boxes_ltrb` has a shape as `self.default_boxes`, the last dimension is [x1, y1, x2, y2].
    """
    def __init__(self):
        fk = config.IMG_SHAPE[0] / np.array(config.STEPS)
        self.default_boxes = []
        for idex, feature_size in enumerate(config.FEATURE_SIZE):
            sk1 = config.SCALES[idex] / config.IMG_SHAPE[0]
            sk2 = config.SCALES[idex + 1] / config.IMG_SHAPE[0]
            sk3 = math.sqrt(sk1 * sk2)

            if config.NUM_DEFAULT[idex] == 3:
                all_sizes = [(0.5, 1.0), (1.0, 1.0), (1.0, 0.5)]
            else:
                all_sizes = [(sk1, sk1), (sk3, sk3)]
                for aspect_ratio in config.ASPECT_RATIOS[idex]:
                    w, h = sk1 * math.sqrt(aspect_ratio), sk1 / math.sqrt(aspect_ratio)
                    all_sizes.append((w, h))
                    all_sizes.append((h, w))

            assert len(all_sizes) == config.NUM_DEFAULT[idex]

            for i, j in it.product(range(feature_size), repeat=2):
                for w, h in all_sizes:
                    cx, cy = (j + 0.5) / fk[idex], (i + 0.5) / fk[idex]
                    box = [np.clip(k, 0, 1) for k in (cx, cy, w, h)]
                    self.default_boxes.append(box)

        def to_ltrb(cx, cy, w, h):
            return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

        # For IoU calculation
        self.default_boxes_ltrb = np.array(tuple(to_ltrb(*i) for i in self.default_boxes), dtype='float32')
        self.default_boxes = np.array(self.default_boxes, dtype='float32')


default_boxes_ltrb = GeneratDefaultBoxes().default_boxes_ltrb
default_boxes = GeneratDefaultBoxes().default_boxes
x1, y1, x2, y2 = np.split(default_boxes_ltrb[:, :4], 4, axis=-1)
vol_anchors = (x2 - x1) * (y2 - y1)
matching_threshold = config.MATCH_THRESHOLD


def ssd_bboxes_encode(boxes):
    """
    Labels anchors with ground truth inputs.

    Args:
        boxex: ground truth with shape [N, 5], for each row, it stores [x, y, w, h, cls].

    Returns:
        gt_loc: location ground truth with shape [num_anchors, 4].
        gt_label: class ground truth with shape [num_anchors, 1].
        num_matched_boxes: number of positives in an image.
    """

    def jaccard_with_anchors(bbox):
        """Compute jaccard score a box and the anchors."""
        # Intersection bbox and volume.
        xmin = np.maximum(x1, bbox[0])
        ymin = np.maximum(y1, bbox[1])
        xmax = np.minimum(x2, bbox[2])
        ymax = np.minimum(y2, bbox[3])
        w = np.maximum(xmax - xmin, 0.)
        h = np.maximum(ymax - ymin, 0.)

        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
        jaccard = inter_vol / union_vol
        return np.squeeze(jaccard)

    pre_scores = np.zeros((config.NUM_SSD_BOXES), dtype=np.float32)
    t_boxes = np.zeros((config.NUM_SSD_BOXES, 4), dtype=np.float32)
    t_label = np.zeros((config.NUM_SSD_BOXES), dtype=np.int64)
    for bbox in boxes:
        label = int(bbox[4])
        scores = jaccard_with_anchors(bbox)
        mask = (scores > matching_threshold)
        if not np.any(mask):
            mask[np.argmax(scores)] = True

        mask = mask & (scores > pre_scores)
        pre_scores = np.maximum(pre_scores, scores)
        t_label = mask * label + (1 - mask) * t_label
        for i in range(4):
            t_boxes[:, i] = mask * bbox[i] + (1 - mask) * t_boxes[:, i]

    index = np.nonzero(t_label)

    # Transform to ltrb.
    bboxes = np.zeros((config.NUM_SSD_BOXES, 4), dtype=np.float32)
    bboxes[:, [0, 1]] = (t_boxes[:, [0, 1]] + t_boxes[:, [2, 3]]) / 2
    bboxes[:, [2, 3]] = t_boxes[:, [2, 3]] - t_boxes[:, [0, 1]]

    # Encode features.
    bboxes_t = bboxes[index]
    default_boxes_t = default_boxes[index]
    bboxes_t[:, :2] = (bboxes_t[:, :2] - default_boxes_t[:, :2]) / (default_boxes_t[:, 2:] * config.PRIOR_SCALING[0])
    bboxes_t[:, 2:4] = np.log(bboxes_t[:, 2:4] / default_boxes_t[:, 2:4]) / config.PRIOR_SCALING[1]
    bboxes[index] = bboxes_t

    num_match_num = np.array([len(np.nonzero(t_label)[0])], dtype=np.int32)
    return bboxes, t_label.astype(np.int32), num_match_num

def ssd_bboxes_decode(boxes, index):
    """Decode predict boxes to [x, y, w, h]"""
    boxes_t = boxes[index]
    default_boxes_t = default_boxes[index]
    boxes_t[:, :2] = boxes_t[:, :2] * config.PRIOR_SCALING[0] * default_boxes_t[:, 2:] + default_boxes_t[:, :2]
    boxes_t[:, 2:4] = np.exp(boxes_t[:, 2:4] * config.PRIOR_SCALING[1]) * default_boxes_t[:, 2:4]

    bboxes = np.zeros((len(boxes_t), 4), dtype=np.float32)

    bboxes[:, [0, 1]] = boxes_t[:, [0, 1]] - boxes_t[:, [2, 3]] / 2
    bboxes[:, [2, 3]] = boxes_t[:, [0, 1]] + boxes_t[:, [2, 3]] / 2

    return bboxes

def preprocess_fn(image, box, is_training):
    """Preprocess function for dataset."""

    def _rand(a=0., b=1.):
        """Generate random."""
        return np.random.rand() * (b - a) + a

    def _infer_data(image, input_shape, box):
        img_h, img_w, _ = image.shape
        input_h, input_w = input_shape

        scale = min(float(input_w) / float(img_w), float(input_h) / float(img_h))
        nw = int(img_w * scale)
        nh = int(img_h * scale)

        image = cv2.resize(image, (nw, nh))

        new_image = np.zeros((input_h, input_w, 3), np.float32)
        dh = (input_h - nh) // 2
        dw = (input_w - nw) // 2
        new_image[dh: (nh + dh), dw: (nw + dw), :] = image
        image = new_image

        #When the channels of image is 1
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        box = box.astype(np.float32)

        box[:, [0, 2]] = (box[:, [0, 2]] * scale + dw) / input_w
        box[:, [1, 3]] = (box[:, [1, 3]] * scale + dh) / input_h
        return image, np.array((img_h, img_w), np.float32), box

    def _data_aug(image, box, is_training, image_size=(300, 300)):
        """Data augmentation function."""
        ih, iw, _ = image.shape
        w, h = image_size

        if not is_training:
            return _infer_data(image, image_size, box)
        # Random settings
        scale_w = _rand(0.75, 1.25)
        scale_h = _rand(0.75, 1.25)

        flip = _rand() < .5
        nw = iw * scale_w
        nh = ih * scale_h
        scale = min(w / nw, h / nh)
        nw = int(scale * nw)
        nh = int(scale * nh)

        # Resize image
        image = cv2.resize(image, (nw, nh))

        # place image
        new_image = np.zeros((h, w, 3), dtype=np.float32)
        dw = (w - nw) // 2
        dh = (h - nh) // 2
        new_image[dh:dh + nh, dw:dw + nw, :] = image
        image = new_image

        # Flip image or not
        if flip:
            image = cv2.flip(image, 1, dst=None)

        # Convert image to gray or not
        gray = _rand() < .25
        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # When the channels of image is 1
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        box = box.astype(np.float32)

        # Transform box with shape[x1, y1, x2, y2].
        box[:, [0, 2]] = (box[:, [0, 2]] * scale * scale_w + dw) / w
        box[:, [1, 3]] = (box[:, [1, 3]] * scale * scale_h + dh) / h

        if flip:
            box[:, [0, 2]] = 1 - box[:, [2, 0]]

        box, label, num_match_num = ssd_bboxes_encode(box)
        return image, box, label, num_match_num
    return _data_aug(image, box, is_training, image_size=config.IMG_SHAPE)


def create_coco_label(is_training):
    """Get image path and annotation from COCO."""
    from pycocotools.coco import COCO

    coco_root = config.COCO_ROOT
    data_type = config.VAL_DATA_TYPE
    if is_training:
        data_type = config.TRAIN_DATA_TYPE

    #Classes need to train or test.
    train_cls = config.COCO_CLASSES
    train_cls_dict = {}
    for i, cls in enumerate(train_cls):
        train_cls_dict[cls] = i

    anno_json = os.path.join(coco_root, config.INSTANCES_SET.format(data_type))

    coco = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["id"]] = cat["name"]

    image_ids = coco.getImgIds()
    image_files = []
    image_anno_dict = {}

    for img_id in image_ids:
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        for label in anno:
            bbox = label["bbox"]
            class_name = classs_dict[label["category_id"]]
            if class_name in train_cls:
                x_min, x_max = bbox[0], bbox[0] + bbox[2]
                y_min, y_max = bbox[1], bbox[1] + bbox[3]
                annos.append(list(map(round, [x_min, y_min, x_max, y_max])) + [train_cls_dict[class_name]])
        if len(annos) >= 1:
            image_files.append(image_path)
            image_anno_dict[image_path] = np.array(annos)
    return image_files, image_anno_dict


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
        image_path = os.path.join(image_dir, file_name)
        if os.path.isfile(image_path):
            image_anno_dict[image_path] = anno_parser(line_split[1:])
            image_files.append(image_path)
    return image_files, image_anno_dict


def data_to_mindrecord_byte_image(dataset="coco", is_training=True, prefix="ssd.mindrecord", file_num=8):
    """Create MindRecord file."""
    mindrecord_dir = config.MINDRECORD_DIR
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)
    if dataset == "coco":
        image_files, image_anno_dict = create_coco_label(is_training)
    else:
        image_files, image_anno_dict = filter_valid_data(config.IMAGE_DIR, config.ANNO_PATH)

    ssd_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
    }
    writer.add_schema(ssd_json, "ssd_json")

    for image_name in image_files:
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        row = {"image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()


def create_ssd_dataset(mindrecord_file, batch_size=32, repeat_num=10, device_num=1, rank=0,
                       is_training=True, num_parallel_workers=4):
    """Creatr SSD dataset with MindDataset."""
    ds = de.MindDataset(mindrecord_file, columns_list=["image", "annotation"], num_shards=device_num, shard_id=rank,
                        num_parallel_workers=num_parallel_workers, shuffle=is_training)
    decode = C.Decode()
    ds = ds.map(input_columns=["image"], operations=decode)
    compose_map_func = (lambda image, annotation: preprocess_fn(image, annotation, is_training))

    if is_training:
        hwc_to_chw = C.HWC2CHW()
        ds = ds.map(input_columns=["image", "annotation"],
                    output_columns=["image", "box", "label", "num_match_num"],
                    columns_order=["image", "box", "label", "num_match_num"],
                    operations=compose_map_func, python_multiprocessing=True, num_parallel_workers=num_parallel_workers)
        ds = ds.map(input_columns=["image"], operations=hwc_to_chw, python_multiprocessing=True,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.repeat(repeat_num)
    else:
        hwc_to_chw = C.HWC2CHW()
        ds = ds.map(input_columns=["image", "annotation"],
                    output_columns=["image", "image_shape", "annotation"],
                    columns_order=["image", "image_shape", "annotation"],
                    operations=compose_map_func)
        ds = ds.map(input_columns=["image"], operations=hwc_to_chw, num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.repeat(repeat_num)
    return ds
