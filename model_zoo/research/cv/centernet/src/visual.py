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
"""
Data operations, will be used in train.py
"""

import os
import json
import random
import cv2
import numpy as np
import pycocotools.coco as COCO
from .config import dataset_config as data_cfg
from .image import get_affine_transform, affine_transform

_NUM_JOINTS = data_cfg.num_joints

def coco_box_to_bbox(box):
    """convert height/width to position coordinates"""
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
    return bbox

def resize_image(image, anns, width, height):
    """resize image to specified scale"""
    h, w = image.shape[0], image.shape[1]
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    s = max(image.shape[0], image.shape[1]) * 1.0
    trans_output = get_affine_transform(c, s, 0, [width, height])
    out_img = cv2.warpAffine(image, trans_output, (width, height), flags=cv2.INTER_LINEAR)

    num_objects = len(anns)
    resize_anno = []
    for i in range(num_objects):
        ann = anns[i]
        bbox = coco_box_to_bbox(ann['bbox'])
        pts = np.array(ann['keypoints'], np.float32).reshape(_NUM_JOINTS, 3)

        bbox[:2] = affine_transform(bbox[:2], trans_output)
        bbox[2:] = affine_transform(bbox[2:], trans_output)
        bbox[0::2] = np.clip(bbox[0::2], 0, width - 1)
        bbox[1::2] = np.clip(bbox[1::2], 0, height - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if (h > 0 and w > 0):
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            for j in range(_NUM_JOINTS):
                pts[j, :2] = affine_transform(pts[j, :2], trans_output)

            bbox = [ct[0] - w / 2, ct[1] - h / 2, w, h, 1]
            keypoints = pts.reshape(_NUM_JOINTS * 3).tolist()
            ann["bbox"] = bbox
            ann["keypoints"] = keypoints
            gt = ann
            resize_anno.append(gt)
    return out_img, resize_anno


def merge_pred(ann_path, mode="val", name="merged_annotations"):
    """merge annotation info of each image together"""
    files = os.listdir(ann_path)
    data_files = []
    for file_name in files:
        if "json" in file_name:
            data_files.append(os.path.join(ann_path, file_name))
    pred = {"images": [], "annotations": []}
    for file in data_files:
        anno = json.load(open(file, 'r'))
        if "images" in anno:
            for img in anno["images"]:
                pred["images"].append(img)
        if "annotations" in anno:
            for ann in anno["annotations"]:
                pred["annotations"].append(ann)
    json.dump(pred, open('{}/{}_{}.json'.format(ann_path, name, mode), 'w'))


def visual(ann_path, image_path, save_path, ratio=1, mode="val", name="merged_annotations"):
    """visulize all images based on dataset and annotations info"""
    merge_pred(ann_path, mode, name)
    ann_path = os.path.join(ann_path, name + '_' + mode + '.json')
    visual_allimages(ann_path, image_path, save_path, ratio)


def visual_allimages(anno_file, image_path, save_path, ratio=1):
    """visualize all images and annotations info"""
    coco = COCO.COCO(anno_file)
    image_ids = coco.getImgIds()
    images = []
    anns = {}
    for img_id in image_ids:
        idxs = coco.getAnnIds(imgIds=[img_id])
        if idxs:
            images.append(img_id)
            anns[img_id] = idxs

    for img_id in images:
        file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(image_path, file_name)
        annos = coco.loadAnns(anns[img_id])
        img = cv2.imread(img_path)
        return visual_image(img, annos, save_path, ratio)


def visual_image(img, annos, save_path, ratio=None, height=None, width=None, name=None, score_threshold=0.01):
    """visualize image and annotations info"""
    # annos: list type, in which all the element is dict
    h, w = img.shape[0], img.shape[1]
    if height is not None and width is not None and (height != h or width != w):
        img, annos = resize_image(img, annos, width, height)
    elif ratio not in (None, 1):
        img, annos = resize_image(img, annos, w * ratio, h * ratio)

    h, w = img.shape[0], img.shape[1]
    num_objects = len(annos)
    num = 0

    def define_color(pair):
        """define line color"""
        left_part = [0, 1, 3, 5, 7, 9, 11, 13, 15]
        right_part = [0, 2, 4, 6, 8, 10, 12, 14, 16]
        if pair[0] in left_part and pair[1] in left_part:
            color = (255, 0, 0)
        elif pair[0] in right_part and pair[1] in right_part:
            color = (0, 0, 255)
        else:
            color = (139, 0, 255)
        return color

    def visible(a, w, h):
        return a[0] >= 0 and a[0] < w and a[1] >= 0 and a[1] < h

    for i in range(num_objects):
        ann = annos[i]
        bbox = coco_box_to_bbox(ann['bbox'])
        if "score" in ann and (ann["score"] >= score_threshold or num == 0):
            num += 1
            txt = ("p" + "{:.2f}".format(ann["score"]))
            cv2.putText(img, txt, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        ct = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        cv2.circle(img, ct, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        bbox = np.array(bbox, dtype=np.int32).tolist()
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        keypoints = ann["keypoints"]
        keypoints = np.array(keypoints, dtype=np.int32).reshape(_NUM_JOINTS, 3).tolist()

        for pair in data_cfg.edges:
            partA = pair[0]
            partB = pair[1]
            color = define_color(pair)
            p_a = tuple(keypoints[partA][:2])
            p_b = tuple(keypoints[partB][:2])
            mask_a = keypoints[partA][2]
            mask_b = keypoints[partB][2]
            if (visible(p_a, w, h) and visible(p_b, w, h) and mask_a * mask_b > 0):
                cv2.line(img, p_a, p_b, color, 2)
                cv2.circle(img, p_a, 3, color, thickness=-1, lineType=cv2.FILLED)
                cv2.circle(img, p_b, 3, color, thickness=-1, lineType=cv2.FILLED)

    img_id = annos[0]["image_id"] if annos and "image_id" in annos[0] else random.randint(0, 9999999)
    image_name = "cv_image_" + str(img_id) + ".png" if name is None else "cv_image_" + str(img_id) + name + ".png"
    cv2.imwrite("{}/{}".format(save_path, image_name), img)
